"""Los nodos del grafo. Cada uno lee el estado, hace su trabajo y lo actualiza."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from core.prompts import EXTEND_SUMMARY_PROMPT, SUMMARY_PROMPT
from graph.state import AgentState
from graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
    get_voice_response_chain,
)
from graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from modules.memory.cache import get_cached_response, set_cached_response
from modules.memory.long_term import get_memory_manager
from settings import settings
from tools import logger
from tools.tracing import observe


def _ultimo_humano(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _historial_breve(state: AgentState, n: int = 4) -> str:
    partes = []
    for m in state["messages"][-n:]:
        quien = "Usuario" if isinstance(m, HumanMessage) else "Aura"
        partes.append(quien + ": " + str(m.content))
    return "\n".join(partes)


@observe(name="memory_extraction")
async def memory_extraction_node(state: AgentState) -> dict:
    """Saca hechos del usuario y los guarda en la memoria a largo plazo."""
    texto = _ultimo_humano(state)
    hecho = await get_memory_manager().extract_and_store(texto, get_chat_model(0.0))
    if hecho:
        logger.info("Nuevo hecho aprendido: %s", hecho)
    return {}


@observe(name="router")
async def router_node(state: AgentState) -> dict:
    """Decide si la respuesta va en texto, imagen o voz."""
    ventana = state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]
    try:
        r = await get_router_chain().ainvoke({"messages": ventana})
        flujo = r.response_type
    except Exception as e:  # noqa: BLE001
        logger.warning("El router fallo (%s); se responde en texto", e)
        flujo = "conversation"
    logger.info("Router -> %s", flujo)
    return {"workflow": flujo}


@observe(name="memory_injection")
async def memory_injection_node(state: AgentState) -> dict:
    """Recupera lo que sabemos del usuario y lo deja listo para el prompt."""
    contexto = await get_memory_manager().format_context(_ultimo_humano(state))
    return {"memory_context": contexto}


@observe(name="conversation")
async def conversation_node(state: AgentState) -> dict:
    """Responde en texto. Es la unica rama que usa cache."""
    pregunta = _ultimo_humano(state)
    primer_turno = len(state["messages"]) <= 1

    if primer_turno:
        cacheada = await get_cached_response(pregunta)
        if cacheada:
            return {"messages": [AIMessage(content=cacheada)]}

    cadena = get_character_response_chain(state.get("summary", ""))
    r = await cadena.ainvoke({
        "messages": state["messages"],
        "memory_context": state.get("memory_context", "- (nada aun)"),
    })
    respuesta = r.content.strip()

    if primer_turno:
        await set_cached_response(pregunta, respuesta)
    return {"messages": [AIMessage(content=respuesta)]}


@observe(name="image")
async def image_node(state: AgentState) -> dict:
    """Genera una imagen y contesta con una frase natural.

    La ruta va en `image_path`, NUNCA en el texto: si se cuela en el historial,
    el modelo empieza a inventarse ficheros en los turnos siguientes.
    """
    tti = get_text_to_image_module()
    llm = get_chat_model(0.5)

    escena = await tti.create_scenario(_historial_breve(state), llm)
    prompt = await tti.enhance_prompt(escena.image_prompt, llm)
    ruta = await asyncio.to_thread(tti.generate, prompt)

    texto = escena.narrative.strip()
    if not texto:
        cadena = get_character_response_chain(
            state.get("summary", ""),
            "Acabas de crear esa imagen y se la estas ensenando. Comentala en una o dos "
            "frases naturales. No menciones ficheros, rutas ni que eres una IA.",
        )
        r = await cadena.ainvoke({
            "messages": state["messages"],
            "memory_context": state.get("memory_context", "- (nada aun)"),
        })
        texto = r.content.strip()
    return {"messages": [AIMessage(content=texto)], "image_path": str(ruta)}


@observe(name="audio")
async def audio_node(state: AgentState) -> dict:
    """Responde con voz. El WAV va en el estado, no en el texto."""
    # Salida estructurada, no prompt: ver get_voice_response_chain para el porque
    # y las mediciones que llevaron a esta decision.
    #
    # Se le pasa una ventana CORTA del historial, no todo. Con el historial entero
    # el modelo copiaba palabra por palabra su propia respuesta anterior: si el
    # turno previo fue una imagen, la locucion repetia la descripcion del dibujo.
    ventana = state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]
    r = await get_voice_response_chain(state.get("summary", "")).ainvoke({
        "messages": ventana,
        "memory_context": state.get("memory_context", "- (nada aun)"),
    })
    texto = r.frase.strip()

    # Guarda por si aun asi se repite: se pide de nuevo diciendoselo.
    anterior = next(
        (m.content for m in reversed(state["messages"][:-1])
         if isinstance(m, AIMessage)), ""
    )
    if anterior and texto[:60].lower() == str(anterior)[:60].lower():
        logger.warning("La locucion repetia la respuesta anterior; se reintenta")
        r = await get_voice_response_chain(
            state.get("summary", "")
            + "\n\nNO repitas lo que acabas de decir. Di algo NUEVO."
        ).ainvoke({
            "messages": state["messages"][-1:],
            "memory_context": state.get("memory_context", "- (nada aun)"),
        })
        texto = r.frase.strip()
    wav = await asyncio.to_thread(get_text_to_speech_module().synthesize, texto)
    return {"messages": [AIMessage(content=texto)], "audio_buffer": wav}


@observe(name="summarize")
async def summarize_conversation_node(state: AgentState) -> dict:
    """Resume y recorta el historial para que el contexto no crezca sin fin."""
    previo = state.get("summary", "")
    instruccion = (
        EXTEND_SUMMARY_PROMPT.format(summary=previo) if previo else SUMMARY_PROMPT
    )
    r = await get_chat_model(0.2).ainvoke(
        [*state["messages"], HumanMessage(content=instruccion)]
    )
    resumen = r.content.strip()

    conservar = settings.TOTAL_MESSAGES_AFTER_SUMMARY
    borrar = [RemoveMessage(id=m.id) for m in state["messages"][:-conservar]]
    logger.info("Conversacion resumida; se conservan %d mensajes", conservar)
    return {"summary": resumen, "messages": borrar}
