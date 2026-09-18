"""
Prueba funcional del stack completo, con la topologia de grafo del enunciado.

No es el proyecto: es la maqueta que demuestra que las piezas encajan ANTES de
escribir los 30 ficheros del modulo 20. Todo local, coste 0 EUR.

    uv run --no-project --python .venv/Scripts/python.exe notebooks/prueba_funcional.py

Grafo:
    START -> memory_extraction -> router -> memory_injection
          -> {conversation | image | audio} -> (summarize?) -> END
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path
from typing import Literal

import ollama
import psycopg
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

URI = "postgresql://root:password@127.0.0.1:5432/agent"
MODELO_TEXTO = "qwen3:14b"
MODELO_VISION = "gemma3:4b"
MODELO_EMBED = "nomic-embed-text"
DIM = 768
VOZ = "es_ES-davefx-medium"
RESUMIR_A_PARTIR_DE = 6          # el enunciado usa 20; aqui menos para poder verlo
RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
SALIDA = RAIZ / "generated_images"

_paso = 0


def paso(txt: str) -> None:
    global _paso
    _paso += 1
    print(f"\n{'=' * 74}\n[{_paso}] {txt}\n{'=' * 74}")


def ok(txt: str) -> None:
    print(f"   OK  {txt}")


def info(txt: str) -> None:
    print(f"       {txt}")


# --------------------------------------------------------------------------
# Memoria larga sobre pgvector
# --------------------------------------------------------------------------

class MemoriaLarga:
    """Vector store minimo sobre pgvector: guardar hechos y recuperarlos."""

    def __init__(self, uri: str):
        self.uri = uri
        self.emb = OllamaEmbeddings(model=MODELO_EMBED)
        with psycopg.connect(self.uri) as cx:
            cx.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cx.execute(
                f"""CREATE TABLE IF NOT EXISTS memoria_larga (
                       id       bigserial PRIMARY KEY,
                       texto    text NOT NULL,
                       creado   timestamptz DEFAULT now(),
                       vector   vector({DIM})
                   )"""
            )
            cx.commit()

    # Umbral MEDIDO con nomic-embed-text (17/09/2026):
    #   duplicados reales .... 0,097 - 0,195
    #   hechos distintos ..... 0,452 - 0,486
    # 0,30 cae en el hueco. Con 0,15 no deduplicaba 'Se llama X' / 'Su nombre es X'.
    def guardar(self, texto: str, umbral: float = 0.30) -> bool:
        """Guarda si no existe ya algo muy parecido. Devuelve True si guardo."""
        v = self.emb.embed_query(texto)
        with psycopg.connect(self.uri) as cx:
            register_vector(cx)
            fila = cx.execute(
                "SELECT texto, vector <=> %s::vector AS d FROM memoria_larga "
                "ORDER BY d LIMIT 1", (v,)
            ).fetchone()
            if fila and fila[1] < umbral:
                return False
            cx.execute("INSERT INTO memoria_larga (texto, vector) VALUES (%s, %s)", (texto, v))
            cx.commit()
            return True

    def buscar(self, consulta: str, k: int = 3) -> list[str]:
        v = self.emb.embed_query(consulta)
        with psycopg.connect(self.uri) as cx:
            register_vector(cx)
            filas = cx.execute(
                "SELECT texto FROM memoria_larga ORDER BY vector <=> %s::vector LIMIT %s",
                (v, k),
            ).fetchall()
        return [f[0] for f in filas]

    def limpiar(self) -> None:
        with psycopg.connect(self.uri) as cx:
            cx.execute("DROP TABLE IF EXISTS memoria_larga")
            cx.commit()


# --------------------------------------------------------------------------
# Modulos multimodales (los de modules/ en el proyecto real)
# --------------------------------------------------------------------------

class Voz:
    def __init__(self):
        from piper import PiperVoice
        from piper.download_voices import download_voice
        vd = DATA / "voices"
        vd.mkdir(parents=True, exist_ok=True)
        if not (vd / f"{VOZ}.onnx").exists():
            download_voice(VOZ, vd)
        self.voz = PiperVoice.load(vd / f"{VOZ}.onnx")
        self._whisper = None

    def sintetizar(self, texto: str, destino: Path) -> Path:
        with wave.open(str(destino), "wb") as w:
            self.voz.synthesize_wav(texto, w)
        return destino

    def transcribir(self, ruta: Path) -> str:
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(
                "small", device="cpu", compute_type="int8",
                download_root=str(DATA / "whisper"))
        segs, _ = self._whisper.transcribe(str(ruta), language="es")
        return "".join(s.text for s in segs).strip()


class Imagen:
    def __init__(self):
        self._pipe = None

    def generar(self, prompt: str, destino: Path) -> Path:
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo", dtype=torch.float32,
                cache_dir=str(DATA / "hf")).to("cpu")
            self._pipe.set_progress_bar_config(disable=True)
        img = self._pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        img.save(destino)
        return destino

    @staticmethod
    def describir(ruta: Path) -> str:
        r = ollama.chat(model=MODELO_VISION, messages=[{
            "role": "user",
            "content": "Describe en una frase que se ve en esta imagen.",
            "images": [ruta.read_bytes()]}])
        return r["message"]["content"].strip()


# --------------------------------------------------------------------------
# El grafo
# --------------------------------------------------------------------------

class Estado(MessagesState):
    summary: str
    workflow: str
    memory_context: str
    audio_path: str
    image_path: str


class RespuestaRouter(BaseModel):
    response_type: Literal["conversation", "image", "audio"] = Field(
        description="tipo de respuesta que toca dar")


PERSONAJE = (
    "Eres Aura, una asistente cercana y directa. Hablas en espanol de Espana, "
    "con frases cortas y naturales. No eres pomposa ni repites la pregunta.\n"
    "Lo que sabes del usuario de conversaciones anteriores:\n{memoria}\n"
    "Si no sabes algo, lo dices.\n"
    "REGLA IMPORTANTE: puedes enviar mensajes de voz y puedes enviar imagenes. "
    "NUNCA digas que no puedes mandar audios ni imagenes, porque si puedes."
)


def construir_grafo(llm, llm_router, memoria: MemoriaLarga, voz: Voz, imagen: Imagen):
    router = llm_router.with_structured_output(RespuestaRouter)
    trazas: list[str] = []

    async def memory_extraction_node(state: Estado):
        trazas.append("memory_extraction")
        ultimo = state["messages"][-1].content
        r = await llm.ainvoke(
            "Extrae UN hecho personal del usuario de este mensaje, en tercera persona "
            "y en 6 palabras o menos.\n"
            "Extrae SOLO hechos reales: nombre, lugar, trabajo, estudios, gustos, "
            "circunstancias vitales.\n"
            "NO son hechos las peticiones ni las ordenes: 'mandame un audio', "
            "'dibujame algo' o 'acuerdate de esto' NO se guardan.\n"
            "Si no hay ningun hecho personal, responde exactamente NADA.\n\n"
            f"Mensaje: {ultimo}")
        hecho = r.content.strip().strip('".')
        if hecho and hecho.upper() != "NADA" and len(hecho) < 90:
            if memoria.guardar(hecho):
                info(f"memoria + '{hecho}'")
            else:
                info(f"memoria = ya conocia algo igual ('{hecho}')")
        return {}

    async def router_node(state: Estado):
        trazas.append("router")
        ultimo = state["messages"][-1].content
        r = await router.ainvoke(
            f"Decide el tipo de respuesta para el mensaje del usuario: {ultimo}")
        info(f"router -> {r.response_type}")
        return {"workflow": r.response_type}

    async def memory_injection_node(state: Estado):
        trazas.append("memory_injection")
        recuerdos = memoria.buscar(state["messages"][-1].content, k=3)
        return {"memory_context": "\n".join(f"- {r}" for r in recuerdos) or "- (nada aun)"}

    def _sistema(state: Estado) -> str:
        s = PERSONAJE.format(memoria=state.get("memory_context", "-"))
        if state.get("summary"):
            s += f"\n\nResumen de lo hablado antes: {state['summary']}"
        return s

    async def conversation_node(state: Estado):
        trazas.append("conversation")
        r = await llm.ainvoke([{"role": "system", "content": _sistema(state)}, *state["messages"]])
        return {"messages": [AIMessage(content=r.content.strip())]}

    async def image_node(state: Estado):
        trazas.append("image")
        r = await llm.ainvoke(
            "Convierte esta peticion en un prompt VISUAL en ingles, una sola linea, "
            f"sin comillas ni explicaciones: {state['messages'][-1].content}")
        prompt = r.content.strip().split("\n")[0][:200]
        info(f"prompt de imagen: {prompt}")
        SALIDA.mkdir(exist_ok=True)
        destino = SALIDA / f"prueba_{int(time.time())}.png"
        t = time.time()
        await asyncio.to_thread(imagen.generar, prompt, destino)
        info(f"imagen generada en {time.time() - t:.1f}s -> {destino.name}")
        # El nombre del fichero va en el ESTADO, nunca en el texto del mensaje.
        # Si se cuela en el historial, el modelo aprende el patron y luego se inventa
        # rutas tipo /audio_123.mp3 cuando le piden voz. Por eso el enunciado tiene
        # image_path y audio_buffer como campos de estado: el artefacto lo adjunta la
        # capa de presentacion, no el LLM.
        return {"messages": [AIMessage(content="Te la he dibujado, aqui la tienes.")],
                "image_path": str(destino)}

    async def audio_node(state: Estado):
        trazas.append("audio")
        # Sin esta instruccion el modelo "manda" una ruta inventada tipo /audio_123.mp3,
        # que Piper lee literalmente. Es la regla del enunciado sobre respuestas en
        # texto plano, sin meta-comentarios: existe por este fallo exacto.
        r = await llm.ainvoke([{"role": "system", "content": _sistema(state) +
                                "\nTu respuesta se va a leer en voz alta TAL CUAL. "
                                "Escribe solo la frase que hay que pronunciar, en 25 "
                                "palabras o menos. Nada de rutas, nombres de fichero, "
                                "markdown ni acotaciones."},
                               *state["messages"]])
        texto = r.content.strip()
        destino = DATA / f"respuesta_{int(time.time())}.wav"
        await asyncio.to_thread(voz.sintetizar, texto, destino)
        info(f"audio sintetizado -> {destino.name}")
        return {"messages": [AIMessage(content=texto)], "audio_path": str(destino)}

    async def summarize_conversation_node(state: Estado):
        trazas.append("summarize")
        r = await llm.ainvoke(
            [*state["messages"],
             HumanMessage(content="Resume la conversacion en 2 frases, en tercera persona.")])
        info(f"resumen: {r.content.strip()[:90]}")
        # conserva los 2 ultimos mensajes
        from langchain_core.messages import RemoveMessage
        borrar = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": r.content.strip(), "messages": borrar}

    def select_workflow(state: Estado) -> Literal["conversation_node", "image_node", "audio_node"]:
        return {"image": "image_node", "audio": "audio_node"}.get(
            state.get("workflow", "conversation"), "conversation_node")

    def should_summarize(state: Estado) -> Literal["summarize_conversation_node", "__end__"]:
        if len(state["messages"]) > RESUMIR_A_PARTIR_DE:
            return "summarize_conversation_node"
        return END

    g = StateGraph(Estado)
    g.add_node("memory_extraction_node", memory_extraction_node)
    g.add_node("router_node", router_node)
    g.add_node("memory_injection_node", memory_injection_node)
    g.add_node("conversation_node", conversation_node)
    g.add_node("image_node", image_node)
    g.add_node("audio_node", audio_node)
    g.add_node("summarize_conversation_node", summarize_conversation_node)

    g.add_edge(START, "memory_extraction_node")
    g.add_edge("memory_extraction_node", "router_node")
    g.add_edge("router_node", "memory_injection_node")
    g.add_conditional_edges("memory_injection_node", select_workflow)
    for n in ("conversation_node", "image_node", "audio_node"):
        g.add_conditional_edges(n, should_summarize)
    g.add_edge("summarize_conversation_node", END)
    return g, trazas


# --------------------------------------------------------------------------
# La prueba
# --------------------------------------------------------------------------

async def main() -> int:
    fallos: list[str] = []
    t0 = time.time()

    paso("Servicios en marcha")
    try:
        v = ollama.list()
        nombres = {m.model for m in v.models}
        ok(f"Ollama responde · {len(nombres)} modelos")
        for m in (MODELO_TEXTO, MODELO_VISION, MODELO_EMBED):
            if not any(n.startswith(m.split(":")[0]) for n in nombres):
                fallos.append(f"falta el modelo {m}")
            else:
                info(f"modelo disponible: {m}")
    except Exception as e:
        fallos.append(f"Ollama no responde: {e}")
        print("   FALLO  Ollama no responde. Lanza 'ollama app.exe' y reintenta.")
        return 1
    try:
        with psycopg.connect(URI) as cx:
            ver = cx.execute("SELECT version()").fetchone()[0].split(",")[0]
        ok(f"PostgreSQL responde · {ver}")
    except Exception as e:
        fallos.append(f"PostgreSQL no responde: {e}")
        print("   FALLO  Arranca la base: podman machine start && podman start agente-pg")
        return 1

    llm = ChatOllama(model=MODELO_TEXTO, temperature=0.3, reasoning=False)
    llm_router = ChatOllama(model=MODELO_TEXTO, temperature=0, reasoning=False)

    paso("Memoria larga sobre pgvector")
    memoria = MemoriaLarga(URI)
    memoria.limpiar()
    memoria = MemoriaLarga(URI)
    memoria.guardar("Se llama Ana")
    memoria.guardar("Vive en Sevilla")
    dup = memoria.guardar("Su nombre es Ana")
    ok("guardados 2 hechos en pgvector")
    info(f"deduplicacion por similitud: {'funciona' if not dup else 'NO detecto el duplicado'}")
    if dup:
        fallos.append("la deduplicacion de memoria no detecto un hecho repetido")
    rec = memoria.buscar("como se llama el usuario?", k=1)
    ok(f"recuperado por similitud: {rec}")
    if not rec or "Ana" not in rec[0]:
        fallos.append("la memoria no recupero el hecho correcto")

    paso("Voz: sintetizar y volver a transcribir")
    voz = Voz()
    frase = "Me llamo Ana y me interesa la ciberseguridad."
    wav = DATA / "prueba_entrada.wav"
    t = time.time()
    voz.sintetizar(frase, wav)
    with wave.open(str(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    ok(f"Piper: {dur:.1f}s de audio en {time.time() - t:.2f}s")
    t = time.time()
    transcrito = voz.transcribir(wav)
    ok(f"Whisper en {time.time() - t:.1f}s: '{transcrito}'")
    if "Ana" not in transcrito:
        fallos.append("la transcripcion no recupero el contenido")

    paso("Construccion del grafo y persistencia en Postgres")
    imagen = Imagen()
    g, trazas = construir_grafo(llm, llm_router, memoria, voz, imagen)

    async with AsyncPostgresSaver.from_conn_string(URI) as saver:
        await saver.setup()
        app = g.compile(checkpointer=saver)
        ok(f"grafo compilado con {len(app.get_graph().nodes)} nodos y checkpointer en Postgres")
        cfg = {"configurable": {"thread_id": "prueba-funcional"}}

        async def turno(texto: str, etiqueta: str):
            print(f"\n   --- {etiqueta} ---")
            print(f"   usuario > {texto}")
            t = time.time()
            r = await app.ainvoke({"messages": [HumanMessage(content=texto)]}, cfg)
            print(f"   bot     > {r['messages'][-1].content[:220]}")
            info(f"({time.time() - t:.1f}s · rama '{r.get('workflow')}')")
            return r

        paso("Turno 1 · texto (rama conversation)")
        r1 = await turno(transcrito, "entrada venida de la voz transcrita")
        if r1.get("workflow") != "conversation":
            fallos.append(f"turno 1 fue a la rama '{r1.get('workflow')}', se esperaba conversation")

        paso("Turno 2 · el bot debe recordar")
        r2 = await turno("Te acuerdas de como me llamo?", "prueba de memoria larga")
        if "ana" not in r2["messages"][-1].content.lower():
            fallos.append("el bot no recupero el nombre de la memoria larga")
        else:
            ok("recordo el nombre desde pgvector")

        paso("Turno 3 · peticion de imagen (rama image)")
        r3 = await turno("Dibujame un faro en un acantilado al atardecer", "generacion de imagen")
        if r3.get("workflow") != "image":
            fallos.append(f"turno 3 fue a la rama '{r3.get('workflow')}', se esperaba image")
        else:
            ruta = Path(r3["image_path"])
            if ruta.exists() and ruta.stat().st_size > 20_000:
                ok(f"imagen en disco: {ruta.name} ({ruta.stat().st_size // 1024} KB)")
                d = await asyncio.to_thread(Imagen.describir, ruta)
                ok(f"vision la describe: '{d[:120]}'")
            else:
                fallos.append("la imagen no se escribio correctamente")

        paso("Turno 4 · peticion de voz (rama audio)")
        r4 = await turno("Respondeme con un audio, quiero oir tu voz", "sintesis de voz")
        if r4.get("workflow") != "audio":
            fallos.append(f"turno 4 fue a la rama '{r4.get('workflow')}', se esperaba audio")
        else:
            wav_out = Path(r4["audio_path"])
            if wav_out.exists():
                with wave.open(str(wav_out)) as w:
                    d = w.getnframes() / w.getframerate()
                ok(f"audio en disco: {wav_out.name} ({d:.1f}s)")
                vuelta = await asyncio.to_thread(voz.transcribir, wav_out)
                ok(f"y se vuelve a entender: '{vuelta[:90]}'")
                dicho = r4["messages"][-1].content
                if any(x in dicho.lower() for x in
                       ("no puedo enviar", "no puedo mandar", "no puedo enviarte")):
                    fallos.append("el bot NIEGA poder mandar audio mientras lo esta mandando")
                elif any(x in dicho.lower() for x in (".mp3", ".wav", ".ogg")) or "/" in dicho:
                    fallos.append(f"el bot devolvio una ruta en vez de habla: '{dicho[:60]}'")
                elif len(dicho.split()) < 4:
                    fallos.append(f"la respuesta de voz es demasiado corta: '{dicho}'")
                else:
                    ok("es habla de verdad: ni niega poder mandarlo, ni inventa una ruta")
            else:
                fallos.append("el audio no se escribio")

        paso("Persistencia: recuperar el hilo desde Postgres")
        estado = await app.aget_state(cfg)
        n = len(estado.values["messages"])
        ok(f"el hilo tiene {n} mensajes guardados en el checkpointer")
        if estado.values.get("summary"):
            ok(f"el nodo de resumen se disparo: '{estado.values['summary'][:90]}'")
        else:
            info("el resumen aun no se disparo (depende del numero de mensajes)")

    paso("Resultado")
    print(f"   nodos recorridos: {' -> '.join(trazas)}")
    print(f"   tiempo total: {time.time() - t0:.0f}s")
    print(f"   coste: 0,00 EUR")
    if fallos:
        print(f"\n   {len(fallos)} FALLO(S):")
        for f in fallos:
            print(f"     - {f}")
        return 1
    print("\n   TODO CORRECTO: el stack libre funciona de punta a punta.")
    return 0


if __name__ == "__main__":
    # En Windows el bucle por defecto es ProactorEventLoop y psycopg en modo async
    # lo rechaza. Hay que forzar SelectorEventLoop. Afecta a AsyncPostgresSaver,
    # asi que el proyecto real (parte 2 del enunciado) necesitara lo mismo.
    if sys.platform == "win32":
        raise SystemExit(asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop))
    raise SystemExit(asyncio.run(main()))
