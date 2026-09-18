"""Las cadenas LCEL del agente: enrutado y respuesta en personaje."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from graph.utils.helpers import get_chat_model


class RouterResponse(BaseModel):
    """Salida estructurada del router: no puede devolver nada fuera de estos tres."""

    response_type: Literal["conversation", "image", "audio"] = Field(
        description="tipo de respuesta que toca dar"
    )


def get_router_chain():
    """Decide si la respuesta va en texto, imagen o voz."""
    modelo = get_chat_model(temperature=0.0).with_structured_output(RouterResponse)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | modelo


class Locucion(BaseModel):
    """Lo que Aura va a decir en voz alta."""

    frase: str = Field(
        description=(
            "Las palabras exactas que se van a pronunciar en voz alta. Debe RESPONDER "
            "de verdad a lo ultimo que ha dicho el usuario y aportar algo: una pregunta, "
            "un comentario o una opinion. Entre 10 y 40 palabras, espanol natural.\n"
            "No vale despachar con 'claro, un momento' ni con 'ya te lo envio': la "
            "locucion YA se esta grabando, asi que no se anuncia, se dice.\n"
            "Sin acotaciones, sin corchetes, sin comillas, sin nombres de fichero y sin "
            "mencionar limitaciones tecnicas."
        )
    )


def get_voice_response_chain(summary: str = ""):
    """La cadena de la rama de audio.

    Va con SALIDA ESTRUCTURADA a proposito. Con un prompt normal, por muy explicito
    que fuera, el modelo contestaba "no puedo enviar audios" mientras Piper le
    sintetizaba la voz: su prior de "soy un modelo de texto" gana a la instruccion.
    Medido sobre cinco peticiones distintas:

        afirmar la capacidad ..... 3/5 (colaba acotaciones tipo [Audio: ...])
        reformular como guion .... 2/5
        guion + ejemplos ......... 1/5 (copiaba los ejemplos literalmente)
        salida estructurada ...... 5/5

    El esquema no deja hueco para el meta-comentario, que es justo el problema.
    """
    sistema = CHARACTER_CARD_PROMPT
    if summary:
        sistema += f"\n\n# Resumen de lo hablado antes\n{summary}"
    sistema += "\n\n# Este turno\nEstas grabando un mensaje de voz para el usuario."
    prompt = ChatPromptTemplate.from_messages(
        [("system", sistema), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | get_chat_model(0.5).with_structured_output(Locucion)


def get_character_response_chain(summary: str = "", extra: str = ""):
    """La cadena que responde en personaje, con la memoria ya inyectada."""
    sistema = CHARACTER_CARD_PROMPT
    if summary:
        sistema += f"\n\n# Resumen de lo hablado antes\n{summary}"
    if extra:
        sistema += f"\n\n# Instrucciones para este turno\n{extra}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", sistema), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | get_chat_model()
