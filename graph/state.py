"""El estado compartido del grafo."""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Memoria de trabajo del grafo. Hereda `messages` de MessagesState.

    Los artefactos (imagen, audio) viajan por AQUI y nunca dentro del texto del
    mensaje: si una ruta se cuela en el historial, el modelo aprende el patron y
    empieza a inventarse nombres de fichero.
    """

    summary: str          # resumen de lo hablado, cuando la conversacion se alarga
    workflow: str         # conversation | image | audio
    memory_context: str   # hechos del usuario inyectados en el system prompt
    image_path: str       # ruta de la imagen generada en este turno
    audio_buffer: bytes   # WAV de la respuesta hablada en este turno
