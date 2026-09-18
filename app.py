"""Front de pruebas con Chainlit.

Habla por texto o por voz, sube imagenes y ficheros, y pide imagenes.
Arranque:

    chainlit run app.py -w
"""

import asyncio
import io
import selectors
import sys
import tempfile
import wave
from pathlib import Path

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from graph import graph_builder
from graph.utils.helpers import get_image_to_text_module, get_speech_to_text_module
from modules.memory.cache import init_cache
from settings import settings
from tools import logger
from tools.tracing import flush

# En Windows el bucle por defecto (Proactor) hace que psycopg en modo async
# falle al conectar. Chainlit crea su propio bucle, asi que hay que fijar la
# politica ANTES de que arranque.
if sys.platform == "win32":
    class _Politica(asyncio.WindowsSelectorEventLoopPolicy):  # type: ignore[misc]
        pass

    asyncio.set_event_loop_policy(_Politica())

IMAGENES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


async def _ejecutar(contenido: str, thread_id: str) -> dict:
    """Pasa el mensaje por el grafo con el checkpointer en PostgreSQL."""
    async with AsyncPostgresSaver.from_conn_string(settings.URI_PSYCOPG) as saver:
        app = graph_builder.compile(checkpointer=saver)
        return await app.ainvoke(
            {"messages": [HumanMessage(content=contenido)]},
            {"configurable": {"thread_id": thread_id}},
        )


async def _responder(estado: dict) -> None:
    """Envia a la interfaz lo que haya producido el grafo."""
    texto = estado["messages"][-1].content
    elementos = []

    if estado.get("image_path"):
        elementos.append(cl.Image(path=estado["image_path"], display="inline",
                                  name="imagen"))
    if estado.get("audio_buffer"):
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        tmp.write_bytes(estado["audio_buffer"])
        elementos.append(cl.Audio(path=str(tmp), mime="audio/wav", name="voz",
                                  auto_play=True))

    await cl.Message(content=texto, elements=elementos).send()
    flush()


@cl.on_chat_start
async def on_chat_start() -> None:
    await init_cache()
    cl.user_session.set("thread_id", cl.context.session.id)
    await cl.Message(
        content=(
            "Hola, soy Aura. Puedes escribirme, mandarme un audio con el microfono, "
            "subirme una imagen o un PDF, o pedirme que te dibuje algo.\n\n"
            "Todo corre en local: no sale nada de este ordenador."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Texto, imagenes y ficheros adjuntos."""
    contenido = message.content or ""

    for el in message.elements or []:
        ruta = Path(getattr(el, "path", "") or "")
        if not ruta.exists():
            continue
        itt = get_image_to_text_module()
        async with cl.Step(name="Leyendo el adjunto", type="tool"):
            if ruta.suffix.lower() in IMAGENES:
                desc = await asyncio.to_thread(itt.describe_image, ruta, contenido)
                contenido += f"\n\n[Imagen adjunta] {desc}"
            else:
                desc = await asyncio.to_thread(itt.describe_file, ruta, contenido)
                contenido += f"\n\n[Documento '{ruta.name}'] {desc}"

    if not contenido.strip():
        return

    async with cl.Step(name="Pensando", type="run"):
        estado = await _ejecutar(contenido, cl.user_session.get("thread_id"))
    await _responder(estado)


@cl.on_audio_start
async def on_audio_start() -> bool:
    cl.user_session.set("audio_buffer", io.BytesIO())
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:
    buf = cl.user_session.get("audio_buffer")
    if buf is None:
        buf = io.BytesIO()
        cl.user_session.set("audio_buffer", buf)
    buf.write(chunk.data)


@cl.on_audio_end
async def on_audio_end() -> None:
    """Cuando el usuario suelta el microfono: transcribir y pasar por el grafo."""
    buf: io.BytesIO | None = cl.user_session.get("audio_buffer")
    if buf is None or buf.getbuffer().nbytes == 0:
        logger.warning("Audio vacio")
        return

    buf.seek(0)
    crudo = buf.read()
    cl.user_session.set("audio_buffer", None)

    # Chainlit entrega PCM 16 bits mono a 24 kHz; hay que envolverlo en WAV.
    wav_path = Path(tempfile.mktemp(suffix=".wav"))
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(crudo)

    await cl.Message(
        content="", elements=[cl.Audio(path=str(wav_path), mime="audio/wav",
                                       name="tu mensaje")]
    ).send()

    async with cl.Step(name="Transcribiendo", type="tool"):
        texto = await asyncio.to_thread(
            get_speech_to_text_module().transcribe, wav_path
        )
    if not texto:
        await cl.Message(content="No he entendido nada, perdona. Repites?").send()
        return

    await cl.Message(content=texto, author="Tu (voz)").send()
    async with cl.Step(name="Pensando", type="run"):
        estado = await _ejecutar(texto, cl.user_session.get("thread_id"))
    await _responder(estado)
