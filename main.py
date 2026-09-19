"""API de integracion con FastAPI.

    python main.py          <- ASI en Windows
    uvicorn main:app        <- solo vale en Linux/macOS

En Windows hay que arrancarla con `python main.py`. El motivo: psycopg en modo
async no funciona sobre el ProactorEventLoop, que es el que usa Windows por
defecto, y con `uvicorn main:app` el bucle YA esta creado cuando se importa este
modulo, asi que fijar la politica aqui arriba llega tarde. El bloque
`if __name__ == "__main__"` del final la fija ANTES de levantar el servidor.

Endpoints:
    GET  /               estado del servicio
    POST /chat/text      mensaje de texto
    POST /chat/image     imagen + prompt opcional
    POST /chat/file      documento + prompt opcional
    POST /chat/audio     audio + se transcribe y se responde
"""

import asyncio
import base64
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from graph import graph_builder
from graph.utils.helpers import get_image_to_text_module, get_speech_to_text_module
from modules.memory.cache import init_cache
from modules.memory.long_term import get_vector_store
from settings import settings
from tools import logger
from tools.tracing import flush, traza_conversacion

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

IMAGENES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class ChatRequest(BaseModel):
    message: str = Field(description="lo que dice el usuario")
    thread_id: str = Field(default="api", description="identificador de la conversacion")


class ChatResponse(BaseModel):
    response: str
    workflow: str
    thread_id: str
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_cache()
    get_vector_store()
    logger.info("API lista")
    yield
    # Al apagar la API, lo que quede en el buffer de LangFuse se envia. Sin esto
    # las trazas del ultimo rato se pierden en silencio.
    flush()


app = FastAPI(title="Chat API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def execute_graph_workflow(content: str, thread_id: str) -> ChatResponse:
    """Pasa el contenido por el grafo y empaqueta la respuesta."""
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="El mensaje esta vacio")
    try:
        async with AsyncPostgresSaver.from_conn_string(settings.URI_PSYCOPG) as saver:
            grafo = graph_builder.compile(checkpointer=saver)
            with traza_conversacion(entrada=content, sesion=thread_id) as traza:
                estado = await grafo.ainvoke(
                    {"messages": [HumanMessage(content=content)]},
                    {"configurable": {"thread_id": thread_id}},
                )
                if traza is not None:
                    traza.update(output=estado["messages"][-1].content)
    except Exception as e:  # noqa: BLE001
        logger.exception("Fallo el grafo")
        raise HTTPException(status_code=500, detail=str(e)) from e

    imagen = None
    if estado.get("image_path"):
        p = Path(estado["image_path"])
        if p.exists():
            imagen = base64.b64encode(p.read_bytes()).decode()

    audio = None
    if estado.get("audio_buffer"):
        audio = base64.b64encode(estado["audio_buffer"]).decode()

    return ChatResponse(
        response=estado["messages"][-1].content,
        workflow=estado.get("workflow", "conversation"),
        thread_id=thread_id,
        image_base64=imagen,
        audio_base64=audio,
    )


async def _guardar_temporal(fichero: UploadFile) -> Path:
    sufijo = Path(fichero.filename or "adjunto").suffix or ".bin"
    destino = Path(tempfile.mktemp(suffix=sufijo))
    destino.write_bytes(await fichero.read())
    return destino


@app.get("/")
async def health() -> dict:
    return {
        "message": "healthy",
        "modelo": settings.TEXT_MODEL_NAME,
        "vision": settings.ITT_MODEL_NAME,
        "recuerdos": get_vector_store().count(),
    }


@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(req: ChatRequest) -> ChatResponse:
    return await execute_graph_workflow(req.message, req.thread_id)


@app.post("/chat/image", response_model=ChatResponse)
async def chat_image(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    thread_id: str = Form("api"),
) -> ChatResponse:
    ruta = await _guardar_temporal(file)
    try:
        desc = await asyncio.to_thread(
            get_image_to_text_module().describe_image, ruta, prompt
        )
    finally:
        ruta.unlink(missing_ok=True)
    contenido = f"{prompt}\n\n[Imagen adjunta] {desc}".strip()
    return await execute_graph_workflow(contenido, thread_id)


@app.post("/chat/file", response_model=ChatResponse)
async def chat_file(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    thread_id: str = Form("api"),
) -> ChatResponse:
    ruta = await _guardar_temporal(file)
    try:
        desc = await asyncio.to_thread(
            get_image_to_text_module().describe_file, ruta, prompt
        )
    finally:
        ruta.unlink(missing_ok=True)
    contenido = f"{prompt}\n\n[Documento '{file.filename}'] {desc}".strip()
    return await execute_graph_workflow(contenido, thread_id)


@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    file: UploadFile = File(...),
    thread_id: str = Form("api"),
) -> ChatResponse:
    ruta = await _guardar_temporal(file)
    try:
        texto = await asyncio.to_thread(
            get_speech_to_text_module().transcribe, ruta
        )
    finally:
        ruta.unlink(missing_ok=True)
    if not texto:
        raise HTTPException(status_code=422, detail="No se entendio nada del audio")
    return await execute_graph_workflow(texto, thread_id)


if __name__ == "__main__":
    import uvicorn

    servidor = uvicorn.Server(
        uvicorn.Config(
            app,
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            # 'none' = no montes tu propio bucle. Con cualquier otro valor uvicorn
            # reimpone el ProactorEventLoop en Windows y psycopg async deja de
            # conectar, aunque hayamos fijado la politica antes.
            loop="none",
        )
    )

    if sys.platform == "win32":
        # El bucle lo creamos NOSOTROS, del tipo que psycopg necesita.
        asyncio.run(servidor.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(servidor.serve())
