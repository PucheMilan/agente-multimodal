"""Extraccion y recuperacion de hechos del usuario.

El LLM decide si un mensaje contiene un hecho personal. La salida es
estructurada a proposito: asi no puede inventarse campos ni formatos.
"""

from functools import lru_cache

from pydantic import BaseModel, Field

from core.prompts import MEMORY_ANALYSIS_PROMPT
from modules.memory.long_term.vector_store import Memory, get_vector_store
from settings import settings
from tools import logger


class AnalisisMemoria(BaseModel):
    """Resultado de analizar un mensaje del usuario."""

    is_important: bool = Field(description="si el mensaje contiene un hecho personal")
    formatted_memory: str | None = Field(
        default=None, description="el hecho en tercera persona, o null"
    )


class MemoryManager:
    """Puente entre la conversacion y el almacen vectorial."""

    def __init__(self) -> None:
        self.store = get_vector_store()

    async def extract_and_store(self, mensaje: str, llm) -> str | None:
        """Analiza el mensaje y guarda el hecho si lo hay. Devuelve el hecho."""
        if not mensaje or len(mensaje.strip()) < 3:
            return None
        try:
            analizador = llm.with_structured_output(AnalisisMemoria)
            r: AnalisisMemoria = await analizador.ainvoke(
                MEMORY_ANALYSIS_PROMPT.format(message=mensaje)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("No se pudo analizar la memoria: %s", e)
            return None

        if not r.is_important or not r.formatted_memory:
            return None
        hecho = r.formatted_memory.strip()
        if not hecho or len(hecho) > 120:
            return None
        await self.store.store_memory(hecho, {"origen": "conversacion"})
        return hecho

    async def get_relevant(self, consulta: str, k: int | None = None) -> list[Memory]:
        return await self.store.search_memories(consulta, k or settings.MEMORY_TOP_K)

    async def format_context(self, consulta: str) -> str:
        """Lo que se inyecta en el system prompt."""
        recuerdos = await self.get_relevant(consulta)
        if not recuerdos:
            return "- (todavia no se nada de esta persona)"
        return "\n".join(f"- {m.text}" for m in recuerdos)


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    return MemoryManager()
