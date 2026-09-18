"""Memoria a largo plazo sobre PostgreSQL + pgvector.

Guarda hechos del usuario como vectores y los recupera por similitud.
Deduplica antes de insertar: sin eso, cada vez que el usuario repite algo
se crea una entrada nueva y la memoria se llena de ruido.
"""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import psycopg
from langchain_ollama import OllamaEmbeddings
from pgvector.psycopg import register_vector, register_vector_async

from core.exceptions import MemoryError_
from settings import settings
from tools import logger


@dataclass
class Memory:
    """Un hecho guardado en la memoria a largo plazo."""

    id: int
    text: str
    created_at: datetime
    score: float | None = None


class PostgreSQLVectorStore:
    """Almacen vectorial. Singleton: una sola instancia por proceso."""

    _instancia: "PostgreSQLVectorStore | None" = None

    def __new__(cls) -> "PostgreSQLVectorStore":
        if cls._instancia is None:
            cls._instancia = super().__new__(cls)
            cls._instancia._listo = False
        return cls._instancia

    def __init__(self) -> None:
        if self._listo:
            return
        self.uri = settings.URI_PSYCOPG
        self.emb = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
        )
        self._create_tables()
        self._listo = True

    # ---------------------------------------------------------------- tablas

    def _create_tables(self) -> None:
        try:
            with psycopg.connect(self.uri) as cx:
                cx.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cx.execute(
                    f"""CREATE TABLE IF NOT EXISTS long_term_memory (
                            id         bigserial PRIMARY KEY,
                            text       text NOT NULL,
                            metadata   jsonb DEFAULT '{{}}'::jsonb,
                            created_at timestamptz NOT NULL DEFAULT now(),
                            embedding  vector({settings.EMBEDDING_DIM})
                        )"""
                )
                cx.execute(
                    "CREATE INDEX IF NOT EXISTS long_term_memory_emb_idx "
                    "ON long_term_memory USING hnsw (embedding vector_cosine_ops)"
                )
                cx.commit()
            logger.info("Memoria a largo plazo lista (pgvector)")
        except Exception as e:  # noqa: BLE001
            raise MemoryError_(f"No se pudo preparar la memoria: {e}") from e

    # ------------------------------------------------------------------ api

    async def find_similar_memory(self, texto: str) -> Memory | None:
        """Devuelve el recuerdo mas parecido si esta por debajo del umbral."""
        v = await self.emb.aembed_query(texto)
        async with await psycopg.AsyncConnection.connect(self.uri) as cx:
            await register_vector_async(cx)
            cur = await cx.execute(
                "SELECT id, text, created_at, embedding <=> %s::vector AS d "
                "FROM long_term_memory ORDER BY d LIMIT 1",
                (v,),
            )
            fila = await cur.fetchone()
        if fila and fila[3] < settings.SIMILARITY_THRESHOLD:
            return Memory(id=fila[0], text=fila[1], created_at=fila[2], score=fila[3])
        return None

    async def store_memory(self, memory_text: str, metadata: dict | None = None) -> bool:
        """Guarda el hecho si no habia ya uno equivalente. True si guardo."""
        import json

        existente = await self.find_similar_memory(memory_text)
        if existente:
            logger.info("Memoria repetida, no se guarda: '%s'", memory_text)
            return False
        v = await self.emb.aembed_query(memory_text)
        async with await psycopg.AsyncConnection.connect(self.uri) as cx:
            await register_vector_async(cx)
            await cx.execute(
                "INSERT INTO long_term_memory (text, metadata, embedding) "
                "VALUES (%s, %s::jsonb, %s)",
                (memory_text, json.dumps(metadata or {}), v),
            )
            await cx.commit()
        logger.info("Memoria guardada: '%s'", memory_text)
        return True

    async def search_memories(self, query: str, k: int | None = None) -> list[Memory]:
        """Los k recuerdos mas parecidos a la consulta."""
        k = k or settings.MEMORY_TOP_K
        v = await self.emb.aembed_query(query)
        async with await psycopg.AsyncConnection.connect(self.uri) as cx:
            await register_vector_async(cx)
            cur = await cx.execute(
                "SELECT id, text, created_at, embedding <=> %s::vector AS d "
                "FROM long_term_memory ORDER BY d LIMIT %s",
                (v, k),
            )
            filas = await cur.fetchall()
        return [Memory(id=f[0], text=f[1], created_at=f[2], score=f[3]) for f in filas]

    # ------------------------------------------------------- envoltorios sync

    def search_memories_sync(self, query: str, k: int | None = None) -> list[Memory]:
        k = k or settings.MEMORY_TOP_K
        v = self.emb.embed_query(query)
        with psycopg.connect(self.uri) as cx:
            register_vector(cx)
            filas = cx.execute(
                "SELECT id, text, created_at, embedding <=> %s::vector AS d "
                "FROM long_term_memory ORDER BY d LIMIT %s",
                (v, k),
            ).fetchall()
        return [Memory(id=f[0], text=f[1], created_at=f[2], score=f[3]) for f in filas]

    def count(self) -> int:
        with psycopg.connect(self.uri) as cx:
            return cx.execute("SELECT count(*) FROM long_term_memory").fetchone()[0]

    def clear(self) -> None:
        with psycopg.connect(self.uri) as cx:
            cx.execute("TRUNCATE long_term_memory")
            cx.commit()


@lru_cache(maxsize=1)
def get_vector_store() -> PostgreSQLVectorStore:
    return PostgreSQLVectorStore()
