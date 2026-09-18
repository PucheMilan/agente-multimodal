"""Cache de respuestas en PostgreSQL, con caducidad.

Evita repetir el trabajo del LLM cuando llega exactamente la misma pregunta.
Se cachea SOLO la rama de conversacion: una imagen o un audio producen un
fichero nuevo cada vez, asi que cachearlos no tendria sentido.
"""

import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, String, Text, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from settings import settings
from tools import logger


class Base(DeclarativeBase):
    pass


class ResponseCache(Base):
    __tablename__ = "response_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    query_text: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


_engine = create_async_engine(settings.URI, echo=False, pool_pre_ping=True)
_Session = async_sessionmaker(_engine, expire_on_commit=False)


def _hash(texto: str) -> str:
    return hashlib.sha256(texto.strip().lower().encode("utf-8")).hexdigest()


async def init_cache() -> None:
    """Crea la tabla si no existe."""
    async with _engine.begin() as cx:
        await cx.run_sync(Base.metadata.create_all)
    logger.info("Cache de respuestas lista")


async def get_cached_response(user_query: str) -> str | None:
    if not settings.CACHE_ENABLED:
        return None
    ahora = datetime.now(timezone.utc)
    async with _Session() as s:
        fila = (
            await s.execute(
                select(ResponseCache).where(
                    ResponseCache.query_hash == _hash(user_query),
                    ResponseCache.expires_at > ahora,
                )
            )
        ).scalar_one_or_none()
    if fila:
        logger.info("Cache HIT: '%s'", user_query[:60])
        return fila.response
    return None


async def set_cached_response(user_query: str, response: str) -> None:
    if not settings.CACHE_ENABLED or not response.strip():
        return
    ahora = datetime.now(timezone.utc)
    h = _hash(user_query)
    async with _Session() as s:
        await s.execute(delete(ResponseCache).where(ResponseCache.query_hash == h))
        s.add(
            ResponseCache(
                query_hash=h,
                query_text=user_query[:2000],
                response=response,
                created_at=ahora,
                expires_at=ahora + timedelta(minutes=settings.CACHE_TTL_MINUTES),
            )
        )
        await s.commit()
    logger.info("Cache guardada: '%s'", user_query[:60])


async def purge_expired() -> int:
    async with _Session() as s:
        r = await s.execute(
            delete(ResponseCache).where(ResponseCache.expires_at <= datetime.now(timezone.utc))
        )
        await s.commit()
    return r.rowcount or 0
