"""Trazabilidad con LangFuse, opcional.

Si no hay claves configuradas, @observe se convierte en un decorador que no hace
nada. Asi el proyecto arranca sin cuenta en ningun sitio y sigue costando 0 EUR,
y basta rellenar dos variables del .env para que empiece a trazar.
"""

from contextlib import contextmanager

from settings import settings
from tools.tools import logger

_cliente = None

if settings.LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse, observe  # type: ignore

        _cliente = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        logger.info("LangFuse activo -> %s", settings.LANGFUSE_BASE_URL)
    except Exception as e:  # noqa: BLE001
        logger.warning("LangFuse configurado pero no se pudo iniciar: %s", e)
        _cliente = None

if _cliente is None:
    logger.info("LangFuse desactivado (sin claves): las trazas no se envian")

    def observe(*d_args, **d_kwargs):  # type: ignore[no-redef]
        """No-op con la misma firma que el decorador de LangFuse."""

        def _wrap(fn):
            return fn

        # Permite usarlo como @observe y como @observe(name="x")
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _wrap


@contextmanager
def traza_conversacion(nombre: str = "conversacion", entrada=None, sesion: str | None = None):
    """Abre la traza PADRE de una vuelta entera por el grafo.

    Sin esto, el @observe de cada nodo crea su propia traza raiz y en LangFuse
    aparecen sueltas: 16 trazas para 4 mensajes, sin forma de ver por donde paso
    una peticion concreta. Con esto, los nodos cuelgan de una sola.

    Si LangFuse esta apagado no hace nada, igual que @observe.
    """
    if _cliente is None:
        yield None
        return
    meta = {"langfuse_session_id": sesion} if sesion else None
    with _cliente.start_as_current_observation(
        name=nombre, as_type="agent", input=entrada, metadata=meta
    ) as span:
        yield span


def flush() -> None:
    if _cliente is not None:
        try:
            _cliente.flush()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["observe", "traza_conversacion", "flush"]
