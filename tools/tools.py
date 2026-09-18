"""Utilidades transversales: por ahora, el logger."""

import logging
import sys

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(nombre: str = "agente", nivel: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger configurado una sola vez (no duplica handlers)."""
    log = logging.getLogger(nombre)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FORMATO, datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(nivel)
        log.propagate = False
    return log


logger = get_logger()
