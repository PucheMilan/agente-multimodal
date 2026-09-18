"""Texto a voz con Piper, en local y sin coste."""

import io
import re
import wave
from functools import lru_cache

from core.exceptions import TextToSpeechError
from settings import settings
from tools import logger


@lru_cache(maxsize=1)
def _voz():
    """Carga (y descarga la primera vez) la voz de Piper."""
    from piper import PiperVoice
    from piper.download_voices import download_voice

    settings.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    onnx = settings.VOICES_DIR / f"{settings.TTS_VOICE}.onnx"
    if not onnx.exists():
        logger.info("Descargando voz '%s'...", settings.TTS_VOICE)
        download_voice(settings.TTS_VOICE, settings.VOICES_DIR)
    logger.info("Cargando voz '%s'", settings.TTS_VOICE)
    return PiperVoice.load(onnx)


_ANUNCIOS = re.compile(
    r"^.{0,60}?\b(diciendo|que dice|con el mensaje)\s*:\s*", re.IGNORECASE
)
_MARCAS = re.compile(r"[\[\]\*`_<>|]")


def limpiar_para_voz(texto: str) -> str:
    """Quita lo que no se debe pronunciar.

    El modelo a veces anuncia el audio en vez de decirlo ("aqui tienes un audio
    diciendo: '...'") y deja comillas que Piper leeria en voz alta. Esto no se
    arregla solo con el prompt: es una limpieza determinista y barata.
    """
    t = _ANUNCIOS.sub("", texto.strip())
    t = _MARCAS.sub("", t)
    t = t.strip().strip("\"'“”«»").strip()
    return re.sub(r"\s+", " ", t)


class TextToSpeech:
    """Sintetiza texto en voz y devuelve un WAV en memoria."""

    def synthesize(self, texto: str) -> bytes:
        if not texto or not texto.strip():
            raise TextToSpeechError("No hay texto que sintetizar")
        limpio = limpiar_para_voz(texto)
        if not limpio:
            raise TextToSpeechError("Tras limpiar el texto no queda nada que decir")
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                _voz().synthesize_wav(limpio, w)
            datos = buf.getvalue()
            logger.info("Audio sintetizado: %d bytes", len(datos))
            return datos
        except Exception as e:  # noqa: BLE001
            raise TextToSpeechError(f"No se pudo sintetizar la voz: {e}") from e
