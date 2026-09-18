"""Voz a texto con faster-whisper, en local y sin coste."""

import tempfile
from functools import lru_cache
from pathlib import Path

from core.exceptions import SpeechToTextError
from settings import settings
from tools import logger


@lru_cache(maxsize=1)
def _modelo():
    """Carga el modelo una sola vez. Tarda ~8 s la primera vez."""
    from faster_whisper import WhisperModel

    settings.WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cargando faster-whisper '%s'...", settings.STT_MODEL_NAME)
    return WhisperModel(
        settings.STT_MODEL_NAME,
        device="cpu",
        compute_type=settings.STT_COMPUTE_TYPE,
        download_root=str(settings.WHISPER_DIR),
    )


class SpeechToText:
    """Transcribe audio a texto."""

    def transcribe(self, audio: bytes | str | Path) -> str:
        """Acepta bytes de un WAV o la ruta de un fichero de audio."""
        try:
            if isinstance(audio, (str, Path)):
                ruta = str(audio)
                borrar = None
            else:
                if not audio:
                    raise SpeechToTextError("El audio recibido esta vacio")
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio)
                tmp.close()
                ruta = borrar = tmp.name

            segmentos, info = _modelo().transcribe(
                ruta, language=settings.STT_LANGUAGE, vad_filter=True
            )
            texto = "".join(s.text for s in segmentos).strip()
            logger.info("Transcrito (%s): %s", info.language, texto[:80])

            if borrar:
                Path(borrar).unlink(missing_ok=True)
            return texto
        except SpeechToTextError:
            raise
        except Exception as e:  # noqa: BLE001
            raise SpeechToTextError(f"No se pudo transcribir el audio: {e}") from e
