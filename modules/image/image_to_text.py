"""Imagen (y ficheros) a texto con un modelo multimodal local."""

from pathlib import Path

import ollama

from core.exceptions import ImageToTextError
from core.prompts import FILE_DESCRIPTION_PROMPT, IMAGE_DESCRIPTION_PROMPT
from settings import settings
from tools import logger

MAX_CARACTERES_FICHERO = 12_000


class ImageToText:
    """Describe imagenes y extrae el contenido de documentos."""

    def __init__(self) -> None:
        self._cli = ollama.Client(host=settings.OLLAMA_BASE_URL)

    def describe_image(self, imagen: bytes | str | Path, user_prompt: str = "") -> str:
        try:
            datos = (
                Path(imagen).read_bytes() if isinstance(imagen, (str, Path)) else imagen
            )
            if not datos:
                raise ImageToTextError("La imagen recibida esta vacia")
            r = self._cli.chat(
                model=settings.ITT_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": IMAGE_DESCRIPTION_PROMPT.format(
                        user_prompt=user_prompt or "(el usuario no ha escrito nada)"
                    ),
                    "images": [datos],
                }],
            )
            desc = r["message"]["content"].strip()
            logger.info("Imagen descrita: %s", desc[:80])
            return desc
        except ImageToTextError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ImageToTextError(f"No se pudo interpretar la imagen: {e}") from e

    def describe_file(self, ruta: str | Path, user_prompt: str = "") -> str:
        """Extrae texto de PDF/TXT/MD y lo resume. Las imagenes van por describe_image."""
        try:
            p = Path(ruta)
            if not p.exists():
                raise ImageToTextError(f"No existe el fichero {p}")
            ext = p.suffix.lower()

            if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                return self.describe_image(p, user_prompt)

            if ext == ".pdf":
                from pypdf import PdfReader

                texto = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
            else:
                texto = p.read_text(encoding="utf-8", errors="replace")

            texto = texto.strip()
            if not texto:
                return f"El documento '{p.name}' no contiene texto legible."

            r = self._cli.chat(
                model=settings.TEXT_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": FILE_DESCRIPTION_PROMPT.format(
                        content=texto[:MAX_CARACTERES_FICHERO]
                    ),
                }],
                think=False,
            )
            resumen = r["message"]["content"].strip()
            logger.info("Fichero '%s' resumido (%d caracteres)", p.name, len(texto))
            return resumen
        except ImageToTextError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ImageToTextError(f"No se pudo leer el fichero: {e}") from e
