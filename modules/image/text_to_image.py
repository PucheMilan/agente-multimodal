"""Texto a imagen con Stable Diffusion Turbo, en local y sin coste."""

import json
import time
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from core.exceptions import TextToImageError
from core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from settings import settings
from tools import logger


class EscenaImagen(BaseModel):
    """Lo que devuelve el LLM al inventarse una escena."""

    narrative: str = Field(description="respuesta narrativa en primera persona")
    image_prompt: str = Field(description="prompt visual en ingles")


@lru_cache(maxsize=1)
def _pipeline():
    """Carga sd-turbo una sola vez. Tarda ~35 s la primera vez."""
    import torch
    from diffusers import AutoPipelineForText2Image

    settings.HF_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cargando %s (puede tardar la primera vez)...", settings.TTI_MODEL_NAME)
    pipe = AutoPipelineForText2Image.from_pretrained(
        settings.TTI_MODEL_NAME, dtype=torch.float32, cache_dir=str(settings.HF_DIR)
    ).to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


class TextToImage:
    """Genera imagenes y ayuda a construir el prompt visual."""

    async def create_scenario(self, chat_history: str, llm) -> EscenaImagen:
        """Inventa una escena a partir de la conversacion."""
        try:
            r = await llm.ainvoke(
                IMAGE_SCENARIO_PROMPT.format(chat_history=chat_history)
                + "\n\nResponde SOLO con el JSON, sin nada mas."
            )
            texto = r.content.strip()
            if "{" in texto:
                texto = texto[texto.index("{"): texto.rindex("}") + 1]
            datos = json.loads(texto)
            return EscenaImagen(**datos)
        except Exception as e:  # noqa: BLE001
            logger.warning("No se pudo crear la escena (%s); se usa el historial", e)
            return EscenaImagen(narrative="", image_prompt=chat_history[:200])

    async def enhance_prompt(self, prompt: str, llm) -> str:
        """Enriquece el prompt visual. Si falla, devuelve el original."""
        try:
            r = await llm.ainvoke(IMAGE_ENHANCEMENT_PROMPT.format(prompt=prompt))
            mejorado = r.content.strip().strip('"').split("\n")[0]
            return mejorado[:300] or prompt
        except Exception as e:  # noqa: BLE001
            logger.warning("No se pudo mejorar el prompt (%s)", e)
            return prompt

    def generate(self, prompt: str, destino: Path | None = None) -> Path:
        """Genera la imagen y devuelve su ruta en disco."""
        if not prompt or not prompt.strip():
            raise TextToImageError("El prompt de imagen esta vacio")
        try:
            destino = destino or settings.IMAGES_DIR / f"img_{int(time.time() * 1000)}.png"
            destino.parent.mkdir(parents=True, exist_ok=True)
            t = time.time()
            img = _pipeline()(
                prompt=prompt,
                num_inference_steps=settings.TTI_STEPS,
                guidance_scale=settings.TTI_GUIDANCE,
            ).images[0]
            img.save(destino)
            logger.info("Imagen generada en %.1fs -> %s", time.time() - t, destino.name)
            return destino
        except Exception as e:  # noqa: BLE001
            raise TextToImageError(f"No se pudo generar la imagen: {e}") from e
