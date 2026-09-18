"""Fabricas de modelos y modulos. Un solo sitio donde se decide que se usa."""

from functools import lru_cache

from langchain_ollama import ChatOllama

from modules.image import ImageToText, TextToImage
from modules.speech import SpeechToText, TextToSpeech
from settings import settings


def get_chat_model(temperature: float = 0.4) -> ChatOllama:
    """El modelo de texto. `reasoning=False` evita que qwen3 razone en voz alta:
    se come el contexto y trunca las respuestas."""
    return ChatOllama(
        model=settings.TEXT_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        reasoning=False,
    )


@lru_cache(maxsize=1)
def get_text_to_speech_module() -> TextToSpeech:
    return TextToSpeech()


@lru_cache(maxsize=1)
def get_speech_to_text_module() -> SpeechToText:
    return SpeechToText()


@lru_cache(maxsize=1)
def get_text_to_image_module() -> TextToImage:
    return TextToImage()


@lru_cache(maxsize=1)
def get_image_to_text_module() -> ImageToText:
    return ImageToText()
