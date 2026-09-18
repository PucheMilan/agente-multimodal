"""Excepciones propias, una por modulo, para saber que se ha roto sin leer la traza."""


class SpeechToTextError(Exception):
    """Fallo al transcribir audio a texto."""

    pass


class TextToSpeechError(Exception):
    """Fallo al sintetizar texto en voz."""

    pass


class TextToImageError(Exception):
    """Fallo al generar una imagen a partir de texto."""

    pass


class ImageToTextError(Exception):
    """Fallo al interpretar una imagen o un fichero."""

    pass


class MemoryError_(Exception):
    """Fallo al guardar o recuperar memoria a largo plazo."""

    pass
