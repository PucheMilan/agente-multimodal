"""Configuracion central del agente.

Todo lo ajustable vive aqui y se puede sobrescribir desde `.env`.
El stack es local y gratuito: ver `readme.md` -> "Stack libre".
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    # --- Modelos de lenguaje (Ollama, en local) -----------------------------
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    TEXT_MODEL_NAME: str = "qwen3:14b"        # conversacion, router, memoria, resumen
    ITT_MODEL_NAME: str = "gemma3:4b"         # image-to-text: multimodal
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIM: int = 768

    # --- Voz ---------------------------------------------------------------
    STT_MODEL_NAME: str = "small"             # faster-whisper
    STT_COMPUTE_TYPE: str = "int8"
    STT_LANGUAGE: str = "es"
    TTS_VOICE: str = "es_ES-davefx-medium"    # Piper

    # --- Imagen ------------------------------------------------------------
    TTI_MODEL_NAME: str = "stabilityai/sd-turbo"
    TTI_STEPS: int = 1                        # sd-turbo va bien con 1-4 pasos
    TTI_GUIDANCE: float = 0.0                 # los modelos "turbo" no usan guidance

    # --- Memoria -----------------------------------------------------------
    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5
    # Distancia coseno por debajo de la cual dos recuerdos se consideran el mismo.
    # MEDIDO con nomic-embed-text: duplicados 0,10-0,20 · distintos 0,45-0,49.
    SIMILARITY_THRESHOLD: float = 0.30

    # --- Base de datos -----------------------------------------------------
    # SQLAlchemy async necesita el prefijo +asyncpg; el checkpointer de LangGraph
    # usa psycopg y lo quiere sin el (ver propiedad URI_PSYCOPG).
    URI: str = "postgresql+asyncpg://root:password@127.0.0.1:5432/agent"

    # --- Cache de respuestas -----------------------------------------------
    CACHE_ENABLED: bool = True
    CACHE_TTL_MINUTES: int = 60

    # --- Observabilidad (opcional: si no hay claves, no se traza) ----------
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"

    # --- Rutas -------------------------------------------------------------
    DATA_DIR: Path = RAIZ / "data"
    IMAGES_DIR: Path = RAIZ / "generated_images"

    @property
    def URI_PSYCOPG(self) -> str:
        """La misma URI, en el formato que espera psycopg (sin +asyncpg)."""
        return self.URI.replace("+asyncpg", "")

    @property
    def LANGFUSE_ENABLED(self) -> bool:
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

    @property
    def VOICES_DIR(self) -> Path:
        return self.DATA_DIR / "voices"

    @property
    def WHISPER_DIR(self) -> Path:
        return self.DATA_DIR / "whisper"

    @property
    def HF_DIR(self) -> Path:
        return self.DATA_DIR / "hf"


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
