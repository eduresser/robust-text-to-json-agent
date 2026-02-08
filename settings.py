from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: SecretStr

    CHAT_MODEL: str = "openai:gpt-4o-mini"
    CHAT_MAX_TOKENS: int = 8192
    EMBEDDING_MODEL: str = "openai:text-embedding-3-small"

    SQLITE_DB_PATH: str = ".cache/checkpoints.db"

    MAX_ITERATIONS_PER_CHUNK: int = 50

    # --- Truncator limits (chars) ---
    TRUNCATE_SKELETON_LIMIT: int = 6000
    TRUNCATE_GUIDANCE_LIMIT: int = 6000
    TRUNCATE_READ_VALUE_LIMIT: int = 6000

    # --- Truncator config ---
    TRUNCATE_INDENTATION: int = 4
    TRUNCATE_MIN_STRING_LEN: int = 23
    TRUNCATE_ELLIPSIS_SIZE: int = 3
    TRUNCATE_MIN_ARRAY_ITEMS: int = 2
    TRUNCATE_MIN_OBJECT_KEYS: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
