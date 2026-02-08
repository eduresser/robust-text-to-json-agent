from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from settings import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return an instance of the configured embeddings model."""
    settings = get_settings()
    return init_embeddings(
        settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return an instance of the configured chat model."""
    settings = get_settings()
    return init_chat_model(
        settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )


@lru_cache(maxsize=1)
def get_tool_calling_model() -> BaseChatModel:
    """Return an instance of the chat model with tools bound for extraction.

    This replaces the old get_json_chat_model(). Instead of forcing JSON
    output, it binds the extraction tools so the model uses native tool
    calling to interact with the JSON document.
    """
    from tools.definitions import ALL_TOOLS

    settings = get_settings()
    model = init_chat_model(
        settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        temperature=0,
        max_tokens=settings.CHAT_MAX_TOKENS,
    )
    return model.bind_tools(ALL_TOOLS, tool_choice="required")


def get_checkpointer() -> SqliteSaver:
    """Return an instance of the checkpointer SQLite."""
    settings = get_settings()
    
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    return SqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


def reset_clients_cache() -> None:
    """Clear the cache of the clients."""
    get_embeddings.cache_clear()
    get_chat_model.cache_clear()
    get_tool_calling_model.cache_clear()
