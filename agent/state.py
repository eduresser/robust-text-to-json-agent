from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages


def messages_reducer(
    current: list[BaseMessage], new: list[BaseMessage]
) -> list[BaseMessage]:
    """
    Reducer for messages that allows reset between chunks.

    If the new list starts with SystemMessage, interprets it as the start of
    a new chat session (new chunk) and replaces all messages.
    Otherwise, uses the default behavior of adding messages.
    """
    if new and isinstance(new[0], SystemMessage):
        return new
    return add_messages(current, new)


def token_usage_reducer(
    current: dict[str, int], new: dict[str, int]
) -> dict[str, int]:
    """Accumulate token usage counters across LLM calls."""
    if not current:
        current = {}
    if not new:
        return current
    return {
        "input_tokens": current.get("input_tokens", 0) + new.get("input_tokens", 0),
        "output_tokens": current.get("output_tokens", 0) + new.get("output_tokens", 0),
        "total_tokens": current.get("total_tokens", 0) + new.get("total_tokens", 0),
        "llm_calls": current.get("llm_calls", 0) + new.get("llm_calls", 0),
        "cache_creation_input_tokens": (
            current.get("cache_creation_input_tokens", 0)
            + new.get("cache_creation_input_tokens", 0)
        ),
        "cache_read_input_tokens": (
            current.get("cache_read_input_tokens", 0)
            + new.get("cache_read_input_tokens", 0)
        ),
    }


class Guidance(TypedDict, total=False):
    """State continuity passed between chunks.

    Designed to carry maximum context in minimum tokens so the next chunk
    can continue extraction seamlessly.
    """

    last_path: str
    sections_snapshot: str
    items_added: str
    open_section: str
    text_excerpt: str
    next_expectations: str
    pending_data: str
    extracted_entities_count: int


class AgentState(TypedDict, total=False):
    """Complete state of the extraction agent."""

    text: str
    target_schema: Optional[dict[str, Any]]

    chunks: list[str]
    current_chunk_idx: int
    current_chunk: str

    json_document: dict[str, Any]

    guidance: Guidance

    # Messages include SystemMessage, HumanMessage, AIMessage (with tool_calls),
    # and ToolMessage responses. The reducer handles chunk resets.
    messages: Annotated[list[BaseMessage], messages_reducer]

    is_chunk_finalized: bool
    iteration_count: int
    max_iterations: int

    token_usage: Annotated[dict[str, int], token_usage_reducer]

    error: Optional[str]
