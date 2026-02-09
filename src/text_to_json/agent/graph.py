from typing import Any, Optional

from langgraph.graph import END, StateGraph

from settings import get_settings

from agent.nodes import (
    call_llm_node,
    chunk_text_node,
    execute_tools_node,
    finalize_chunk_node,
    has_more_chunks,
    is_chunk_done,
    prepare_chunk_node,
)
from agent.state import AgentState


def create_graph() -> StateGraph:
    """
    Create the LangGraph for structured data extraction.

    Returns:
        The compiled graph ready to execute.
    """
    graph = StateGraph(AgentState)

    graph.add_node("chunk_text", chunk_text_node)
    graph.add_node("prepare_chunk", prepare_chunk_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("finalize_chunk", finalize_chunk_node)

    graph.set_entry_point("chunk_text")

    graph.add_edge("chunk_text", "prepare_chunk")

    graph.add_conditional_edges(
        "prepare_chunk",
        has_more_chunks,
        {
            "call_llm": "call_llm",
            "__end__": END,
        },
    )

    graph.add_edge("call_llm", "execute_tools")

    graph.add_conditional_edges(
        "execute_tools",
        is_chunk_done,
        {
            "finalize_chunk": "finalize_chunk",
            "call_llm": "call_llm",
        },
    )

    graph.add_edge("finalize_chunk", "prepare_chunk")

    return graph.compile()


def extract(
    text: str,
    schema: Optional[dict[str, Any]] = None,
    max_iterations_per_chunk: Optional[int] = None,
) -> dict[str, Any]:
    """
    Extract structured data from a text.

    This is the main function of the agent API.

    Args:
        text: The input text for extraction.
        schema: Optional target schema JSON. If not provided, the agent
               will infer a logical structure based on the content.
        max_iterations_per_chunk: Maximum number of iterations of the agent
                                  by chunk before forcing finalization.
                                  If None, uses the value from Settings.MAX_ITERATIONS_PER_CHUNK.

    Returns:
        A dictionary with:
        - "json_document": The extracted JSON document.
        - "metadata": Information about the processing.
        - "error": Error message, if there is one.

    Example:
        >>> from agent import extract
        >>> result = extract(
        ...     text="John Doe, 30 years old, works at Acme Corp...",
        ...     schema={"type": "object", "properties": {"name": {}, "age": {}, "company": {}}}
        ... )
        >>> print(result["json_document"])
        {"name": "John Doe", "age": 30, "company": "Acme Corp"}
    """
    settings = get_settings()

    if max_iterations_per_chunk is None:
        max_iterations_per_chunk = settings.MAX_ITERATIONS_PER_CHUNK

    app = create_graph()

    initial_state: AgentState = {
        "text": text,
        "target_schema": schema,
        "max_iterations": max_iterations_per_chunk,
        "max_chunk_retries": settings.MAX_CHUNK_RETRIES,
        "chunks": [],
        "current_chunk_idx": 0,
        "json_document": {},
        "guidance": {},
        "messages": [],
        "is_chunk_finalized": False,
        "iteration_count": 0,
        "chunk_retry_count": 0,
        "token_usage": {},
    }

    final_state = app.invoke(initial_state)

    return {
        "json_document": final_state.get("json_document", {}),
        "metadata": {
            "total_chunks": len(final_state.get("chunks", [])),
            "final_guidance": final_state.get("guidance", {}),
            "token_usage": final_state.get("token_usage", {}),
        },
        "error": final_state.get("error"),
    }

def rich_extract(
    text: str,
    schema: Optional[dict[str, Any]],
    max_iterations_per_chunk: int,
) -> dict[str, Any]:
    """Execute extraction with progress visualization (uses Rich via cli/)."""
    from agent.graph import create_graph
    from agent.state import AgentState
    from cli import (
        print_error_panel,
        print_result_panel,
        print_start_panel,
        run_live_progress,
    )

    settings = get_settings()
    model_name = settings.CHAT_MODEL
    app = create_graph()

    initial_state: AgentState = {
        "text": text,
        "target_schema": schema,
        "max_iterations": max_iterations_per_chunk,
        "chunks": [],
        "current_chunk_idx": 0,
        "json_document": {},
        "guidance": {},
        "messages": [],
        "is_chunk_finalized": False,
        "iteration_count": 0,
        "token_usage": {},
    }

    print_start_panel(model_name, len(text), schema is not None)
    final_state = run_live_progress(
        app, initial_state, model_name, max_iterations_per_chunk
    )

    token_usage = final_state.get("token_usage", {})

    result = {
        "json_document": final_state.get("json_document", {}),
        "metadata": {
            "total_chunks": len(final_state.get("chunks", [])),
            "final_guidance": final_state.get("guidance", {}),
            "token_usage": token_usage,
        },
        "error": final_state.get("error"),
    }

    if result.get("error"):
        print_error_panel(result["error"])
    else:
        print_result_panel(
            result["metadata"]["total_chunks"],
            len(result["json_document"]),
            token_usage,
        )

    return result
