from typing import Any, Optional

from settings import get_settings


def extract(
    text: str,
    schema: Optional[dict[str, Any]] = None,
    max_iterations_per_chunk: Optional[int] = None,
    show_progress: bool = False,
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
        show_progress: If True, show detailed progress visualization.

    Returns:
        A dictionary with:
        - "json_document": The extracted JSON document.
        - "metadata": Information about the processing:
          - "total_chunks": Total number of chunks processed.
          - "final_guidance": Final guidance used.
          - "token_usage": Token usage information.
        - "error": Error message, if there is one.

    Example:
        >>> from main import extract
        >>> result = extract(
        ...     text="John Doe, 30 years old, works at Acme Corp...",
        ...     schema={"type": "object", "properties": {"name": {}, "age": {}, "company": {}}}
        ... )
        >>> print(result["json_document"])
        {"name": "John Doe", "age": 30, "company": "Acme Corp"}
    """
    if max_iterations_per_chunk is None:
        max_iterations_per_chunk = get_settings().MAX_ITERATIONS_PER_CHUNK

    if show_progress:
        return _extract_with_progress(text, schema, max_iterations_per_chunk)

    from agent.graph import extract as agent_extract
    return agent_extract(text, schema, max_iterations_per_chunk)


def _extract_with_progress(
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


def main():
    """Entry point of the CLI."""
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
