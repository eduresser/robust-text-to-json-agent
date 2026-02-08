from typing import Any, Optional

from clients import reset_clients_cache
from settings import get_settings, reset_settings_cache


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
        from agent.graph import rich_extract
        result = rich_extract(text, schema, max_iterations_per_chunk)
    else:
        from agent.graph import extract
        result = extract(text, schema, max_iterations_per_chunk)

    reset_clients_cache()
    reset_settings_cache()

    return result


def main():
    """Entry point of the CLI."""
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
