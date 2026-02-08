from collections import Counter
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class TokenUsage:
    """Accumulated token usage across all LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    llm_calls: int = 0

    def add(self, usage_metadata: dict) -> None:
        """Accumulate token counts from a single LLM response."""
        if not usage_metadata:
            return
        self.input_tokens += usage_metadata.get("input_tokens", 0)
        self.output_tokens += usage_metadata.get("output_tokens", 0)
        self.total_tokens += usage_metadata.get("total_tokens", 0)
        self.cache_creation_input_tokens += usage_metadata.get(
            "cache_creation_input_tokens", 0
        )
        self.cache_read_input_tokens += usage_metadata.get(
            "cache_read_input_tokens", 0
        )
        self.llm_calls += 1

    def to_dict(self) -> dict:
        """Return a dictionary with the token usage."""
        d = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }
        if self.cache_creation_input_tokens:
            d["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        if self.cache_read_input_tokens:
            d["cache_read_input_tokens"] = self.cache_read_input_tokens
        return d


def _format_token_count(n: int) -> str:
    """Format a token count with thousands separator."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def create_progress_display(
    current_node: str,
    chunk_idx: int,
    total_chunks: int,
    iteration: int,
    max_iterations: int,
    tools_used: Counter,
    text_preview: str,
    model_name: str,
    token_usage: TokenUsage | None = None,
) -> Table:
    """Build the progress visualization table."""
    node_descriptions = {
        "chunk_text": "Dividing text into chunks...",
        "prepare_chunk": "Preparing chunk for processing...",
        "call_llm": "Calling language model...",
        "execute_tools": "Executing tools...",
        "finalize_chunk": "Finalizing chunk...",
        "__end__": "Processing completed!",
    }

    table = Table(
        title="[bold cyan]Text-to-JSON Agent[/bold cyan]",
        box=box.ROUNDED,
        show_header=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Info", style="bold", width=20)
    table.add_column("Value", style="white")

    status_text = node_descriptions.get(current_node, f"⏳ {current_node}")
    table.add_row("Status", Text(status_text, style="yellow"))

    table.add_row("Model", Text(model_name, style="cyan"))

    if total_chunks > 0:
        chunk_progress = f"[green]{chunk_idx + 1}[/green] / [blue]{total_chunks}[/blue]"
        table.add_row("Chunk", chunk_progress)
    else:
        table.add_row("Chunk", "[dim]Calculating...[/dim]")

    iter_style = "red" if iteration >= max_iterations - 2 else "green"
    table.add_row(
        "Iteration",
        f"[{iter_style}]{iteration}[/{iter_style}] / [blue]{max_iterations}[/blue]",
    )

    if tools_used:
        sorted_tools = tools_used.most_common()
        tools_parts = [f"{name} ({count})" for name, count in sorted_tools]
        tools_text = ", ".join(tools_parts)
        total_calls = sum(tools_used.values())
        table.add_row(
            f"Tools ({total_calls})",
            Text(tools_text, style="magenta"),
        )
    else:
        table.add_row("Tools", "[dim]None yet[/dim]")

    if token_usage and token_usage.total_tokens > 0:
        total_fmt = _format_token_count(token_usage.total_tokens)
        input_fmt = _format_token_count(token_usage.input_tokens)
        output_fmt = _format_token_count(token_usage.output_tokens)

        parts = [
            f"[bold white]{total_fmt}[/bold white]",
            f"[dim](in:[/dim] [green]{input_fmt}[/green]",
            f"[dim]out:[/dim] [yellow]{output_fmt}[/yellow][dim])[/dim]",
        ]

        if token_usage.cache_read_input_tokens > 0:
            cache_fmt = _format_token_count(token_usage.cache_read_input_tokens)
            parts.append(f"[dim]cache:[/dim] [blue]{cache_fmt}[/blue]")

        token_text = " ".join(parts)
        table.add_row(f"Tokens ({token_usage.llm_calls} calls)", token_text)
    else:
        table.add_row("Tokens", "[dim]None yet[/dim]")

    if text_preview:
        preview = text_preview[:60] + "..." if len(text_preview) > 60 else text_preview
        preview = preview.replace("\n", " ")
        table.add_row("Text", Text(preview, style="dim"))

    return table


def print_start_panel(model_name: str, text_len: int, has_schema: bool) -> None:
    """Print the start panel of the extraction."""
    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/bold] {model_name}\n"
            f"[bold]Text:[/bold] {text_len} characters\n"
            f"[bold]Schema:[/bold] {'Provided' if has_schema else 'Automatic inference'}",
            title="[bold cyan]Starting Extraction[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()


def print_result_panel(
    total_chunks: int,
    num_fields: int,
    token_usage: dict | None = None,
) -> None:
    """Print the success panel at the end of the extraction."""
    lines = [
        "[bold green]Extraction completed successfully![/bold green]\n",
        f"[bold]Chunks processed:[/bold] {total_chunks}",
        f"[bold]Fields extracted:[/bold] {num_fields}",
    ]

    if token_usage and token_usage.get("total_tokens", 0) > 0:
        total = _format_token_count(token_usage["total_tokens"])
        inp = _format_token_count(token_usage["input_tokens"])
        out = _format_token_count(token_usage["output_tokens"])
        calls = token_usage.get("llm_calls", 0)

        lines.append("")
        lines.append(f"[bold]Tokens:[/bold] {total} total  [dim]([/dim]input: {inp}  output: {out}[dim])[/dim]")
        lines.append(f"[bold]LLM calls:[/bold] {calls}")

        if token_usage.get("cache_read_input_tokens", 0) > 0:
            cache = _format_token_count(token_usage["cache_read_input_tokens"])
            lines.append(f"[bold]Cache read:[/bold] {cache} tokens")
        if token_usage.get("cache_creation_input_tokens", 0) > 0:
            cache_c = _format_token_count(token_usage["cache_creation_input_tokens"])
            lines.append(f"[bold]Cache creation:[/bold] {cache_c} tokens")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold green]Result[/bold green]",
            border_style="green",
        )
    )
    console.print()


def print_error_panel(message: str) -> None:
    """Print the error panel."""
    console.print(
        Panel(
            f"[red]{message}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        )
    )
    console.print()


def print_json_panel(json_document: dict) -> None:
    """Print the extracted JSON in a panel with syntax highlighting."""
    import json

    syntax = Syntax(
        json.dumps(json_document, indent=2, ensure_ascii=False),
        "json",
        theme="monokai",
        line_numbers=True,
    )
    console.print(
        Panel(syntax, title="[bold]Extracted JSON[/bold]", border_style="blue")
    )


def run_live_progress(app, initial_state: dict, model_name: str, max_iterations: int):
    """
    Run the stream of the graph with Live from Rich, updating the progress table.
    Return the final state (with json_document, chunks, guidance, error, token_usage).
    """
    current_node = "chunk_text"
    total_chunks = 0
    chunk_idx = 0
    iteration = 0
    tools_used: Counter = Counter()
    current_chunk_text = ""
    token_usage = TokenUsage()
    final_state = None

    with Live(
        create_progress_display(
            current_node,
            chunk_idx,
            total_chunks,
            iteration,
            max_iterations,
            tools_used,
            current_chunk_text,
            model_name,
            token_usage,
        ),
        console=console,
        refresh_per_second=4,
        transient=True,
    ) as live:
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in event.items():
                current_node = node_name

                if "chunks" in state_update and state_update["chunks"]:
                    total_chunks = len(state_update["chunks"])

                if "current_chunk_idx" in state_update:
                    new_chunk_idx = state_update["current_chunk_idx"]
                    if new_chunk_idx != chunk_idx:
                        chunk_idx = new_chunk_idx
                        iteration = 0
                        tools_used = Counter()

                if "current_chunk" in state_update:
                    current_chunk_text = state_update["current_chunk"]

                if "iteration_count" in state_update:
                    iteration = state_update["iteration_count"]

                if "messages" in state_update:
                    for msg in state_update["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get("name", "")
                                if tool_name:
                                    tools_used[tool_name] += 1

                        usage = getattr(msg, "usage_metadata", None)
                        if usage:
                            token_usage.add(usage)

                if "json_document" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["json_document"] = state_update["json_document"]
                if "chunks" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["chunks"] = state_update["chunks"]
                if "guidance" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["guidance"] = state_update["guidance"]
                if "error" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["error"] = state_update["error"]

                live.update(
                    create_progress_display(
                        current_node,
                        chunk_idx,
                        total_chunks,
                        iteration,
                        max_iterations,
                        tools_used,
                        current_chunk_text,
                        model_name,
                        token_usage,
                    )
                )

    result = final_state or {}
    result["token_usage"] = token_usage.to_dict()
    return result
