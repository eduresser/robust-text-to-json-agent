import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cli.rich_display import console, print_json_panel
from settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Extract structured data from text in JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from direct text
  text-to-json --text "John Doe, 30 years old, works at Acme Corp"

  # Extract from file
  text-to-json --file document.txt

  # Use specific schema
  text-to-json --file doc.txt --schema schema.json

  # Save result to file
  text-to-json --file doc.txt --output resultado.json
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t", type=str, help="Direct text for extraction"
    )
    input_group.add_argument(
        "--file", "-f", type=Path, help="Text file for extraction"
    )

    parser.add_argument(
        "--schema",
        "-s",
        type=Path,
        help="Target schema JSON file (optional)",
    )
    settings = get_settings()
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.MAX_ITERATIONS_PER_CHUNK,
        help=f"Maximum number of iterations per chunk (default: {settings.MAX_ITERATIONS_PER_CHUNK})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for JSON (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        "-p",
        action="store_true",
        help="Format JSON with indentation",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show detailed progress visualization",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Silent mode (only final result)",
    )

    return parser


def _read_input_text(args: argparse.Namespace) -> str:
    """Read the input text from args (direct text or file)."""
    if args.text:
        return args.text

    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    return args.file.read_text(encoding="utf-8")


def _read_schema(args: argparse.Namespace) -> dict[str, Any] | None:
    """Read and parse the optional JSON schema from args."""
    if not args.schema:
        return None

    if not args.schema.exists():
        print(f"Error: Schema file not found: {args.schema}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(args.schema.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid schema: {e}", file=sys.stderr)
        sys.exit(1)


def _handle_output(
    result: dict[str, Any],
    args: argparse.Namespace,
    show_progress: bool,
) -> None:
    """Handle the output of the extraction result."""
    indent = 2 if args.pretty else None
    output_json = json.dumps(
        result["json_document"], indent=indent, ensure_ascii=False
    )

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        if not args.quiet:
            if show_progress:
                console.print(f"[green]Result saved in:[/green] {args.output}")
            else:
                print(f"Result saved in: {args.output}")
                print(f"Chunks processed: {result['metadata']['total_chunks']}")
    else:
        if show_progress:
            print_json_panel(result["json_document"])
        else:
            print(output_json)


def main():
    """Entry point of the CLI."""
    from main import extract

    parser = build_parser()
    args = parser.parse_args()

    text = _read_input_text(args)
    schema = _read_schema(args)

    show_progress = args.progress and not args.quiet
    try:
        result = extract(
            text=text,
            schema=schema,
            max_iterations_per_chunk=args.max_iterations,
            show_progress=show_progress,
        )
    except Exception as e:
        if not args.quiet:
            console.print(f"[red]Error during extraction: {e}[/red]")
        else:
            print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

    if result.get("error"):
        if not args.quiet:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    _handle_output(result, args, show_progress)
