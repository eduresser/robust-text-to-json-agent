from main import extract
import rich

# Simple extraction (no schema — agent infers structure)
rich.print("Simple extraction (no schema — agent infers structure)")
result = extract(
    text="John Doe, 30 years old, works at Acme Corp as an engineer.",
)
rich.print(result["json_document"])
rich.print(f"Chunks processed: {result['metadata']['total_chunks']}")
rich.print(f"Token usage: {result['metadata']['token_usage']}")
# {"name": "John Doe", "age": 30, "company": "Acme Corp", "role": "engineer"}

rich.print("With a JSON Schema")
schema = {
    "type": "object",
    "properties": {
        "employees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "company": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    }
}

rich.print("With live progress")
result = extract(
    text="John Doe, 30 years old, works at Acme Corp as an engineer.",
    schema=schema,
    show_progress=True,  # Rich live display
)

rich.print(result["json_document"])
rich.print(f"Chunks processed: {result['metadata']['total_chunks']}")
rich.print(f"Token usage: {result['metadata']['token_usage']}")