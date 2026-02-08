"""
LangChain tool definitions for the extraction agent.

These tools are bound to the LLM via bind_tools(). The actual execution
with document/target_schema from state is done in execute_tools_node.
"""

from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# --- Input schemas (only what the LLM sends) ---


class InspectKeysInput(BaseModel):
    """Input for inspect_keys: path in the JSON document or schema."""

    source: Literal["document", "schema"] = Field(
        default="document",
        description='Which data source to inspect: "document" (JSON being built) or "schema" (target schema). Default: "document".',
    )
    path: str = Field(
        default="",
        description="JSON Pointer to the location (e.g. '/sections', '/metadata'). Use '' or '/' for root. Paths must start with / for non-root.",
    )


class ReadValueInput(BaseModel):
    """Input for read_value: path and optional truncation limits."""

    source: Literal["document", "schema"] = Field(
        default="document",
        description='Which data source to read: "document" (JSON being built) or "schema" (target schema). Default: "document".',
    )
    path: str = Field(
        default="",
        description="JSON Pointer to the value (e.g. '/sections', '/metadata'). Use '' or '/' for root. Paths must start with / for non-root.",
    )
    max_string_length: int = Field(
        default=160,
        description="Maximum length of strings before truncating.",
    )
    max_depth: int = Field(default=6, description="Maximum depth for nested objects.")
    max_array_items: int = Field(default=50, description="Maximum number of array items to return.")
    max_object_keys: int = Field(default=50, description="Maximum number of object keys to return.")


class SearchPointerInput(BaseModel):
    """Input for search_pointer: query and search options."""

    source: Literal["document", "schema"] = Field(
        default="document",
        description='Which data source to search: "document" (JSON being built) or "schema" (target schema). Default: "document".',
    )
    query: str = Field(description="Search query (key or value to find).")
    type: Literal["key", "value"] = Field(
        default="value",
        description="Search in 'key' or 'value'.",
    )
    fuzzy_match: bool = Field(
        default=False,
        description="If True, fuzzy search (all words present).",
    )
    limit: int = Field(default=20, description="Maximum number of results.")
    max_value_length: int = Field(default=120, description="Max length of value preview in results.")


class ApplyPatchesInput(BaseModel):
    """Input for apply_patches: list of JSON Patch operations (RFC 6902)."""

    patches: list[dict[str, Any]] = Field(
        description="Non-empty list of patch operations: add, remove, replace, move, copy. Each has 'op', 'path', and optionally 'value' or 'from'. Do NOT call this tool with an empty list; if you have no changes, call update_guidance to finalize instead.",
    )


class UpdateGuidanceInput(BaseModel):
    """Input for update_guidance: finalize current chunk and pass state to next."""

    last_processed_path: str = Field(
        default="",
        description="The last JSON path that was processed.",
    )
    current_context: str = Field(
        default="",
        description="Summary of what is being built (e.g. list of clients). Include part of the text chunk if relevant.",
    )
    pending_action: str = Field(
        default="",
        description="Pending action for the next chunk (e.g. expecting_contract_signature).",
    )
    extracted_entities_count: int = Field(
        default=0,
        description="Number of entities extracted in this chunk.",
    )


# --- Stub implementation (never called; execute_tools_node dispatches by name) ---


def _stub_invoke(**kwargs: Any) -> dict[str, Any]:
    """Not used; execution is done in execute_tools_node with state."""
    return {"_executed_by": "execute_tools_node"}


# --- Tool definitions for bind_tools ---


def get_extraction_tools() -> list[StructuredTool]:
    """Return the list of tools to bind to the chat model."""
    return [
        StructuredTool.from_function(
            name="inspect_keys",
            description="Returns the keys of an object or the length of an array at a path in the document or schema. Use this FIRST for arrays to get length/type. Set source='schema' to explore the target schema when it is truncated in the prompt.",
            args_schema=InspectKeysInput,
            func=_stub_invoke,
        ),
        StructuredTool.from_function(
            name="read_value",
            description="Retrieves the value at ONE specific path in the document or schema. Use only for narrow paths you need to patch or verify. Set source='schema' to read schema details when the TargetSchema is truncated in the prompt.",
            args_schema=ReadValueInput,
            func=_stub_invoke,
        ),
        StructuredTool.from_function(
            name="search_pointer",
            description="Searches the document or schema for a key or value and returns JSON Pointers. MANDATORY before creating new list items (e.g. check if an entity already exists to avoid duplicates). Set source='schema' to find properties or types in the target schema.",
            args_schema=SearchPointerInput,
            func=_stub_invoke,
        ),
        StructuredTool.from_function(
            name="apply_patches",
            description="Applies changes to the JSON document using RFC 6902 (JSON Patch). Operations: add (new key/item), replace (overwrite), remove (delete), move (relocate), copy (duplicate). Batch multiple small patches when safe. Always verify with read_value before replace/remove/move.",
            args_schema=ApplyPatchesInput,
            func=_stub_invoke,
        ),
        StructuredTool.from_function(
            name="update_guidance",
            description="Finalizes the current chunk and saves state for the next chunk. Call ONLY when the current chunk is fully processed, all writes confirmed, and you are ready to move to the next chunk. This MUST be your only tool call when finalizing.",
            args_schema=UpdateGuidanceInput,
            func=_stub_invoke,
        ),
    ]
