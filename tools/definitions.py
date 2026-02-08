from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ======================================================================
# inspect_keys
# ======================================================================
class InspectKeysArgs(BaseModel):
    source: Literal["document", "schema"] = Field(
        default="document",
        description=(
            'Which data source to inspect: "document" (JSON being built) '
            'or "schema" (target schema). Default: "document".'
        ),
    )
    path: str = Field(
        default="",
        description=(
            "JSON Pointer (RFC 6901) to the location to inspect. "
            'Use "" for root, "/sections" for sections array, "/sections/0" for first section, etc.'
        ),
    )


@tool("inspect_keys", args_schema=InspectKeysArgs)
def inspect_keys_tool(source: str = "document", path: str = "") -> str:
    """Returns the keys of an object or length of an array at a specific path in the JSON Document or Schema.

    Use this to:
    - Check array lengths before appending (to know current indices).
    - Verify parent containers exist before patching.
    - Navigate the structure without loading full data.
    - Explore the target schema when it is truncated in the prompt (use source="schema").

    For arrays, returns the length — essential to know before using "add" with "/-".
    """
    raise NotImplementedError("Execution handled by execute_tools_node")


# ======================================================================
# search_pointer
# ======================================================================
class SearchPointerArgs(BaseModel):
    source: Literal["document", "schema"] = Field(
        default="document",
        description=(
            'Which data source to search: "document" (JSON being built) '
            'or "schema" (target schema). Default: "document".'
        ),
    )
    query: str = Field(description="Search query string.")
    type: Literal["key", "value"] = Field(
        default="value",
        description='Search type: "key" to search in keys, "value" to search in values.',
    )
    fuzzy_match: bool = Field(
        default=False,
        description="Enable fuzzy matching (Levenshtein-based).",
    )
    limit: int = Field(
        default=20,
        description="Maximum number of results to return.",
    )
    max_value_length: int = Field(
        default=120,
        description="Maximum length of value preview in results.",
    )


@tool("search_pointer", args_schema=SearchPointerArgs)
def search_pointer_tool(
    source: str = "document",
    query: str = "",
    type: str = "value",
    fuzzy_match: bool = False,
    limit: int = 20,
    max_value_length: int = 120,
) -> str:
    """Searches the JSON Document or Schema for a key or value and returns JSON Pointers to matching locations.

    MANDATORY before creating new list items to avoid duplicates (e.g., check if "Client X" already exists).
    Also useful to locate data that needs correction or to find properties/types in the target schema (use source="schema").
    """
    raise NotImplementedError("Execution handled by execute_tools_node")


# ======================================================================
# read_value
# ======================================================================
class ReadValueArgs(BaseModel):
    source: Literal["document", "schema"] = Field(
        default="document",
        description=(
            'Which data source to read: "document" (JSON being built) '
            'or "schema" (target schema). Default: "document".'
        ),
    )
    path: str = Field(
        description="JSON Pointer (RFC 6901) path to read.",
    )
    max_string_length: int = Field(
        default=160,
        description="Maximum length of strings before truncating.",
    )
    max_depth: int = Field(
        default=6,
        description="Maximum depth for nested objects.",
    )
    max_array_items: int = Field(
        default=50,
        description="Maximum number of array items to return.",
    )
    max_object_keys: int = Field(
        default=50,
        description="Maximum number of object keys to return.",
    )


@tool("read_value", args_schema=ReadValueArgs)
def read_value_tool(
    source: str = "document",
    path: str = "",
    max_string_length: int = 160,
    max_depth: int = 6,
    max_array_items: int = 50,
    max_object_keys: int = 50,
) -> str:
    """Retrieves the exact value at a specific path in the JSON Document or Schema.

    Use for verification before updates. Don't read whole arrays; read specific indices.
    Essential before corrections (replace, remove, move).
    Use source="schema" to read schema details when the TargetSchema is truncated in the prompt.
    """
    raise NotImplementedError("Execution handled by execute_tools_node")


# ======================================================================
# apply_patches
# ======================================================================
class ApplyPatchesArgs(BaseModel):
    patches: list[dict[str, Any]] = Field(
        description=(
            "List of JSON Patch operations (RFC 6902). Each operation must have: "
            '"op" (add|replace|remove|copy|move), "path" (JSON Pointer starting with /), '
            '"value" (for add/replace), "from" (for copy/move). '
            "CRITICAL: To APPEND to an array, path MUST end with /-  "
            'Example: {"op":"add","path":"/sections/0/fields/-","value":{...}}'
        ),
    )


@tool("apply_patches", args_schema=ApplyPatchesArgs)
def apply_patches_tool(patches: list[dict[str, Any]]) -> str:
    """Applies changes to the JSON Document using RFC 6902 (JSON Patch).

    CRITICAL ARRAY RULES:
    - To APPEND an item to an array, use "/-" at the end of the path:
      {"op":"add","path":"/sections/0/fields/-","value":{...}}
    - NEVER use "add" directly on a path that holds an existing array — this REPLACES the entire array and DESTROYS all previous data!
    - To add multiple items, use separate operations each with "/-":
      [{"op":"add","path":"/sections/-","value":{...}}, {"op":"add","path":"/sections/-","value":{...}}]

    Operations:
    - "add" with "/-": Append item to an array. "add" with "/key": Set a new property on an object.
    - "replace": Overwrite a SINGLE existing value (only for scalars, never for arrays).
    - "remove": Delete a specific key or array element by index.
    - "move"/"copy": Relocate or duplicate values.

    Best practices:
    - ALWAYS use "/-" to append to arrays.
    - Batch multiple appends in a single call.
    - Verify state with read_value before replace/remove/move.
    """
    raise NotImplementedError("Execution handled by execute_tools_node")


# ======================================================================
# update_guidance
# ======================================================================
class UpdateGuidanceArgs(BaseModel):
    last_path: str = Field(
        default="",
        description=(
            "Last JSON Pointer written to. "
            'Example: "/sections/2/fields/-"'
        ),
    )
    sections_snapshot: str = Field(
        default="",
        description=(
            "Compact map of current document state: section names with item "
            "counts and build status. Use abbreviations. "
            'Example: "[0]INFO_GERAIS(12flds) [1]MERCADO(8flds,1tbl) [2]DESEMP(building)"'
        ),
    )
    items_added: str = Field(
        default="",
        description=(
            "Compact list of items ADDED in this chunk, including key values. "
            'Example: "3 fields→DESEMPENHO: Rent.Mensal=2.3%, DY=0.8%, PL=R$1.2bi"'
        ),
    )
    open_section: str = Field(
        default="",
        description=(
            "Section still being built (name + path + what is incomplete). "
            'Example: "DESEMPENHO @ /sections/2 — rentabilidade table incomplete, 6/12 months"'
        ),
    )
    text_excerpt: str = Field(
        default="",
        description=(
            "Key text fragment from the END of the chunk that provides "
            "continuity for the next chunk — e.g. a table cut mid-row, "
            "an incomplete sentence, trailing data. Max ~200 chars. "
            'Example: "...aluguel mensal: R$ 18,50/m² | vacância: 5,2% | ..."'
        ),
    )
    next_expectations: str = Field(
        default="",
        description=(
            "What the next chunk likely contains based on document flow, "
            "so the agent knows what to look for. "
            'Example: "expect remaining rentabilidade rows (Jul-Dec), then CARTEIRA DE ATIVOS section"'
        ),
    )
    pending_data: str = Field(
        default="",
        description=(
            "Partial, ambiguous or unresolved data from this chunk that "
            "may be clarified later (values TBD, cross-references, etc). "
            'Example: "aluguel \'a definir\'; ref contrato #42 sem detalhes"'
        ),
    )
    extracted_entities_count: int = Field(
        default=0,
        description="Number of entities (fields, rows, items) extracted in this chunk.",
    )


@tool("update_guidance", args_schema=UpdateGuidanceArgs)
def update_guidance_tool(
    last_path: str = "",
    sections_snapshot: str = "",
    items_added: str = "",
    open_section: str = "",
    text_excerpt: str = "",
    next_expectations: str = "",
    pending_data: str = "",
    extracted_entities_count: int = 0,
) -> str:
    """Finalizes the current chunk and creates rich context for the next chunk.

    This MUST be your FINAL tool call when the TextChunk is fully processed and
    all writes are confirmed. It MUST be the ONLY tool call in the response.

    Fill every field with dense, abbreviated info. This is the ONLY bridge
    between chunks — the next invocation sees ONLY this guidance plus the
    document skeleton. Be specific: include actual values, section names,
    paths, and text fragments that matter for continuity.
    """
    raise NotImplementedError("Execution handled by execute_tools_node")


# ======================================================================
# List of all tools for bind_tools
# ======================================================================
ALL_TOOLS = [
    inspect_keys_tool,
    search_pointer_tool,
    read_value_tool,
    apply_patches_tool,
    update_guidance_tool,
]
