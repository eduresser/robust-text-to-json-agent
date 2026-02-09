from __future__ import annotations

from typing import Any


def update_guidance(
    last_path: str = "",
    sections_snapshot: str = "",
    items_added: str = "",
    open_section: str = "",
    text_excerpt: str = "",
    next_expectations: str = "",
    pending_data: str = "",
    extracted_entities_count: int = 0,
) -> dict[str, Any]:
    """
    Finalize the processing of the current chunk and create guidance for the next.

    Args:
        last_path: Last JSON Pointer written to (e.g. "/data/0/item/-").
        sections_snapshot: Compact map of document state — section names with
            item counts and status.
            Example: "[0]SECTION_A(5items) [1]SECTION_B(3items,1pending) [2]SECTION_C(incomplete)"
        items_added: Compact list of items added in THIS chunk, with key values.
            Example: "2 items→SECTION_B: Key1=Value1, Key2=Value2"
        open_section: Name and path of section still being built, if any.
            Example: "SECTION_B @ /data/1 — section incomplete"
        text_excerpt: Relevant text fragment from the END of the chunk that
            provides continuity (e.g. item cut mid-entry, a sentence that continues).
            Example: "...lorem ipsum dolor sit amet, consectetur adip..."
        next_expectations: What the next chunk likely contains based on document
            flow, so the agent knows what to look for.
            Example: "expect more items for SECTION_B, then possible new section SECTION_C"
        pending_data: Partial, ambiguous or unresolved data from this chunk that
            may be clarified in the next (e.g. "value pending", references).
            Example: "key 'amount' to be confirmed; unresolved reference to ID#123"
        extracted_entities_count: Number of entities extracted in this chunk.

    Returns:
        The new guidance object to be passed to the next chunk.
    """
    return {
        "finalized": True,
        "guidance": {
            "last_path": last_path,
            "sections_snapshot": sections_snapshot,
            "items_added": items_added,
            "open_section": open_section,
            "text_excerpt": text_excerpt,
            "next_expectations": next_expectations,
            "pending_data": pending_data,
            "extracted_entities_count": extracted_entities_count,
        },
    }
