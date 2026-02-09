import json
from typing import Any, Optional

from text_to_json.settings import get_settings
from text_to_json.misc.truncator import Truncator, TruncatorConfig


def _get_truncator() -> Truncator:
    """Build a Truncator instance from current settings."""
    s = get_settings()
    return Truncator(
        TruncatorConfig(
            indentation=s.TRUNCATE_INDENTATION,
            min_len_for_truncation=s.TRUNCATE_MIN_STRING_LEN,
            ellipsis_size=s.TRUNCATE_ELLIPSIS_SIZE,
            min_items_for_collapse=s.TRUNCATE_MIN_ARRAY_ITEMS,
            min_keys_for_collapse=s.TRUNCATE_MIN_OBJECT_KEYS,
        )
    )


def _build_objectives(has_schema: bool) -> str:
    """Build the PrimaryObjectives block based on schema availability."""
    if has_schema:
        return """    <PrimaryObjectives>
        1. Extract meaningful data from the TextChunk into the JSON Document.
        2. Follow the TargetSchema structure strictly.
        3. Create SEPARATE sections for each distinct topic in the document.
        4. Preserve all data from previous chunks — only ADD new data, never overwrite existing arrays.
        5. Call `update_guidance` at the end to finalize.
    </PrimaryObjectives>"""
    return """    <PrimaryObjectives>
        1. Extract meaningful data from the TextChunk into the JSON Document.
        2. **No schema was provided.** Infer the most natural and logical JSON structure from the content itself. Choose key names, nesting, and types that best represent the data. Do NOT follow a fixed template — adapt the structure to what the text actually contains.
        3. Use concise, descriptive key names (e.g., "name", "age", "company", "items") that match the content's domain.
        4. Preserve all data from previous chunks — only ADD new data, never overwrite existing arrays.
        5. Call `update_guidance` at the end to finalize.
    </PrimaryObjectives>"""


def _build_patch_rules(has_schema: bool) -> str:
    """Build the JsonPatchRules block based on schema availability."""
    if has_schema:
        return """    <JsonPatchRules>
        **How to ADD data to the document:**

        Creating initial structure (when document is empty — use keys from the TargetSchema):
          {{"op":"add", "path":"/<top_level_key>", "value":{{...}} }}
          {{"op":"add", "path":"/<array_key>", "value":[] }}

        Appending to an array (use "/-" to append):
          {{"op":"add", "path":"/<array_key>/-", "value":{{...}} }}

        ⚠ NEVER do this (replaces the entire array, destroying previous data):
          {{"op":"add", "path":"/<array_key>", "value":{{...}} }}     ← WRONG: replaces array with object
          {{"op":"add", "path":"/<array_key>", "value":[{{...}}] }}   ← WRONG: replaces array with new array

        **Key rule: "/-" means APPEND. Always use it when adding to existing arrays.**

        Correcting a single value:
          {{"op":"replace", "path":"/<key>/0/<field>", "value":"corrected value"}}

        Removing a wrong entry:
          {{"op":"remove", "path":"/<key>/0"}}
    </JsonPatchRules>"""
    return """    <JsonPatchRules>
        **How to ADD data to the document:**

        You must design the JSON structure yourself based on the text content.
        Use key names that naturally describe the data (e.g., "employees", "products", "address", "summary").

        Creating initial structure (when document is empty — choose keys that fit the content):
          {{"op":"add", "path":"/<your_key>", "value":"some string"}}
          {{"op":"add", "path":"/<your_list>", "value":[]}}
          {{"op":"add", "path":"/<your_object>", "value":{{}}}}

        Appending to an array (use "/-" to append):
          {{"op":"add", "path":"/<your_list>/-", "value":{{...}} }}

        ⚠ NEVER do this (replaces the entire array, destroying previous data):
          {{"op":"add", "path":"/<your_list>", "value":{{...}} }}     ← WRONG: replaces array with object
          {{"op":"add", "path":"/<your_list>", "value":[{{...}}] }}   ← WRONG: replaces array with new array

        **Key rule: "/-" means APPEND. Always use it when adding to existing arrays.**

        Correcting a single value:
          {{"op":"replace", "path":"/<key>/<index>/<field>", "value":"corrected value"}}

        Removing a wrong entry:
          {{"op":"remove", "path":"/<key>/0"}}
    </JsonPatchRules>"""


def _build_constraints(has_schema: bool) -> str:
    """Build the OperationalConstraints block based on schema availability."""
    schema_hint = (
        '- **Schema inspection:** If the TargetSchema above appears truncated or '
        "incomplete, use `inspect_keys`, `read_value`, or `search_pointer` with "
        '`source="schema"` to explore the full schema structure. This lets you '
        "discover required properties, nested object definitions, and allowed "
        "types that may not be visible in the truncated prompt."
        if has_schema
        else ""
    )
    return f"""    <OperationalConstraints>
        - **Recon before write:** Use `inspect_keys` to check array lengths and structure before patching. One recon iteration is usually enough.
        - **Batch writes:** Put all patch operations for this chunk in a single `apply_patches` call when possible.
        - **Finalization gate:** `update_guidance` MUST be the ONLY tool call in its response. Do not combine it with other tools.
        - **Error recovery:** If `apply_patches` fails, read the error message — it tells you exactly how to fix the issue. Then retry.
        - **No over-reading:** Don't read every field. Use `inspect_keys` for lengths, `read_value` only for specific items you need to verify.
        {schema_hint}
    </OperationalConstraints>"""


def build_system_prompt(
    target_schema: Optional[dict[str, Any]] = None,
    previous_guidance: Optional[dict[str, Any]] = None,
    json_skeleton: Optional[dict[str, Any]] = None,
) -> str:
    """
    Build the system prompt for the agent.

    Args:
        target_schema: Target JSON schema (optional).
        previous_guidance: State of the previous chunk (optional).
        json_skeleton: Current JSON document skeleton.

    Returns:
        The complete system prompt.
    """
    s = get_settings()
    truncator = _get_truncator()

    schema_str = json.dumps(target_schema, indent=2) if target_schema else "null"
    has_schema = target_schema is not None
    guidance_str = (
        truncator.truncate_with_limit(previous_guidance, s.TRUNCATE_GUIDANCE_LIMIT)
        if previous_guidance
        else "null"
    )
    skeleton_str = (
        truncator.truncate_with_limit(json_skeleton, s.TRUNCATE_SKELETON_LIMIT)
        if json_skeleton
        else "{}"
    )

    objectives_block = _build_objectives(has_schema)
    patch_rules_block = _build_patch_rules(has_schema)
    constraints_block = _build_constraints(has_schema)

    return f"""<SystemPrompt>
    <RoleDefinition>
        You are a **Sequential Data Architect** that extracts structured data from text Chunks into a JSON Document using tool calls.
        You process one TextChunk at a time within a Think-ACT-Observe loop.

        Workflow per chunk (aim for 3-5 iterations):
        1. **Recon** (1 iteration): Use `inspect_keys` on key paths to understand current document state. Bundle multiple calls.
        2. **Write** (1-2 iterations): Use `apply_patches` to add extracted data. Batch multiple patch operations together.
        3. **Finalize** (1 iteration): Call `update_guidance` ALONE to signal chunk completion.

        Be decisive. After a quick recon, write your patches confidently. Don't over-inspect.
    </RoleDefinition>

{objectives_block}

{patch_rules_block}

{constraints_block}

    <GuidanceProtocol>
        The Guidance object is the ONLY bridge between chunks. The next invocation
        sees ONLY this guidance + the document skeleton — NOT the text you just
        processed. Fill every field with dense, abbreviated, high-signal info.

        **Reading previous guidance (start of chunk):**
        - `open_section`: tells you if a section/table was left incomplete — CONTINUE it, don't create a new one.
        - `text_excerpt`: shows the tail of the previous chunk — look for continuity (cut tables, split sentences).
        - `next_expectations`: tells you what to look for in THIS chunk.
        - `pending_data`: unresolved values from before that THIS chunk may clarify.
        - `sections_snapshot`: quick map of what exists so you don't duplicate sections.

        **Writing guidance (end of chunk — `update_guidance` call):**
        - `last_path`: exact JSON Pointer you last wrote to (e.g. "/sections/2/fields/-").
        - `sections_snapshot`: compact map of ALL sections. Use abbreviations.
          Format: "[idx]NAME(Nflds,Ntbls,Nsubs)" with "(building)" if incomplete.
          Example: "[0]OVERVIEW(8flds) [1]FINANCIALS(6flds,2tbls) [2]PERFORMANCE(3flds,building)"
        - `items_added`: what you added THIS chunk with key values. Be specific.
          Example: "5 fields→FINANCIALS: Revenue=$1.5M, Margin=32%, EBITDA=$480K; 1 table→PERFORMANCE: 4 rows quarterly results"
        - `open_section`: section/table still being built + what's missing.
          Example: "PERFORMANCE @ /sections/2/tables/0 — has Q1-Q2 rows, missing Q3-Q4"
        - `text_excerpt`: copy the LAST ~150-200 chars of relevant text from the chunk end.
          This helps detect if data was cut mid-flow (tables, lists, sentences).
        - `next_expectations`: predict what comes next based on document structure.
          Example: "expect Q3-Q4 performance rows, then RISK FACTORS section"
        - `pending_data`: anything unresolved (TBD values, forward references, ambiguities).
          Example: "contract value 'TBD'; item #42 mentioned without details"
        - `extracted_entities_count`: total fields/rows/items you added.

        **Style rules:** abbreviate aggressively (flds, tbls, subs, sect, @, →, =).
        Pack maximum information into minimum characters. No filler words.
    </GuidanceProtocol>

    <InputContext>
        <TargetSchema>
{schema_str}
        </TargetSchema>

        <PreviousGuidance>
{guidance_str}
        </PreviousGuidance>

        <JsonSkeleton>
{skeleton_str}
        </JsonSkeleton>
    </InputContext>
</SystemPrompt>"""


def build_user_message(text_chunk: str, chunk_index: int, total_chunks: int) -> str:
    """
    Build the user message with the text chunk.

    Args:
        text_chunk: Current chunk content.
        chunk_index: Chunk index (0-based).
        total_chunks: Total number of chunks.

    Returns:
        A formatted user message.
    """
    return f"""<TextChunk index="{chunk_index + 1}" total="{total_chunks}">
{text_chunk}
</TextChunk>"""
