import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from text_to_json.agent.prompts import build_system_prompt, build_user_message
from text_to_json.agent.state import AgentState
from text_to_json.clients import get_chat_model
from text_to_json.chunking.semantic import chunk_with_fallback
from text_to_json.settings import get_settings
from text_to_json.misc.truncator import Truncator, TruncatorConfig
from text_to_json.tools.apply_patches import apply_patches
from text_to_json.tools.inspect_keys import inspect_keys
from text_to_json.tools.json_pointer import parse_json_pointer_lenient
from text_to_json.tools.read_value import read_value
from text_to_json.tools.search_pointer import search_pointer
from text_to_json.tools.update_guidance import update_guidance

logger = logging.getLogger(__name__)

# ── Guard thresholds ─────────────────────────────────────────────────
# Minimum number of leaf values in a document before the shrinkage guard
# kicks in. Small documents are not checked.
SHRINKAGE_GUARD_MIN_ITEMS: int = 10
# Maximum ratio of (new / old) leaf count allowed after a patch batch.
# Below this ratio the patch is rejected as a likely destructive overwrite.
SHRINKAGE_GUARD_RATIO: float = 0.5
# Same concept but for the pre-validation replace check on objects.
DATA_LOSS_MIN_ITEMS: int = 5
# Default number of recent LLM rounds kept when trimming context.
DEFAULT_KEEP_LAST_N_ROUNDS: int = 2


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


def chunk_text_node(state: AgentState) -> dict[str, Any]:
    """
    Node that divides the text into semantic chunks.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state with the chunks.
    """
    text = state.get("text", "")

    if not text:
        return {
            "chunks": [],
            "current_chunk_idx": 0,
            "json_document": {},
            "error": "No text provided for processing.",
        }

    chunks = chunk_with_fallback(text)

    return {
        "chunks": chunks,
        "current_chunk_idx": 0,
        "json_document": {},
        "guidance": {},
        "is_chunk_finalized": False,
        "iteration_count": 0,
        "max_iterations": state.get("max_iterations", 20),
    }


def prepare_chunk_node(state: AgentState) -> dict[str, Any]:
    """
    Node that prepares the next chunk for processing.

    Resets the message history with a new SystemMessage + HumanMessage
    for the new chunk. The messages_reducer detects the SystemMessage
    and replaces all previous messages.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state for the new chunk.
    """
    chunks = state.get("chunks", [])
    current_idx = state.get("current_chunk_idx", 0)

    if current_idx >= len(chunks):
        return {"current_chunk": ""}

    current_chunk = chunks[current_idx]
    retry_count = state.get("chunk_retry_count", 0)
    max_retries = state.get("max_chunk_retries", 0)

    system_prompt = build_system_prompt(
        target_schema=state.get("target_schema"),
        previous_guidance=state.get("guidance"),
        json_skeleton=state.get("json_document", {}),
    )

    user_message = build_user_message(
        text_chunk=current_chunk,
        chunk_index=current_idx,
        total_chunks=len(chunks),
    )

    if retry_count > 0:
        logger.warning(
            "Chunk %d/%d: retry %d/%d (hit max iterations without finalizing)",
            current_idx + 1,
            len(chunks),
            retry_count,
            max_retries,
        )
        user_message += (
            f"\n\n⚠️ RETRY ATTEMPT {retry_count}/{max_retries}: "
            f"The previous processing of this chunk reached the maximum "
            f"number of iterations without finalizing. The document already "
            f"contains partial progress from the previous attempt. "
            f"Focus on extracting any remaining data and MAKE SURE to call "
            f"update_guidance to finalize this chunk."
        )

    return {
        "current_chunk": current_chunk,
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
        "is_chunk_finalized": False,
        "iteration_count": 0,
    }


_TRIM_SUMMARY = (
    "[CONTEXT TRIMMED: {removed} previous iteration(s) removed to free "
    "context space. All successful patches are already applied to the "
    "document. Use inspect_keys to check the current document state before "
    "continuing extraction.]"
)


def _trim_messages(
    messages: list[BaseMessage],
    keep_last_n_rounds: int = DEFAULT_KEEP_LAST_N_ROUNDS,
) -> list[BaseMessage] | None:
    """Trim old message rounds to free context space.

    Keeps the SystemMessage, the original HumanMessage (text chunk), and the
    last *keep_last_n_rounds* complete rounds (AIMessage + its ToolMessages).
    Injects a compact summary HumanMessage so the model knows context was
    trimmed.

    A "round" starts with an AIMessage and includes all consecutive
    ToolMessages that follow it.  Rounds are never split — this preserves
    the AI/Tool pairing required by the OpenAI API.

    Returns the trimmed list, or ``None`` if there are not enough rounds
    to trim (caller should treat None as "trimming won't help").
    """
    # ── 1. Separate prefix (before first AIMessage) from the rest ──
    prefix: list[BaseMessage] = []
    rest: list[BaseMessage] = []
    hit_ai = False

    for msg in messages:
        if not hit_ai and isinstance(msg, AIMessage):
            hit_ai = True
        if hit_ai:
            rest.append(msg)
        else:
            prefix.append(msg)

    # ── 2. Group rest into rounds (AIMessage + consecutive ToolMessages) ──
    rounds: list[list[BaseMessage]] = []
    current_round: list[BaseMessage] = []

    for msg in rest:
        if isinstance(msg, AIMessage) and current_round:
            rounds.append(current_round)
            current_round = [msg]
        else:
            current_round.append(msg)
    if current_round:
        rounds.append(current_round)

    if len(rounds) <= keep_last_n_rounds:
        return None  # Not enough rounds to trim

    # ── 3. Build clean prefix (SystemMessage + first HumanMessage only) ──
    #    Skips any previously injected summary HumanMessages.
    clean_prefix: list[BaseMessage] = []
    found_human = False

    for msg in prefix:
        if isinstance(msg, SystemMessage):
            clean_prefix.append(msg)
        elif isinstance(msg, HumanMessage) and not found_human:
            clean_prefix.append(msg)
            found_human = True

    # ── 4. Assemble trimmed list ──
    kept_rounds = rounds[-keep_last_n_rounds:]
    removed_count = len(rounds) - keep_last_n_rounds

    summary = HumanMessage(
        content=_TRIM_SUMMARY.format(removed=removed_count)
    )

    trimmed: list[BaseMessage] = list(clean_prefix) + [summary]
    for rnd in kept_rounds:
        trimmed.extend(rnd)

    return trimmed


def _extract_token_usage(*responses: BaseMessage) -> dict[str, int]:
    """Build a token_usage delta dict from one or more LLM responses."""
    usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    for resp in responses:
        meta = getattr(resp, "usage_metadata", None)
        if not meta:
            continue
        usage["input_tokens"] += meta.get("input_tokens", 0)
        usage["output_tokens"] += meta.get("output_tokens", 0)
        usage["total_tokens"] += meta.get("total_tokens", 0)
        usage["cache_creation_input_tokens"] += meta.get(
            "cache_creation_input_tokens", 0
        )
        usage["cache_read_input_tokens"] += meta.get(
            "cache_read_input_tokens", 0
        )
        usage["llm_calls"] += 1
    return usage


def call_llm_node(state: AgentState) -> dict[str, Any]:
    """
    Node that calls the LLM with tools bound.

    The model receives the full message history (including previous
    ToolMessages from prior iterations) and can produce AIMessages
    with tool_calls.

    If the model responds without tool calls (typically because the context
    window is nearly full), trims old messages and retries once.  If the
    retry also produces no tool calls, the response is returned as-is and
    ``execute_tools_node`` will force-finalize the chunk.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state with the LLM response.
    """
    messages = state.get("messages", [])
    llm = get_chat_model()
    iteration = state.get("iteration_count", 0)

    # ── First attempt ──
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001 — intentionally broad
        # LLM SDKs raise diverse exception hierarchies (rate-limit,
        # context-too-long, network, JSON decoding, etc.).  The except
        # is intentionally broad so the trim-and-retry fallback can
        # handle *any* transient failure.
        logger.warning("LLM call failed (%s: %s), attempting trim+retry", type(exc).__name__, exc)
        trimmed = _trim_messages(messages)
        if trimmed is None:
            raise  # Can't trim further — propagate the error

        retry_response = llm.invoke(trimmed)
        # trimmed starts with SystemMessage → reducer replaces all messages
        return {
            "messages": trimmed + [retry_response],
            "iteration_count": iteration + 1,
            "token_usage": _extract_token_usage(retry_response),
        }

    # ── Check for tool calls ──
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "token_usage": _extract_token_usage(response),
        }

    trimmed = _trim_messages(messages)
    if trimmed is None:
        # Not enough rounds to trim — return as-is (force-finalize)
        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "token_usage": _extract_token_usage(response),
        }

    retry_response = llm.invoke(trimmed)
    # trimmed starts with SystemMessage → reducer replaces all messages
    return {
        "messages": trimmed + [retry_response],
        "iteration_count": iteration + 1,
        "token_usage": _extract_token_usage(response, retry_response),
    }


def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """
    Node that executes the tool calls from the LLM response.

    Reads tool_calls from the last AIMessage, dispatches each to
    the corresponding implementation function (with document state),
    and creates ToolMessage responses.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state with tool results, updated document, etc.
    """
    messages = state.get("messages", [])

    # Find the last AIMessage
    last_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_message = msg
            break

    if not last_message:
        return {"error": "No LLM response found."}

    # Check for tool calls
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return {"is_chunk_finalized": True}

    document = state.get("json_document", {})
    target_schema = state.get("target_schema")
    tool_messages: list[ToolMessage] = []
    new_document = document
    is_finalized = False
    new_guidance = state.get("guidance", {})

    settings = get_settings()
    truncator = _get_truncator()

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        call_id = tc["id"]

        try:
            result = _dispatch_tool(
                name, args, new_document, target_schema
            )

            # Handle side effects
            if name == "apply_patches":
                if result.get("ok"):
                    candidate = result.get("finalDoc", new_document)
                    # Post-patch shrinkage guard: reject if document
                    # lost significant content after the patch
                    old_count = _count_nested_items(new_document)
                    new_count = _count_nested_items(candidate)
                    if (
                        old_count > SHRINKAGE_GUARD_MIN_ITEMS
                        and new_count < old_count * SHRINKAGE_GUARD_RATIO
                    ):
                        result = {
                            "ok": False,
                            "errors": [
                                {
                                    "opIndex": -1,
                                    "op": None,
                                    "pointer": "/",
                                    "message": (
                                        f"SHRINKAGE GUARD: patches would reduce "
                                        f"document from {old_count} to "
                                        f"{new_count} values "
                                        f"({100 - int(new_count / old_count * 100)}% data loss). "
                                        f"This likely means you replaced a container "
                                        f"instead of appending. Use \"/-\" to "
                                        f"append to arrays, or update individual "
                                        f"fields instead of replacing objects."
                                    ),
                                }
                            ],
                            "finalDoc": new_document,
                        }
                    else:
                        new_document = candidate

            elif name == "update_guidance":
                new_guidance = result.get("guidance", new_guidance)
                is_finalized = True

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.warning("Tool '%s' raised %s: %s", name, type(e).__name__, e)
            result = {"error": f"Tool execution error: {e}"}

        # Truncate read_value results to avoid blowing up context
        if name == "read_value":
            content = truncator.truncate_with_limit(
                result, settings.TRUNCATE_READ_VALUE_LIMIT
            )
        elif name == "apply_patches":
            response = {k: v for k, v in result.items() if k != "finalDoc"}
            content = json.dumps(response, ensure_ascii=False, default=str)
        else:
            content = json.dumps(result, ensure_ascii=False, default=str)

        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=call_id,
            )
        )

    updates: dict[str, Any] = {
        "messages": tool_messages,
        "json_document": new_document,
    }

    if is_finalized:
        updates["guidance"] = new_guidance
        updates["is_chunk_finalized"] = True

    return updates


def _resolve_path(document: Any, path: str) -> tuple[bool, Any]:
    """Navigate a JSON document following a JSON Pointer path.

    Returns ``(found, value)`` where *found* indicates whether the path
    resolved to an existing location in the document.
    """
    if not path or path == "/":
        return True, document
    try:
        tokens = parse_json_pointer_lenient(path)
    except ValueError:
        return False, None
    current = document
    for t in tokens:
        if isinstance(current, dict) and t in current:
            current = current[t]
        elif isinstance(current, list):
            if t == "-":
                return False, None
            try:
                idx = int(t)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return False, None
            except ValueError:
                return False, None
        else:
            return False, None
    return True, current


def _count_nested_items(value: Any) -> int:
    """Count the total number of leaf values inside a nested structure."""
    if isinstance(value, dict):
        return sum(_count_nested_items(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_nested_items(v) for v in value)
    return 1


def _make_error(
    index: int, patch: dict, path: str, message: str
) -> dict[str, Any]:
    return {
        "opIndex": index,
        "op": patch,
        "pointer": path,
        "message": message,
    }


def _filter_duplicate_appends(
    patches: list[dict[str, Any]],
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Filter out 'add' operations that would append an item identical to one
    already present in the target array (or already queued in the same batch).

    Uses canonical JSON serialization (sorted keys) for O(1) lookups.

    Args:
        patches: List of JSON Patch operations.
        document: Current JSON document.

    Returns:
        Tuple of (filtered_patches, skipped_messages).
    """
    filtered: list[dict[str, Any]] = []
    skipped: list[str] = []

    # Pre-build canonical sets for arrays already in the document.
    # Lazily populated per array_path on first access.
    _existing_cache: dict[str, set[str]] = {}

    def _get_existing_set(array_path: str) -> set[str]:
        if array_path not in _existing_cache:
            found, current_array = _resolve_path(document, array_path)
            if found and isinstance(current_array, list):
                _existing_cache[array_path] = {
                    json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                    for item in current_array
                }
            else:
                _existing_cache[array_path] = set()
        return _existing_cache[array_path]

    # Track items queued for addition in this batch (catches intra-batch dupes)
    batch_additions: dict[str, set[str]] = {}

    for patch in patches:
        if not (
            isinstance(patch, dict)
            and patch.get("op") == "add"
            and isinstance(patch.get("path"), str)
            and patch["path"].endswith("/-")
        ):
            # Not an array-append operation — keep as-is
            filtered.append(patch)
            continue

        array_path = patch["path"][:-2]  # strip trailing /-
        new_value = patch.get("value")
        canonical = json.dumps(
            new_value, sort_keys=True, ensure_ascii=False, default=str
        )

        existing_set = _get_existing_set(array_path)
        batch_set = batch_additions.get(array_path, set())

        if canonical in existing_set or canonical in batch_set:
            preview = canonical[:120]
            skipped.append(
                f'DUPLICATE SKIPPED at "{patch["path"]}": '
                f"identical item already exists in the array. "
                f"Preview: {preview}"
            )
            continue

        # Accept the patch and record it for intra-batch dedup
        batch_additions.setdefault(array_path, set()).add(canonical)
        filtered.append(patch)

    return filtered, skipped


def _check_invalid_path(
    i: int, patch: dict, path: str,
) -> dict[str, Any] | None:
    """Check 1: path must start with '/'."""
    if path and not path.startswith("/"):
        return _make_error(
            i, patch, path,
            f'Invalid JSON Pointer: "{path}" must start with "/". '
            f'Did you mean "/{path}"?',
        )
    return None


def _check_add_on_existing_array(
    i: int, patch: dict, path: str, value: Any, document: dict[str, Any],
) -> dict[str, Any] | None:
    """Check 2: 'add' that would overwrite an existing array."""
    if not path:
        return None
    found, current = _resolve_path(document, path)
    if not found or not isinstance(current, list):
        return None

    n = len(current)
    if isinstance(value, list):
        return _make_error(
            i, patch, path,
            f'DESTRUCTIVE OVERWRITE: "{path}" already contains an '
            f"array with {n} items. Your \"add\" would REPLACE ALL "
            f"existing data with a new array of {len(value)} items. "
            f"To APPEND items, use \"{path}/-\" for each: "
            f'[{{"op":"add","path":"{path}/-","value":item1}}, ...]',
        )
    if isinstance(value, dict):
        return _make_error(
            i, patch, path,
            f'DESTRUCTIVE OVERWRITE: "{path}" already contains an '
            f"array with {n} items. Your \"add\" would REPLACE the "
            f"entire array with a single object. "
            f'To APPEND, use "{path}/-": '
            f'{{"op":"add","path":"{path}/-","value":{{...}}}}',
        )
    # Scalar value replacing an array
    return _make_error(
        i, patch, path,
        f'TYPE DOWNGRADE: "{path}" already contains an '
        f"array with {n} items. Your \"add\" would REPLACE "
        f"the entire array with a {type(value).__name__}. "
        f'To APPEND, use "{path}/-": '
        f'{{"op":"add","path":"{path}/-","value":...}}',
    )


def _check_add_at_root(
    i: int, patch: dict, path: str, document: dict[str, Any],
) -> dict[str, Any] | None:
    """Check 3: 'add' at root replaces entire document."""
    item_count = _count_nested_items(document)
    if item_count > 0:
        return _make_error(
            i, patch, path,
            f"DESTRUCTIVE: \"add\" at root would REPLACE the entire "
            f"document ({item_count} existing values). "
            f"Add to specific paths instead (e.g., /metadata, /sections/-).",
        )
    return None


def _check_replace_container(
    i: int, patch: dict, path: str, value: Any, document: dict[str, Any],
) -> dict[str, Any] | None:
    """Check 4: 'replace' on a container (array/object)."""
    if not path:
        return None
    found, current = _resolve_path(document, path)
    if not found:
        return None

    if isinstance(current, list) and len(current) > 0:
        return _make_error(
            i, patch, path,
            f'DESTRUCTIVE REPLACE: "{path}" is an array with '
            f"{len(current)} items. Replacing it would DISCARD all "
            f"existing data. To update specific items, use "
            f'"replace" on individual indices (e.g., "{path}/0/value"). '
            f'To append new items, use "add" with "{path}/-".',
        )
    if isinstance(current, dict) and len(current) > 0:
        nested = _count_nested_items(current)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return _make_error(
                i, patch, path,
                f'TYPE DOWNGRADE: "{path}" is an object with '
                f"{len(current)} keys ({nested} nested values). "
                f"Replacing it with a {type(value).__name__} would "
                f"DESTROY all nested data. To update a specific "
                f"field, use \"{path}/fieldName\" as the path.",
            )
        if isinstance(value, dict):
            old_count = nested
            new_count = _count_nested_items(value)
            if new_count < old_count * SHRINKAGE_GUARD_RATIO and old_count > DATA_LOSS_MIN_ITEMS:
                return _make_error(
                    i, patch, path,
                    f'SIGNIFICANT DATA LOSS: replacing "{path}" '
                    f"would reduce content from {old_count} to "
                    f"{new_count} values ({100 - int(new_count / old_count * 100)}% loss). "
                    f"Consider updating individual fields instead.",
                )
    return None


def _check_remove_container(
    i: int, patch: dict, path: str, document: dict[str, Any],
) -> dict[str, Any] | None:
    """Check 5: 'remove' on a container with significant data."""
    if not path:
        return None
    found, current = _resolve_path(document, path)
    if not found:
        return None
    path_depth = len(path.split("/")) - 1  # /a/b/c → 3
    nested = _count_nested_items(current)

    if isinstance(current, list) and len(current) > 0:
        return _make_error(
            i, patch, path,
            f'DATA LOSS WARNING: removing "{path}" would delete '
            f"an array with {len(current)} items ({nested} total "
            f"nested values). If you need to remove specific items, "
            f"use their full path (e.g., \"{path}/0\").",
        )
    if (
        isinstance(current, dict)
        and nested > 2
        and path_depth <= 3
    ):
        return _make_error(
            i, patch, path,
            f'DATA LOSS WARNING: removing "{path}" would delete '
            f"an object with {len(current)} keys ({nested} total "
            f"nested values). If you need to remove specific fields, "
            f"use their full path (e.g., \"{path}/fieldName\").",
        )
    return None


def _check_type_downgrade(
    i: int, patch: dict, path: str, value: Any, document: dict[str, Any],
) -> dict[str, Any] | None:
    """Check 6: scalar replacing a container (type downgrade)."""
    if not path or value is None:
        return None
    found, current = _resolve_path(document, path)
    if not found or current is None:
        return None
    cur_is_container = isinstance(current, (list, dict))
    val_is_scalar = isinstance(value, (str, int, float, bool))
    if cur_is_container and val_is_scalar:
        nested = _count_nested_items(current)
        if nested > 1:
            ctype = "array" if isinstance(current, list) else "object"
            return _make_error(
                i, patch, path,
                f'TYPE DOWNGRADE: "{path}" is a {ctype} with '
                f"{nested} nested values. Replacing it with a "
                f"{type(value).__name__} ({repr(value)[:60]}) would "
                f"DESTROY all nested data. Update specific fields instead.",
            )
    return None


def _pre_validate_patches(
    patches: list[dict[str, Any]],
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pre-validate patches for common LLM mistakes before sending to the
    full schema validator.

    Catches destructive operations that would silently discard accumulated
    data, returning prescriptive error messages.
    """
    errors: list[dict[str, Any]] = []

    for i, patch in enumerate(patches):
        if not isinstance(patch, dict):
            continue
        op = patch.get("op")
        path = patch.get("path", "")
        value = patch.get("value")

        # 1. Invalid path format
        err = _check_invalid_path(i, patch, path)
        if err:
            errors.append(err)
            continue

        # 2. "add" on existing array → destructive overwrite
        if op == "add":
            err = _check_add_on_existing_array(i, patch, path, value, document)
            if err:
                errors.append(err)
                continue

        # 3. "add" at root → replaces entire document
        if op == "add" and (path == "" or path == "/"):
            err = _check_add_at_root(i, patch, path, document)
            if err:
                errors.append(err)
                continue

        # 4. "replace" on a container
        if op == "replace":
            err = _check_replace_container(i, patch, path, value, document)
            if err:
                errors.append(err)
                continue

        # 5. "remove" on a container with significant data
        if op == "remove":
            err = _check_remove_container(i, patch, path, document)
            if err:
                errors.append(err)
                continue

        # 6. Type downgrade (scalar replacing container)
        if op in ("add", "replace"):
            err = _check_type_downgrade(i, patch, path, value, document)
            if err:
                errors.append(err)
                continue

    return errors


def _resolve_source(
    source: str,
    document: dict[str, Any],
    target_schema: Any,
) -> dict[str, Any]:
    """Select the data source based on the 'source' parameter.

    Args:
        source: "document" or "schema".
        document: Current JSON document (skeleton).
        target_schema: Optional target schema.

    Returns:
        The selected data source dict.

    Raises:
        ValueError: If source is "schema" but no target_schema is available.
    """
    if source == "schema":
        if target_schema is None:
            raise ValueError(
                "No target schema available. The workflow was started without "
                "a schema, so source='schema' cannot be used. "
                "Use source='document' instead."
            )
        return target_schema
    return document


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    document: dict[str, Any],
    target_schema: Any,
) -> dict[str, Any]:
    """
    Dispatch a tool call to the corresponding implementation function.

    Args:
        name: Tool name.
        args: Tool arguments from the LLM.
        document: Current JSON document.
        target_schema: Optional target schema.

    Returns:
        Tool result dict.
    """
    if name == "inspect_keys":
        source_doc = _resolve_source(
            args.get("source", "document"), document, target_schema
        )
        return inspect_keys(
            source_doc,
            args.get("path", ""),
        )

    if name == "search_pointer":
        source_doc = _resolve_source(
            args.get("source", "document"), document, target_schema
        )
        return search_pointer(
            source_doc,
            {
                "query": args.get("query", ""),
                "type": args.get("type", "value"),
                "fuzzy_match": args.get("fuzzy_match", False),
                "limit": args.get("limit", 20),
                "max_value_length": args.get("max_value_length", 120),
            },
        )

    if name == "read_value":
        source_doc = _resolve_source(
            args.get("source", "document"), document, target_schema
        )
        return read_value(
            source_doc,
            {
                "path": args.get("path", ""),
                "max_string_length": args.get("max_string_length", 160),
                "max_depth": args.get("max_depth", 6),
                "max_array_items": args.get("max_array_items", 50),
                "max_object_keys": args.get("max_object_keys", 50),
            },
        )

    if name == "apply_patches":
        if "patches" not in args or not args["patches"]:
            return {
                "ok": False,
                "errors": [
                    {
                        "opIndex": -1,
                        "op": None,
                        "pointer": "/",
                        "message": (
                            "No patches provided. You must include a 'patches' "
                            "array with at least one operation. "
                            'Example: {"op":"add","path":"/sections/-","value":{"section_name":"..."}}'
                        ),
                    }
                ],
                "finalDoc": document,
            }
        patches = args["patches"]
        # Pre-validate common mistakes and provide prescriptive hints
        pre_errors = _pre_validate_patches(patches, document)
        if pre_errors:
            return {
                "ok": False,
                "errors": pre_errors,
                "finalDoc": document,
            }
        # Filter out exact-duplicate appends (same item already in array)
        patches, dup_messages = _filter_duplicate_appends(patches, document)
        if not patches:
            # All patches were duplicates — nothing to apply
            return {
                "ok": True,
                "errors": [],
                "finalDoc": document,
                "duplicates_skipped": dup_messages,
            }
        result = apply_patches(document, patches, target_schema)
        if dup_messages:
            result["duplicates_skipped"] = dup_messages
        return result

    if name == "update_guidance":
        return update_guidance(
            last_path=args.get("last_path", ""),
            sections_snapshot=args.get("sections_snapshot", ""),
            items_added=args.get("items_added", ""),
            open_section=args.get("open_section", ""),
            text_excerpt=args.get("text_excerpt", ""),
            next_expectations=args.get("next_expectations", ""),
            pending_data=args.get("pending_data", ""),
            extracted_entities_count=args.get("extracted_entities_count", 0),
        )

    return {"error": f"Unknown tool: {name}"}


def finalize_chunk_node(state: AgentState) -> dict[str, Any]:
    """
    Node that finalizes the current chunk processing and advances to the next.

    If the chunk was force-finalized (hit max_iterations without the LLM
    calling update_guidance) and there are retries remaining, the chunk index
    is NOT advanced so that ``prepare_chunk_node`` will reprocess the same
    chunk with fresh messages but the accumulated document state.

    Clears any chunk-level error so it doesn't prevent subsequent chunks
    from being processed.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state for the next chunk.
    """
    current_idx = state.get("current_chunk_idx", 0)
    is_properly_finalized = state.get("is_chunk_finalized", False)
    retry_count = state.get("chunk_retry_count", 0)
    max_retries = state.get("max_chunk_retries", 0)

    # Check if chunk was force-finalized (max iterations or error, without
    # the LLM explicitly calling update_guidance).
    # Note: is_chunk_finalized is set to True only when update_guidance is
    # called in execute_tools_node. When is_chunk_done routes here due to
    # max_iterations, is_chunk_finalized remains False.
    chunks = state.get("chunks", [])
    total_chunks = len(chunks)

    if not is_properly_finalized and retry_count < max_retries:
        # Retry: keep same chunk index, increment retry counter
        logger.info(
            "Chunk %d/%d: scheduling retry %d/%d",
            current_idx + 1,
            total_chunks,
            retry_count + 1,
            max_retries,
        )
        return {
            "is_chunk_finalized": False,
            "chunk_retry_count": retry_count + 1,
            "error": None,
        }

    if not is_properly_finalized and retry_count >= max_retries:
        logger.warning(
            "Chunk %d/%d: max retries exhausted (%d), advancing to next chunk",
            current_idx + 1,
            total_chunks,
            max_retries,
        )

    # Normal finalization or retries exhausted: advance to next chunk
    return {
        "current_chunk_idx": current_idx + 1,
        "is_chunk_finalized": False,
        "chunk_retry_count": 0,
        "error": None,
    }


def has_more_chunks(state: AgentState) -> Literal["call_llm", "__end__"]:
    """
    Check if there are more chunks to process.

    Args:
        state: Current state of the agent.

    Returns:
        "call_llm" if there are more chunks, "__end__" if not.
    """
    chunks = state.get("chunks", [])
    current_idx = state.get("current_chunk_idx", 0)

    if state.get("error"):
        return "__end__"

    if current_idx < len(chunks):
        return "call_llm"

    return "__end__"


def is_chunk_done(state: AgentState) -> Literal["finalize_chunk", "call_llm"]:
    """
    Check if the current chunk has been finalized or needs more iterations.

    Args:
        state: Current state of the agent.

    Returns:
        "finalize_chunk" if finalized, "call_llm" if needs more iterations.
    """
    if state.get("is_chunk_finalized"):
        return "finalize_chunk"

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 20)

    if iteration_count >= max_iterations:
        return "finalize_chunk"

    if state.get("error"):
        return "finalize_chunk"

    return "call_llm"
