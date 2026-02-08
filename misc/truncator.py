"""Smart JSON truncator with character limit.

Faithful Python port of the N8N Truncator class. Progressively truncates
JSON data to fit within a character limit using three strategies applied
in order:

1. **String truncation** – shorten long strings (binary-search for the
   optimal max length that still fits the limit).
2. **Array collapse** – remove middle items from arrays at the deepest
   level first, replacing with a ``...`` marker.
3. **Object collapse** – remove middle keys from objects at the deepest
   level first, replacing with a ``...`` marker.

The output is a custom-formatted string (not standard JSON) that uses
``...``, ``[...]`` and ``{...}`` markers for truncated sections.
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field
from typing import Any


# Sentinel value used internally to mark truncated locations.
_TRUNCATION_TOKEN = "__TRUNCATED__"


@dataclass(frozen=True)
class TruncatorConfig:
    """Configuration knobs for the truncation algorithm."""

    indentation: int = 4
    min_len_for_truncation: int = 23
    ellipsis_size: int = 3
    min_items_for_collapse: int = 2
    min_keys_for_collapse: int = 2


@dataclass
class _Node:
    """Collected metadata about a single node in the JSON tree."""

    type: str  # "string" | "array" | "object"
    value: Any
    depth: int
    path: list[str | int] = field(default_factory=list)
    # string-specific
    length: int = 0
    # array-specific
    size: int = 0
    # object-specific
    keys: int = 0


class Truncator:
    """Smart JSON truncator that progressively shrinks data to a char limit."""

    def __init__(self, config: TruncatorConfig | None = None) -> None:
        self._cfg = config or TruncatorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def truncate_with_limit(self, data: Any, limit: int) -> str:
        """Truncate *data* so its custom-formatted string fits in *limit* chars.

        Returns the custom-formatted string (with ``...`` markers where
        content was removed).
        """
        if data is None:
            return "null"
        working = copy.deepcopy(data)
        result = self._smart_truncate(working, limit)
        text = self._custom_stringify(result)
        return text.replace("...,\n", "...\n")

    # ------------------------------------------------------------------
    # Custom stringify (mirrors the JS #customStringify)
    # ------------------------------------------------------------------

    def _custom_stringify(self, data: Any, indent_level: int = 0) -> str:
        indent_str = " " * self._cfg.indentation
        indent = indent_str * indent_level

        if data == _TRUNCATION_TOKEN:
            return "..."

        if isinstance(data, list):
            if len(data) == 0:
                return "[]"
            if len(data) == 1 and data[0] == _TRUNCATION_TOKEN:
                return "[...]"

            items: list[str] = []
            for item in data:
                if item == _TRUNCATION_TOKEN:
                    items.append(f"{indent}{indent_str}...")
                else:
                    items.append(
                        f"{indent}{indent_str}"
                        f"{self._custom_stringify(item, indent_level + 1).strip()}"
                    )
            inner = ",\n".join(items)
            return f"[\n{inner}\n{indent}]"

        if isinstance(data, dict):
            keys = list(data.keys())
            if len(keys) == 0:
                return "{}"
            if len(keys) == 1 and keys[0] == _TRUNCATION_TOKEN:
                return "{...}"

            props: list[str] = []
            for key in keys:
                if key == _TRUNCATION_TOKEN:
                    props.append(f"{indent}{indent_str}...")
                else:
                    val = data[key]
                    if val == _TRUNCATION_TOKEN:
                        val_str = "..."
                    else:
                        val_str = self._custom_stringify(
                            val, indent_level + 1
                        ).strip()
                    props.append(f'{indent}{indent_str}"{key}": {val_str}')
            inner = ",\n".join(props)
            return f"{{\n{inner}\n{indent}}}"

        # Primitives – use json.dumps for proper escaping
        return json.dumps(data, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_size(self, obj: Any) -> int:
        return len(self._custom_stringify(obj))

    @staticmethod
    def _set_in(obj: Any, path: list[str | int], value: Any) -> Any:
        """Immutably set a deeply nested value in *obj* at *path*."""
        if len(path) == 0:
            return value

        head, tail = path[0], path[1:]

        if isinstance(obj, list):
            new_arr = list(obj)
            new_arr[head] = Truncator._set_in(obj[head], tail, value)  # type: ignore[index]
            return new_arr

        if isinstance(obj, dict):
            return {
                **obj,
                head: Truncator._set_in(obj[head], tail, value),  # type: ignore[index]
            }

        return obj

    def _collect_nodes(
        self,
        obj: Any,
        depth: int = 0,
        path: list[str | int] | None = None,
    ) -> list[_Node]:
        if path is None:
            path = []

        if obj == _TRUNCATION_TOKEN:
            return []

        nodes: list[_Node] = []

        if isinstance(obj, str):
            nodes.append(
                _Node(
                    type="string",
                    value=obj,
                    depth=depth,
                    path=list(path),
                    length=len(obj),
                )
            )

        elif isinstance(obj, list):
            nodes.append(
                _Node(
                    type="array",
                    value=obj,
                    depth=depth,
                    path=list(path),
                    size=len(obj),
                )
            )
            for i, item in enumerate(obj):
                nodes.extend(self._collect_nodes(item, depth + 1, path + [i]))

        elif isinstance(obj, dict):
            obj_keys = list(obj.keys())
            if len(obj_keys) == 1 and obj_keys[0] == _TRUNCATION_TOKEN:
                return nodes
            nodes.append(
                _Node(
                    type="object",
                    value=obj,
                    depth=depth,
                    path=list(path),
                    keys=len(obj_keys),
                )
            )
            for key in obj_keys:
                nodes.extend(
                    self._collect_nodes(obj[key], depth + 1, path + [key])
                )

        return nodes

    # ------------------------------------------------------------------
    # Strategy: arrays
    # ------------------------------------------------------------------

    def _apply_array_strategy(
        self, nodes: list[_Node]
    ) -> list[dict[str, Any]] | None:
        candidates = sorted(
            [
                n
                for n in nodes
                if n.type == "array"
                and n.size > 1
                and not (n.size == 1 and n.value[0] == _TRUNCATION_TOKEN)
            ],
            key=lambda n: (-n.depth, -n.size),
        )

        if not candidates:
            return None

        max_depth = candidates[0].depth
        target_nodes = [n for n in candidates if n.depth == max_depth]

        updates: list[dict[str, Any]] = []
        for node in target_nodes:
            arr = node.value
            try:
                idx = arr.index(_TRUNCATION_TOKEN)
            except ValueError:
                idx = -1

            if idx == -1:
                mid = len(arr) // 2
                new_val = list(arr)
                new_val[mid] = _TRUNCATION_TOKEN
            else:
                real_items = len(arr) - 1
                if real_items > self._cfg.min_items_for_collapse:
                    left_count = idx
                    right_count = len(arr) - 1 - idx
                    new_val = list(arr)
                    if left_count > right_count:
                        if idx > 0:
                            new_val.pop(idx - 1)
                    else:
                        new_val.pop(idx + 1)
                else:
                    new_val = [_TRUNCATION_TOKEN]

            if json.dumps(arr, ensure_ascii=False, default=str) != json.dumps(
                new_val, ensure_ascii=False, default=str
            ):
                updates.append({"path": node.path, "value": new_val})

        return updates if updates else None

    # ------------------------------------------------------------------
    # Strategy: objects
    # ------------------------------------------------------------------

    def _apply_object_strategy(
        self, nodes: list[_Node]
    ) -> list[dict[str, Any]] | None:
        candidates = sorted(
            [
                n
                for n in nodes
                if n.type == "object"
                and n.keys > 1
                and not (n.keys == 1 and _TRUNCATION_TOKEN in n.value)
            ],
            key=lambda n: (-n.depth, -n.keys),
        )

        if not candidates:
            return None

        max_depth = candidates[0].depth
        target_nodes = [n for n in candidates if n.depth == max_depth]

        updates: list[dict[str, Any]] = []
        for node in target_nodes:
            obj = node.value
            keys = list(obj.keys())

            try:
                idx = keys.index(_TRUNCATION_TOKEN)
            except ValueError:
                idx = -1

            if idx == -1:
                mid = len(keys) // 2
                next_keys = list(keys)
                next_keys[mid] = _TRUNCATION_TOKEN

                new_val: dict[str, Any] = {}
                for k in next_keys:
                    if k == _TRUNCATION_TOKEN:
                        new_val[_TRUNCATION_TOKEN] = _TRUNCATION_TOKEN
                    else:
                        new_val[k] = obj[k]
            else:
                real_keys_count = len(keys) - 1
                if real_keys_count > self._cfg.min_keys_for_collapse:
                    left_count = idx
                    right_count = len(keys) - 1 - idx
                    remove_idx = (idx - 1) if left_count > right_count else (idx + 1)
                    key_to_remove = keys[remove_idx] if remove_idx < len(keys) else None
                    new_val = dict(obj)
                    if key_to_remove and key_to_remove in new_val:
                        del new_val[key_to_remove]
                else:
                    new_val = {_TRUNCATION_TOKEN: True}

            if json.dumps(obj, ensure_ascii=False, default=str) != json.dumps(
                new_val, ensure_ascii=False, default=str
            ):
                updates.append({"path": node.path, "value": new_val})

        return updates if updates else None

    # ------------------------------------------------------------------
    # Core recursive truncation
    # ------------------------------------------------------------------

    def _smart_truncate(self, data: Any, limit: int) -> Any:
        if self._get_size(data) <= limit:
            return data

        nodes = self._collect_nodes(data)

        # --- Strategy 1: truncate strings ---
        string_candidates = [
            n
            for n in nodes
            if n.type == "string"
            and n.length > self._cfg.min_len_for_truncation
            and n.value != _TRUNCATION_TOKEN
        ]

        if string_candidates:

            def _get_updates(max_len: int) -> list[dict[str, Any]]:
                return [
                    {
                        "path": n.path,
                        "value": n.value[
                            : max(0, max_len - self._cfg.ellipsis_size)
                        ]
                        + "...",
                    }
                    for n in string_candidates
                    if n.length > max_len
                ]

            def _apply_updates(
                upds: list[dict[str, Any]], current: Any
            ) -> Any:
                result = current
                for u in upds:
                    result = self._set_in(result, u["path"], u["value"])
                return result

            base_len = self._cfg.min_len_for_truncation
            base_updates = _get_updates(base_len)
            base_data = _apply_updates(base_updates, data)

            if self._get_size(base_data) > limit:
                return self._smart_truncate(base_data, limit)
            else:
                low = base_len
                high = max(n.length for n in string_candidates)
                best_data = base_data

                while low <= high:
                    mid = (low + high) // 2
                    updates = _get_updates(mid)
                    attempt_data = _apply_updates(updates, data)

                    if self._get_size(attempt_data) <= limit:
                        best_data = attempt_data
                        low = mid + 1
                    else:
                        high = mid - 1

                return best_data

        # --- Strategy 2: collapse arrays ---
        array_updates = self._apply_array_strategy(nodes)
        if array_updates:
            next_data = data
            for update in array_updates:
                next_data = self._set_in(
                    next_data, update["path"], update["value"]
                )
            return self._smart_truncate(next_data, limit)

        # --- Strategy 3: collapse objects ---
        object_updates = self._apply_object_strategy(nodes)
        if object_updates:
            next_data = data
            for update in object_updates:
                next_data = self._set_in(
                    next_data, update["path"], update["value"]
                )
            return self._smart_truncate(next_data, limit)

        return data
