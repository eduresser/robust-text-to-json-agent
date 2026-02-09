from __future__ import annotations

import math
import re
from typing import Any, Optional


class ReadValue:
    _DEFAULTS = {
        "max_string_length": 160,
        "max_depth": 6,
        "max_array_items": 50,
        "max_object_keys": 50,
    }

    @classmethod
    def read(
        cls,
        root: Any,
        input_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if input_data is None:
            input_data = {}

        path = input_data.get("path")
        if not isinstance(path, str):
            return {"found": False, "error": "Missing input.path (string)", "path": ""}

        opts = cls._read_options(input_data)

        try:
            tokens = cls._parse_json_pointer(path)
        except ValueError as e:
            return {"found": False, "error": str(e), "path": path}

        cur = root

        if len(tokens) == 0:
            return cls._build_found_result(cur, path, opts)

        for i, tok in enumerate(tokens):
            if cur is None:
                return {
                    "found": False,
                    "error": (
                        f"Cannot traverse '{tok}': encountered "
                        f"{'null' if cur is None else 'undefined'} "
                        f"at token index {i}"
                    ),
                    "path": path,
                }

            if isinstance(cur, list):
                idx_result = cls._parse_array_index(tok, i)
                if not idx_result["ok"]:
                    return {"found": False, "error": idx_result["error"], "path": path}
                idx = idx_result["index"]
                if idx < 0 or idx >= len(cur):
                    return {
                        "found": False,
                        "error": (
                            f"Array index out of range: {idx} "
                            f"(length {len(cur)})"
                        ),
                        "path": path,
                    }
                cur = cur[idx]
                continue

            if isinstance(cur, dict):
                if tok not in cur:
                    return {
                        "found": False,
                        "error": (
                            f"Property not found: '{tok}' "
                            f"at token index {i}"
                        ),
                        "path": path,
                    }
                cur = cur[tok]
                continue

            return {
                "found": False,
                "error": (
                    f"Cannot traverse into non-container type "
                    f"'{cls._describe_type(cur)}' at token index {i}"
                ),
                "path": path,
            }

        return cls._build_found_result(cur, path, opts)

    @classmethod
    def _read_options(cls, input_data: dict[str, Any]) -> dict[str, Any]:
        d = cls._DEFAULTS

        def _get_finite_int(key: str, default: int) -> int:
            v = input_data.get(key)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                return max(0, int(v))
            return default

        return {
            "max_string_length": _get_finite_int("max_string_length", d["max_string_length"]),
            "max_depth": _get_finite_int("max_depth", d["max_depth"]),
            "max_array_items": _get_finite_int("max_array_items", d["max_array_items"]),
            "max_object_keys": _get_finite_int("max_object_keys", d["max_object_keys"]),
        }

    @classmethod
    def _build_found_result(
        cls,
        value: Any,
        path: str,
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        value_type = cls._describe_type(value)
        seen: set[int] = set()
        sanitized = cls._sanitize_for_json(value, opts, seen, 0)

        return {
            "found": True,
            "path": path,
            "valueType": value_type,
            "value": sanitized["jsonValue"],
            "valueTruncated": sanitized["truncated"],
            "notes": sanitized["notes"],
            "stats": sanitized["stats"],
            "limits": {
                "max_string_length": opts["max_string_length"],
                "max_depth": opts["max_depth"],
                "max_array_items": opts["max_array_items"],
                "max_object_keys": opts["max_object_keys"],
            },
        }

    @staticmethod
    def _unescape_json_pointer_token(token: str) -> str:
        return str(token).replace("~1", "/").replace("~0", "~")

    @classmethod
    def _parse_json_pointer(cls, path: str) -> list[str]:
        if not isinstance(path, str):
            raise ValueError("Invalid path: must be a string")
        if path == "" or path == "/":
            return []
        if path[0] != "/":
            path = "/" + path
        raw = path.split("/")[1:]
        return [cls._unescape_json_pointer_token(t) for t in raw]

    @staticmethod
    def _describe_type(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, list):
            return "array"
        if isinstance(v, dict):
            return "object"
        if isinstance(v, bool):
            return "boolean"
        if isinstance(v, (int, float)):
            return "number"
        if isinstance(v, str):
            return "string"
        return type(v).__name__

    @staticmethod
    def _parse_array_index(tok: str, token_index: int) -> dict[str, Any]:
        if tok == "-":
            return {
                "ok": False,
                "error": "Invalid array index '-': not readable for read_value",
            }
        if not re.match(r"^(0|[1-9]\d*)$", tok):
            return {
                "ok": False,
                "error": (
                    f"Invalid array index token '{tok}' "
                    f"at token index {token_index}"
                ),
            }
        return {"ok": True, "index": int(tok)}

    @classmethod
    def _sanitize_for_json(
        cls,
        value: Any,
        opts: dict[str, Any],
        seen: set[int],
        depth: int,
    ) -> dict[str, Any]:
        vtype = cls._describe_type(value)
        truncated = False
        notes: list[str] = []

        # null
        if value is None:
            return {
                "jsonValue": None,
                "truncated": False,
                "notes": [],
                "stats": {"type": vtype},
            }

        # string
        if isinstance(value, str):
            if len(value) > opts["max_string_length"]:
                truncated = True
                notes.append(
                    f"String truncated to max_string_length={opts['max_string_length']}"
                )
                return {
                    "jsonValue": value[: opts["max_string_length"]] + "\u2026",
                    "truncated": truncated,
                    "notes": notes,
                    "stats": {"type": vtype, "originalLength": len(value)},
                }
            return {
                "jsonValue": value,
                "truncated": False,
                "notes": [],
                "stats": {"type": vtype, "length": len(value)},
            }

        # number / boolean
        if isinstance(value, (bool, int, float)):
            return {
                "jsonValue": value,
                "truncated": False,
                "notes": [],
                "stats": {"type": vtype},
            }

        # Non-container types shouldn't appear in JSON but handle gracefully
        if not isinstance(value, (dict, list)):
            notes.append(f"Value was {vtype}; encoded as string for JSON compatibility")
            try:
                repr_str = str(value)
            except Exception:
                repr_str = f"[{vtype}]"
            if len(repr_str) > opts["max_string_length"]:
                truncated = True
                repr_str = repr_str[: opts["max_string_length"]] + "\u2026"
                notes.append(
                    f"Representation truncated to max_string_length={opts['max_string_length']}"
                )
            return {
                "jsonValue": repr_str,
                "truncated": truncated,
                "notes": notes,
                "stats": {"type": vtype},
            }

        obj_id = id(value)
        if obj_id in seen:
            notes.append("Circular reference replaced with '[Circular]'")
            return {
                "jsonValue": "[Circular]",
                "truncated": True,
                "notes": notes,
                "stats": {"type": vtype},
            }
        seen.add(obj_id)

        # Max depth
        if depth >= opts["max_depth"]:
            notes.append(
                f"Max depth reached (max_depth={opts['max_depth']}); "
                f"replaced with '[MaxDepth]'"
            )
            return {
                "jsonValue": "[MaxDepth]",
                "truncated": True,
                "notes": notes,
                "stats": {"type": vtype},
            }

        # array
        if isinstance(value, list):
            original_length = len(value)
            limit = opts["max_array_items"]
            out = []
            take = min(original_length, limit)
            nested_changed = False

            for i in range(take):
                child = cls._sanitize_for_json(value[i], opts, seen, depth + 1)
                if child["truncated"] or (child["notes"] and len(child["notes"]) > 0):
                    nested_changed = True
                if child["truncated"]:
                    truncated = True
                out.append(child["jsonValue"])

            if original_length > limit:
                truncated = True
                notes.append(f"Array truncated to max_array_items={limit}")
            if nested_changed:
                notes.append(
                    "Nested values were sanitized/truncated for JSON safety"
                )
                truncated = True

            return {
                "jsonValue": out,
                "truncated": truncated,
                "notes": notes,
                "stats": {
                    "type": vtype,
                    "originalLength": original_length,
                    "returnedLength": len(out),
                },
            }

        # object (dict)
        keys = list(value.keys())
        original_key_count = len(keys)
        limit = opts["max_object_keys"]
        out_dict: dict[str, Any] = {}
        take = min(original_key_count, limit)
        nested_changed = False

        for i in range(take):
            k = keys[i]
            child = cls._sanitize_for_json(value[k], opts, seen, depth + 1)
            if child["truncated"] or (child["notes"] and len(child["notes"]) > 0):
                nested_changed = True
            if child["truncated"]:
                truncated = True
            out_dict[k] = child["jsonValue"]

        if original_key_count > limit:
            truncated = True
            notes.append(f"Object truncated to max_object_keys={limit}")
        if nested_changed:
            notes.append(
                "Nested values were sanitized/truncated for JSON safety"
            )
            truncated = True

        return {
            "jsonValue": out_dict,
            "truncated": truncated,
            "notes": notes,
            "stats": {
                "type": vtype,
                "originalKeyCount": original_key_count,
                "returnedKeyCount": take,
            },
        }


def read_value(
    document: dict[str, Any],
    input_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Read a value at a specific path in the JSON document.

    This is the public API matching the n8n ReadValue behavior.

    Args:
        document: The root JSON document.
        input_data: Dict with path, max_string_length, max_depth,
                    max_array_items, max_object_keys.

    Returns:
        Read result dict with found, path, valueType, value, stats, etc.
    """
    return ReadValue.read(document, input_data)
