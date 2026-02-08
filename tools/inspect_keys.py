from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import unquote


class JsonInspector:
    _DEFAULTS: dict[str, Any] = {
        "maxKeys": 50,
        "maxArrayItems": 20,
        "maxStringLength": 300,
        "maxDepthPreview": 2,
        "includeValue": True,
        "tryUrlDecode": True,
    }

    # ------------------------------------------------------------------ public
    @classmethod
    def inspect(
        cls,
        document: Any,
        pointer: Optional[str],
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        opts = cls._normalize_options(options)
        original_pointer = pointer

        if pointer is None:
            pointer = ""

        parsed = cls._parse_json_pointer(pointer, opts["tryUrlDecode"])
        if not parsed["ok"]:
            return {
                "ok": False,
                "found": False,
                "pointer": original_pointer,
                "error": parsed["error"],
                "note": 'Pass "" or "/" to inspect the root.',
            }

        current = document
        walked = ""

        for i, token in enumerate(parsed["tokens"]):
            cur_type = cls._safe_type(current)
            escaped = str(token).replace("~", "~0").replace("/", "~1")
            next_ptr = f"{walked}/{escaped}"

            if cur_type == "array":
                if not re.match(r"^(0|[1-9]\d*)$", str(token)):
                    return {
                        "ok": True,
                        "found": False,
                        "pointer": original_pointer,
                        "atPointer": next_ptr,
                        "message": (
                            f"Expected a numeric index for array, "
                            f"but received token '{token}'."
                        ),
                        "containerType": "array",
                        "containerLength": len(current),
                    }
                idx = int(token)
                if idx < 0 or idx >= len(current):
                    return {
                        "ok": True,
                        "found": False,
                        "pointer": original_pointer,
                        "atPointer": next_ptr,
                        "message": (
                            f"The index is out of range: {idx} "
                            f"(len={len(current)})."
                        ),
                        "containerType": "array",
                        "containerLength": len(current),
                    }
                current = current[idx]
                walked = next_ptr
                continue

            if cur_type == "object":
                if token not in current:
                    keys = list(current.keys()) if isinstance(current, dict) else []
                    take = min(len(keys), opts["maxKeys"])
                    return {
                        "ok": True,
                        "found": False,
                        "pointer": original_pointer,
                        "atPointer": next_ptr,
                        "message": f"The key was not found: '{token}'.",
                        "containerType": "object",
                        "availableKeysPreview": keys[:take],
                        "availableKeysTruncated": take < len(keys),
                    }
                current = current[token]
                walked = next_ptr
                continue

            return {
                "ok": True,
                "found": False,
                "pointer": original_pointer,
                "atPointer": walked or "/",
                "message": (
                    f"It's not possible to navigate inside a value "
                    f"of type '{cur_type}'."
                ),
                "encounteredType": cur_type,
            }

        summary = cls._summarize(current, opts, 0)
        return {
            "ok": True,
            "found": True,
            "pointer": original_pointer,
            "resolvedPointer": "/" if pointer == "" else pointer,
            **summary,
        }

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _safe_type(x: Any) -> str:
        if x is None:
            return "null"
        if isinstance(x, list):
            return "array"
        if isinstance(x, dict):
            return "object"
        if isinstance(x, bool):
            return "boolean"
        if isinstance(x, (int, float)):
            return "number"
        if isinstance(x, str):
            return "string"
        return type(x).__name__

    @staticmethod
    def _clamp_int(n: Any, fallback: int) -> int:
        try:
            v = int(n)
        except (TypeError, ValueError):
            return fallback
        return max(0, v)

    @classmethod
    def _normalize_options(cls, options: Optional[dict[str, Any]]) -> dict[str, Any]:
        o = options if isinstance(options, dict) else {}
        d = cls._DEFAULTS
        return {
            "maxKeys": cls._clamp_int(o.get("maxKeys"), d["maxKeys"]),
            "maxArrayItems": cls._clamp_int(o.get("maxArrayItems"), d["maxArrayItems"]),
            "maxStringLength": cls._clamp_int(
                o.get("maxStringLength"), d["maxStringLength"]
            ),
            "maxDepthPreview": cls._clamp_int(
                o.get("maxDepthPreview"), d["maxDepthPreview"]
            ),
            "includeValue": (
                o["includeValue"]
                if isinstance(o.get("includeValue"), bool)
                else d["includeValue"]
            ),
            "tryUrlDecode": (
                o["tryUrlDecode"]
                if isinstance(o.get("tryUrlDecode"), bool)
                else d["tryUrlDecode"]
            ),
        }

    @staticmethod
    def _decode_pointer_token(token: str, try_url_decode: bool) -> str:
        t = str(token).replace("~1", "/").replace("~0", "~")
        if not try_url_decode:
            return t
        try:
            if "%" in t:
                t = unquote(t)
        except Exception:
            pass
        return t

    @classmethod
    def _parse_json_pointer(
        cls, pointer: str, try_url_decode: bool
    ) -> dict[str, Any]:
        if pointer == "" or pointer == "/":
            return {"ok": True, "tokens": []}
        if not pointer.startswith("/"):
            pointer = "/" + pointer
        raw_tokens = pointer.split("/")[1:]
        tokens = [cls._decode_pointer_token(t, try_url_decode) for t in raw_tokens]
        return {"ok": True, "tokens": tokens}

    @classmethod
    def _preview_string(cls, s: str, max_len: int) -> str:
        text = str(s)
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"\u2026(truncated, len={len(text)})"

    @classmethod
    def _preview_primitive(cls, value: Any, opts: dict[str, Any]) -> Any:
        t = cls._safe_type(value)
        if t == "string":
            return cls._preview_string(value, opts["maxStringLength"])
        if t in ("number", "boolean", "null"):
            return value
        if value is None:
            return None
        return f"[{t}]"

    @classmethod
    def _summarize(
        cls, value: Any, opts: dict[str, Any], depth: int
    ) -> dict[str, Any]:
        t = cls._safe_type(value)

        if t not in ("object", "array"):
            result: dict[str, Any] = {"type": t}
            if opts["includeValue"]:
                result["valuePreview"] = cls._preview_primitive(value, opts)
            return result

        if t == "array":
            arr = value
            length = len(arr)
            if depth >= opts["maxDepthPreview"]:
                return {
                    "type": "array",
                    "length": length,
                    "itemsPreview": (
                        f"[preview depth limit {opts['maxDepthPreview']}]"
                    ),
                }
            take = min(length, opts["maxArrayItems"])
            items = []
            for i in range(take):
                it = arr[i]
                it_type = cls._safe_type(it)
                if it_type in ("object", "array"):
                    items.append({"index": i, "type": it_type})
                else:
                    entry: dict[str, Any] = {"index": i, "type": it_type}
                    if opts["includeValue"]:
                        entry["valuePreview"] = cls._preview_primitive(it, opts)
                    items.append(entry)
            return {
                "type": "array",
                "length": length,
                "previewCount": take,
                "truncated": take < length,
                "itemsPreview": items,
            }

        # object
        obj = value
        keys = list(obj.keys()) if isinstance(obj, dict) else []
        count = len(keys)
        take = min(count, opts["maxKeys"])
        keys_preview = keys[:take]

        shallow_preview = None
        if depth < opts["maxDepthPreview"]:
            shallow_preview = {}
            for k in keys_preview:
                v = obj[k]
                vt = cls._safe_type(v)
                if vt in ("object", "array"):
                    shallow_preview[k] = {"type": vt}
                else:
                    entry = {"type": vt}
                    if opts["includeValue"]:
                        entry["valuePreview"] = cls._preview_primitive(v, opts)
                    shallow_preview[k] = entry

        return {
            "type": "object",
            "count": count,
            "previewCount": take,
            "truncated": take < count,
            "keysPreview": keys_preview,
            "shallowPreview": shallow_preview,
        }


def inspect_keys(
    document: dict[str, Any],
    path: str = "",
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Inspect a JSON document at the given JSON Pointer path.

    This is the public API matching the n8n JsonInspector behavior.

    Args:
        document: The root JSON document.
        path: JSON Pointer (RFC 6901) to inspect.
        options: Optional dict with maxKeys, maxArrayItems, maxStringLength,
                 maxDepthPreview, includeValue, tryUrlDecode.

    Returns:
        Inspection result dict with ok, found, type info, previews, etc.
    """
    return JsonInspector.inspect(document, path, options)
