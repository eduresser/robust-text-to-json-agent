"""
Port of the n8n SearchPointer class.

Searches the JSON document for keys or values matching a query,
returning JSON Pointers to matching locations.
Supports exact and fuzzy (Levenshtein) matching.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Literal, Optional


class SearchPointer:
    """Faithful port of the n8n SearchPointer class."""

    @classmethod
    def search(
        cls,
        root: Any,
        input_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if input_data is None:
            input_data = {}

        query = str(input_data.get("query", "")) if input_data.get("query") is not None else ""
        search_type: str = input_data.get("type", "value")
        if search_type not in ("key", "value"):
            search_type = "value"
        fuzzy: bool = bool(input_data.get("fuzzy_match", False))
        include_pointers: bool = bool(input_data.get("include_pointers", False))

        raw_limit = input_data.get("limit")
        if raw_limit is not None and isinstance(raw_limit, (int, float)) and raw_limit == raw_limit:
            limit = max(0, int(raw_limit))
        else:
            limit = float("inf")

        raw_mvl = input_data.get("max_value_length")
        if raw_mvl is not None and isinstance(raw_mvl, (int, float)) and raw_mvl == raw_mvl:
            max_value_length = max(0, int(raw_mvl))
        else:
            max_value_length = 120

        matches: list[dict[str, Any]] = []
        seen: set[int] = set()

        state = {
            "query": query,
            "type": search_type,
            "fuzzy": fuzzy,
            "matches": matches,
            "seen": seen,
            "limit": limit,
            "truncated": False,
            "maxValueLength": max_value_length,
        }

        cls._visit(root, "", state)

        result: dict[str, Any] = {
            "matches": matches,
            "count": len(matches),
            "truncated": state["truncated"],
            "limit": limit if limit != float("inf") else None,
            "max_value_length": max_value_length,
        }

        if include_pointers:
            result["pointers"] = [m["pointer"] for m in matches]

        return result

    @classmethod
    def _visit(cls, node: Any, ptr: str, state: dict[str, Any]) -> None:
        if len(state["matches"]) >= state["limit"]:
            state["truncated"] = True
            return
        if node is None or not isinstance(node, (dict, list)):
            return

        node_id = id(node)
        if node_id in state["seen"]:
            return
        state["seen"].add(node_id)

        if isinstance(node, list):
            for i, item in enumerate(node):
                if len(state["matches"]) >= state["limit"]:
                    state["truncated"] = True
                    break
                child_ptr = cls._join_pointer(ptr, str(i))
                cls._maybe_collect_value(item, child_ptr, state)
                cls._visit(item, child_ptr, state)
            return

        # dict
        for key in node:
            if len(state["matches"]) >= state["limit"]:
                state["truncated"] = True
                break
            child_ptr = cls._join_pointer(ptr, key)
            value = node[key]

            cls._maybe_collect_key(key, child_ptr, state)
            cls._maybe_collect_value(value, child_ptr, state)
            cls._visit(value, child_ptr, state)

    @classmethod
    def _maybe_collect_key(
        cls, key: str, pointer: str, state: dict[str, Any]
    ) -> None:
        if state["type"] != "key":
            return
        if not cls._matches_query(str(key), state):
            return
        if len(state["matches"]) >= state["limit"]:
            state["truncated"] = True
            return
        state["matches"].append({"pointer": pointer, "kind": "key", "key": key})

    @classmethod
    def _maybe_collect_value(
        cls, value: Any, pointer: str, state: dict[str, Any]
    ) -> None:
        if state["type"] != "value" or not cls._is_primitive(value):
            return
        if len(state["matches"]) >= state["limit"]:
            state["truncated"] = True
            return

        comparable = cls._value_to_comparable_string(value)
        if not cls._matches_query(comparable, state):
            return

        stored_value, value_truncated = cls._value_to_stored_value(
            value, state["maxValueLength"]
        )
        value_type = "null" if value is None else type(value).__name__
        if value_type == "str":
            value_type = "string"
        elif value_type in ("int", "float"):
            value_type = "number"
        elif value_type == "bool":
            value_type = "boolean"

        state["matches"].append(
            {
                "pointer": pointer,
                "kind": "value",
                "value": stored_value,
                "valueType": value_type,
                "valueTruncated": value_truncated,
            }
        )

    @classmethod
    def _matches_query(cls, candidate: str, state: dict[str, Any]) -> bool:
        if not state["fuzzy"]:
            return str(candidate) == state["query"]
        return cls._fuzzy_match(str(candidate), state["query"])

    @classmethod
    def _fuzzy_match(cls, a: str, b: str) -> bool:
        na = cls._normalize_for_match(a)
        nb = cls._normalize_for_match(b)
        if na == nb:
            return True
        if na in nb or nb in na:
            return True

        max_len = max(len(na), len(nb))
        if max_len == 0:
            return True
        if max_len > 64:
            return False

        dist = cls._levenshtein(na, nb)
        min_len = min(len(na), len(nb))
        threshold = min(3, max(1, -(-int(min_len * 0.34) // 1)))
        return dist <= threshold

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        """Normalize string for matching: NFD decompose, strip accents, lowercase, strip."""
        text = str(value)
        nfd = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return stripped.lower().strip()

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        left = str(a)
        right = str(b)
        n, m = len(left), len(right)
        if n == 0:
            return m
        if m == 0:
            return n

        prev = list(range(m + 1))
        cur = [0] * (m + 1)

        for i in range(1, n + 1):
            cur[0] = i
            ca = ord(left[i - 1])
            for j in range(1, m + 1):
                cb = ord(right[j - 1])
                cost = 0 if ca == cb else 1
                cur[j] = min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            prev, cur = cur, prev

        return prev[m]

    @staticmethod
    def _join_pointer(base: str, token: str) -> str:
        escaped = str(token).replace("~", "~0").replace("/", "~1")
        if base == "":
            return "/" + escaped
        return base + "/" + escaped

    @staticmethod
    def _is_primitive(v: Any) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    @staticmethod
    def _value_to_comparable_string(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, str):
            return v
        return str(v)

    @staticmethod
    def _value_to_stored_value(
        v: Any, max_len: int
    ) -> tuple[Any, bool]:
        if isinstance(v, str) and len(v) > max_len:
            return v[:max_len] + "\u2026", True
        return v, False


def search_pointer(
    document: dict[str, Any],
    input_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Search for keys or values in the JSON document and return JSON Pointers.

    This is the public API matching the n8n SearchPointer behavior.

    Args:
        document: The root JSON document.
        input_data: Dict with query, type, fuzzy_match, include_pointers,
                    limit, max_value_length.

    Returns:
        Search result dict with matches, count, truncated, etc.
    """
    return SearchPointer.search(document, input_data)
