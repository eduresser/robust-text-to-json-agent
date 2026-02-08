"""Tests for tools/inspect_keys.py — JsonInspector."""

from __future__ import annotations

import pytest

from tools.inspect_keys import inspect_keys


class TestInspectKeysRoot:
    """Inspect root of the document."""

    def test_empty_doc(self):
        result = inspect_keys({}, "")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "object"
        assert result["count"] == 0

    def test_root_object(self):
        doc = {"a": 1, "b": 2, "c": 3}
        result = inspect_keys(doc, "")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "object"
        assert result["count"] == 3
        assert set(result["keysPreview"]) == {"a", "b", "c"}

    def test_root_with_slash(self):
        doc = {"x": 1}
        result = inspect_keys(doc, "/")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "object"


class TestInspectKeysPath:
    """Inspect specific paths."""

    def test_object_path(self):
        doc = {"metadata": {"title": "Test", "author": "Bot"}}
        result = inspect_keys(doc, "/metadata")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "object"
        assert "title" in result["keysPreview"]
        assert "author" in result["keysPreview"]

    def test_array_path(self):
        doc = {"items": [1, 2, 3, 4, 5]}
        result = inspect_keys(doc, "/items")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "array"
        assert result["length"] == 5

    def test_scalar_path(self):
        doc = {"name": "Alice"}
        result = inspect_keys(doc, "/name")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "string"

    def test_nested_array_index(self):
        doc = {"items": [{"name": "A"}, {"name": "B"}]}
        result = inspect_keys(doc, "/items/0")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "object"
        assert "name" in result["keysPreview"]

    def test_deep_nesting(self):
        doc = {"a": {"b": {"c": {"d": 42}}}}
        result = inspect_keys(doc, "/a/b/c")
        assert result["ok"] is True
        assert result["found"] is True
        assert result["type"] == "object"
        assert "d" in result["keysPreview"]


class TestInspectKeysErrors:
    """Handle missing/invalid paths gracefully."""

    def test_missing_key(self):
        doc = {"a": 1}
        result = inspect_keys(doc, "/nonexistent")
        assert result["ok"] is True
        assert result["found"] is False

    def test_array_index_out_of_range(self):
        doc = {"items": [1, 2]}
        result = inspect_keys(doc, "/items/99")
        assert result["ok"] is True
        assert result["found"] is False

    def test_non_numeric_array_index(self):
        doc = {"items": [1, 2]}
        result = inspect_keys(doc, "/items/abc")
        assert result["ok"] is True
        assert result["found"] is False

    def test_traverse_scalar(self):
        doc = {"name": "Alice"}
        result = inspect_keys(doc, "/name/nested")
        assert result["ok"] is True
        assert result["found"] is False


class TestInspectKeysOptions:
    """Test with custom options."""

    def test_max_keys_limit(self):
        doc = {f"k{i}": i for i in range(100)}
        result = inspect_keys(doc, "", {"maxKeys": 5})
        assert result["ok"] is True
        assert result["previewCount"] == 5
        assert result["truncated"] is True

    def test_include_value_false(self):
        doc = {"name": "Alice"}
        result = inspect_keys(doc, "", {"includeValue": False})
        assert result["ok"] is True
        # shallowPreview should not have valuePreview
        for v in result.get("shallowPreview", {}).values():
            assert "valuePreview" not in v
