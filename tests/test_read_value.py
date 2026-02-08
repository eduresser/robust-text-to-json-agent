"""Tests for tools/read_value.py — ReadValue."""

from __future__ import annotations

import pytest

from tools.read_value import read_value


@pytest.fixture
def doc():
    return {
        "name": "Alice",
        "age": 30,
        "active": True,
        "address": {
            "street": "Main St",
            "city": "NY",
            "zip": "10001",
        },
        "tags": ["engineer", "manager", "lead"],
        "scores": [95, 88, 72, 61],
    }


class TestReadValuePrimitives:
    """Read primitive values."""

    def test_read_string(self, doc):
        result = read_value(doc, {"path": "/name"})
        assert result["found"] is True
        assert result["value"] == "Alice"
        assert result["valueType"] == "string"

    def test_read_integer(self, doc):
        result = read_value(doc, {"path": "/age"})
        assert result["found"] is True
        assert result["value"] == 30
        assert result["valueType"] == "number"

    def test_read_boolean(self, doc):
        result = read_value(doc, {"path": "/active"})
        assert result["found"] is True
        assert result["value"] is True
        assert result["valueType"] == "boolean"

    def test_read_null(self):
        result = read_value({"x": None}, {"path": "/x"})
        assert result["found"] is True
        assert result["value"] is None
        assert result["valueType"] == "null"


class TestReadValueContainers:
    """Read objects and arrays."""

    def test_read_object(self, doc):
        result = read_value(doc, {"path": "/address"})
        assert result["found"] is True
        assert result["valueType"] == "object"
        assert result["value"]["street"] == "Main St"

    def test_read_array(self, doc):
        result = read_value(doc, {"path": "/tags"})
        assert result["found"] is True
        assert result["valueType"] == "array"
        assert len(result["value"]) == 3

    def test_read_array_element(self, doc):
        result = read_value(doc, {"path": "/tags/0"})
        assert result["found"] is True
        assert result["value"] == "engineer"

    def test_read_root(self, doc):
        result = read_value(doc, {"path": ""})
        assert result["found"] is True
        assert result["valueType"] == "object"

    def test_read_root_slash(self, doc):
        result = read_value(doc, {"path": "/"})
        assert result["found"] is True
        assert result["valueType"] == "object"


class TestReadValueErrors:
    """Handle missing paths and errors."""

    def test_missing_path(self, doc):
        result = read_value(doc, {"path": "/nonexistent"})
        assert result["found"] is False

    def test_array_index_out_of_range(self, doc):
        result = read_value(doc, {"path": "/tags/99"})
        assert result["found"] is False

    def test_traverse_scalar(self, doc):
        result = read_value(doc, {"path": "/name/nested"})
        assert result["found"] is False

    def test_missing_path_input(self, doc):
        result = read_value(doc, {})
        assert result["found"] is False

    def test_dash_index_not_readable(self, doc):
        result = read_value(doc, {"path": "/tags/-"})
        assert result["found"] is False


class TestReadValueTruncation:
    """Test truncation limits."""

    def test_string_truncated(self):
        doc = {"text": "a" * 500}
        result = read_value(doc, {"path": "/text", "max_string_length": 20})
        assert result["found"] is True
        assert result["valueTruncated"] is True
        assert len(result["value"]) <= 25  # 20 + ellipsis char

    def test_array_truncated(self):
        doc = {"items": list(range(100))}
        result = read_value(doc, {"path": "/items", "max_array_items": 5})
        assert result["found"] is True
        assert result["valueTruncated"] is True
        assert len(result["value"]) == 5

    def test_object_truncated(self):
        doc = {"data": {f"k{i}": i for i in range(100)}}
        result = read_value(doc, {"path": "/data", "max_object_keys": 5})
        assert result["found"] is True
        assert result["valueTruncated"] is True
        assert len(result["value"]) == 5

    def test_max_depth(self):
        doc = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        result = read_value(doc, {"path": "", "max_depth": 2})
        assert result["found"] is True
        assert result["valueTruncated"] is True
