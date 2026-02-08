"""Tests for misc/truncator.py — Truncator."""

from __future__ import annotations

import json

import pytest

from misc.truncator import Truncator, TruncatorConfig


@pytest.fixture
def truncator():
    """Truncator with default config."""
    return Truncator()


@pytest.fixture
def small_config_truncator():
    """Truncator with aggressive truncation settings."""
    return Truncator(TruncatorConfig(
        indentation=2,
        min_len_for_truncation=10,
        ellipsis_size=3,
        min_items_for_collapse=1,
        min_keys_for_collapse=1,
    ))


class TestTruncateWithLimit:
    """Test the main truncate_with_limit API."""

    def test_null_returns_null(self, truncator):
        assert truncator.truncate_with_limit(None, 100) == "null"

    def test_small_data_unchanged(self, truncator):
        data = {"name": "Alice"}
        result = truncator.truncate_with_limit(data, 10000)
        # Should contain the data without truncation
        assert '"name"' in result
        assert '"Alice"' in result

    def test_strings_truncated_first(self, small_config_truncator):
        data = {
            "short": "hi",
            "long": "a" * 200,
        }
        result = small_config_truncator.truncate_with_limit(data, 80)
        assert "..." in result
        assert '"short"' in result  # short string preserved

    def test_arrays_collapsed(self, small_config_truncator):
        data = {"items": list(range(50))}
        result = small_config_truncator.truncate_with_limit(data, 60)
        assert "..." in result

    def test_objects_collapsed(self, small_config_truncator):
        data = {f"key_{i}": f"val_{i}" for i in range(20)}
        result = small_config_truncator.truncate_with_limit(data, 80)
        assert "..." in result

    def test_nested_structure_fits_limit(self, truncator):
        data = {
            "sections": [
                {"name": "A" * 100, "fields": [{"v": "x" * 100} for _ in range(10)]},
                {"name": "B" * 100, "fields": [{"v": "y" * 100} for _ in range(10)]},
            ]
        }
        limit = 200
        result = truncator.truncate_with_limit(data, limit)
        assert len(result) <= limit + 50  # Allow some margin for final formatting

    def test_empty_dict(self, truncator):
        result = truncator.truncate_with_limit({}, 100)
        assert result == "{}"

    def test_empty_list(self, truncator):
        result = truncator.truncate_with_limit([], 100)
        assert result == "[]"

    def test_primitive_values(self, truncator):
        assert truncator.truncate_with_limit(42, 100) == "42"
        assert truncator.truncate_with_limit(True, 100) == "true"
        assert truncator.truncate_with_limit("hello", 100) == '"hello"'


class TestCustomStringify:
    """Test internal custom formatting."""

    def test_simple_object(self, truncator):
        result = truncator._custom_stringify({"a": 1, "b": 2})
        assert '"a": 1' in result
        assert '"b": 2' in result

    def test_simple_array(self, truncator):
        result = truncator._custom_stringify([1, 2, 3])
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_nested(self, truncator):
        result = truncator._custom_stringify({"arr": [1, 2], "obj": {"x": 1}})
        assert '"arr"' in result
        assert '"obj"' in result
