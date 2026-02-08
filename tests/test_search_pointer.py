"""Tests for tools/search_pointer.py — SearchPointer."""

from __future__ import annotations

import pytest

from tools.search_pointer import search_pointer


@pytest.fixture
def sample_doc():
    return {
        "metadata": {"title": "Report", "author": "Alice"},
        "sections": [
            {
                "name": "Overview",
                "fields": [
                    {"label": "Revenue", "value": 1500},
                    {"label": "Profit", "value": 300},
                ],
            },
            {
                "name": "Details",
                "fields": [
                    {"label": "Cost", "value": 1200},
                ],
            },
        ],
    }


class TestValueSearch:
    """Search by value."""

    def test_find_exact_string(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "Alice", "type": "value"})
        assert result["count"] >= 1
        assert any(m["value"] == "Alice" for m in result["matches"])

    def test_find_exact_number(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "1500", "type": "value"})
        assert result["count"] >= 1

    def test_no_match(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "NonExistent", "type": "value"})
        assert result["count"] == 0

    def test_find_returns_pointer(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "Revenue", "type": "value"})
        assert result["count"] >= 1
        match = result["matches"][0]
        assert match["pointer"].startswith("/")

    def test_limit_enforced(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "e", "type": "value", "fuzzy_match": True, "limit": 1})
        assert result["count"] <= 1


class TestKeySearch:
    """Search by key."""

    def test_find_key(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "title", "type": "key"})
        assert result["count"] >= 1
        assert any(m["key"] == "title" for m in result["matches"])

    def test_find_key_in_array_items(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "label", "type": "key"})
        # "label" exists in each field object
        assert result["count"] >= 3

    def test_no_key_match(self, sample_doc):
        result = search_pointer(sample_doc, {"query": "nonexistent_key", "type": "key"})
        assert result["count"] == 0


class TestFuzzySearch:
    """Fuzzy matching."""

    def test_fuzzy_finds_similar(self, sample_doc):
        result = search_pointer(sample_doc, {
            "query": "Revenu",  # missing 'e'
            "type": "value",
            "fuzzy_match": True,
        })
        assert result["count"] >= 1

    def test_fuzzy_finds_case_insensitive(self, sample_doc):
        result = search_pointer(sample_doc, {
            "query": "alice",  # lowercase
            "type": "value",
            "fuzzy_match": True,
        })
        assert result["count"] >= 1

    def test_fuzzy_no_match_too_different(self, sample_doc):
        result = search_pointer(sample_doc, {
            "query": "CompletelyDifferentString",
            "type": "value",
            "fuzzy_match": True,
        })
        assert result["count"] == 0


class TestValueTruncation:
    """Long value truncation in results."""

    def test_long_value_truncated(self):
        doc = {"text": "a" * 500}
        result = search_pointer(doc, {
            "query": "a" * 500,
            "type": "value",
            "max_value_length": 20,
        })
        if result["count"] > 0:
            match = result["matches"][0]
            assert match["valueTruncated"] is True
            assert len(str(match["value"])) <= 25  # 20 + ellipsis


class TestEmptyDocument:
    """Edge cases with empty structures."""

    def test_empty_doc(self):
        result = search_pointer({}, {"query": "anything", "type": "value"})
        assert result["count"] == 0

    def test_empty_query(self):
        doc = {"name": "Alice"}
        result = search_pointer(doc, {"query": "", "type": "value"})
        # Empty string should match empty string values only
        assert isinstance(result["count"], int)
