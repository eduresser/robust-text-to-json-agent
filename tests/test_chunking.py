"""Tests for chunking/semantic.py — _merge_small_chunks and chunk_with_fallback."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chunking.semantic import _merge_small_chunks, chunk_with_fallback


# ======================================================================
# _merge_small_chunks
# ======================================================================
class TestMergeSmallChunks:

    def test_empty(self):
        assert _merge_small_chunks([], 100) == []

    def test_single_chunk(self):
        assert _merge_small_chunks(["hello"], 100) == ["hello"]

    def test_no_merging_needed(self):
        chunks = ["a" * 200, "b" * 200, "c" * 200]
        result = _merge_small_chunks(chunks, 100)
        assert len(result) == 3

    def test_merges_small_first(self):
        chunks = ["small", "a" * 200, "b" * 200]
        result = _merge_small_chunks(chunks, 100)
        # "small" should be merged with "a"*200
        assert len(result) == 2
        assert "small" in result[0]

    def test_merges_small_last(self):
        chunks = ["a" * 200, "b" * 200, "tiny"]
        result = _merge_small_chunks(chunks, 100)
        # "tiny" should be merged with previous
        assert len(result) == 2
        assert "tiny" in result[-1]

    def test_all_small(self):
        chunks = ["a", "b", "c", "d"]
        result = _merge_small_chunks(chunks, 100)
        # All should be merged into one
        assert len(result) == 1
        assert "a" in result[0]
        assert "d" in result[0]

    def test_merge_preserves_content(self):
        chunks = ["hello", "world"]
        result = _merge_small_chunks(chunks, 100)
        combined = result[0]
        assert "hello" in combined
        assert "world" in combined

    def test_separator_is_double_newline(self):
        chunks = ["hello", "world"]
        result = _merge_small_chunks(chunks, 100)
        assert "\n\n" in result[0]


# ======================================================================
# chunk_with_fallback (mocking the semantic chunker)
# ======================================================================
class TestChunkWithFallback:

    def test_fallback_on_error(self):
        """When semantic chunking fails, falls back to recursive splitter."""
        text = "Hello world. " * 1000  # ~13k chars

        with patch(
            "chunking.semantic.semantic_chunk",
            side_effect=Exception("Embedding API failed"),
        ):
            result = chunk_with_fallback(text)
            assert len(result) > 0
            # All original text should be covered
            combined = " ".join(result)
            assert "Hello world" in combined

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk via fallback."""
        text = "Short text."

        with patch(
            "chunking.semantic.semantic_chunk",
            side_effect=Exception("fail"),
        ):
            result = chunk_with_fallback(text)
            assert len(result) == 1
            assert result[0] == text

    def test_returns_list_of_strings(self):
        text = "Test. " * 500

        with patch(
            "chunking.semantic.semantic_chunk",
            return_value=["chunk 1", "chunk 2"],
        ):
            result = chunk_with_fallback(text)
            assert isinstance(result, list)
            assert all(isinstance(c, str) for c in result)
