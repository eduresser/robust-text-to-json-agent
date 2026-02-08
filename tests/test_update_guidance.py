"""Tests for tools/update_guidance.py."""

from __future__ import annotations

from tools.update_guidance import update_guidance


class TestUpdateGuidance:

    def test_basic(self):
        result = update_guidance(
            last_path="/sections/0",
            sections_snapshot="[0]OVERVIEW(5flds)",
            items_added="5 fields",
            extracted_entities_count=5,
        )
        assert result["finalized"] is True
        assert result["guidance"]["last_path"] == "/sections/0"
        assert result["guidance"]["sections_snapshot"] == "[0]OVERVIEW(5flds)"
        assert result["guidance"]["extracted_entities_count"] == 5

    def test_defaults(self):
        result = update_guidance()
        assert result["finalized"] is True
        assert result["guidance"]["last_path"] == ""
        assert result["guidance"]["sections_snapshot"] == ""
        assert result["guidance"]["extracted_entities_count"] == 0

    def test_all_fields(self):
        result = update_guidance(
            last_path="/data/0",
            sections_snapshot="[0]A(3flds)",
            items_added="3 items",
            open_section="A @ /data/0 — building",
            text_excerpt="...end of chunk...",
            next_expectations="expect more items",
            pending_data="value TBD",
            extracted_entities_count=3,
        )
        g = result["guidance"]
        assert g["last_path"] == "/data/0"
        assert g["open_section"] == "A @ /data/0 — building"
        assert g["text_excerpt"] == "...end of chunk..."
        assert g["next_expectations"] == "expect more items"
        assert g["pending_data"] == "value TBD"
