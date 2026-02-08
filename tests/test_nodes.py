"""Tests for agent/nodes.py — _pre_validate_patches, _trim_messages, helpers."""

from __future__ import annotations

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.nodes import (
    _count_nested_items,
    _pre_validate_patches,
    _resolve_path,
    _trim_messages,
)


# ======================================================================
# _count_nested_items
# ======================================================================
class TestCountNestedItems:

    def test_empty_dict(self):
        assert _count_nested_items({}) == 0

    def test_flat_dict(self):
        assert _count_nested_items({"a": 1, "b": 2}) == 2

    def test_nested_dict(self):
        assert _count_nested_items({"a": {"b": 1, "c": 2}, "d": 3}) == 3

    def test_array(self):
        assert _count_nested_items([1, 2, 3]) == 3

    def test_nested_array(self):
        assert _count_nested_items([{"a": 1}, {"b": 2}]) == 2

    def test_scalar(self):
        assert _count_nested_items(42) == 1
        assert _count_nested_items("hello") == 1
        assert _count_nested_items(None) == 1

    def test_complex(self):
        doc = {
            "sections": [
                {"fields": [{"v": 1}, {"v": 2}]},
                {"fields": [{"v": 3}]},
            ],
            "meta": {"title": "X"},
        }
        # 1 + 2 + 3 + "X" = 4 leaf values
        assert _count_nested_items(doc) == 4


# ======================================================================
# _resolve_path
# ======================================================================
class TestResolvePath:

    def test_root(self):
        doc = {"a": 1}
        found, value = _resolve_path(doc, "")
        assert found is True
        assert value == doc

    def test_root_slash(self):
        doc = {"a": 1}
        found, value = _resolve_path(doc, "/")
        assert found is True
        assert value == doc

    def test_simple_key(self):
        doc = {"name": "Alice"}
        found, value = _resolve_path(doc, "/name")
        assert found is True
        assert value == "Alice"

    def test_nested(self):
        doc = {"a": {"b": {"c": 42}}}
        found, value = _resolve_path(doc, "/a/b/c")
        assert found is True
        assert value == 42

    def test_array_index(self):
        doc = {"items": ["a", "b", "c"]}
        found, value = _resolve_path(doc, "/items/1")
        assert found is True
        assert value == "b"

    def test_missing_key(self):
        doc = {"a": 1}
        found, _ = _resolve_path(doc, "/missing")
        assert found is False

    def test_array_dash(self):
        doc = {"items": [1, 2]}
        found, _ = _resolve_path(doc, "/items/-")
        assert found is False

    def test_out_of_range(self):
        doc = {"items": [1]}
        found, _ = _resolve_path(doc, "/items/5")
        assert found is False


# ======================================================================
# _pre_validate_patches
# ======================================================================
class TestPreValidatePatches:

    def test_valid_patches_pass(self):
        doc = {"items": [1, 2], "name": "Alice"}
        errors = _pre_validate_patches([
            {"op": "add", "path": "/items/-", "value": 3},
            {"op": "replace", "path": "/name", "value": "Bob"},
        ], doc)
        assert errors == []

    def test_invalid_path_format(self):
        errors = _pre_validate_patches([
            {"op": "add", "path": "no-leading-slash", "value": 1},
        ], {})
        assert len(errors) == 1
        assert "must start with" in errors[0]["message"]

    def test_add_on_existing_array(self):
        doc = {"items": [1, 2, 3]}
        errors = _pre_validate_patches([
            {"op": "add", "path": "/items", "value": [4]},
        ], doc)
        assert len(errors) == 1
        assert "DESTRUCTIVE OVERWRITE" in errors[0]["message"]

    def test_add_object_on_existing_array(self):
        doc = {"items": [1, 2, 3]}
        errors = _pre_validate_patches([
            {"op": "add", "path": "/items", "value": {"x": 1}},
        ], doc)
        assert len(errors) == 1
        assert "DESTRUCTIVE OVERWRITE" in errors[0]["message"]

    def test_add_at_root(self):
        doc = {"a": 1, "b": 2}
        errors = _pre_validate_patches([
            {"op": "add", "path": "/", "value": {"c": 3}},
        ], doc)
        assert len(errors) == 1
        assert "DESTRUCTIVE" in errors[0]["message"]

    def test_replace_container_array(self):
        doc = {"items": [1, 2, 3, 4, 5]}
        errors = _pre_validate_patches([
            {"op": "replace", "path": "/items", "value": []},
        ], doc)
        assert len(errors) == 1
        assert "DESTRUCTIVE REPLACE" in errors[0]["message"]

    def test_type_downgrade_object_to_scalar(self):
        doc = {"meta": {"a": 1, "b": 2}}
        errors = _pre_validate_patches([
            {"op": "replace", "path": "/meta", "value": "string"},
        ], doc)
        assert len(errors) == 1
        assert "TYPE DOWNGRADE" in errors[0]["message"]

    def test_remove_array_container(self):
        doc = {"items": [1, 2, 3]}
        errors = _pre_validate_patches([
            {"op": "remove", "path": "/items"},
        ], doc)
        assert len(errors) == 1
        assert "DATA LOSS" in errors[0]["message"]

    def test_remove_leaf_allowed(self):
        doc = {"items": [1, 2, 3]}
        errors = _pre_validate_patches([
            {"op": "remove", "path": "/items/0"},
        ], doc)
        assert errors == []

    def test_type_downgrade_container_to_scalar(self):
        doc = {"data": [1, 2, 3]}
        errors = _pre_validate_patches([
            {"op": "add", "path": "/data", "value": 42},
        ], doc)
        # Caught by both add-on-existing-array and type downgrade checks
        assert len(errors) >= 1


# ======================================================================
# _trim_messages
# ======================================================================
class TestTrimMessages:

    def _make_round(self, tool_name: str = "inspect_keys") -> list:
        """Create a single AI + ToolMessage round."""
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": tool_name, "args": {}}],
        )
        tool = ToolMessage(content='{"ok": true}', tool_call_id="tc1")
        return [ai, tool]

    def test_not_enough_rounds_returns_none(self):
        sys = SystemMessage(content="system")
        human = HumanMessage(content="chunk 1")
        r1 = self._make_round()
        messages = [sys, human] + r1
        result = _trim_messages(messages, keep_last_n_rounds=2)
        assert result is None  # only 1 round, can't trim

    def test_trims_old_rounds(self):
        sys = SystemMessage(content="system")
        human = HumanMessage(content="chunk 1")
        rounds = []
        for _ in range(5):
            rounds.extend(self._make_round())
        messages = [sys, human] + rounds

        result = _trim_messages(messages, keep_last_n_rounds=2)
        assert result is not None

        # Should have: SystemMessage + HumanMessage + summary + 2 rounds (4 msgs)
        sys_count = sum(1 for m in result if isinstance(m, SystemMessage))
        human_count = sum(1 for m in result if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in result if isinstance(m, AIMessage))

        assert sys_count == 1
        assert human_count == 2  # original + summary
        assert ai_count == 2  # kept 2 rounds

    def test_summary_injected(self):
        sys = SystemMessage(content="system")
        human = HumanMessage(content="chunk 1")
        rounds = []
        for _ in range(4):
            rounds.extend(self._make_round())
        messages = [sys, human] + rounds

        result = _trim_messages(messages, keep_last_n_rounds=1)
        assert result is not None

        # Find the summary message
        summaries = [
            m for m in result
            if isinstance(m, HumanMessage) and "CONTEXT TRIMMED" in m.content
        ]
        assert len(summaries) == 1
        assert "3 previous iteration" in summaries[0].content

    def test_preserves_system_and_original_human(self):
        sys = SystemMessage(content="my system prompt")
        human = HumanMessage(content="my chunk")
        rounds = []
        for _ in range(5):
            rounds.extend(self._make_round())
        messages = [sys, human] + rounds

        result = _trim_messages(messages, keep_last_n_rounds=1)
        assert result is not None
        assert result[0].content == "my system prompt"
        assert result[1].content == "my chunk"

    def test_no_messages_returns_none(self):
        result = _trim_messages([])
        assert result is None

    def test_only_prefix_no_ai(self):
        sys = SystemMessage(content="system")
        human = HumanMessage(content="chunk")
        result = _trim_messages([sys, human])
        assert result is None
