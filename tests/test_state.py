"""Tests for agent/state.py — reducers."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import messages_reducer, token_usage_reducer


# ======================================================================
# messages_reducer
# ======================================================================
class TestMessagesReducer:

    def test_append_mode(self):
        """Non-SystemMessage start → append to current."""
        current = [HumanMessage(content="old")]
        new = [AIMessage(content="response")]
        result = messages_reducer(current, new)
        assert len(result) == 2
        assert result[0].content == "old"
        assert result[1].content == "response"

    def test_reset_on_system_message(self):
        """SystemMessage at start → replace all messages."""
        current = [
            SystemMessage(content="old system"),
            HumanMessage(content="old human"),
            AIMessage(content="old response"),
        ]
        new = [
            SystemMessage(content="new system"),
            HumanMessage(content="new human"),
        ]
        result = messages_reducer(current, new)
        assert len(result) == 2
        assert result[0].content == "new system"
        assert result[1].content == "new human"

    def test_empty_new(self):
        current = [HumanMessage(content="old")]
        result = messages_reducer(current, [])
        assert len(result) == 1

    def test_empty_current(self):
        result = messages_reducer([], [HumanMessage(content="first")])
        assert len(result) == 1


# ======================================================================
# token_usage_reducer
# ======================================================================
class TestTokenUsageReducer:

    def test_accumulates(self):
        current = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "llm_calls": 1}
        new = {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280, "llm_calls": 1}
        result = token_usage_reducer(current, new)
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 130
        assert result["total_tokens"] == 430
        assert result["llm_calls"] == 2

    def test_empty_current(self):
        result = token_usage_reducer({}, {"input_tokens": 100, "total_tokens": 100, "llm_calls": 1})
        assert result["input_tokens"] == 100
        assert result["llm_calls"] == 1

    def test_empty_new(self):
        current = {"input_tokens": 100, "total_tokens": 100, "llm_calls": 1}
        result = token_usage_reducer(current, {})
        assert result == current

    def test_both_empty(self):
        result = token_usage_reducer({}, {})
        assert result == {}

    def test_cache_tokens(self):
        current = {
            "input_tokens": 100,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 30,
        }
        new = {
            "input_tokens": 200,
            "cache_creation_input_tokens": 60,
            "cache_read_input_tokens": 40,
        }
        result = token_usage_reducer(current, new)
        assert result["cache_creation_input_tokens"] == 110
        assert result["cache_read_input_tokens"] == 70
