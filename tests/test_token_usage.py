from __future__ import annotations

from types import SimpleNamespace

from runtime.token_usage import estimate_token_usage, record_llm_token_usage, token_usage_summary


class _DummyEncoding:
    def encode(self, text: str):
        return str(text).split()


def test_estimate_token_usage_counts_prompt_and_completion_tokens(monkeypatch):
    monkeypatch.setattr("runtime.token_usage._encoding_for_model", lambda model: _DummyEncoding())

    usage = estimate_token_usage(
        [("system", "Return JSON."), ("user", "你好")],
        '{"ok": true}',
        model="gpt-4o-mini",
    )

    assert usage is not None
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["method"] == "tiktoken"


def test_record_llm_token_usage_uses_tiktoken_without_provider_usage(monkeypatch):
    monkeypatch.setattr("runtime.token_usage._encoding_for_model", lambda model: _DummyEncoding())

    request_state = SimpleNamespace(completion_state={})
    response = SimpleNamespace(content='{"action": "terminate"}')

    entry = record_llm_token_usage(
        request_state,
        source="unit",
        response=response,
        messages=[("user", "hello")],
        output_text=response.content,
    )

    assert entry is not None
    assert entry["estimated"]["total_tokens"] > 0
    assert entry["provider"] is None
    assert "duration_ms" not in entry
    assert entry["tool_name"] == "terminate"
    assert request_state.completion_state["token_usage"]["totals"]["call_count"] == 1
    assert request_state.completion_state["token_usage"]["by_tool"]["terminate"]["call_count"] == 1


def test_record_llm_token_usage_retains_duration(monkeypatch):
    monkeypatch.setattr("runtime.token_usage._encoding_for_model", lambda model: _DummyEncoding())

    request_state = SimpleNamespace(completion_state={})
    response = SimpleNamespace(content='{"action": "terminate"}')

    entry = record_llm_token_usage(
        request_state,
        source="unit",
        response=response,
        messages=[("user", "hello")],
        output_text=response.content,
        duration_ms=1234,
    )

    assert entry["duration_ms"] == 1234
    assert request_state.completion_state["token_usage"]["calls"][0]["duration_ms"] == 1234


def test_token_usage_summary_groups_sql_generation_by_tool(monkeypatch):
    monkeypatch.setattr("runtime.token_usage._encoding_for_model", lambda model: _DummyEncoding())

    request_state = SimpleNamespace(completion_state={})
    response = SimpleNamespace(content='{"query": "SELECT 1", "query_language": "sql"}')

    record_llm_token_usage(
        request_state,
        source="sql_query.generation",
        response=response,
        messages=[("user", "generate query")],
        output_text=response.content,
    )

    summary = token_usage_summary(request_state)

    assert summary["by_tool"]["sql_query"]["call_count"] == 1
    assert summary["by_tool"]["sql_query"]["total_tokens"] > 0
