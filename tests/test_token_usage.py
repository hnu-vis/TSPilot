from __future__ import annotations

from types import SimpleNamespace

from runtime.token_usage import estimate_token_usage, record_llm_token_usage


def test_estimate_token_usage_counts_prompt_and_completion_tokens():
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


def test_record_llm_token_usage_uses_tiktoken_without_provider_usage():
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
    assert request_state.completion_state["token_usage"]["totals"]["call_count"] == 1
