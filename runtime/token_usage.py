"""Token usage accounting helpers for LLM calls."""
from __future__ import annotations

import json
from typing import Any

import tiktoken


def record_llm_token_usage(
    request_state,
    *,
    source: str,
    response: Any,
    messages: list | None = None,
    output_text: str | None = None,
    model: str | None = None,
) -> dict | None:
    """Estimate LLM token usage with tiktoken and retain provider usage when present."""

    if request_state is None:
        return None
    provider_usage = extract_provider_token_usage(response)
    response_text = output_text if output_text is not None else _response_text(response)
    estimated_usage = estimate_token_usage(messages or [], response_text, model=model or _response_model(response))
    if estimated_usage is None and provider_usage is None:
        return None
    token_state = request_state.completion_state.setdefault("token_usage", _empty_usage())
    entry = {
        "source": source,
        "estimated": estimated_usage,
        "provider": provider_usage,
    }
    token_state["calls"].append(entry)
    totals = token_state["totals"]
    counted = estimated_usage or provider_usage or {}
    totals["prompt_tokens"] += int(counted.get("prompt_tokens") or 0)
    totals["completion_tokens"] += int(counted.get("completion_tokens") or 0)
    totals["total_tokens"] += int(counted.get("total_tokens") or 0)
    totals["call_count"] = len(token_state["calls"])
    totals["counting_method"] = "tiktoken_estimate" if estimated_usage else "provider_usage"
    return entry


def token_usage_summary(request_state) -> dict | None:
    usage = (request_state.completion_state or {}).get("token_usage") if request_state is not None else None
    if not isinstance(usage, dict):
        return None
    return {
        "totals": dict(usage.get("totals") or {}),
        "calls": list(usage.get("calls") or []),
    }


def estimate_token_usage(messages: list, output_text: str, *, model: str | None = None) -> dict | None:
    encoding = _encoding_for_model(model)
    prompt_tokens = _count_messages(messages, encoding)
    completion_tokens = len(encoding.encode(output_text or ""))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "model": model,
        "method": "tiktoken",
    }


def extract_provider_token_usage(response: Any) -> dict | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    response_metadata = getattr(response, "response_metadata", None)
    token_usage = None
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not isinstance(token_usage, dict):
        token_usage = None

    raw_usage = usage_metadata if isinstance(usage_metadata, dict) else token_usage
    if not isinstance(raw_usage, dict):
        return None

    prompt_tokens = _first_int(raw_usage, "prompt_tokens", "input_tokens")
    completion_tokens = _first_int(raw_usage, "completion_tokens", "output_tokens")
    total_tokens = _first_int(raw_usage, "total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    model_name = None
    if isinstance(response_metadata, dict):
        model_name = response_metadata.get("model_name") or response_metadata.get("model")
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "model": model_name,
        "method": "provider",
        "raw_usage": raw_usage,
    }


def _empty_usage() -> dict:
    return {
        "totals": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "counting_method": "tiktoken_estimate",
        },
        "calls": [],
    }


def _encoding_for_model(model: str | None):
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    return tiktoken.get_encoding("cl100k_base")


def _count_messages(messages: list, encoding) -> int:
    total = 0
    for message in messages:
        if isinstance(message, tuple) and len(message) >= 2:
            role, content = message[0], message[1]
            total += 4
            total += len(encoding.encode(str(role)))
            total += len(encoding.encode(_content_to_text(content)))
            continue
        if isinstance(message, dict):
            total += 4
            for key, value in message.items():
                total += len(encoding.encode(str(key)))
                total += len(encoding.encode(_content_to_text(value)))
            continue
        total += len(encoding.encode(_content_to_text(message)))
    return total + 2


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        return str(content)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _response_model(response: Any) -> str | None:
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        model = response_metadata.get("model_name") or response_metadata.get("model")
        return str(model) if model else None
    return None


def _first_int(mapping: dict, *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None
