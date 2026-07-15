"""Deterministic query repair helpers."""
from __future__ import annotations


def classify_query_error(error: Exception | str) -> dict:
    message = str(error)
    normalized = message.lower()
    retryable = any(
        token in normalized
        for token in (
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
        )
    )
    suggestion = None
    if "not found" in normalized or "unknown metric" in normalized:
        suggestion = "inspect_schema_or_metrics"
    elif "timeout" in normalized:
        suggestion = "retry_with_lower_cardinality"
    elif "parse" in normalized or "syntax" in normalized:
        suggestion = "repair_query_syntax"
    return {
        "message": message,
        "retryable": retryable,
        "suggestion": suggestion,
    }


def should_retry_query(repair_result: dict, attempts: int) -> bool:
    return bool(repair_result.get("retryable")) and attempts < 1
