"""Database query error classification for ReAct observations."""
from __future__ import annotations


def classify_query_error(error: Exception | str) -> dict:
    message = str(error)
    normalized = message.lower()
    transient = any(
        token in normalized
        for token in (
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
        )
    )
    suggestion = None
    if "undefined identifier date" in normalized:
        suggestion = "regenerate_with_required_flux_imports"
    elif "tried to produce more than one result" in normalized:
        suggestion = "regenerate_with_named_or_single_flux_result"
    elif "not found" in normalized or "unknown metric" in normalized:
        suggestion = "repeat_schema_linking_before_regeneration"
    elif "timeout" in normalized:
        suggestion = "regenerate_with_lower_cardinality"
    elif "parse" in normalized or "syntax" in normalized:
        suggestion = "regenerate_query_from_execution_error"
    retryable = transient or suggestion is not None
    return {
        "message": message,
        "retryable": retryable,
        "suggestion": suggestion,
    }
