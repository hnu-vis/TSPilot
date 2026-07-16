"""Deterministic query repair helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRepairResult:
    query: str
    changed: bool = False
    reason: str | None = None
    hint: str | None = None

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
    if "undefined identifier date" in normalized:
        suggestion = "add_flux_date_import"
    elif "tried to produce more than one result" in normalized:
        suggestion = "name_or_split_flux_results"
    elif "not found" in normalized or "unknown metric" in normalized:
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


def repair_read_only_query(
    *,
    query: str,
    query_language: str | None = None,
    error: Exception | str | None = None,
) -> QueryRepairResult:
    """Return one conservative repair for a read-only query.

    Repairs must be local and semantics-preserving. Anything that changes tables,
    filters, grouping, or analysis intent belongs to ReAct, not this helper.
    """

    repaired = query.strip()
    language = str(query_language or "").lower()
    looks_like_flux = language == "flux" or "|>" in repaired or repaired.startswith("from(")
    error_text = str(error or "").lower()

    if looks_like_flux:
        date_result = _repair_flux_date_import(repaired, error_text)
        if date_result.changed:
            return date_result
        yield_result = _repair_flux_duplicate_default_result(repaired, error_text)
        if yield_result.changed:
            return yield_result

    if repaired != query:
        return QueryRepairResult(query=repaired, changed=True, reason="trim_query_whitespace")
    return QueryRepairResult(query=query)


def _repair_flux_date_import(query: str, error_text: str) -> QueryRepairResult:
    if "date." not in query:
        return QueryRepairResult(query=query)
    if 'import "date"' in query or "import 'date'" in query:
        return QueryRepairResult(query=query)
    if error_text and "undefined identifier date" not in error_text:
        return QueryRepairResult(query=query)
    return QueryRepairResult(
        query='import "date"\n' + query,
        changed=True,
        reason="add_flux_date_import",
        hint='Added Flux import "date" because the query uses date.* functions.',
    )


def _repair_flux_duplicate_default_result(query: str, error_text: str) -> QueryRepairResult:
    if "tried to produce more than one result" not in error_text:
        return QueryRepairResult(query=query)
    if "yield(" in query:
        return QueryRepairResult(
            query=query,
            hint="Flux produced multiple default results; split the query or give each result a unique yield(name).",
        )
    parts = [part.strip() for part in re.split(r"\n\s*\n(?=from\s*\()", query) if part.strip()]
    if len(parts) < 2:
        return QueryRepairResult(
            query=query,
            hint="Flux produced multiple default results; split the query or give each result a unique yield(name).",
        )
    repaired_parts = [
        f'{part}\n  |> yield(name: "result_{index}")'
        for index, part in enumerate(parts, start=1)
    ]
    return QueryRepairResult(
        query="\n\n".join(repaired_parts),
        changed=True,
        reason="name_flux_results",
        hint="Added unique yield names to multiple Flux result streams.",
    )
