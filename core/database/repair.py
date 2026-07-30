"""Query repair helpers.

Dialect-specific repairs live in core.database.dialects.  This module keeps a
small compatibility entrypoint for callers and common error classification.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dialects import dialect_for_database


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
    dialect = dialect_for_database(_database_type_from_language(query_language, repaired))
    dialect_result = dialect.repair_query(
        query=repaired,
        query_language=query_language,
        error=error,
    )
    if dialect_result.changed or dialect_result.hint:
        return dialect_result

    if repaired != query:
        return QueryRepairResult(query=repaired, changed=True, reason="trim_query_whitespace")
    return QueryRepairResult(query=query)


def _database_type_from_language(query_language: str | None, query: str | None = None) -> str:
    language = str(query_language or "").strip().lower()
    query_text = str(query or "").strip()
    if language == "flux" or "|>" in query_text or query_text.startswith("from("):
        return "influxdb"
    if language == "promql":
        return "prometheus"
    return language or "sql"
