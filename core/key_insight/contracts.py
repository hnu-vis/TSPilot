"""Tool capability contracts for semantic Key Insight requests."""
from __future__ import annotations

from schemas.key_insight import KeyInsightRequest


SQL_INSIGHT_TYPES = {
    "count",
    "record_count",
    "row_count",
    "time_boundary",
    "boundary_time",
    "point_value",
    "extreme",
    "extrema",
    "extreme_time",
}


def insight_request_contract_error(request: KeyInsightRequest, tool_name: str) -> str | None:
    """Return a precise contract error without deriving missing semantics."""

    if tool_name in {"forecast", "anomaly"}:
        return (
            f"{tool_name} produces an analysis artifact, not a verified Key Insight. "
            "Keep its points in the artifact and request deterministic Key Insights from sql_query or code_interpreter."
        )
    if tool_name == "code_interpreter":
        # Result validation requires database rows or verified Key Insight parents and
        # a calculation trace, so scalar calculations remain grounded.
        return None
    if tool_name != "sql_query":
        return None
    if request.insight_type not in SQL_INSIGHT_TYPES:
        return (
            f"SQL does not support insight_type '{request.insight_type}' for '{request.insight_key}'. "
            "Use point_value/time_boundary/extreme/count or move the Key Insight to code_interpreter."
        )
    requirements = request.requirements or {}
    if request.insight_type == "point_value" and requirements.get("time_position") not in {"start", "end"}:
        return f"SQL point_value Key Insight '{request.insight_key}' requires requirements.time_position start or end."
    if request.insight_type in {"time_boundary", "boundary_time"} and requirements.get("time_position") not in {"start", "end"}:
        return f"SQL time boundary Key Insight '{request.insight_key}' requires requirements.time_position start or end."
    if request.insight_type in {"time_boundary", "boundary_time"}:
        semantic_class = str(request.semantic_class or "").strip().lower()
        boundary_classes = {"", "dataset_boundary", "observation_boundary", "time_boundary"}
        if semantic_class not in boundary_classes or "measure" in requirements:
            return (
                f"SQL time boundary Key Insight '{request.insight_key}' can describe only a queried dataset/observation boundary. "
                "Calculated event or decision times must be produced by code_interpreter."
            )
    if request.insight_type in {"extreme", "extrema", "extreme_time"} and requirements.get("operator") not in {"min", "max"}:
        return f"SQL extreme Key Insight '{request.insight_key}' requires requirements.operator min or max."
    return None
