"""Tool capability contracts for semantic Data Fact requests."""
from __future__ import annotations

from schemas.data_fact import DataFactRequest


SQL_FACT_TYPES = {
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


def fact_request_contract_error(request: DataFactRequest, tool_name: str) -> str | None:
    """Return a precise contract error without deriving missing semantics."""

    if tool_name == "code_interpreter":
        if request.fact_type in SQL_FACT_TYPES and not request.derived_from:
            return (
                f"Code Interpreter cannot create atomic database Fact '{request.fact_key}' without parent Facts. "
                "Produce it with sql_query, or declare verified parent fact_key values in derived_from."
            )
        return None
    if tool_name != "sql_query":
        return None
    if request.fact_type not in SQL_FACT_TYPES:
        return (
            f"SQL does not support fact_type '{request.fact_type}' for '{request.fact_key}'. "
            "Use point_value/time_boundary/extreme/count or move the Fact to code_interpreter."
        )
    requirements = request.requirements or {}
    if request.fact_type == "point_value" and requirements.get("time_position") not in {"start", "end"}:
        return f"SQL point_value Fact '{request.fact_key}' requires requirements.time_position start or end."
    if request.fact_type in {"time_boundary", "boundary_time"} and requirements.get("time_position") not in {"start", "end"}:
        return f"SQL time boundary Fact '{request.fact_key}' requires requirements.time_position start or end."
    if request.fact_type in {"extreme", "extrema", "extreme_time"} and requirements.get("operator") not in {"min", "max"}:
        return f"SQL extreme Fact '{request.fact_key}' requires requirements.operator min or max."
    return None
