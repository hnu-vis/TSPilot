"""Subprocess worker for generated Python analysis code."""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

from core.analysis.python_runner import validate_analysis_result_payload
from sandbox.analysis_context import canonical_namespace_values


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m sandbox.worker <input.json> <output.json>", file=sys.stderr)
        return 2
    input_path = Path(args[0])
    output_path = Path(args[1])
    started = time.perf_counter()
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = _execute(payload)
        output = {
            "status": "succeeded",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "result": result,
        }
    except Exception as exc:
        output = {
            "status": "failed",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "error": str(exc),
        }
    output_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return 0


def _execute(payload: dict) -> dict:
    code = str(payload.get("code") or "")
    rows = [dict(row) for row in payload.get("rows") or [] if isinstance(row, dict)]
    points = [dict(point) for point in payload.get("points") or [] if isinstance(point, dict)]
    columns = list(payload.get("columns") or [])
    metadata = dict(payload.get("metadata") or {})
    diagnostics = dict(payload.get("diagnostics") or {})
    input_insights = [dict(insight) for insight in payload.get("input_insights") or [] if isinstance(insight, dict)]
    insight_by_key = {
        str(insight.get("insight_key") or insight.get("insight_id") or insight.get("name")): insight
        for insight in input_insights
        if insight.get("insight_key") or insight.get("insight_id") or insight.get("name")
    }
    canonical_values = canonical_namespace_values(
        {
            "rows": rows,
            "points": points,
            "columns": columns,
            "metadata": metadata,
            "diagnostics": diagnostics,
        }
    )
    supplied_context = payload.get("analysis_context")
    if isinstance(supplied_context, dict):
        canonical_values["analysis_context"] = supplied_context
    database_evidence = {
        "rows": rows,
        "points": points,
        "data": {
            "rows": rows,
            "points": points,
            "series": [{"points": points}] if points else [],
        },
        "columns": columns,
        "metadata": metadata,
        "diagnostics": diagnostics,
        "input_insights": input_insights,
        "insight_by_key": insight_by_key,
    }
    data = dict(database_evidence["data"])
    df = canonical_values.get("df")
    try:
        if df is not None:
            for column in getattr(df, "columns", []):
                data[str(column)] = list(df[column])
    except Exception:
        pass
    pd = None
    np = None
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    namespace = {
        "rows": rows,
        "points": points,
        "columns": columns,
        "database_evidence": database_evidence,
        "data": data,
        "metadata": metadata,
        "diagnostics": diagnostics,
        "input_insights": input_insights,
        "insight_by_key": insight_by_key,
        "math": math,
        "statistics": statistics,
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "pstdev": statistics.pstdev,
        "sqrt": math.sqrt,
        "pd": pd,
        "np": np,
        **canonical_values,
    }
    exec(compile(code, "<sandbox_analysis_code>", "exec"), namespace, namespace)
    return validate_analysis_result_payload(namespace.get("result"))


if __name__ == "__main__":
    raise SystemExit(main())
