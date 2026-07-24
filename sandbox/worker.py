"""Subprocess worker for generated Python analysis code."""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

from core.analysis.python_runner import validate_analysis_result_payload


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
    database_evidence = {
        "data": {"rows": rows, "points": points},
        "columns": columns,
        "metadata": metadata,
        "diagnostics": diagnostics,
    }
    namespace = {
        "rows": rows,
        "points": points,
        "columns": columns,
        "database_evidence": database_evidence,
        "metadata": metadata,
        "diagnostics": diagnostics,
        "math": math,
        "statistics": statistics,
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "pstdev": statistics.pstdev,
        "sqrt": math.sqrt,
    }
    exec(compile(code, "<sandbox_analysis_code>", "exec"), namespace, namespace)
    return validate_analysis_result_payload(namespace.get("result"))


if __name__ == "__main__":
    raise SystemExit(main())
