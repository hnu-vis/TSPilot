"""Restricted Python execution for generated row analysis."""
from __future__ import annotations

import collections
import datetime
import decimal
import enum
import json
import math
import signal
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from runtime.timeout_policy import load_timeout_policy

from .code_policy import (
    AnalysisPolicyError,
    prepare_analysis_code,
    resolve_import_bindings,
    safe_analysis_builtins,
)


class AnalysisCodeError(ValueError):
    """Raised when generated analysis code is unsafe or invalid."""


@dataclass
class ExecutionOutput:
    result: dict
    runtime_ms: int


_SAFE_IMPORT_MODULES = {
    "collections": collections,
    "datetime": datetime,
    "math": math,
    "statistics": statistics,
}


def execute_python_rows_v1(
    *,
    code: str,
    rows: list[dict],
    points: list[dict],
    columns: list[str],
    metadata: dict,
    diagnostics: dict,
    timeout_seconds: int | float | None = None,
) -> ExecutionOutput:
    """Execute generated analysis code over normalized evidence rows."""

    timeout_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else load_timeout_policy().tool("code_interpreter").stage_seconds("sandbox_seconds")
    )

    try:
        prepared = prepare_analysis_code(
            code,
            allowed_import_modules=frozenset(_SAFE_IMPORT_MODULES),
        )
        imported_names = resolve_import_bindings(prepared.imports, _SAFE_IMPORT_MODULES)
    except AnalysisPolicyError as exc:
        raise AnalysisCodeError(str(exc)) from exc
    safe_metadata = _safe_mapping(metadata)
    safe_diagnostics = _safe_mapping(diagnostics)
    globals_dict = {
        "__builtins__": safe_analysis_builtins(),
        "math": math,
        "statistics": statistics,
    }
    locals_dict: dict[str, Any] = {
        "rows": [dict(row) for row in rows],
        "points": [dict(point) for point in points],
        "columns": list(columns),
        "database_evidence": {
            "rows": [dict(row) for row in rows],
            "points": [dict(point) for point in points],
            "data": {
                "rows": [dict(row) for row in rows],
                "points": [dict(point) for point in points],
                "series": [{"points": [dict(point) for point in points]}] if points else [],
            },
            "columns": list(columns),
            "metadata": safe_metadata,
            "diagnostics": safe_diagnostics,
        },
        "metadata": safe_metadata,
        "diagnostics": safe_diagnostics,
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "pstdev": statistics.pstdev,
        "sqrt": math.sqrt,
    }
    locals_dict.update(imported_names)
    started = time.perf_counter()
    try:
        with _time_limit(timeout_seconds):
            exec(compile(prepared.code, "<analysis_code>", "exec"), globals_dict, locals_dict)
    except TimeoutError as exc:
        raise AnalysisCodeError(f"analysis_code exceeded {timeout_seconds}s timeout") from exc
    except AnalysisCodeError:
        raise
    except Exception as exc:
        raise AnalysisCodeError(f"analysis_code execution failed: {exc}") from exc
    runtime_ms = int((time.perf_counter() - started) * 1000)
    result = validate_analysis_result_payload(locals_dict.get("result"))
    return ExecutionOutput(result=result, runtime_ms=runtime_ms)


def _safe_mapping(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, "", [], ()):
        return {}
    return {"value": value}


def validate_analysis_result_payload(result) -> dict:
    """Validate the computation-only payload emitted inside the sandbox."""

    if not isinstance(result, dict):
        raise AnalysisCodeError("analysis_code must assign a dict to variable 'result'.")
    computed = result.get("computed_insights")
    if not isinstance(computed, list) or not computed or any(not isinstance(item, dict) for item in computed):
        raise AnalysisCodeError("analysis result must include non-empty computed_insights list.")
    derived = result.get("derived_evidence", [])
    if not isinstance(derived, list) or any(not isinstance(item, dict) for item in derived):
        raise AnalysisCodeError("analysis result field derived_evidence must be a list of objects.")
    normalized = {
        "computed_insights": _json_safe_value(computed, path="computed_insights"),
        "derived_evidence": _json_safe_value(derived, path="derived_evidence"),
    }
    try:
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except TypeError as exc:
        raise AnalysisCodeError("analysis result must be JSON serializable.") from exc
    except ValueError as exc:
        raise AnalysisCodeError("analysis result must be strict JSON without NaN or Infinity.") from exc
    return normalized


def _json_safe_value(value: Any, *, path: str, seen: set[int] | None = None) -> Any:
    """Normalize common analysis-library values to strict JSON primitives."""

    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, decimal.Decimal):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, enum.Enum):
        return _json_safe_value(value.value, path=path, seen=seen)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()

    value_id = id(value)
    if value_id in seen:
        raise AnalysisCodeError(f"analysis result field '{path}' contains a circular reference.")

    scalar = _library_scalar(value)
    if scalar is not _UNHANDLED:
        return _json_safe_value(scalar, path=path, seen=seen)

    converted = _library_container(value)
    if converted is not _UNHANDLED:
        return _json_safe_value(converted, path=path, seen=seen)

    if isinstance(value, dict):
        seen.add(value_id)
        try:
            return {
                str(key): _json_safe_value(item, path=f"{path}.{key}", seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.remove(value_id)
    if isinstance(value, (list, tuple)):
        seen.add(value_id)
        try:
            return [
                _json_safe_value(item, path=f"{path}[{index}]", seen=seen)
                for index, item in enumerate(value)
            ]
        finally:
            seen.remove(value_id)
    raise AnalysisCodeError(
        f"analysis result field '{path}' is not JSON serializable: {type(value).__name__}."
    )


_UNHANDLED = object()


def _library_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return _UNHANDLED
    return _UNHANDLED


def _library_container(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            type_name = type(value).__name__.lower()
            if "dataframe" in type_name:
                return value.to_dict(orient="records")
            return to_dict()
        except Exception:
            return _UNHANDLED
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return to_list()
        except Exception:
            return _UNHANDLED
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            return _UNHANDLED
    return _UNHANDLED


@contextmanager
def _time_limit(seconds: int):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handler(_signum, _frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
