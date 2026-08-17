"""Restricted Python execution for generated row analysis."""
from __future__ import annotations

import ast
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


class AnalysisCodeError(ValueError):
    """Raised when generated analysis code is unsafe or invalid."""


@dataclass
class ExecutionOutput:
    result: dict
    runtime_ms: int


_BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "exit",
    "vars",
}

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_BLOCKED_NODE_TYPES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)

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
    timeout_seconds: int = 2,
) -> ExecutionOutput:
    """Execute generated analysis code over normalized evidence rows."""

    code, imported_names = _prepare_code(code)
    _validate_code(code)
    safe_metadata = _safe_mapping(metadata)
    safe_diagnostics = _safe_mapping(diagnostics)
    globals_dict = {
        "__builtins__": _SAFE_BUILTINS,
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
            exec(compile(code, "<analysis_code>", "exec"), globals_dict, locals_dict)
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
    """Validate the stable result contract consumed by later harness stages."""

    if not isinstance(result, dict):
        raise AnalysisCodeError("analysis_code must assign a dict to variable 'result'.")
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise AnalysisCodeError("analysis result must include non-empty string field 'summary'.")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise AnalysisCodeError("analysis result must include object field 'metrics'.")
    details = result.get("details")
    if not isinstance(details, dict):
        raise AnalysisCodeError("analysis result must include object field 'details'.")
    normalized = dict(result)
    normalized["summary"] = summary.strip()
    normalized["metrics"] = _json_safe_value(dict(metrics), path="metrics")
    normalized["details"] = _json_safe_value(dict(details), path="details")
    insights = result.get("insights", [])
    if not isinstance(insights, list) or any(not isinstance(insight, dict) for insight in insights):
        raise AnalysisCodeError("analysis result field 'insights' must be a list of objects when provided.")
    normalized["insights"] = _json_safe_value(insights, path="insights")
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


def _prepare_code(code: str) -> tuple[str, dict[str, Any]]:
    if not code or not code.strip():
        raise AnalysisCodeError("analysis_code cannot be empty.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise AnalysisCodeError(f"analysis_code syntax error: {exc}") from exc

    imported_names: dict[str, Any] = {}
    sanitized_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            _collect_safe_import(node, imported_names)
            continue
        if isinstance(node, ast.ImportFrom):
            _collect_safe_import_from(node, imported_names)
            continue
        sanitized_body.append(node)
    tree.body = sanitized_body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), imported_names


def _collect_safe_import(node: ast.Import, imported_names: dict[str, Any]) -> None:
    for alias in node.names:
        module_name = alias.name
        if module_name not in _SAFE_IMPORT_MODULES:
            raise AnalysisCodeError(f"analysis_code imports unsupported module: {module_name}.")
        imported_names[alias.asname or module_name] = _SAFE_IMPORT_MODULES[module_name]


def _collect_safe_import_from(node: ast.ImportFrom, imported_names: dict[str, Any]) -> None:
    module_name = node.module or ""
    if node.level or module_name not in _SAFE_IMPORT_MODULES:
        raise AnalysisCodeError(f"analysis_code imports unsupported module: {module_name}.")
    module = _SAFE_IMPORT_MODULES[module_name]
    for alias in node.names:
        if alias.name == "*":
            raise AnalysisCodeError("analysis_code cannot use wildcard imports.")
        if not hasattr(module, alias.name):
            raise AnalysisCodeError(f"analysis_code imports unsupported name: {module_name}.{alias.name}.")
        imported_names[alias.asname or alias.name] = getattr(module, alias.name)


def _validate_code(code: str) -> None:
    if not code or not code.strip():
        raise AnalysisCodeError("analysis_code cannot be empty.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise AnalysisCodeError(f"analysis_code syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODE_TYPES):
            raise AnalysisCodeError(f"analysis_code uses blocked syntax: {type(node).__name__}.")
        if isinstance(node, ast.Name) and (node.id in _BLOCKED_NAMES or node.id.startswith("__")):
            raise AnalysisCodeError(f"analysis_code uses blocked name: {node.id}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise AnalysisCodeError(f"analysis_code uses blocked attribute: {node.attr}.")
        if isinstance(node, ast.Call):
            _validate_call(node)


def _validate_call(node: ast.Call) -> None:
    func = node.func
    if isinstance(func, ast.Name) and (func.id in _BLOCKED_NAMES or func.id.startswith("__")):
        raise AnalysisCodeError(f"analysis_code calls blocked function: {func.id}.")
    if isinstance(func, ast.Attribute):
        if func.attr.startswith("__"):
            raise AnalysisCodeError(f"analysis_code calls blocked attribute: {func.attr}.")
        if isinstance(func.value, ast.Name) and func.value.id not in {"math", "statistics"}:
            return


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
