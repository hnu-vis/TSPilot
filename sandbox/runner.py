"""Host-side runner for subprocess-isolated generated Python analysis."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from core.analysis.python_runner import AnalysisCodeError, ExecutionOutput, validate_analysis_result_payload

from .analysis_context import build_canonical_analysis_context
from .policy import MAX_RESULT_BYTES, MAX_STDIO_CHARS, clamp_timeout


@dataclass
class SandboxPaths:
    work_dir: Path
    input_path: Path
    output_path: Path


def execute_python_sandbox_v1(
    *,
    code: str,
    rows: list[dict],
    points: list[dict],
    columns: list[str],
    metadata: dict,
    diagnostics: dict,
    input_insights: list[dict] | None = None,
    analysis_context: dict | None = None,
    timeout_seconds: int = 5,
    work_dir: str | Path | None = None,
) -> ExecutionOutput:
    """Execute generated analysis code in a short-lived Python subprocess."""

    if not code or not code.strip():
        raise AnalysisCodeError("analysis_code cannot be empty.")
    timeout = clamp_timeout(timeout_seconds)
    started = time.perf_counter()
    paths, temp_dir = _prepare_paths(work_dir)
    payload = {
        "code": code,
        "rows": [dict(row) for row in rows],
        "points": [dict(point) for point in points],
        "columns": list(columns),
        "metadata": dict(metadata),
        "diagnostics": dict(diagnostics),
        "input_insights": [dict(insight) for insight in input_insights or [] if isinstance(insight, dict)],
    }
    payload["analysis_context"] = dict(analysis_context or build_canonical_analysis_context(
        rows=payload["rows"], points=payload["points"], columns=payload["columns"],
        metadata=payload["metadata"], diagnostics=payload["diagnostics"],
    ))
    try:
        try:
            paths.input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "sandbox.worker",
                str(paths.input_path),
                str(paths.output_path),
            ]
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AnalysisCodeError(f"analysis_code exceeded {timeout}s sandbox timeout") from exc

        stdout = _truncate(completed.stdout)
        stderr = _truncate(completed.stderr)
        if completed.returncode != 0:
            detail = stderr or stdout or f"worker exited with code {completed.returncode}"
            raise AnalysisCodeError(f"analysis_code sandbox failed: {detail}")
        if not paths.output_path.exists():
            raise AnalysisCodeError("analysis_code sandbox did not produce an output file.")
        if paths.output_path.stat().st_size > MAX_RESULT_BYTES:
            raise AnalysisCodeError("analysis_code sandbox result exceeded maximum output size.")

        try:
            output_payload = json.loads(paths.output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AnalysisCodeError("analysis_code sandbox produced invalid JSON.") from exc
        if not isinstance(output_payload, dict):
            raise AnalysisCodeError("analysis_code sandbox output must be a JSON object.")
        if output_payload.get("status") != "succeeded":
            error = str(output_payload.get("error") or "unknown sandbox error")
            raise AnalysisCodeError(f"analysis_code sandbox failed: {error}")
        result = validate_analysis_result_payload(output_payload.get("result"))
        _validate_result_size(result)
        runtime_ms = int((time.perf_counter() - started) * 1000)
        return ExecutionOutput(result=result, runtime_ms=runtime_ms)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _prepare_paths(work_dir: str | Path | None) -> tuple[SandboxPaths, tempfile.TemporaryDirectory | None]:
    if work_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="tspilot_sandbox_")
        base = Path(temp_dir.name)
    else:
        temp_dir = None
        base = Path(work_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
    return (
        SandboxPaths(
            work_dir=base,
            input_path=base / "input.json",
            output_path=base / "output.json",
        ),
        temp_dir,
    )


def _validate_result_size(result: dict) -> None:
    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        raise AnalysisCodeError("analysis result exceeded maximum output size.")


def _truncate(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= MAX_STDIO_CHARS:
        return value
    return value[:MAX_STDIO_CHARS] + "...[truncated]"
