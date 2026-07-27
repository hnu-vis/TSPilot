from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.analysis.python_runner import AnalysisCodeError, execute_python_rows_v1
from sandbox.runner import execute_python_sandbox_v1
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.database_context import DatabaseContext
from schemas.tool import ToolObservation
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool


class _AnalysisSpec:
    result_target = "analysis"


def _evidence() -> DatabaseEvidence:
    rows = [
        {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
        {"timestamp": "2023-01-01T01:00:00Z", "value": 25.0},
        {"timestamp": "2023-01-01T02:00:00Z", "value": 30.0},
    ]
    return DatabaseEvidence(
        evidence_id="evi_generated",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:test",
        summary="demo evidence",
        data={
            "points": [{"timestamp": row["timestamp"], "value": row["value"]} for row in rows],
            "rows": rows,
            "time_field": "timestamp",
            "value_field": "value",
        },
        columns=["timestamp", "value"],
        metadata={},
        diagnostics={},
    )


def _request_state():
    evidence = _evidence()
    request_state = build_request_state(
        ChatRequest(
            message="计算 value 高于 20 的比例",
            database_context=DatabaseContext(database_id="demo", database_type="unit"),
        ),
        get_settings(),
    )
    request_state.latest_database_evidence = evidence
    request_state.database_evidence_artifacts[evidence.evidence_id] = evidence
    return request_state


def test_generated_insight_executes_code_over_full_evidence():
    request_state = _request_state()

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence="evi_generated",
                analysis_goal="threshold share",
                code=(
                    "total = len(rows)\n"
                    "count = sum(1 for row in rows if float(row['value']) > 20)\n"
                    "result = {'summary': f'{count}/{total} rows are above 20', "
                    "'metrics': {'count': count, 'total': total, 'proportion': count / total}, 'details': {}}"
                ),
            ),
            request_state=request_state,
        )
    )

    assert result["status"] == "succeeded"
    assert result["input_row_count"] == 3
    assert result["result"]["metrics"]["count"] == 2
    assert result["result"]["metrics"]["proportion"] == pytest.approx(2 / 3)


def test_multiple_generated_insights_accumulate_in_analysis_workspace():
    request_state = _request_state()
    first = {
        "analysis_id": "ana_first",
        "analysis_goal": "first",
        "code_type": "python_rows_v1",
        "code_hash": "sha256:first",
        "input_evidence_id": "evi_generated",
        "input_row_count": 3,
        "status": "succeeded",
        "summary": "first summary",
        "result": {"summary": "first summary", "metrics": {"a": 1}, "details": {}},
        "diagnostics": {},
    }
    second = {
        **first,
        "analysis_id": "ana_second",
        "analysis_goal": "second",
        "code_hash": "sha256:second",
        "summary": "second summary",
        "result": {"summary": "second summary", "metrics": {"b": 2}, "details": {}},
    }
    apply_observation(request_state, ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}), first, _AnalysisSpec())
    apply_observation(request_state, ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}), second, _AnalysisSpec())

    context = DataAgentPromptBuilder().build_context(
        request_state,
        build_conversation_state(ChatRequest(message="x"), "conv"),
    )

    assert set(request_state.analysis_artifacts) == {"ana_first", "ana_second"}
    workspace = context["outputs"]["analysis_workspace"]
    assert workspace["analysis_count"] == 2
    assert [item["analysis_id"] for item in workspace["analyses"]] == ["ana_first", "ana_second"]


def test_python_rows_runner_allows_safe_imports_and_lambda_sorting():
    output = execute_python_rows_v1(
        code=(
            "import statistics as stats\n"
            "from collections import Counter\n"
            "from math import sqrt\n"
            "from datetime import datetime\n"
            "ordered = sorted(rows, key=lambda row: row['value'])\n"
            "values = [row['value'] for row in ordered]\n"
            "counter = Counter([datetime.fromisoformat('2023-01-01T00:00:00').year])\n"
            "evidence_rows = database_evidence['data']['rows']\n"
            "result = {'summary': f'mean={stats.mean(values):.1f}', "
            "'metrics': {'mean': stats.mean(values), 'sqrt': sqrt(4), 'year': counter[2023], 'rows': len(evidence_rows)}, 'details': {}}\n"
        ),
        rows=[{"value": 3.0}, {"value": 1.0}, {"value": 2.0}],
        points=[],
        columns=["value"],
        metadata={},
        diagnostics={},
    )

    assert output.result["summary"] == "mean=2.0"
    assert output.result["metrics"]["sqrt"] == 2.0
    assert output.result["metrics"]["year"] == 1
    assert output.result["metrics"]["rows"] == 3


def test_python_rows_runner_rejects_unsafe_imports_and_requires_result_summary():
    with pytest.raises(AnalysisCodeError):
        execute_python_rows_v1(
            code="import os\nresult = {'summary': 'bad', 'metrics': {}, 'details': {}}",
            rows=[],
            points=[],
            columns=[],
            metadata={},
            diagnostics={},
        )
    with pytest.raises(AnalysisCodeError):
        execute_python_rows_v1(
            code="result = {'metrics': {}}",
            rows=[],
            points=[],
            columns=[],
            metadata={},
            diagnostics={},
        )

    with pytest.raises(AnalysisCodeError, match="details"):
        execute_python_rows_v1(
            code="result = {'summary': 'computed', 'metrics': {}}",
            rows=[],
            points=[],
            columns=[],
            metadata={},
            diagnostics={},
        )


def test_python_sandbox_v1_executes_in_subprocess():
    output = execute_python_sandbox_v1(
        code=(
            "values = [float(row['value']) for row in rows]\n"
            "result = {'summary': f'{len(values)} sandbox rows', "
            "'metrics': {'total': sum(values), 'count': len(values)}, 'details': {}}\n"
        ),
        rows=[{"value": 2.0}, {"value": 3.0}],
        points=[],
        columns=["value"],
        metadata={},
        diagnostics={},
        timeout_seconds=5,
    )

    assert output.result["summary"] == "2 sandbox rows"
    assert output.result["metrics"]["total"] == 5.0


def test_python_sandbox_v1_requires_stable_result_shape():
    with pytest.raises(AnalysisCodeError, match="details"):
        execute_python_sandbox_v1(
            code="result = {'summary': 'computed', 'metrics': {}}",
            rows=[],
            points=[],
            columns=[],
            metadata={},
            diagnostics={},
            timeout_seconds=5,
        )


def test_python_sandbox_v1_times_out():
    with pytest.raises(AnalysisCodeError, match="sandbox timeout"):
        execute_python_sandbox_v1(
            code="while True:\n    pass\nresult = {'summary': 'never', 'metrics': {}, 'details': {}}",
            rows=[],
            points=[],
            columns=[],
            metadata={},
            diagnostics={},
            timeout_seconds=1,
        )


def test_generated_insight_supports_python_sandbox_v1():
    request_state = _request_state()

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence="evi_generated",
                analysis_goal="sandbox threshold share",
                code_type="python_sandbox_v1",
                code=(
                    "total = len(rows)\n"
                    "count = sum(1 for row in rows if float(row['value']) > 20)\n"
                    "result = {'summary': f'{count}/{total} rows are above 20', "
                    "'metrics': {'count': count, 'total': total}, 'details': {}}\n"
                ),
                constraints={"timeout_seconds": 5},
            ),
            request_state=request_state,
        )
    )

    assert result["status"] == "succeeded"
    assert result["code_type"] == "code_interpreter_v1"
    assert result["diagnostics"]["sandbox"] == "subprocess_code_interpreter_v1"
    assert result["input_row_count"] == 3
    assert result["result"]["metrics"]["count"] == 2
