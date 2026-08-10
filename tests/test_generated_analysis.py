from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
from schemas.data_fact import DataFact, DataFactRequest, FactEvidenceRef
from schemas.tool import ToolObservation
from tools.code_interpreter import (
    CodeInterpreterInput,
    CodeInterpreterTool,
    _preflight_analysis_code,
    _validate_fact_output_contract,
    _validate_result_has_numeric_analysis,
)


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


def test_generated_analysis_executes_code_over_full_evidence():
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


def test_code_interpreter_composes_structured_fact_from_verified_parent_facts():
    request_state = _request_state()
    request_state.fact_set.facts = [
        DataFact(
            fact_id="fact_start",
            fact_key="price.start",
            name="start_price",
            fact_type="point_value",
            statement="Start price is 10.",
            value=10.0,
            method="sql_query",
            evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi_generated")],
        ),
        DataFact(
            fact_id="fact_end",
            fact_key="price.end",
            name="end_price",
            fact_type="point_value",
            statement="End price is 30.",
            value=30.0,
            method="sql_query",
            evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi_generated")],
        ),
    ]
    code = (
        "start = float(fact_by_key['price.start']['value'])\n"
        "end = float(fact_by_key['price.end']['value'])\n"
        "change = (end - start) / start * 100\n"
        "result = {'summary': 'price change computed', 'metrics': {'percentage_change': change}, 'details': {}, "
        "'facts': [{'fact_key': 'price.percentage_change', 'name': 'percentage_change', "
        "'fact_type': 'difference', 'statement': f'Price changed by {change}%.', 'value': change, "
        "'derived_from': ['price.start', 'price.end'], "
        "'calculation_trace': {'formula': '(end - start) / start * 100'}}]}"
    )

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence="evi_generated",
                analysis_goal="percentage change",
                code=code,
                fact_requests=[
                    {
                        "fact_key": "price.percentage_change",
                        "name": "percentage_change",
                        "fact_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ],
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["facts"][0]["fact_key"] == "price.percentage_change"
    assert result["result"]["facts"][0]["value"] == pytest.approx(200.0)


def test_code_interpreter_marks_root_fact_partial_when_database_evidence_is_empty():
    request = DataFactRequest(
        fact_key="price.start",
        name="start_price",
        fact_type="point_value",
    )
    result = {
        "facts": [
            {
                "fact_key": "price.start",
                "value": 16838.35,
                "statement": "Start price is 16838.35.",
                "calculation_trace": {"source": "generated"},
            }
        ]
    }

    diagnostics = _validate_fact_output_contract(result, [request], input_row_count=0, input_facts=[])

    assert diagnostics == {
        "bound": ["price.start"],
        "missing": [],
        "rejected": [],
        "partial": [{"fact_key": "price.start", "quality_flags": ["ungrounded_candidate"]}],
    }
    assert result["facts"][0]["status"] == "partial"
    assert result["facts"][0]["quality_flags"] == ["ungrounded_candidate"]


def test_code_interpreter_reports_missing_fact_without_failing_analysis():
    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence=_evidence(),
                analysis_goal="calculate a summary while a requested fact remains unavailable",
                code=(
                    "result = {'summary': 'analysis completed', "
                    "'metrics': {'row_count': len(rows)}, 'details': {}, 'facts': []}"
                ),
                fact_requests=[
                    {
                        "fact_key": "price.unsatisfied",
                        "name": "unsatisfied fact",
                        "fact_type": "custom",
                    }
                ],
            )
        )
    )

    assert result["status"] == "succeeded"
    assert result["result"]["facts"] == []
    assert result["diagnostics"]["fact_binding"]["missing"] == ["price.unsatisfied"]


def test_multiple_generated_analyses_accumulate_in_analysis_workspace():
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
    assert context["state"]["artifact_inventory"]["analysis_count"] == 2
    assert context["artifacts"]["refs"]["analysis"] == ["analysis:ana_first", "analysis:ana_second"]
    assert context["artifacts"]["refs"]["latest_analysis"] == "analysis:ana_second"


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


def test_python_rows_runner_normalizes_common_non_json_analysis_values():
    output = execute_python_rows_v1(
        code=(
            "from datetime import datetime\n"
            "result = {\n"
            "  'summary': 'normalized common values',\n"
            "  'metrics': {'when': datetime(2023, 1, 1), 'bad': float('nan')},\n"
            "  'details': {'values': (1, 2, 3)},\n"
            "}\n"
        ),
        rows=[],
        points=[],
        columns=[],
        metadata={},
        diagnostics={},
    )

    assert output.result["metrics"]["when"] == "2023-01-01T00:00:00"
    assert output.result["metrics"]["bad"] is None
    assert output.result["details"]["values"] == [1, 2, 3]


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


def test_python_sandbox_v1_provides_canonical_timeseries_inputs():
    output = execute_python_sandbox_v1(
        code=(
            "result = {\n"
            "  'summary': 'canonical inputs available',\n"
            "  'metrics': {'mean_value': float(value.mean()), 'point_count': int(len(series))},\n"
            "  'details': {'value_col': value_col, 'time_col': time_col, 'first_timestamp': str(time.iloc[0])},\n"
            "}\n"
        ),
        rows=[],
        points=[
            {"_time": "2023-01-01T01:00:00Z", "price": "3.0"},
            {"_time": "2023-01-01T00:00:00Z", "price": "1.0"},
        ],
        columns=["_time", "price"],
        metadata={},
        diagnostics={},
        timeout_seconds=5,
    )

    assert output.result["metrics"]["mean_value"] == 2.0
    assert output.result["metrics"]["point_count"] == 2
    assert output.result["details"]["value_col"] == "price"
    assert output.result["details"]["time_col"] == "_time"
    assert output.result["details"]["first_timestamp"].startswith("2023-01-01 00:00:00")


def test_python_sandbox_v1_data_supports_column_array_aliases():
    output = execute_python_sandbox_v1(
        code=(
            "values = data['appliances_energy_wh']\n"
            "result = {'summary': 'column alias available', "
            "'metrics': {'max_value': float(max(values))}, "
            "'details': {'value_col': value_col}}\n"
        ),
        rows=[
            {"timestamp": "2023-01-01T00:00:00Z", "appliances_energy_wh": "2"},
            {"timestamp": "2023-01-01T01:00:00Z", "appliances_energy_wh": "5"},
        ],
        points=[],
        columns=["timestamp", "appliances_energy_wh"],
        metadata={},
        diagnostics={},
        timeout_seconds=5,
    )

    assert output.result["metrics"]["max_value"] == 5.0
    assert output.result["details"]["value_col"] == "appliances_energy_wh"


def test_python_sandbox_v1_normalizes_pandas_and_numpy_result_values():
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    output = execute_python_sandbox_v1(
        code=(
            "top = df.nsmallest(2, value_col)\n"
            "result = {\n"
            "  'summary': 'lowest rows',\n"
            "  'metrics': {'min_value': value.min(), 'count': len(top)},\n"
            "  'details': {'rows': top[[time_col, value_col]], 'values': value.to_numpy()},\n"
            "}\n"
        ),
        rows=[
            {"timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat(), "value": "2"},
            {"timestamp": datetime(2023, 1, 2, tzinfo=timezone.utc).isoformat(), "value": "1"},
        ],
        points=[],
        columns=["timestamp", "value"],
        metadata={},
        diagnostics={},
        timeout_seconds=5,
    )

    assert output.result["metrics"]["min_value"] == 1.0
    assert output.result["metrics"]["count"] == 2
    assert output.result["details"]["values"] == [2.0, 1.0]
    assert output.result["details"]["rows"][0]["value"] == 1.0


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


def test_code_interpreter_numeric_validation_allows_details_only_outputs():
    _validate_result_has_numeric_analysis(
        {
            "summary": "lowest rows",
            "metrics": {},
            "details": {
                "lowest_three": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 1.0},
                ],
            },
        },
        requires_numeric_series=True,
        input_rows=3,
    )

    with pytest.raises(AnalysisCodeError, match="no non-empty computed output"):
        _validate_result_has_numeric_analysis(
            {"summary": "empty", "metrics": {}, "details": {}},
            requires_numeric_series=True,
            input_rows=3,
        )


def test_code_interpreter_preflight_handles_python_local_scopes():
    error = _preflight_analysis_code(
        "clean = [x for x in value if isinstance(x, (int, float))]\n"
        "finite = list(filter(lambda v: v == v, clean))\n"
        "has_df = 'df' in globals()\n"
        "result = {'summary': str(has_df), 'metrics': {}, 'details': {'values': finite}}\n",
    )

    assert error is None


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


def test_generated_analysis_supports_python_sandbox_v1():
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
    assert result["diagnostics"]["executed_code"].startswith("total = len(rows)")
    assert result["input_row_count"] == 3
    assert result["result"]["metrics"]["count"] == 2
