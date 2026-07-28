from __future__ import annotations

import asyncio

import pytest

from core.analysis.python_runner import AnalysisCodeError
from app.settings import get_settings
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.tool import ToolObservation
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool


class _ToolSpec:
    result_target = "evidence"


def _build_full_evidence_payload():
    points = [
        {"timestamp": f"2023-01-01T00:{index:02d}:00Z", "value": float(index)}
        for index in range(40)
    ]
    return {
        "evidence_id": "evi_demo_full",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo",
        "summary": "Loaded 40 points.",
        "data": {
            "points": points,
            "rows": [{"timestamp": item["timestamp"], "value": item["value"]} for item in points],
            "series": [
                {
                    "series_name": "value",
                    "value_field": "value",
                    "time_field": "timestamp",
                    "points": points,
                    "labels": {},
                }
            ],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {"query_trace": {"raw_result_summary": {"row_count": 40, "columns": ["timestamp", "value"]}}},
    }


def _build_large_evidence_payload(size: int = 20_000):
    rows = [
        {
            "timestamp": f"2023-01-{(index % 31) + 1:02d}T00:00:00Z",
            "value": float(index),
            "marker": f"middle-row-{index}",
        }
        for index in range(size)
    ]
    points = [
        {"timestamp": row["timestamp"], "value": row["value"], "marker": row["marker"]}
        for row in rows
    ]
    return {
        "evidence_id": "evi_demo_large",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo large query",
        "summary": f"Loaded {size} rows.",
        "data": {
            "points": points,
            "rows": rows,
            "series": [
                {
                    "series_name": "value",
                    "value_field": "value",
                    "time_field": "timestamp",
                    "points": points,
                    "labels": {"currency": "USD"},
                }
            ],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value", "marker"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {
            "query_trace": {"raw_result_summary": {"row_count": size, "columns": ["timestamp", "value", "marker"]}}
        },
    }


def _build_rows_only_price_evidence_payload():
    rows = [
        {"timestamp": "2023-01-01T00:00:00Z", "price": 10.0, "code": "USD"},
        {"timestamp": "2023-01-02T00:00:00Z", "price": 12.0, "code": "USD"},
    ]
    return {
        "evidence_id": "evi_rows_only_price",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo rows only query",
        "summary": "Loaded 2 rows.",
        "data": {"rows": rows, "time_field": "timestamp"},
        "columns": ["timestamp", "price", "code"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {},
    }


def test_request_state_keeps_summary_evidence_and_full_artifact():
    settings = get_settings()
    request = ChatRequest(message="分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})

    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    latest = request_state.latest_database_evidence
    assert latest is not None
    assert latest.evidence_id == "evi_demo_full"
    assert len(latest.data["points"]) == 24
    assert latest.diagnostics["artifact_ref"] == "evidence:evi_demo_full"
    assert request_state.database_evidence_artifacts["evi_demo_full"].data["points"][0]["timestamp"] == "2023-01-01T00:00:00Z"
    assert len(request_state.database_evidence_artifacts["evi_demo_full"].data["points"]) == 40
    assert len(request_state.observations[-1].payload["data"]["points"]) == 24
    assert len(request_state.observations[-1].payload["data"]["series"][0]["points"]) == 12


def test_code_interpreter_supports_database_evidence_point_aliases_for_rows_only_price_data():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 计算价格变化")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_rows_only_price_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="rows-only price compatibility",
                code=(
                    "pts = database_evidence['points']\n"
                    "series_pts = database_evidence['data']['series'][0]['points']\n"
                    "result = {\n"
                    "    'summary': 'computed from aliased points',\n"
                    "    'metrics': {\n"
                    "        'point_count': len(pts),\n"
                    "        'series_point_count': len(series_pts),\n"
                    "        'start_value': float(pts[0]['value']),\n"
                    "        'end_value': float(pts[-1]['value']),\n"
                    "    },\n"
                    "    'details': {},\n"
                    "}\n"
                ),
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["metrics"] == {
        "point_count": 2,
        "series_point_count": 2,
        "start_value": 10.0,
        "end_value": 12.0,
    }


def test_large_evidence_observation_and_prompt_are_prompt_safe():
    settings = get_settings()
    request = ChatRequest(message="分析大结果集趋势")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})

    apply_observation(request_state, observation, _build_large_evidence_payload(), _ToolSpec())

    full_artifact = request_state.database_evidence_artifacts["evi_demo_large"]
    stored_payload = request_state.observations[-1].payload
    assert len(full_artifact.data["rows"]) == 20_000
    assert len(full_artifact.data["series"][0]["points"]) == 20_000
    assert len(stored_payload["data"]["rows"]) == 12
    assert len(stored_payload["data"]["points"]) == 24
    assert len(stored_payload["data"]["series"][0]["points"]) == 12
    assert stored_payload["diagnostics"]["summary_stats"]["rows_count"] == 20_000

    prompt = DataAgentPromptBuilder().build_user_prompt(request_state, conversation_state)
    assert len(prompt) < 80_000
    assert "middle-row-10000" not in prompt
    assert "evidence:evi_demo_large" in prompt


def test_code_interpreter_uses_full_evidence_artifact_from_request_state():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    request_state.latest_database_evidence = None
    request_state.database_evidence_artifacts = {}
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="count full artifact rows with code interpreter",
                code=(
                    "import itertools\n"
                    "pairs = list(itertools.pairwise([row['value'] for row in rows]))\n"
                    "result = {'summary': f'{len(rows)} rows analyzed by code interpreter', "
                    "'metrics': {'row_count': len(rows), 'pair_count': len(pairs)}, 'details': {}}\n"
                ),
            ),
            request_state=request_state,
        )
    )

    assert result["code_type"] == "code_interpreter_v1"
    assert result["diagnostics"]["sandbox"] == "subprocess_code_interpreter_v1"
    assert result["input_row_count"] == 40
    assert result["result"]["metrics"]["row_count"] == 40
    assert result["result"]["metrics"]["pair_count"] == 39


def test_code_interpreter_accepts_analysis_code_alias():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    tool_input = CodeInterpreterInput.model_validate(
        {
            "analysis_goal": "count rows via alias",
            "analysis_code": (
                "result = {'summary': f'{len(rows)} rows analyzed', "
                "'metrics': {'row_count': len(rows)}, 'details': {}}\n"
            ),
        }
    )
    result = asyncio.run(
        CodeInterpreterTool().execute(
            tool_input,
            request_state=request_state,
        )
    )

    assert result["input_row_count"] == 40
    assert result["result"]["metrics"]["row_count"] == 40


def test_code_interpreter_requires_stable_result_shape():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(AnalysisCodeError, match="metrics"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="missing metrics",
                    code="result = {'summary': 'computed', 'details': {}}\n",
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_enforces_expected_result_schema():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="structured trend summary",
                code=(
                    "values = [float(row['value']) for row in rows]\n"
                    "result = {\n"
                    "    'summary': 'structured trend result',\n"
                    "    'metrics': {'row_count': len(rows), 'min_value': min(values), 'max_value': max(values)},\n"
                    "    'details': {\n"
                    "        'extrema': {'min_timestamp': rows[0]['timestamp'], 'max_timestamp': rows[-1]['timestamp']},\n"
                    "        'findings': [{'label': 'range', 'value': max(values) - min(values), 'evidence_ref': 'rows'}],\n"
                    "    },\n"
                    "}\n"
                ),
                expected_result_schema={
                    "summary": "str",
                    "metrics": {"row_count": "int", "min_value": "number", "max_value": "number"},
                    "details": {
                        "extrema": {"min_timestamp": "str", "max_timestamp": "str"},
                        "findings": [{"label": "str", "value": "number", "evidence_ref": "str"}],
                    },
                },
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["details"]["findings"][0]["value"] == 39.0


def test_code_interpreter_rejects_opaque_outlier_treatment():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 处理异常值并计算指标")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(AnalysisCodeError, match="outlier treatment"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="opaque outlier treatment",
                    code=(
                        "result = {\n"
                        "    'summary': 'filtered outliers',\n"
                        "    'metrics': {'has_outlier_like_values': True, 'used_point_count': 38},\n"
                        "    'details': {'outlier_like_values': rows[:2]},\n"
                        "}\n"
                    ),
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_rejects_text_only_outlier_treatment_note():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 处理异常值并计算指标")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(AnalysisCodeError, match="outlier treatment"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="text-only outlier treatment",
                    code=(
                        "result = {\n"
                        "    'summary': '已基于数据库证据计算区间涨跌幅。',\n"
                        "    'metrics': {'row_count': len(rows), 'start_value': rows[2]['value']},\n"
                        "    'details': {'note': '原始结果中前两条为异常巨大值，因此起始值采用区间内首个有效价格。'},\n"
                        "}\n"
                    ),
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_rejects_outlier_treatment_without_excluded_row_list():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 处理异常值并计算指标")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(AnalysisCodeError, match="excluded_rows as a list"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="bad excluded rows type",
                    code=(
                        "result = {\n"
                        "    'summary': 'filtered outliers with count only',\n"
                        "    'metrics': {'outlier_count': 2},\n"
                        "    'details': {\n"
                        "        'outlier_rule': 'value > threshold',\n"
                        "        'threshold_or_formula': 'value > threshold',\n"
                        "        'rationale': 'extreme values',\n"
                        "        'excluded_rows': 2,\n"
                        "        'raw_metrics': {'row_count': len(rows)},\n"
                        "        'adjusted_metrics': {'row_count': len(rows) - 2},\n"
                        "    },\n"
                        "}\n"
                    ),
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_rejects_duplicate_excluded_rows():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 处理异常值并计算指标")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    duplicate = {"timestamp": "2023-01-01T00:00:00Z", "value": 1000001.0}
    with pytest.raises(AnalysisCodeError, match="duplicate details.excluded_rows"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="duplicate excluded rows",
                    code=(
                        f"duplicate = {duplicate!r}\n"
                        "result = {\n"
                        "    'summary': 'filtered duplicate outliers',\n"
                        "    'metrics': {'outlier_count': 2},\n"
                        "    'details': {\n"
                        "        'outlier_rule': 'value > threshold',\n"
                        "        'threshold_or_formula': 'value > threshold',\n"
                        "        'rationale': 'extreme values',\n"
                        "        'excluded_rows': [duplicate, duplicate],\n"
                        "        'raw_metrics': {'row_count': len(rows)},\n"
                        "        'adjusted_metrics': {'row_count': len(rows) - 1},\n"
                        "    },\n"
                        "}\n"
                    ),
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_allows_transparent_outlier_treatment():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 处理异常值并计算指标")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="transparent outlier treatment",
                code=(
                    "excluded = rows[:2]\n"
                    "result = {\n"
                    "    'summary': 'filtered outliers with explicit rule',\n"
                    "    'metrics': {'has_outlier_like_values': True, 'used_point_count': len(rows) - len(excluded)},\n"
                    "    'details': {\n"
                    "        'outlier_rule': 'manual demonstration rule',\n"
                    "        'threshold_or_formula': 'first two rows are excluded for this test fixture',\n"
                    "        'rationale': 'test-only fixture validates transparent reporting',\n"
                    "        'excluded_rows': excluded,\n"
                    "        'raw_metrics': {'row_count': len(rows)},\n"
                    "        'adjusted_metrics': {'row_count': len(rows) - len(excluded)},\n"
                    "    },\n"
                    "}\n"
                ),
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["details"]["excluded_rows"] == _build_full_evidence_payload()["data"]["rows"][:2]


def test_code_interpreter_rejects_result_missing_expected_detail_field():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(AnalysisCodeError, match="details\\.findings"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="missing structured findings",
                    code=(
                        "result = {\n"
                        "    'summary': 'structured trend result',\n"
                        "    'metrics': {'row_count': len(rows)},\n"
                        "    'details': {'extrema': {}},\n"
                        "}\n"
                    ),
                    expected_result_schema={
                        "summary": "str",
                        "metrics": {"row_count": "int"},
                        "details": {"findings": [{"label": "str", "value": "number"}]},
                    },
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_rejects_unknown_evidence_reference():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(ValueError, match="could not resolve database_evidence"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    database_evidence="evidence:missing",
                    analysis_goal="bad ref",
                    code="result = {'summary': 'computed', 'metrics': {}, 'details': {}}\n",
                ),
                request_state=request_state,
            )
        )
