from __future__ import annotations

import asyncio
import json

import pytest

from core.analysis.python_runner import AnalysisCodeError
from app.settings import get_settings
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.tool import ToolObservation
from schemas.timeseries import AnomalyResult
from tools.base import StructuredToolError
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool


class _ToolSpec:
    result_target = "evidence"


class _CapturingCodeLLM:
    def __init__(self):
        self.payload = None

    async def ainvoke(self, messages, **kwargs):
        self.payload = json.loads(messages[-1][1])
        code = (
            "result = {"
            "'summary': 'computed transparent outlier analysis', "
            "'metrics': {'mean': float(value.mean()) if hasattr(value, 'mean') else 0.0}, "
            "'details': {"
            "'outlier_rule': 'IQR upper fence', "
            "'threshold_or_formula': 'Q3 + 1.5 * IQR', "
            "'rationale': 'robust high-value screening', "
            "'excluded_rows': [], "
            "'raw_metrics': {'mean': float(value.mean()) if hasattr(value, 'mean') else 0.0}, "
            "'adjusted_metrics': {'mean': float(value.mean()) if hasattr(value, 'mean') else 0.0}}, "
            "'insights': []}"
        )
        return type("_Response", (), {"content": json.dumps({"code": code})})()


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


def test_code_interpreter_generation_consumes_repair_contract():
    llm = _CapturingCodeLLM()
    repair_contract = {
        "mode": "analysis_artifact_repair",
        "required_details_fields": [
            "outlier_rule",
            "threshold_or_formula",
            "rationale",
            "excluded_rows",
            "raw_metrics",
            "adjusted_metrics",
        ],
    }

    result = asyncio.run(
        CodeInterpreterTool(llm=llm).execute(
            CodeInterpreterInput(
                database_evidence=_build_full_evidence_payload(),
                analysis_goal="identify high outliers with a custom robust rule",
                analysis_request={"required_outputs": ["custom robust outlier analysis"]},
                required_outputs=["custom robust outlier analysis"],
                repair_contract=repair_contract,
            )
        )
    )

    assert llm.payload["mode"] == "repair"
    assert llm.payload["repair_contract"] == repair_contract
    assert llm.payload["analysis_request"]["required_outputs"] == ["custom robust outlier analysis"]
    assert llm.payload["available_lineage_refs"] == ["evidence:evi_demo_full"]
    assert result["result"]["details"]["outlier_rule"] == "IQR upper fence"
    assert result["diagnostics"]["execution_attempts"] == 1


def test_code_interpreter_allows_insight_composition_within_one_result():
    insights = [
        {
            "insight_key": "trend",
            "name": "trend",
            "insight_type": "analysis",
            "value": "down",
            "statement": "trend is down",
            "calculation_trace": {"method": "slope"},
            "derived_from": [],
        },
        {
            "insight_key": "anomalies",
            "name": "anomalies",
            "insight_type": "analysis",
            "value": 2,
            "statement": "two anomalies",
            "calculation_trace": {"method": "iqr"},
            "derived_from": [],
        },
        {
            "insight_key": "conclusion",
            "name": "conclusion",
            "insight_type": "analysis",
            "value": "down with two anomalies",
            "statement": "series declines with two anomalies",
            "calculation_trace": {"method": "composition"},
            "derived_from": ["trend", "anomalies"],
        },
    ]
    code = "result = " + repr(
        {
            "summary": "computed",
            "metrics": {"anomaly_count": 2},
            "details": {},
            "insights": insights,
        }
    )

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence=_build_full_evidence_payload(),
                analysis_goal="compose analysis insights",
                code=code,
                insight_requests=[
                    {key: value for key, value in insight.items() if key in {"insight_key", "name", "insight_type", "derived_from"}}
                    for insight in insights
                ],
            )
        )
    )

    assert [insight["insight_key"] for insight in result["result"]["insights"]] == ["trend", "anomalies", "conclusion"]


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


def _build_timestamp_only_evidence_payload():
    rows = [
        {"timestamp": "2023-01-01T00:00:00Z"},
        {"timestamp": "2023-01-02T00:00:00Z"},
    ]
    return {
        "evidence_id": "evi_timestamp_only",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo timestamp only query",
        "summary": "Loaded 2 rows.",
        "data": {"rows": rows, "time_field": "timestamp"},
        "columns": ["timestamp"],
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


def test_code_interpreter_rejects_numeric_series_analysis_without_value_column():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 计算收益率、波动率和最大回撤")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_timestamp_only_evidence_payload(), _ToolSpec())

    with pytest.raises(AnalysisCodeError, match="numeric time-series values"):
        asyncio.run(
            CodeInterpreterTool().execute(
                CodeInterpreterInput(
                    analysis_goal="计算收益率、波动率和最大回撤",
                    code=(
                        "result = {'summary': 'empty metrics', "
                        "'metrics': {'row_count': 0, 'percentage_change': None, 'volatility': None, 'max_drawdown': 0.0}, "
                        "'details': {'series_length': 0}}\n"
                    ),
                ),
                request_state=request_state,
            )
        )


def test_code_interpreter_allows_non_numeric_row_count_analysis_without_value_column():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 统计行数")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_timestamp_only_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="count rows",
                code="result = {'summary': 'counted rows', 'metrics': {'row_count': len(rows)}, 'details': {}}\n",
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["metrics"]["row_count"] == 2


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

    with pytest.raises(StructuredToolError, match="metrics"):
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

    with pytest.raises(StructuredToolError, match="outlier treatment"):
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

    with pytest.raises(StructuredToolError, match="outlier treatment"):
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


def test_code_interpreter_allows_cleaned_input_assumption_without_claiming_new_outlier_treatment():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析清洗后的序列")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="metrics on cleaned evidence",
                code=(
                    "values = [float(row['value']) for row in rows]\n"
                    "result = {\n"
                    "    'summary': '已计算清洗后序列的指标。',\n"
                    "    'metrics': {'row_count': len(rows), 'start_value': values[0], 'end_value': values[-1]},\n"
                    "    'details': {\n"
                    "        'series_sorted': True,\n"
                    "        'assumptions': ['使用清洗后的完整序列（剔除明显离群高值）作为输入。'],\n"
                    "    },\n"
                    "}\n"
                ),
                expected_result_schema={
                    "summary": "str",
                    "metrics": {"row_count": "int", "start_value": "number", "end_value": "number"},
                    "details": {"series_sorted": "bool", "assumptions": "list"},
                },
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["metrics"]["row_count"] == 40


def test_code_interpreter_does_not_treat_generic_filtering_as_outlier_removal():
    settings = get_settings()
    request_state = build_request_state(ChatRequest(message="筛选时间范围并计算平均值"), settings)
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        _build_full_evidence_payload(),
        _ToolSpec(),
    )

    result = asyncio.run(
        CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="filter the requested time range and calculate an average",
                code=(
                    "values = [float(row['value']) for row in rows]\n"
                    "result = {\n"
                    "    'summary': 'filtered to the requested time range',\n"
                    "    'metrics': {'mean': sum(values) / len(values)},\n"
                    "    'details': {'filter': 'requested time boundary only'},\n"
                    "}\n"
                ),
            ),
            request_state=request_state,
        )
    )

    assert result["result"]["metrics"]["mean"] == 19.5


def test_code_interpreter_rejects_outlier_treatment_without_excluded_row_list():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 处理异常值并计算指标")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(StructuredToolError, match="excluded_rows as a list"):
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
    with pytest.raises(StructuredToolError, match="duplicate details.excluded_rows"):
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


def test_code_interpreter_uses_authoritative_anomaly_points_and_publishes_clean_view():
    settings = get_settings()
    request_state = build_request_state(ChatRequest(message="排除异常点后计算指标并绘图"), settings)
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        _build_full_evidence_payload(),
        _ToolSpec(),
    )
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_demo_full", detector_name="unit",
        anomaly_points=_build_full_evidence_payload()["data"]["rows"][:2],
        diagnostics={"resolved_evidence_id": "evi_demo_full"},
    )
    request_state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
    request_state.latest_anomaly = anomaly

    result = asyncio.run(CodeInterpreterTool().execute(
        CodeInterpreterInput(
            database_evidence="anomaly:anomaly_evi_demo_full",
            analysis_goal="exclude authoritative anomalies and compute metrics",
            code=(
                "excluded = analysis_context['anomaly_context']['anomaly_points']\n"
                "excluded_keys = {(item['timestamp'], item['value']) for item in excluded}\n"
                "clean = [row for row in rows if (row['timestamp'], row['value']) not in excluded_keys]\n"
                "result = {\n"
                "  'summary': 'computed on the authoritative clean series',\n"
                "  'metrics': {'used_point_count': len(clean)},\n"
                "  'details': {\n"
                "    'outlier_rule': 'authoritative anomaly artifact',\n"
                "    'threshold_or_formula': 'artifact membership',\n"
                "    'rationale': 'single source of truth',\n"
                "    'excluded_rows': excluded,\n"
                "    'raw_metrics': {'row_count': len(rows)},\n"
                "    'adjusted_metrics': {'row_count': len(clean)},\n"
                "  },\n"
                "  'data_views': [{\n"
                "    'view_id': 'clean', 'name': 'Clean series', 'shape': 'timeseries', 'rows': clean,\n"
                "    'schema_fields': [{'name': 'timestamp', 'data_type': 'time'}, {'name': 'value', 'data_type': 'number'}],\n"
                "    'lineage': ['evidence:evi_demo_full', analysis_context['anomaly_context']['source_ref']],\n"
                "    'transform_summary': 'Excluded authoritative anomalies',\n"
                "  }],\n"
                "}\n"
            ),
        ),
        request_state=request_state,
    ))

    assert result["result"]["details"]["excluded_rows"] == anomaly.anomaly_points
    assert result["data_views"][0]["lineage"][-1] == "anomaly:anomaly_evi_demo_full"


def test_code_interpreter_rejects_analysis_that_disagrees_with_authoritative_anomaly_points():
    settings = get_settings()
    request_state = build_request_state(ChatRequest(message="排除异常点后计算指标"), settings)
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        _build_full_evidence_payload(),
        _ToolSpec(),
    )
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_demo_full", detector_name="unit",
        anomaly_points=_build_full_evidence_payload()["data"]["rows"][:2],
        diagnostics={"resolved_evidence_id": "evi_demo_full"},
    )
    request_state.anomaly_artifacts[anomaly.anomaly_id] = anomaly

    with pytest.raises(StructuredToolError, match="exactly match"):
        asyncio.run(CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="exclude a different anomaly set",
                code=(
                    "excluded = rows[1:3]\n"
                    "result = {'summary': 'wrong exclusions', 'metrics': {}, 'details': {"
                    "'outlier_rule': 'local detector', 'threshold_or_formula': 'local', 'rationale': 'local', "
                    "'excluded_rows': excluded, 'raw_metrics': {}, 'adjusted_metrics': {}}, 'data_views': [{"
                    "'view_id': 'clean', 'name': 'Clean', 'shape': 'timeseries', 'rows': rows[3:], "
                    "'schema_fields': [], 'lineage': ['anomaly:anomaly_evi_demo_full']}] }\n"
                ),
            ),
            request_state=request_state,
        ))


def test_code_interpreter_returns_structured_repair_for_invalid_data_view_contract():
    settings = get_settings()
    request_state = build_request_state(ChatRequest(message="计算并绘图"), settings)
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        _build_full_evidence_payload(),
        _ToolSpec(),
    )
    with pytest.raises(StructuredToolError) as caught:
        asyncio.run(CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="publish a typed view",
                code=(
                    "result = {'summary': 'view', 'metrics': {}, 'details': {}, 'data_views': [{"
                    "'view_id': 'bad', 'name': 'Bad', 'shape': 'timeseries', 'rows': rows, "
                    "'schema_fields': ['timestamp', 'value'], 'lineage': {'source': 'rows'}}]}\n"
                ),
            ),
            request_state=request_state,
        ))
    assert caught.value.error_type == "analysis_data_view_invalid"
    assert caught.value.validation_failure["repair_contract"]["data_view_contract"]


def test_code_interpreter_rejects_unknown_data_view_lineage_before_artifact_registration():
    settings = get_settings()
    request_state = build_request_state(ChatRequest(message="计算并绘图"), settings)
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        _build_full_evidence_payload(),
        _ToolSpec(),
    )

    with pytest.raises(StructuredToolError) as caught:
        asyncio.run(CodeInterpreterTool().execute(
            CodeInterpreterInput(
                analysis_goal="publish a grounded typed view",
                code=(
                    "result = {'summary': 'view', 'metrics': {}, 'details': {}, 'data_views': [{"
                    "'view_id': 'returns', 'name': 'Returns', 'shape': 'timeseries', 'rows': rows, "
                    "'lineage': ['evidence:invented_alias']}] }\n"
                ),
            ),
            request_state=request_state,
        ))

    failure = caught.value.validation_failure
    assert caught.value.error_type == "analysis_data_view_invalid"
    assert failure["retry_policy"]["required_action"] == "code_interpreter"
    assert failure["repair_contract"]["allowed_lineage_refs"] == ["evidence:evi_demo_full"]
    assert "evidence:invented_alias" in str(caught.value)


def test_code_interpreter_rejects_result_missing_expected_detail_field():
    settings = get_settings()
    request = ChatRequest(message="用 code interpreter 分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    with pytest.raises(StructuredToolError, match="details\\.findings"):
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
