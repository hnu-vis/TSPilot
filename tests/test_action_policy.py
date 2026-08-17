from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.completion import apply_previous_observation_assessment, evaluate_goal_completion, normalize_todo_for_completion
from runtime.react_loop import ReActLoop
from runtime.action_policy import runtime_action_constraints, validate_action
from runtime.output_selection import select_outputs_for_action
from runtime.request_state import apply_observation, apply_observation_async, enrich_observation_payload
from schemas.agent_turn import PreviousObservationAssessment, ReActTurn
from schemas.action_output import ActionOutput
from schemas.database import DatabaseEvidence
from schemas.state import ConversationStateModel
from schemas.database_context import DatabaseContext
from schemas.state import RequestStateModel
from schemas.task_contract import TaskContract
from schemas.timeseries import AnomalyResult, ForecastPlan, ForecastResult, TimeSeriesPoint
from schemas.tool import ToolCall, ToolObservation
from tools.todowrite import TodoWriteInput, TodoWriteTool


class _EvidenceSpec:
    result_target = "evidence"
    produces_terminal_payload = False


class _AnalysisSpec:
    result_target = "analysis"
    produces_terminal_payload = False


class _NoTargetSpec:
    result_target = "none"
    produces_terminal_payload = False


def test_equivalent_structured_failures_are_counted_and_exhaust_into_terminate():
    request_state = RequestStateModel(
        request_id="req-repair-exhaustion",
        message="分析趋势",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        requested_capabilities=["query", "analysis"],
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_demo",
            result_type="timeseries",
            database="demo",
            summary="Loaded rows.",
            data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        ),
    )
    payload = {
        "error_type": "analysis_transparency_missing",
        "validation_failure": {
            "scope": "artifact_output",
            "capability": "analysis",
            "tool": "code_interpreter",
            "error_code": "analysis_transparency_missing",
            "repair_contract": {
                "mode": "analysis_artifact_repair",
                "required_details_fields": ["raw_metrics", "adjusted_metrics"],
            },
            "retry_policy": {
                "required_action": "code_interpreter",
                "max_equivalent_retries": 2,
                "terminal_after_exhausted": True,
            },
        },
    }

    for expected_count in (1, 2, 3):
        safe = apply_observation(
            request_state,
            ToolObservation(
                tool_name="code_interpreter",
                success=False,
                summary="analysis contract failed",
                payload=payload,
                error="analysis contract failed",
            ),
            payload,
            _NoTargetSpec(),
        )
        assert safe.payload["repeated_failure_count"] == expected_count

    constraints = runtime_action_constraints(request_state)
    assert constraints["required_actions"][0]["action"] == "terminate"
    assert "code_interpreter" in constraints["prohibited_actions"]
    terminal_input = constraints["required_actions"][0]["input_guidance"]
    allowed, reason = validate_action(request_state, "terminate", terminal_input)
    assert allowed is True
    assert reason is None


def _assess_previous_complete(request_state: RequestStateModel, reason: str = "Previous observation satisfies active todo."):
    return apply_previous_observation_assessment(
        request_state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason=reason,
        ),
    )


def test_observation_payload_removes_duplicate_envelope_summary():
    request_state = RequestStateModel(
        request_id="req-observation-dedupe",
        message="查询",
        status="running",
    )
    observation = ToolObservation(
        tool_name="sql_query",
        success=True,
        summary="Loaded 1 row.",
        payload={"summary": "Loaded 1 row.", "evidence_id": "evi_demo"},
    )

    safe_observation = enrich_observation_payload(
        request_state,
        observation,
        {"summary": "Loaded 1 row.", "evidence_id": "evi_demo"},
        _NoTargetSpec(),
    )

    assert safe_observation.summary == "Loaded 1 row."
    assert "summary" not in safe_observation.payload
    assert safe_observation.payload["evidence_id"] == "evi_demo"


def test_observation_payload_keeps_distinct_business_summary():
    request_state = RequestStateModel(
        request_id="req-observation-keep-summary",
        message="分析",
        status="running",
    )
    observation = ToolObservation(
        tool_name="code_interpreter",
        success=True,
        summary="Tool completed.",
        payload={"summary": "业务分析结论。", "analysis_id": "ana_demo"},
    )

    safe_observation = enrich_observation_payload(
        request_state,
        observation,
        {"summary": "业务分析结论。", "analysis_id": "ana_demo"},
        _NoTargetSpec(),
    )

    assert safe_observation.summary == "Tool completed."
    assert safe_observation.payload["summary"] == "业务分析结论。"


def test_terminate_requires_forecast_tool_output_when_forecast_is_requested():
    request_state = RequestStateModel(
        request_id="req-forecast-required",
        message="查询并预测 Bitcoin USD",
        status="running",
        database_context=DatabaseContext(database_id="influxdb2-bitcoin-sample", database_type="influxdb"),
        requested_capabilities=["query", "analysis", "forecast"],
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_btc",
            result_type="timeseries",
            database="influxdb2-bitcoin-sample",
            summary="Loaded Bitcoin rows.",
            data={
                "points": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 1.0},
                    {"timestamp": "2023-01-01T01:00:00Z", "value": 2.0},
                ]
            },
        ),
        latest_analysis_id="ana_stats",
    )

    allowed, reason = validate_action(request_state, "terminate")

    assert allowed is False
    assert reason is not None
    assert "Required specialized tool output is missing" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["forecast"]


def test_downstream_analysis_constraints_follow_active_analysis_todo_only():
    request_state = RequestStateModel(
        request_id="req-downstream-analysis-active-todo",
        message="查询数据，分析趋势，检测异常，预测接下来 6 个点，最后给出结论。",
        status="running",
        requested_capabilities=["query", "analysis", "anomaly", "forecast"],
        todo_list=[
            {"content": "查询数据", "task_type": "query", "status": "completed", "priority": 1},
            {"content": "分析趋势", "task_type": "code_interpreter", "status": "in_progress", "priority": 2},
            {"content": "检测异常", "task_type": "anomaly", "status": "pending", "priority": 3},
            {"content": "预测接下来 6 个点", "task_type": "forecast", "status": "pending", "priority": 4},
            {"content": "给出结论", "task_type": "answer", "status": "pending", "priority": 5},
        ],
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_energy",
            result_type="timeseries",
            database="demo",
            summary="Loaded rows.",
            data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
            diagnostics={
                "task_coverage": {
                    "missing": ["分析趋势", "检测异常", "预测接下来 6 个点", "给出结论"],
                    "query_task_contract": {
                        "downstream_action": "code_interpreter",
                        "preferred_evidence_shape": "raw_series",
                    },
                }
            },
        ),
    )

    constraints = runtime_action_constraints(request_state)

    required = constraints["required_actions"][0]
    assert required["action"] == "code_interpreter"
    guidance = required["input_guidance"]
    assert guidance["analysis_request"]["required_outputs"] == ["分析趋势"]
    assert guidance["analysis_request"]["missing"] == ["分析趋势"]


def test_completed_anomaly_does_not_suppress_declared_downstream_analysis():
    request_state = RequestStateModel(
        request_id="req-anomaly-before-analysis",
        message="排除异常后计算最优交易",
        status="running",
        requested_capabilities=["query", "anomaly", "analysis"],
        task_contract=TaskContract.model_validate({
            "source": "llm", "goal": "排除异常后计算最优交易",
            "required_outputs": [{
                "id": "optimal_trade", "description": "排除异常后的最优单次交易",
                "output_type": "analysis", "evidence_kind": "analysis",
            }],
        }),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_market", result_type="timeseries", database="demo",
            summary="Loaded rows.", data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
            diagnostics={"task_coverage": {
                "missing": ["最优交易"],
                "query_task_contract": {
                    "downstream_action": "code_interpreter",
                    "preferred_evidence_shape": "raw_series",
                },
            }},
        ),
        latest_anomaly=AnomalyResult(
            anomaly_id="ano_market", detector_name="test", status="succeeded",
            summary="No anomalies.", anomaly_points=[], diagnostics={"resolved_evidence_id": "evi_market"},
        ),
    )

    constraints = runtime_action_constraints(request_state)

    assert constraints["required_actions"][0]["action"] == "code_interpreter"


def test_successful_but_contract_incomplete_sql_is_repaired_before_anomaly():
    evidence = DatabaseEvidence(
        evidence_id="evi_limited", result_type="timeseries", database="demo",
        query="SELECT timestamp, value FROM prices LIMIT 48", summary="Loaded 48 rows.",
        data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
        diagnostics={"task_coverage": {
            "runtime_requires_followup": True,
            "runtime_missing": ["raw LIMIT truncates the analysis interval"],
            "next_action_hint": "Query the complete raw interval without LIMIT.",
        }},
    )
    request_state = RequestStateModel(
        request_id="req-repair-before-anomaly", message="排除异常后计算最优交易", status="running",
        requested_capabilities=["query", "anomaly", "analysis"],
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        observations=[ToolObservation(
            tool_name="sql_query", success=True, summary="Loaded limited rows.",
            payload={"evidence_id": evidence.evidence_id},
        )],
    )

    constraints = runtime_action_constraints(request_state)

    required = constraints["required_actions"][0]
    assert required["action"] == "sql_query"
    assert required["input_guidance"]["constraints"]["evidence_shape"] == "raw_timeseries"
    assert "LIMIT" in required["input_guidance"]["message"]


def test_visual_verification_must_be_newer_than_analytical_sources():
    request_state = RequestStateModel(
        request_id="req-current-visual-verification",
        message="展示最高点",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        requested_capabilities=["query", "visualization"],
        task_contract=TaskContract.model_validate({
            "source": "llm", "goal": "展示最高点",
            "required_outputs": [{
                "id": "visual_verification", "description": "完整区间上的最高点可视验证",
                "output_type": "visualization", "evidence_kind": "visualization",
            }],
        }),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_market", result_type="timeseries", database="demo",
            summary="Loaded rows.", data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
        ),
    )
    request_state.visualizations = [SimpleNamespace(visualization_id="viz_old", source_refs=["evidence:evi_market"])]
    request_state.action_outputs = [
        ActionOutput(tool_name="visualization", success=True, content="old", observations={}, meta={"iteration": 1}),
        ActionOutput(tool_name="sql_query", success=True, content="new context", observations={}, meta={"iteration": 2}),
    ]

    stale = evaluate_goal_completion(request_state)
    assert stale.can_answer is False
    assert stale.missing_evidence == ["visualization"]

    request_state.action_outputs.append(
        ActionOutput(tool_name="visualization", success=True, content="current", observations={}, meta={"iteration": 3})
    )
    current = evaluate_goal_completion(request_state)
    assert current.can_answer is True


def test_output_selector_filters_task_contract_by_action_capability():
    request_state = RequestStateModel(
        request_id="req-output-selector-contract",
        message="查询数据，计算指标，检测异常，预测并总结。",
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "multi step",
                "required_outputs": [
                    {"id": "series", "description": "原始序列", "evidence_kind": "database"},
                    {"id": "trend", "description": "趋势指标", "evidence_kind": "analysis"},
                    {"id": "anomalies", "description": "异常点", "evidence_kind": "anomaly"},
                    {"id": "future", "description": "未来 6 个点", "evidence_kind": "forecast"},
                    {"id": "conclusion", "description": "综合结论", "evidence_kind": "answer"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_series",
            result_type="timeseries",
            database="demo",
            summary="Loaded rows.",
            data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        ),
    )

    code_outputs = select_outputs_for_action(request_state, "code_interpreter")
    anomaly_outputs = select_outputs_for_action(request_state, "anomaly")
    forecast_outputs = select_outputs_for_action(request_state, "forecast")

    assert code_outputs["required_outputs"] == ["趋势指标"]
    assert anomaly_outputs["required_outputs"] == ["异常点"]
    assert forecast_outputs["required_outputs"] == ["未来 6 个点"]


def test_output_selector_active_todo_overrides_global_missing_outputs():
    request_state = RequestStateModel(
        request_id="req-output-selector-active-todo",
        message="查询数据，分析趋势，检测异常，预测并总结。",
        status="running",
        todo_list=[
            {"content": "查询数据", "task_type": "query", "status": "completed", "priority": 1},
            {"content": "分析趋势", "task_type": "code_interpreter", "status": "in_progress", "priority": 2},
            {"content": "检测异常", "task_type": "anomaly", "status": "pending", "priority": 3},
            {"content": "预测接下来 6 个点", "task_type": "forecast", "status": "pending", "priority": 4},
            {"content": "给出结论", "task_type": "answer", "status": "pending", "priority": 5},
        ],
    )

    selected = select_outputs_for_action(
        request_state,
        "code_interpreter",
        fallback_outputs=["分析趋势", "检测异常", "预测接下来 6 个点", "给出结论"],
    )

    assert selected["required_outputs"] == ["分析趋势"]
    assert selected["missing"] == ["分析趋势"]


def test_terminate_rejects_incomplete_forecast_artifact_when_forecast_is_requested():
    forecast = ForecastResult(
        forecast_id="forecast_btc",
        model_name="linear_regression",
        horizon=96,
        status="requires_rolling",
        forecast_plan=ForecastPlan(
            mode="requires_rolling",
            horizon_source="duration_from_user",
            requested_steps=96,
            resolved_steps=96,
            sampling_interval_seconds=900,
            forecast_duration_seconds=86400,
            max_direct_steps=48,
            recommended_chunk_steps=48,
        ),
        forecast_points=[],
    )
    request_state = RequestStateModel(
        request_id="req-incomplete-forecast",
        message="预测 Bitcoin USD 接下来 24 小时走势",
        status="running",
        database_context=DatabaseContext(database_id="influxdb2-bitcoin-sample", database_type="influxdb"),
        requested_capabilities=["query", "forecast"],
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_btc",
            result_type="timeseries",
            database="influxdb2-bitcoin-sample",
            summary="Loaded Bitcoin rows.",
            data={
                "points": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 1.0},
                    {"timestamp": "2023-01-01T01:00:00Z", "value": 2.0},
                ]
            },
        ),
        latest_forecast=forecast,
        forecast_artifacts={forecast.forecast_id: forecast},
    )

    allowed, reason = validate_action(request_state, "terminate")

    assert allowed is False
    assert reason is not None
    assert "Required specialized tool output is missing" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["forecast"]


def test_terminate_requires_anomaly_tool_when_contract_requires_anomaly_evidence():
    request_state = RequestStateModel(
        request_id="req-anomaly-contract-required",
        message="检测异常点。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        requested_capabilities=["query", "anomaly"],
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "检测异常点",
                "required_outputs": [
                    {"id": "series", "description": "时序数据", "evidence_kind": "database"},
                    {"id": "anomaly_points", "description": "异常点", "evidence_kind": "anomaly"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_series",
            result_type="timeseries",
            database="demo",
            summary="Loaded series.",
            data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["series"],
                "missing": ["anomaly_points"],
                "can_answer": False,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "未发现异常。"})

    assert allowed is False
    assert reason is not None
    assert "Required specialized tool output is missing" in reason


def test_open_vocabulary_anomaly_detection_contract_requires_anomaly_action():
    request_state = RequestStateModel(
        request_id="req-open-anomaly-contract",
        message="排除异常后分析",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        task_contract=TaskContract.model_validate({
            "source": "llm",
            "goal": "排除异常后分析",
            "required_outputs": [
                {"id": "market_evidence", "description": "原始时序", "evidence_kind": "time_series"},
                {"id": "anomaly_set", "description": "识别并排除异常点", "evidence_kind": "anomaly_detection"},
                {"id": "metric", "description": "派生指标", "evidence_kind": "custom_analysis"},
            ],
        }),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_series", result_type="timeseries", database="demo",
            summary="Loaded series.", data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
        ),
    )

    constraints = runtime_action_constraints(request_state)
    assert constraints["required_actions"][0]["action"] == "anomaly"


def test_llm_contract_describing_excluded_anomaly_points_requires_authoritative_anomaly():
    request_state = RequestStateModel(
        request_id="req-described-anomaly-contract",
        message="排除异常点后计算最优交易并绘图",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        requested_capabilities=["query"],
        task_contract=TaskContract.model_validate({
            "source": "llm",
            "goal": "计算最优交易并绘图",
            "required_outputs": [
                {
                    "id": "cleaned_price_series",
                    "description": "清洗后的行情及被排除异常点所需证据",
                    "output_type": "data_view",
                    "evidence_kind": "database",
                },
                {
                    "id": "trade_analysis",
                    "description": "最优交易",
                    "output_type": "analysis",
                    "evidence_kind": "analysis",
                },
            ],
        }),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_series",
            result_type="timeseries",
            database="demo",
            summary="Loaded series.",
            data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
        ),
    )

    constraints = runtime_action_constraints(request_state)
    allowed, reason = validate_action(request_state, "terminate")

    assert constraints["required_actions"][0]["action"] == "anomaly"
    assert allowed is False
    assert reason is not None
    assert "anomaly" in reason


def test_terminate_requires_authoritative_anomaly_artifact_for_outlier_contract():
    request_state = RequestStateModel(
        request_id="req-outlier-analysis-contract",
        message="计算指标，如果发现明显异常值，请说明规则、剔除行和剔除前后指标。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        requested_capabilities=["query", "analysis", "anomaly"],
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算指标并说明异常值处理",
                "required_outputs": [
                    {"id": "series", "description": "时序数据", "evidence_kind": "database"},
                    {"id": "pct_change", "description": "涨跌幅", "evidence_kind": "analysis"},
                    {"id": "outlier_treatment", "description": "异常规则、剔除行、剔除前后指标", "evidence_kind": "analysis"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_series",
            result_type="timeseries",
            database="demo",
            summary="Loaded series.",
            data={
                "rows": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 1000001.0},
                    {"timestamp": "2023-01-02T00:00:00Z", "value": 10.0},
                    {"timestamp": "2023-01-03T00:00:00Z", "value": 12.0},
                ]
            },
        ),
        latest_analysis_id="ana_outlier",
        analysis_artifacts={
            "ana_outlier": {
                "analysis_id": "ana_outlier",
                "analysis_goal": "计算异常值处理后的指标",
                "code_type": "code_interpreter_v2",
                "code_hash": "sha256:outlier",
                "input_evidence_id": "evi_series",
                "input_row_count": 3,
                "status": "succeeded",
                "summary": "Computed outlier treatment.",
                "computed_insights": [{
                    "insight_key": "pct_change",
                    "value": 20.0,
                    "calculation_trace": "Calculated from the supplied evidence.",
                }],
                "derived_evidence": [],
                "diagnostics": {},
            }
        },
        completion_state={
            "latest_gap_assessment": {
                "covered": ["series", "pct_change", "outlier_treatment"],
                "missing": [],
                "can_answer": True,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "基于 code 结果回答。"})

    assert allowed is False
    assert reason is not None
    assert "anomaly" in reason


def test_terminate_blocked_when_latest_gap_assessment_has_missing_outputs():
    request_state = RequestStateModel(
        request_id="req-gap-missing",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_partial",
            result_type="table",
            database="demo",
            summary="Loaded extrema only.",
            data={"rows": [{"min_value": 1.0, "max_value": 2.0}]},
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["highest", "lowest"],
                "missing": ["start_value", "end_value", "change_pct"],
                "can_answer": False,
                "next_action_reason": "Query boundary values for the same time range.",
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "还缺少首末值。"})

    assert allowed is False
    assert reason is not None
    assert "not fully covered" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == [
        "start_value",
        "end_value",
        "change_pct",
    ]


def test_terminate_blocked_when_task_contract_outputs_not_covered():
    request_state = RequestStateModel(
        request_id="req-contract-missing",
        message="返回总数和最早 3 条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "返回总数和最早 3 条",
                "required_outputs": [
                    {"id": "total_count", "description": "总记录数"},
                    {"id": "earliest_rows", "description": "最早 3 条记录"},
                ],
            }
        ),
        latest_database_evidence={
            "evidence_id": "evi_demo",
            "result_type": "table",
            "database": "demo",
            "query_language": "flux",
            "query": "from(...) |> count()",
            "summary": "Loaded count.",
            "data": {"rows": [{"count": 10}]},
            "columns": ["count"],
            "metadata": {},
            "diagnostics": {},
        },
        completion_state={
            "latest_gap_assessment": {
                "covered": ["total_count"],
                "missing": ["earliest_rows"],
                "can_answer": False,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "总数为 10。"})

    assert allowed is False
    assert "Task contract required outputs" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["earliest_rows"]


def test_terminate_blocks_derived_contract_outputs_without_code_analysis():
    request_state = RequestStateModel(
        request_id="req-derived-analysis-required",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算边界值和涨跌幅",
                "required_outputs": [
                    {"id": "start_value", "description": "起始值", "evidence_kind": "database"},
                    {"id": "end_value", "description": "结束值", "evidence_kind": "database"},
                    {"id": "pct_change", "description": "涨跌幅", "evidence_kind": "derived"},
                    {"id": "high_low", "description": "最高最低", "evidence_kind": "derived"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_prices",
            result_type="timeseries",
            database="demo",
            summary="Loaded requested rows.",
            data={
                "rows": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
                    {"timestamp": "2023-01-02T00:00:00Z", "value": 12.0},
                ]
            },
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["start_value", "end_value", "pct_change", "high_low"],
                "missing": [],
                "can_answer": True,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "涨跌幅 20%。"})

    assert allowed is False
    assert reason is not None
    assert "Required specialized tool output is missing" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["analysis"]


def test_terminate_blocks_analysis_capability_without_task_contract_until_code_runs():
    request_state = RequestStateModel(
        request_id="req-analysis-capability-without-contract",
        message="计算涨跌幅和最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        requested_capabilities=["query", "analysis"],
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_prices",
            result_type="timeseries",
            database="demo",
            summary="Loaded requested rows.",
            data={
                "rows": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
                    {"timestamp": "2023-01-02T00:00:00Z", "value": 12.0},
                ]
            },
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["database_evidence"],
                "missing": [],
                "can_answer": True,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "涨跌幅 20%。"})

    assert allowed is False
    assert reason is not None
    assert "Required specialized tool output is missing" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["analysis"]


def test_terminate_cannot_mark_required_analysis_unavailable_after_code_failure():
    request_state = RequestStateModel(
        request_id="req-derived-analysis-unavailable-bypass",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算边界值和涨跌幅",
                "required_outputs": [
                    {"id": "start_value", "description": "起始值", "evidence_kind": "database"},
                    {"id": "end_value", "description": "结束值", "evidence_kind": "database"},
                    {"id": "pct_change", "description": "涨跌幅", "evidence_kind": "analysis"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_prices",
            result_type="timeseries",
            database="demo",
            summary="Loaded requested rows.",
            data={
                "rows": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
                    {"timestamp": "2023-01-02T00:00:00Z", "value": 12.0},
                ]
            },
        ),
        observations=[
            ToolObservation(
                tool_name="code_interpreter",
                success=False,
                summary="analysis failed",
                payload={},
                error="sandbox failed",
            )
        ],
        completion_state={
            "latest_gap_assessment": {
                "covered": ["start_value", "end_value"],
                "missing": ["pct_change"],
                "can_answer": False,
            }
        },
    )

    allowed, reason = validate_action(
        request_state,
        "terminate",
        {
            "direct_answer": "涨跌幅不可用。",
            "unavailable_outputs": ["pct_change"],
            "unavailable_reason": "code_interpreter failed.",
        },
    )

    assert allowed is False
    assert reason is not None
    assert "Required specialized tool output is missing" in reason
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["analysis"]


def test_terminate_allows_derived_contract_outputs_with_code_analysis():
    request_state = RequestStateModel(
        request_id="req-derived-analysis-present",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算边界值和涨跌幅",
                "required_outputs": [
                    {"id": "start_value", "description": "起始值", "evidence_kind": "database"},
                    {"id": "end_value", "description": "结束值", "evidence_kind": "database"},
                    {"id": "pct_change", "description": "涨跌幅", "evidence_kind": "derived"},
                    {"id": "high_low", "description": "最高最低", "evidence_kind": "derived"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_prices",
            result_type="timeseries",
            database="demo",
            summary="Loaded requested rows.",
            data={
                "rows": [
                    {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
                    {"timestamp": "2023-01-02T00:00:00Z", "value": 12.0},
                ]
            },
        ),
        latest_analysis_id="ana_price_metrics",
        analysis_artifacts={
            "ana_price_metrics": {
                "analysis_id": "ana_price_metrics",
                "analysis_goal": "计算价格指标",
                "code_type": "code_interpreter_v2",
                "code_hash": "sha256:demo",
                "input_evidence_id": "evi_prices",
                "input_row_count": 2,
                "status": "succeeded",
                "summary": "Computed price metrics in code.",
                "computed_insights": [{
                    "insight_key": "pct_change",
                    "value": 20.0,
                    "calculation_trace": "(12 - 10) / 10 * 100",
                }],
                "derived_evidence": [],
                "diagnostics": {},
            }
        },
        completion_state={
            "latest_gap_assessment": {
                "covered": ["start_value", "end_value", "pct_change", "high_low"],
                "missing": [],
                "can_answer": True,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "涨跌幅 20%。"})

    assert allowed is True
    assert reason is None


def test_terminate_allows_answer_when_global_gap_covers_stale_non_answer_todos():
    request_state = RequestStateModel(
        request_id="req-derived-analysis-covered-stale-todos",
        message="查询 Bitcoin USD 2023-01-04 到 2023-02-03，计算起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询 Bitcoin USD 数据", "task_type": "query", "status": "completed", "priority": 1},
            {"content": "整理边界值", "task_type": "answer", "status": "in_progress", "priority": 2},
            {"content": "计算涨跌幅", "task_type": "code_interpreter", "status": "pending", "priority": 3},
        ],
        plan_current_step=2,
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算边界值和涨跌幅",
                "required_outputs": [
                    {"id": "query_text", "description": "查询对象和时间范围", "evidence_kind": "database"},
                    {"id": "row_count", "description": "数据点数量", "evidence_kind": "database"},
                    {"id": "start_value", "description": "起始值", "evidence_kind": "database"},
                    {"id": "end_value", "description": "结束值", "evidence_kind": "database"},
                    {"id": "pct_change", "description": "涨跌幅", "evidence_kind": "derived"},
                    {"id": "max_value", "description": "最高值", "evidence_kind": "derived"},
                    {"id": "min_value", "description": "最低值", "evidence_kind": "derived"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_prices",
            result_type="timeseries",
            database="demo",
            summary="Loaded Bitcoin rows.",
            data={
                "rows": [
                    {"timestamp": "2023-01-04T00:00:00Z", "value": 16858.2362},
                    {"timestamp": "2023-02-03T00:00:00Z", "value": 23428.6802},
                ]
            },
        ),
        latest_analysis_id="ana_price_metrics",
        analysis_artifacts={
            "ana_price_metrics": {
                "analysis_id": "ana_price_metrics",
                "analysis_goal": "计算价格指标",
                "code_type": "code_interpreter_v2",
                "code_hash": "sha256:demo",
                "input_evidence_id": "evi_prices",
                "input_row_count": 2,
                "status": "succeeded",
                "summary": "Computed price metrics in code.",
                "computed_insights": [{
                    "insight_key": "price_metrics",
                    "value": {"pct_change": 38.97468229802119, "max_value": 24104.6943, "min_value": 16702.3044},
                    "calculation_trace": "Calculated return and extrema from the complete interval.",
                }],
                "derived_evidence": [],
                "diagnostics": {},
            }
        },
        completion_state={
            "latest_gap_assessment": {
                "covered": [
                    "query_text",
                    "row_count",
                    "start_value",
                    "end_value",
                    "pct_change",
                    "max_value",
                    "min_value",
                ],
                "missing": [],
                "can_answer": True,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "基于 code 结果回答。"})

    assert allowed is True
    assert reason is None


def test_terminate_still_blocks_stale_non_answer_todos_without_global_gap_coverage():
    request_state = RequestStateModel(
        request_id="req-derived-analysis-stale-todos-not-covered",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询数据", "task_type": "query", "status": "completed", "priority": 1},
            {"content": "整理答案", "task_type": "answer", "status": "in_progress", "priority": 2},
            {"content": "计算涨跌幅", "task_type": "code_interpreter", "status": "pending", "priority": 3},
        ],
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_prices",
            result_type="timeseries",
            database="demo",
            summary="Loaded rows.",
            data={"rows": [{"timestamp": "2023-01-01T00:00:00Z", "value": 10.0}]},
        ),
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "证据不足。"})

    assert allowed is False
    assert reason == "Final answer is blocked because non-answer todo steps are still incomplete."


def test_terminate_allows_unavailable_outputs_from_empty_database_evidence_with_active_query_todo():
    request_state = RequestStateModel(
        request_id="req-empty-derived-unavailable",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询该时间区间的原始价格序列", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "计算涨跌幅", "task_type": "code_interpreter", "status": "pending", "priority": 2},
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 3},
        ],
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算边界值和涨跌幅",
                "required_outputs": [
                    {"id": "start_value", "description": "起始值", "evidence_kind": "database"},
                    {"id": "end_value", "description": "结束值", "evidence_kind": "database"},
                    {"id": "pct_change", "description": "涨跌幅", "evidence_kind": "analysis"},
                    {"id": "high_low", "description": "最高最低", "evidence_kind": "analysis"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_empty",
            result_type="timeseries",
            database="demo",
            summary="The query completed but returned no rows.",
            data={"rows": [], "points": []},
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["empty_database_result"],
                "missing": ["start_value", "end_value", "pct_change", "high_low"],
                "can_answer": False,
                "next_action_reason": "The selected range contains no rows, so requested metrics are unavailable.",
            }
        },
    )

    allowed, reason = validate_action(
        request_state,
        "terminate",
        {
            "direct_answer": "该区间没有数据，无法计算起始值、结束值、涨跌幅、最高最低。",
            "unavailable_outputs": ["start_value", "end_value", "pct_change", "high_low"],
            "unavailable_reason": "The database query returned no rows for the requested time range.",
        },
    )

    assert allowed is True
    assert reason is None
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == [
        "start_value",
        "end_value",
        "pct_change",
        "high_low",
    ]


def test_terminate_allows_human_labels_for_unavailable_outputs_when_database_evidence_is_empty():
    request_state = RequestStateModel(
        request_id="req-empty-contract-human-labels",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "计算边界值和涨跌幅",
                "required_outputs": [
                    {"id": "r1", "description": "指定时间范围内的可用数据", "evidence_kind": "database"},
                    {"id": "r2", "description": "起始值、结束值、最高值、最低值", "evidence_kind": "database_or_analysis"},
                    {"id": "r3", "description": "涨跌幅", "evidence_kind": "analysis"},
                    {"id": "r4", "description": "若查不到数据，明确说明原因", "evidence_kind": "database"},
                ],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_empty",
            result_type="table",
            database="demo",
            summary="The query completed but returned no rows.",
            data={"rows": []},
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["r1", "r4"],
                "missing": ["r2", "r3"],
                "can_answer": False,
                "next_action_reason": "The selected range contains no rows, so requested metrics are unavailable.",
            }
        },
    )

    allowed, reason = validate_action(
        request_state,
        "terminate",
        {
            "direct_answer": "该区间没有数据，无法计算起始值、结束值、涨跌幅、最高最低。",
            "unavailable_outputs": ["起始值", "结束值", "涨跌幅", "最高值", "最低值"],
            "unavailable_reason": "查询返回 0 行，缺少可用于计算的原始价格序列。",
        },
    )

    assert allowed is True
    assert reason is None
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["r2", "r3"]


def test_terminate_allows_explicit_unavailable_task_contract_outputs():
    request_state = RequestStateModel(
        request_id="req-contract-unavailable",
        message="返回总数和最早 3 条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "返回总数和最早 3 条",
                "required_outputs": [
                    {"id": "total_count", "description": "总记录数"},
                    {"id": "earliest_rows", "description": "最早 3 条记录"},
                ],
            }
        ),
        latest_database_evidence={
            "evidence_id": "evi_demo",
            "result_type": "table",
            "database": "demo",
            "query_language": "flux",
            "query": "from(...) |> count()",
            "summary": "Loaded count.",
            "data": {"rows": [{"count": 10}]},
            "columns": ["count"],
            "metadata": {},
            "diagnostics": {},
        },
        completion_state={
            "latest_gap_assessment": {
                "covered": ["total_count"],
                "missing": ["earliest_rows"],
                "can_answer": False,
            }
        },
    )

    allowed, reason = validate_action(
        request_state,
        "terminate",
        {
            "direct_answer": "总数为 10，但无法取得最早 3 条。",
            "unavailable_outputs": ["earliest_rows"],
            "unavailable_reason": "当前数据源只返回聚合结果，原始行不可用。",
        },
    )

    assert allowed is True
    assert reason is None


def test_terminate_allowed_when_latest_gap_assessment_can_answer():
    request_state = RequestStateModel(
        request_id="req-gap-covered",
        message="查询起始值、结束值、涨跌幅、最高最低。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_stats",
            result_type="table",
            database="demo",
            summary="Loaded requested statistics.",
            data={
                "rows": [
                    {
                        "start_value": 1.0,
                        "end_value": 1.2,
                        "change_pct": 20.0,
                        "min_value": 0.9,
                        "max_value": 1.3,
                    }
                ]
            },
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["start_value", "end_value", "change_pct", "min_value", "max_value"],
                "missing": [],
                "can_answer": True,
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "已得到全部统计。"})

    assert allowed is True
    assert reason is None
    assert request_state.completion_state["latest_goal"]["can_answer"] is True


def test_terminate_allows_nonblocking_missing_when_gap_can_answer_true():
    request_state = RequestStateModel(
        request_id="req-gap-answerable-with-caveat",
        message="检测异常并总结趋势。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_series",
            result_type="timeseries",
            database="demo",
            summary="Loaded anomaly-ready series.",
            data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": ["异常检测已完成", "趋势总结可回答"],
                "missing": ["可选异常点逐条解释"],
                "can_answer": True,
                "next_action_reason": "可以回答并说明限制。",
            }
        },
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "未发现异常点。"})

    assert allowed is True
    assert reason is None
    assert request_state.completion_state["latest_goal"]["can_answer"] is True


def test_terminate_allowed_with_explicit_unavailable_gap_outputs():
    request_state = RequestStateModel(
        request_id="req-gap-unavailable",
        message="计算涨跌幅。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_empty",
            result_type="table",
            database="demo",
            summary="No boundary rows returned.",
            data={"rows": []},
        ),
        completion_state={
            "latest_gap_assessment": {
                "covered": [],
                "missing": ["start_value", "end_value", "change_pct"],
                "can_answer": False,
                "next_action_reason": "The selected range has no rows after query repair.",
            }
        },
    )

    allowed, reason = validate_action(
        request_state,
        "terminate",
        {
            "direct_answer": "当前区间没有可用边界值，无法计算涨跌幅。",
            "unavailable_outputs": ["start_value", "end_value", "change_pct"],
            "unavailable_reason": "Repeated grounded queries returned no rows for the selected range.",
        },
    )

    assert allowed is True
    assert reason is None


def test_repeated_failed_action_requires_changed_strategy_reason():
    request_state = RequestStateModel(
        request_id="req-repeat-failure",
        message="查询价格。",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        observations=[
            ToolObservation(tool_name="sql_query", success=False, summary="syntax error", payload={}, error="syntax error"),
            ToolObservation(tool_name="sql_query", success=False, summary="syntax error", payload={}, error="syntax error"),
        ],
    )

    allowed, reason = validate_action(request_state, "sql_query", {"message": "查询价格"})

    assert allowed is False
    assert reason is not None
    assert "failed repeatedly" in reason

    request_state.completion_state["latest_gap_assessment"] = {
        "missing": ["price"],
        "next_action_reason": "Simplify the query and fetch raw rows before aggregation.",
    }

    assert validate_action(request_state, "sql_query", {"message": "查询价格"}) == (True, None)


class _OneTurnAgent:
    async def next_turn(self, request_state, conversation_state):
        return ReActTurn(
            thought="try query first",
            action="sql_query",
            action_input={"message": request_state.message, "database_context": request_state.database_context.model_dump(mode="json")},
        )


class _TerminateAgent:
    async def next_turn(self, request_state, conversation_state):
        return ReActTurn(
            thought="I think I can answer now.",
            action="terminate",
            action_input={"summary_goal": request_state.message, "direct_answer": "partial answer"},
        )


class _UnusedExecutor:
    async def execute(self, *args, **kwargs):
        raise AssertionError("tool executor should not run")


class _TerminalSpec:
    result_target = "presentation"
    produces_terminal_payload = True


class _TerminalExecutor:
    async def execute(self, action_name, action_input, *args, **kwargs):
        assert action_name == "terminate"
        return type(
            "_ExecutionResult",
            (),
            {
                "tool_spec": _TerminalSpec(),
                "observation": ToolObservation(
                    tool_name="terminate",
                    success=True,
                    summary="candidate",
                    payload={"summary": "partial answer", "sections": [], "references": [], "visualizations": []},
                ),
                "full_payload": {
                    "summary": "partial answer",
                    "sections": [],
                    "references": [],
                    "visualizations": [],
                },
            },
        )()


@pytest.mark.asyncio
async def test_gap_assessment_without_active_todo_does_not_block_action_execution():
    request_state = RequestStateModel(
        request_id="req-gap-no-active-todo",
        message="检测异常点。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        max_iterations=1,
    )
    request_state.observations.append(
        ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="Loaded time series.",
            payload={
                "evidence_id": "evi_demo",
                "result_type": "timeseries",
                "data": {"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
            },
        )
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_demo",
        result_type="timeseries",
        database="demo",
        summary="Loaded time series.",
        data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
    )

    class _GapThenActionAgent:
        async def next_turn(self, request_state, conversation_state):
            return ReActTurn(
                thought="上一条查询覆盖输入序列，还需要执行异常检测。",
                previous_observation_assessment=PreviousObservationAssessment(
                    completed_active_todo=True,
                    reason="No active todo; this is a gap assessment only.",
                    evidence_refs=["evidence:evi_demo"],
                    covered=["time_series"],
                    missing=["anomaly_result"],
                    can_answer=False,
                    next_action_reason="Run anomaly on the evidence.",
                ),
                action="anomaly",
                action_input={"database_evidence": "evidence:evi_demo"},
            )

    class _AnomalySpec:
        result_target = "analysis"
        produces_terminal_payload = False

    class _Executor:
        async def execute(self, action_name, action_input, *args, **kwargs):
            assert action_name == "anomaly"
            return type(
                "_ExecutionResult",
                (),
                {
                    "tool_spec": _AnomalySpec(),
                    "observation": ToolObservation(tool_name="anomaly", success=True, summary="ok", payload={}),
                    "full_payload": {
                        "anomaly_id": "anomaly_demo",
                        "detector_name": "unit",
                        "anomaly_points": [],
                        "anomaly_spans": [],
                        "scores": [],
                        "diagnostics": {},
                        "visualizations": [],
                    },
                },
            )()

    loop = ReActLoop(
        data_agent=_GapThenActionAgent(),
        tool_executor=_Executor(),
        settings=type("_Settings", (), {"conversation_log_enabled": False, "resolved_conversation_log_dir": "."})(),
    )

    events = [event async for event in loop._iterate(request_state, ConversationStateModel(conversation_id="conv"))]

    assert any(event.event_type == "observation" and event.payload["tool_name"] == "anomaly" for event in events)
    assert not any(
        event.event_type == "observation"
        and event.payload["tool_name"] == "todo_assessment"
        and event.payload["success"] is False
        for event in events
    )


def test_policy_does_not_force_todowrite_for_complex_initial_request():
    request_state = RequestStateModel(
        request_id="req-policy-complex",
        message=(
            "请判断 Bitcoin USD 在 2023 年 1 月 4 日 23:04:00 UTC 到 "
            "2023 年 2 月 3 日 22:47:00 UTC 这个历史数据集内有没有明显每天或每周"
            "重复的周期性波动。请严格基于数据库数据分析，并展示执行过程。"
        ),
        database_context=DatabaseContext(
            database_id="influxdb2-bitcoin-sample",
            database_type="influxdb",
        ),
        time_range={
            "start": "2023-01-04T23:04:00Z",
            "end": "2023-02-03T22:47:00Z",
        },
        constraints={},
        status="running",
        max_iterations=8,
    )

    allowed, reason = validate_action(request_state, "sql_query")

    assert allowed is True
    assert reason is None


def test_policy_requires_todowrite_for_explicit_multi_deliverable_list():
    request_state = RequestStateModel(
        request_id="req-policy-explicit-list",
        message=(
            "请查询当前数据源中的比特币USD价格数据，并完成以下任务： "
            "1.返回USD价格数据的总记录数； "
            "2.返回按时间升序排列的最早5条原始记录； "
            "3.返回按时间降序排列的最晚5条原始记录； "
            "4.返回整个数据集的最早时间和最晚时间，精确到秒； "
            "5.展示每项结果对应的完整Flux查询语句和实际返回行数。"
        ),
        database_context=DatabaseContext(
            database_id="influxdb2-bitcoin-sample",
            database_type="influxdb",
        ),
        status="running",
    )

    allowed, reason = validate_action(request_state, "sql_query")

    assert allowed is False
    assert "initial todo plan" in (reason or "")
    assert validate_action(request_state, "todowrite") == (True, None)


def test_policy_allows_explicit_initial_todowrite_before_required_evidence():
    request_state = RequestStateModel(
        request_id="req-policy-explicit-todo-first",
        message="请先制定一个todo list：1 查询 bitcoin USD 价格数据；2 计算最大值；3 给出中文结论。",
        database_context=DatabaseContext(
            database_id="influxdb2-bitcoin-sample",
            database_type="influxdb",
        ),
        requested_capabilities=["query", "analysis"],
        status="running",
    )

    allowed, reason = validate_action(request_state, "sql_query", {"message": "查询价格"})

    assert allowed is False
    assert "initial todo plan" in (reason or "")
    assert validate_action(
        request_state,
        "todowrite",
        {
            "todos": ["查询 bitcoin USD 价格数据", "计算最大值", "给出中文结论"],
        },
    ) == (True, None)


def test_policy_rejects_repeated_todowrite_but_allows_next_action():
    request_state = RequestStateModel(
        request_id="req-policy-todo",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "提炼趋势事实", "task_type": "code_interpreter", "status": "in_progress", "priority": 1}
        ],
    )

    allowed, reason = validate_action(request_state, "todowrite")

    assert allowed is False
    assert "already exists" in (reason or "")
    assert validate_action(request_state, "code_interpreter") == (True, None)
    assert validate_action(request_state, "code_interpreter") == (True, None)
    assert validate_action(request_state, "forecast") == (True, None)


def test_policy_does_not_block_model_chosen_action_by_active_todo():
    request_state = RequestStateModel(
        request_id="req-policy-anomaly-step",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "检查异常",
                "task_type": "anomaly",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["anomaly_result"],
            }
        ],
        latest_database_evidence={
            "evidence_id": "evi_demo",
            "result_type": "timeseries",
            "database": "demo",
            "query_language": "flux",
            "query": "demo",
            "summary": "ok",
            "data": {"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
            "columns": ["timestamp", "value"],
            "metadata": {},
            "diagnostics": {},
        },
    )

    assert validate_action(request_state, "sql_query") == (True, None)
    assert validate_action(request_state, "anomaly") == (True, None)


def test_policy_allows_sql_to_fill_missing_query_evidence_for_analysis_step():
    request_state = RequestStateModel(
        request_id="req-policy-anomaly-needs-evidence",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "检查异常",
                "task_type": "anomaly",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["time_series", "anomaly_result"],
            }
        ],
    )

    assert validate_action(request_state, "sql_query") == (True, None)


def test_policy_rejects_unknown_action_only():
    request_state = RequestStateModel(
        request_id="req-policy-unknown",
        message="分析趋势。",
        status="running",
    )

    allowed, reason = validate_action(request_state, "ask_user")

    assert allowed is False
    assert "runtime contract" in (reason or "")


def test_todo_evidence_needed_is_progress_metadata_only():
    todo = normalize_todo_for_completion(
        {
            "content": "查询 appliances_energy_wh 的原始时间序列",
            "task_type": "query",
            "evidence_needed": ["timestamp", "appliances_energy_wh", "time_series"],
        }
    )

    assert "evidence_needed" not in todo


def test_query_todo_clears_internal_schema_need():
    todo = normalize_todo_for_completion(
        {
            "content": "确认数据源与字段并获取基础样本",
            "task_type": "query",
            "evidence_needed": ["schema", "sample_rows"],
        }
    )

    assert "evidence_needed" not in todo


@pytest.mark.asyncio
async def test_todowrite_drops_internal_plan_steps():
    result = await TodoWriteTool().execute(
        TodoWriteInput(
            message="返回总数和最早5条。",
            todos=[
                {
                    "content": "确认数据源与字段并生成可执行Flux查询计划",
                    "task_type": "plan",
                    "status": "in_progress",
                    "priority": 1,
                    "evidence_needed": ["schema"],
                },
                {
                    "content": "查询总记录数",
                    "task_type": "query",
                    "status": "pending",
                    "priority": 2,
                    "evidence_needed": ["count"],
                },
                {
                    "content": "整理最终答案",
                    "task_type": "answer",
                    "status": "pending",
                    "priority": 3,
                },
            ],
        )
    )

    todos = result["todos"]
    assert [todo["content"] for todo in todos] == ["查询总记录数", "整理最终答案"]
    assert todos[0]["status"] == "in_progress"
    assert all(todo["task_type"] != "plan" for todo in todos)
    assert all("evidence_needed" not in todo for todo in todos)


@pytest.mark.asyncio
async def test_todowrite_normalizes_llm_task_type_aliases():
    result = await TodoWriteTool().execute(
        TodoWriteInput(
            message="请先列 todo，然后查询数据，分析趋势，检测异常，预测并总结。",
            todos=[
                {"content": "列出待办事项", "task_type": "list", "status": "in_progress", "priority": 1},
                {"content": "查询 appliances_energy_wh 数据", "task_type": "data", "status": "pending", "priority": 2},
                {"content": "分析趋势", "task_type": "code_interpreter", "status": "pending", "priority": 3},
                {"content": "检测异常", "task_type": "anomaly", "status": "pending", "priority": 4},
                {"content": "预测接下来 6 个点", "task_type": "forecast", "status": "pending", "priority": 5},
                {"content": "给出结论", "task_type": "answer", "status": "pending", "priority": 6},
            ],
        )
    )

    todos = result["todos"]
    assert [todo["content"] for todo in todos] == [
        "查询 appliances_energy_wh 数据",
        "分析趋势",
        "检测异常",
        "预测接下来 6 个点",
        "给出结论",
    ]
    assert [todo["task_type"] for todo in todos] == ["query", "code_interpreter", "anomaly", "forecast", "answer"]
    assert [todo["status"] for todo in todos] == ["in_progress", "pending", "pending", "pending", "pending"]


@pytest.mark.asyncio
async def test_todowrite_accepts_string_todo_items():
    result = await TodoWriteTool().execute(
        TodoWriteInput(
            message="请先制定 todo 后执行。",
            task_contract={
                "required_outputs": [
                    {"evidence_kind": "query"},
                    {"evidence_kind": "analysis"},
                    {"evidence_kind": "answer"},
                ]
            },
            todos=["查询 bitcoin USD 价格数据", "计算最大值", "给出中文结论"],
        )
    )

    todos = result["todos"]
    assert [todo["content"] for todo in todos] == ["查询 bitcoin USD 价格数据", "计算最大值", "给出中文结论"]
    assert [todo["task_type"] for todo in todos] == ["query", "code_interpreter", "answer"]
    assert [todo["status"] for todo in todos] == ["in_progress", "pending", "pending"]


@pytest.mark.asyncio
async def test_todowrite_accepts_description_todo_items():
    result = await TodoWriteTool().execute(
        TodoWriteInput(
            message="请做完整分析。",
            task_contract={
                "required_outputs": [
                    {"evidence_kind": "query"},
                    {"evidence_kind": "analysis"},
                    {"evidence_kind": "answer"},
                ]
            },
            todos=[
                {"id": "1", "description": "查询历史价格"},
                {"id": "2", "description": "计算收益率和波动率"},
                {"id": "3", "description": "给出综合结论"},
            ],
        )
    )

    todos = result["todos"]
    assert [todo["content"] for todo in todos] == ["查询历史价格", "计算收益率和波动率", "给出综合结论"]
    assert [todo["task_type"] for todo in todos] == ["query", "code_interpreter", "answer"]
    assert [todo["status"] for todo in todos] == ["in_progress", "pending", "pending"]


@pytest.mark.asyncio
async def test_todowrite_infers_task_types_without_contract():
    result = await TodoWriteTool().execute(
        TodoWriteInput(
            message="请先写 todo 后执行。",
            todos=[
                "查询比特币USD价格数据",
                "计算最大值、最小值、平均值和最新值",
                "进行异常检测",
                "预测未来5个点",
                "用中文解释关键结论",
            ],
        )
    )

    todos = result["todos"]
    assert [todo["task_type"] for todo in todos] == [
        "query",
        "code_interpreter",
        "anomaly",
        "forecast",
        "answer",
    ]


def test_todo_plan_expands_iteration_budget_for_multistep_workflow():
    request_state = RequestStateModel(
        request_id="req-policy-budget",
        message="先规划再分析趋势、异常和预测。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        max_iterations=10,
    )
    todo_payload = {
        "summary": "plan",
        "todos": [
            {"content": "查询数据", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "分析趋势", "task_type": "code_interpreter", "status": "pending", "priority": 2},
            {"content": "检查异常", "task_type": "anomaly", "status": "pending", "priority": 3},
            {"content": "预测", "task_type": "forecast", "status": "pending", "priority": 4},
            {"content": "回答", "task_type": "answer", "status": "pending", "priority": 5},
        ],
        "current_step": 1,
        "planning_complete": False,
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="todowrite", success=True, summary="plan", payload={}),
        todo_payload,
        type("_PlanSpec", (), {"result_target": "todo"})(),
    )

    assert request_state.max_iterations == 17


@pytest.mark.asyncio
async def test_external_semantic_verdict_no_longer_blocks_todo_progress():
    request_state = RequestStateModel(
        request_id="req-llm-completion-false",
        message="返回总数和最早5条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询总记录数", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "查询最早5条", "task_type": "query", "status": "pending", "priority": 2},
        ],
    )
    payload = {
        "evidence_id": "evi_demo",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": "from(...) |> count()",
        "summary": "Loaded count row.",
        "data": {"rows": [{"count": 5}]},
        "columns": ["count"],
        "metadata": {},
        "diagnostics": {},
    }

    await apply_observation_async(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="rows", payload=payload),
        payload,
        _EvidenceSpec(),
    )

    assert request_state.todo_list[0]["status"] == "in_progress"
    _assess_previous_complete(request_state)
    assert request_state.todo_list[0]["status"] == "completed"
    assert request_state.todo_list[1]["status"] == "in_progress"
    assert request_state.completion_state["latest_step"]["completed"] is True
    assert "completion_verdict" not in request_state.completion_state["latest_step"]


@pytest.mark.asyncio
async def test_successful_query_observation_advances_without_semantic_verdict():
    request_state = RequestStateModel(
        request_id="req-query-observation-first",
        message="分析趋势。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "查询趋势所需的时间序列",
                "task_type": "query",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["time_series"],
            },
            {"content": "分析趋势", "task_type": "code_interpreter", "status": "pending", "priority": 2},
        ],
    )
    payload = {
        "evidence_id": "evi_demo",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "from(...)",
        "summary": "Loaded points.",
        "data": {"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        "columns": ["timestamp", "value"],
        "metadata": {},
        "diagnostics": {},
    }

    await apply_observation_async(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="points", payload=payload),
        payload,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]
    assert request_state.completion_state["latest_step"]["missing_evidence"] == []


@pytest.mark.asyncio
async def test_successful_observation_advances_todo_without_external_verdict():
    request_state = RequestStateModel(
        request_id="req-llm-completion-true",
        message="返回总数和最早5条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询总记录数", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "查询最早5条", "task_type": "query", "status": "pending", "priority": 2},
        ],
    )
    payload = {
        "evidence_id": "evi_demo",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": "from(...) |> count()",
        "summary": "Loaded 1 row.",
        "data": {"rows": [{"count": 10}]},
        "columns": ["count"],
        "metadata": {},
        "diagnostics": {},
    }

    await apply_observation_async(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="count", payload=payload),
        payload,
        _EvidenceSpec(),
    )

    assert request_state.todo_list[0]["status"] == "in_progress"
    _assess_previous_complete(request_state)
    assert request_state.todo_list[0]["status"] == "completed"
    assert request_state.todo_list[1]["status"] == "in_progress"
    assert request_state.completion_state["latest_step"]["completed"] is True


@pytest.mark.asyncio
async def test_runtime_does_not_run_separate_plan_requirement_gate():
    request_state = RequestStateModel(
        request_id="req-no-plan-gate",
        message="请完成以下任务: 1.返回总数; 2.返回最早5条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        max_iterations=1,
    )

    class _Executor:
        async def execute(self, action_name, action_input, *args, **kwargs):
            assert action_name == "sql_query"
            return type(
                "_ExecutionResult",
                (),
                {
                    "tool_spec": _EvidenceSpec(),
                    "observation": ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
                    "full_payload": {
                        "evidence_id": "evi_demo",
                        "result_type": "table",
                        "database": "demo",
                        "query_language": "flux",
                        "query": "from(...)",
                        "summary": "ok",
                        "data": {"rows": [{"count": 1}]},
                        "columns": ["count"],
                        "metadata": {},
                        "diagnostics": {},
                    },
                },
            )()

    loop = ReActLoop(
        data_agent=_OneTurnAgent(),
        tool_executor=_Executor(),
        settings=type("_Settings", (), {"conversation_log_enabled": False, "resolved_conversation_log_dir": "."})(),
    )

    events = [event async for event in loop._iterate(request_state, ConversationStateModel(conversation_id="conv"))]
    observation = next(event for event in events if event.event_type == "observation")

    assert observation.payload["success"] is True
    assert observation.payload["tool_name"] == "sql_query"
    assert "plan_requirement" not in request_state.completion_state


@pytest.mark.asyncio
async def test_terminate_does_not_use_llm_answerability_gate():
    request_state = RequestStateModel(
        request_id="req-no-answerability-gate",
        message="分析趋势。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        latest_database_evidence={
            "evidence_id": "evi_demo",
            "result_type": "timeseries",
            "database": "demo",
            "query_language": "flux",
            "query": "from(...)",
            "summary": "ok",
            "data": {"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
            "columns": ["timestamp", "value"],
            "metadata": {},
            "diagnostics": {},
        },
    )
    loop = ReActLoop(
        data_agent=_OneTurnAgent(),
        tool_executor=_UnusedExecutor(),
        settings=type("_Settings", (), {"conversation_log_enabled": False, "resolved_conversation_log_dir": "."})(),
    )

    allowed, reason = validate_action(request_state, "terminate")

    assert allowed is True
    assert reason is None
    assert "answerability_verdict" not in request_state.completion_state
    assert "semantic_repair_directive" not in request_state.completion_state


@pytest.mark.asyncio
async def test_terminal_candidate_does_not_run_goal_verifier():
    request_state = RequestStateModel(
        request_id="req-no-verifier",
        message="返回最大值。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        max_iterations=1,
        latest_database_evidence={
            "evidence_id": "evi_demo",
            "result_type": "table",
            "database": "demo",
            "query_language": "flux",
            "query": "from(...)",
            "summary": "Loaded rows.",
            "data": {"rows": [{"value": 1.0}]},
            "columns": ["value"],
            "metadata": {},
            "diagnostics": {},
        },
    )
    loop = ReActLoop(
        data_agent=_TerminateAgent(),
        tool_executor=_TerminalExecutor(),
        settings=type("_Settings", (), {"conversation_log_enabled": False, "resolved_conversation_log_dir": "."})(),
    )

    events = [event async for event in loop._iterate(request_state, ConversationStateModel(conversation_id="conv"))]

    assert request_state.final_answer_draft is not None
    assert not any(
        event.event_type == "observation" and event.payload["tool_name"] == "goal_verifier"
        for event in events
    )
    assert "semantic_repair_directive" not in request_state.completion_state
    assert "final_answer" in [event.event_type for event in events]


def test_policy_blocks_terminal_answer_until_database_goal_has_evidence():
    request_state = RequestStateModel(
        request_id="req-policy-format-block",
        message="总共有多少条数据？",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
    )

    allowed, reason = validate_action(request_state, "terminate")

    assert allowed is False
    assert "not complete" in (reason or "")
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["database_evidence"]


def test_policy_allows_terminate_despite_sql_runtime_coverage_hint():
    request_state = RequestStateModel(
        request_id="req-policy-runtime-coverage",
        message="返回最晚一条价格记录。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        latest_database_evidence={
            "evidence_id": "evi_missing_price",
            "result_type": "table",
            "database": "demo",
            "query_language": "flux",
            "query": "from(...) |> last()",
            "summary": "Loaded 1 row.",
            "data": {"rows": [{"timestamp": "2023-01-01T00:00:00Z"}]},
            "columns": ["timestamp"],
            "metadata": {},
            "diagnostics": {
                "task_coverage": {
                    "runtime_missing": [
                        "selected result fields are not present in returned columns: price"
                    ],
                    "runtime_requires_followup": True,
                    "next_action_hint": "Query again and return the price value column.",
                }
            },
        },
        tool_history=[
            ToolCall(
                tool_name="sql_query",
                tool_input={},
                iteration=1,
                reason="load latest price",
            )
        ],
    )

    allowed, reason = validate_action(request_state, "terminate")

    assert allowed is True
    assert reason is None
    latest_goal = request_state.completion_state["latest_goal"]
    assert latest_goal["can_answer"] is True
    assert latest_goal["missing_evidence"] == []


def test_runtime_keeps_query_todo_active_when_sql_missing_projected_value_field():
    request_state = RequestStateModel(
        request_id="req-policy-query-missing-value",
        message="查询价格序列",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询价格序列", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "回答", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    request_state.observations.append(
        ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="Loaded timestamp-only rows.",
            payload={
                "evidence_id": "evi_missing_price",
                "result_type": "table",
                "data": {"rows": [{"timestamp": "2023-01-01T00:00:00Z"}]},
                "columns": ["timestamp"],
                "diagnostics": {
                    "task_coverage": {
                        "runtime_missing": ["selected result fields are not present in returned columns: price"],
                        "runtime_requires_followup": True,
                        "next_action_hint": "Query again and return the price value column.",
                    }
                },
            },
        )
    )

    result = apply_previous_observation_assessment(
        request_state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason="查询已返回行。",
            evidence_refs=["evidence:evi_missing_price"],
            covered=["price"],
            missing=[],
            can_answer=False,
        ),
    )

    assert result.completed is False
    assert "selected result fields" in result.missing_evidence[0]
    assert request_state.todo_list[0]["status"] == "in_progress"


def test_runtime_allows_sql_prerequisite_repair_during_code_interpreter_todo():
    request_state = RequestStateModel(
        request_id="req-policy-code-prereq-sql",
        message="用 code interpreter 计算收益率",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "计算收益率", "task_type": "code_interpreter", "status": "in_progress", "priority": 1},
            {"content": "回答", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    request_state.observations.append(
        ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="Loaded numeric evidence.",
            payload={
                "evidence_id": "evi_price",
                "result_type": "timeseries",
                "data": {"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
                "columns": ["timestamp", "value"],
            },
        )
    )

    result = apply_previous_observation_assessment(
        request_state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason="补齐了 code 所需数值证据。",
            evidence_refs=["evidence:evi_price"],
            covered=["evidence"],
            missing=["metrics"],
            can_answer=False,
        ),
    )

    assert result.completed is False
    assert "prerequisite evidence" in result.reason
    assert request_state.todo_list[0]["status"] == "in_progress"
    assert request_state.completion_state["latest_step"]["completed"] is False


def test_terminal_boundary_defers_anomaly_trace_quality_to_completion_review():
    analysis = {
        "analysis_id": "ana_raw",
        "analysis_goal": "calculate return volatility drawdown",
        "code_type": "code_interpreter_v2",
        "code_hash": "sha256:test",
        "input_evidence_id": "evi_price",
        "input_row_count": 3,
        "status": "succeeded",
        "summary": "raw metrics",
        "computed_insights": [{
            "insight_key": "return_metrics",
            "value": {"percentage_change": -0.99, "max_drawdown": -0.99},
            "calculation_trace": "Calculated directly from the raw series.",
        }],
        "derived_evidence": [],
    }
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_price",
        detector_name="zscore",
        anomaly_points=[{"timestamp": "t1", "value": 1_000_000.0, "score": 50.0}],
        anomaly_spans=[],
        scores=[],
        diagnostics={"resolved_evidence_id": "evi_price"},
    )
    request_state = RequestStateModel(
        request_id="req-policy-analysis-anomaly-conflict",
        message="计算收益率并检测异常",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        requested_capabilities=["analysis", "anomaly"],
        task_contract=TaskContract.model_validate({
            "source": "llm",
            "goal": "calculate return volatility drawdown after anomaly handling",
            "required_outputs": [
                {"id": "analysis", "description": "return metrics", "evidence_kind": "analysis"},
                {"id": "anomaly_detection", "description": "detected anomalies", "evidence_kind": "anomaly"},
            ],
        }),
        latest_analysis_id="ana_raw",
        analysis_artifacts={"ana_raw": analysis},
        latest_anomaly=anomaly,
        anomaly_artifacts={anomaly.anomaly_id: anomaly},
    )

    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "done"})

    assert allowed is True
    assert reason is None


def test_runtime_keeps_query_todo_active_for_schema_only_sql_observation():
    request_state = RequestStateModel(
        request_id="req-policy-count-schema",
        message="总共有多少条数据？",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "查询总行数",
                "task_type": "query",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["count"],
            },
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    schema_evidence = {
        "evidence_id": "evi_schema",
        "result_type": "schema",
        "database": "demo",
        "query_language": "flux",
        "query": None,
        "summary": "schema",
        "data": {"tables_or_measurements": [{"name": "m", "row_count": None}]},
        "columns": ["name"],
        "metadata": {},
        "diagnostics": {},
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="schema", payload={}),
        schema_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    assert request_state.completion_state["latest_step"]["completed"] is False
    assert request_state.completion_state["latest_step"]["missing_evidence"] == ["database_result"]


def test_runtime_completes_count_todo_with_statistics_evidence():
    request_state = RequestStateModel(
        request_id="req-policy-count-stats",
        message="总共有多少条数据？",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "查询总行数",
                "task_type": "query",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["count"],
            },
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    stats_evidence = {
        "evidence_id": "evi_count",
        "result_type": "statistics",
        "database": "demo",
        "query_language": "flux",
        "query": 'from(bucket: "demo") |> range(start: 0) |> count()',
        "summary": "Computed count.",
        "data": {"statistics": {"count": 19735}},
        "columns": ["metric", "value"],
        "metadata": {},
        "diagnostics": {},
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="count", payload={}),
        stats_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]
    assert request_state.todo_list[0]["result_ref"] == "evidence:evi_count"


def test_runtime_completes_count_todo_with_explicit_count_table():
    request_state = RequestStateModel(
        request_id="req-policy-count-table",
        message="总共有多少条数据？",
        database_context=DatabaseContext(database_id="demo", database_type="clickhouse"),
        status="running",
        todo_list=[
            {
                "content": "查询总行数",
                "task_type": "query",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["count"],
            },
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    table_evidence = {
        "evidence_id": "evi_count_table",
        "result_type": "table",
        "database": "demo",
        "query_language": "sql",
        "query": "SELECT COUNT(*) AS total_count FROM events",
        "summary": "1 row",
        "data": {"rows": [{"total_count": 42}]},
        "columns": ["total_count"],
        "metadata": {"sql_query_mode": "explicit"},
        "diagnostics": {},
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="count", payload={}),
        table_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]
    assert request_state.completion_state["latest_step"]["missing_evidence"] == []


def test_runtime_advances_todo_when_generation_coverage_only_mentions_future_work():
    request_state = RequestStateModel(
        request_id="req-policy-generation-followup",
        message="返回总数、最早5条、最晚5条和时间范围。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询总记录数", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "查询最早5条", "task_type": "query", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    count_evidence = {
        "evidence_id": "evi_count_generation_followup",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": 'from(bucket: "bitcoin") |> count()',
        "summary": "Loaded 1 row.",
        "data": {"rows": [{"count": 2680}]},
        "columns": ["count"],
        "metadata": {},
        "diagnostics": {
            "task_coverage": {
                "requires_followup": True,
                "runtime_requires_followup": False,
                "missing": ["未返回按时间升序排列的最早5条原始记录。"],
                "runtime_missing": [],
            }
        },
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="count", payload={}),
        count_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]
    assert request_state.completion_state["latest_step"]["completed"] is True


def test_runtime_keeps_todo_active_for_runtime_coverage_gap():
    request_state = RequestStateModel(
        request_id="req-policy-runtime-followup",
        message="返回总数。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询总记录数", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    incomplete_evidence = {
        "evidence_id": "evi_runtime_gap",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": 'from(bucket: "bitcoin")',
        "summary": "Loaded 1 row.",
        "data": {"rows": [{"timestamp": "2023-01-01T00:00:00Z"}]},
        "columns": ["timestamp"],
        "metadata": {},
        "diagnostics": {
            "task_coverage": {
                "requires_followup": True,
                "runtime_requires_followup": True,
                "missing": ["未返回总记录数。"],
                "runtime_missing": ["selected result fields are not present in returned columns: count"],
            }
        },
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="rows", payload={}),
        incomplete_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    assert request_state.completion_state["latest_step"]["completed"] is False
    assert request_state.completion_state["latest_step"]["missing_evidence"] == [
        "selected result fields are not present in returned columns: count"
    ]


def test_runtime_completes_answer_todo_after_terminal_payload():
    request_state = RequestStateModel(
        request_id="req-policy-answer-terminal",
        message="返回总数。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询总记录数", "task_type": "query", "status": "completed", "priority": 1},
            {"content": "回答问题", "task_type": "answer", "status": "in_progress", "priority": 2},
        ],
        plan_current_step=2,
    )
    final_answer = {
        "summary": "共有 42 条数据。",
        "sections": [],
        "references": [],
        "visualizations": [],
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="terminate", success=True, summary="answered", payload={}),
        final_answer,
        _TerminalSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "completed"]
    assert request_state.planning_complete is True
    assert request_state.completion_state["latest_step"]["completed"] is True
    assert request_state.completion_state["latest_step"]["tool_name"] == "terminate"


def test_runtime_reconciles_stale_non_answer_todos_when_contract_is_covered():
    request_state = RequestStateModel(
        request_id="req-policy-global-reconcile",
        message="查询历史价格并预测。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询历史价格", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "生成预测", "task_type": "forecast", "status": "pending", "priority": 2},
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 3},
        ],
        plan_current_step=1,
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "查询历史价格并预测",
                "required_outputs": [
                    {"id": "historical_price_series", "description": "历史价格序列"},
                    {"id": "forecast_24h", "description": "24小时预测"},
                ],
                "constraints": {},
                "assumptions": [],
                "evidence_quality_notes": [],
            }
        ),
        latest_database_evidence=DatabaseEvidence(
            evidence_id="evi_price",
            result_type="timeseries",
            database="demo",
            summary="Loaded prices.",
            data={"points": [{"timestamp": "t1", "value": 1.0}]},
        ),
        latest_forecast=ForecastResult(
            forecast_id="forecast_price",
            model_name="linear_regression",
            horizon=1,
            status="succeeded",
            forecast_points=[TimeSeriesPoint(timestamp="t2", value=2.0)],
        ),
    )
    request_state.forecast_artifacts[request_state.latest_forecast.forecast_id] = request_state.latest_forecast
    request_state.observations.append(
        ToolObservation(
            tool_name="code_interpreter",
            success=True,
            summary="Forecast explanation ready.",
            payload={"analysis_id": "ana_forecast", "input_evidence_id": "evi_price"},
        )
    )

    result = apply_previous_observation_assessment(
        request_state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason="历史序列和预测输出均已由现有证据覆盖。",
            evidence_refs=["evidence:evi_price"],
            covered=["historical_price_series", "forecast_24h"],
            missing=[],
            completed_todos=[1, 2],
            next_active_todo=3,
            can_answer=True,
        ),
    )

    assert result.completed is True
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "completed", "in_progress"]
    assert request_state.todo_list[0]["result_ref"] == "evidence:evi_price"
    assert request_state.todo_list[1]["result_ref"] == "forecast:forecast_price"
    assert request_state.todo_list[2]["task_type"] == "answer"
    allowed, reason = validate_action(request_state, "terminate", {"direct_answer": "已完成。"})
    assert allowed is True
    assert reason is None


def test_runtime_rejects_global_reconcile_with_unknown_evidence_ref():
    request_state = RequestStateModel(
        request_id="req-policy-global-reconcile-bad-ref",
        message="查询历史价格并预测。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询历史价格", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        task_contract=TaskContract.model_validate(
            {
                "source": "llm",
                "goal": "查询历史价格",
                "required_outputs": [
                    {"id": "historical_price_series", "description": "历史价格序列"},
                ],
                "constraints": {},
                "assumptions": [],
                "evidence_quality_notes": [],
            }
        ),
    )
    request_state.observations.append(
        ToolObservation(tool_name="sql_query", success=True, summary="Loaded prices.", payload={})
    )

    result = apply_previous_observation_assessment(
        request_state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason="历史序列已覆盖。",
            evidence_refs=["evidence:missing"],
            covered=["historical_price_series"],
            missing=[],
            next_active_todo=2,
            can_answer=True,
        ),
    )

    assert result.completed is False
    assert "evidence:missing" in result.missing_evidence
    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]


def test_runtime_does_not_complete_non_answer_todo_after_terminal_payload():
    request_state = RequestStateModel(
        request_id="req-policy-query-terminal",
        message="返回总数。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询总记录数", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    final_answer = {
        "summary": "缺少查询证据，无法确认总数。",
        "sections": [],
        "references": [],
        "visualizations": [],
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="terminate", success=True, summary="answered", payload={}),
        final_answer,
        _TerminalSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    assert request_state.planning_complete is False
    assert request_state.completion_state["latest_step"]["completed"] is False


def test_runtime_completes_count_todo_with_flux_value_column():
    request_state = RequestStateModel(
        request_id="req-policy-count-flux-value",
        message="总共有多少条数据？",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "查询总行数",
                "task_type": "query",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["count"],
            },
            {"content": "回答问题", "task_type": "answer", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    table_evidence = {
        "evidence_id": "evi_count_flux",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": 'from(bucket: "b") |> count(column: "_value")',
        "summary": "Loaded 1 row.",
        "data": {"rows": [{"value": 2680, "field": "price"}]},
        "columns": ["value", "field"],
        "metadata": {},
        "diagnostics": {},
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="count", payload={}),
        table_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]


def test_runtime_completes_timeseries_todo_with_timestamp_value_table():
    request_state = RequestStateModel(
        request_id="req-policy-timeseries-table",
        message="查询时间序列。",
        database_context=DatabaseContext(database_id="demo", database_type="clickhouse"),
        status="running",
        todo_list=[
            {
                "content": "查询时序数据",
                "task_type": "query",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["time_series"],
            },
            {"content": "分析趋势", "task_type": "code_interpreter", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    table_evidence = {
        "evidence_id": "evi_timeseries_table",
        "result_type": "table",
        "database": "demo",
        "query_language": "sql",
        "query": "SELECT timestamp, value FROM metrics",
        "summary": "2 rows",
        "data": {
            "rows": [
                {"timestamp": "2023-01-01T00:00:00Z", "value": 1.0},
                {"timestamp": "2023-01-01T01:00:00Z", "value": 2.0},
            ]
        },
        "columns": ["timestamp", "value"],
        "metadata": {},
        "diagnostics": {},
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="timeseries", payload={}),
        table_evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]


def test_runtime_advances_legacy_plan_step_with_query_evidence():
    request_state = RequestStateModel(
        request_id="req-policy-legacy-plan-step",
        message="返回总数和最早5条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {
                "content": "确认数据源与字段并生成可执行查询计划",
                "task_type": "plan",
                "status": "in_progress",
                "priority": 1,
                "evidence_needed": ["schema"],
            },
            {"content": "查询总记录数", "task_type": "query", "status": "pending", "priority": 2},
        ],
        plan_current_step=1,
    )
    evidence = {
        "evidence_id": "evi_rows",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": "from(...) |> limit(n: 5)",
        "summary": "5 rows",
        "data": {"rows": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        "columns": ["timestamp", "value"],
        "metadata": {},
        "diagnostics": {},
    }

    assert validate_action(request_state, "sql_query") == (True, None)
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="rows", payload={}),
        evidence,
        _EvidenceSpec(),
    )

    assert request_state.todo_list[0]["status"] == "in_progress"
    _assess_previous_complete(request_state)
    assert request_state.todo_list[0]["status"] == "completed"
    assert request_state.todo_list[0]["task_type"] == "query"
    assert request_state.todo_list[1]["status"] == "in_progress"


def test_runtime_advances_todo_after_successful_actions():
    request_state = RequestStateModel(
        request_id="req-policy-progress",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询时序数据", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "提炼趋势事实", "task_type": "code_interpreter", "status": "pending", "priority": 2},
            {"content": "检查异常", "task_type": "anomaly", "status": "pending", "priority": 3},
        ],
        plan_current_step=1,
    )
    evidence = {
        "evidence_id": "evi_demo",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo",
        "summary": "ok",
        "data": {"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}], "rows": []},
        "columns": ["timestamp", "value"],
        "metadata": {},
        "diagnostics": {},
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        evidence,
        _EvidenceSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["in_progress", "pending", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress", "pending"]

    code_interpreter = {
        "analysis_id": "ana_demo",
        "analysis_goal": "趋势",
        "code_type": "python_rows_v1",
        "code_hash": "hash",
        "input_evidence_id": "evi_demo",
        "input_row_count": 1,
        "status": "succeeded",
        "summary": "ok",
        "result": {"summary": "ok", "metrics": {}, "details": {}},
        "diagnostics": {},
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}),
        code_interpreter,
        _AnalysisSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress", "pending"]
    _assess_previous_complete(request_state)
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "completed", "in_progress"]
