from __future__ import annotations

import pytest

from core.completion import apply_previous_observation_assessment, normalize_todo_for_completion
from runtime.react_loop import ReActLoop
from runtime.action_policy import validate_action
from runtime.request_state import apply_observation, apply_observation_async, enrich_observation_payload
from schemas.agent_turn import PreviousObservationAssessment, ReActTurn
from schemas.database import DatabaseEvidence
from schemas.state import ConversationStateModel
from schemas.database_context import DatabaseContext
from schemas.state import RequestStateModel
from schemas.task_contract import TaskContract
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
        "query_language": "reference_dataset",
        "query": "reference_dataset:value:statistics",
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
    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]
    assert request_state.completion_state["latest_step"]["completed"] is True


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
    )
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
