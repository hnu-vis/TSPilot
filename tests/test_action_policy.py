from __future__ import annotations

import pytest

from core.completion import CompletionEvaluation, normalize_todo_for_completion
from core.runtime_evaluator import PlanRequirementVerdict
from runtime.react_loop import ReActLoop
from runtime.action_policy import validate_action
from runtime.request_state import apply_observation, apply_observation_async
from schemas.agent_turn import ReActTurn
from schemas.state import ConversationStateModel
from schemas.database_context import DatabaseContext
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation


class _EvidenceSpec:
    result_target = "evidence"


class _AnalysisSpec:
    result_target = "analysis"


class _CompletionEvaluator:
    def __init__(self, completed: bool):
        self.completed = completed
        self.last_step_verdict = {
            "completed": completed,
            "reason": "test verdict",
            "missing_items": [] if completed else ["missing exact result"],
        }

    async def evaluate_step_completion(self, **kwargs):
        return CompletionEvaluation(
            completed=self.completed,
            reason="test verdict",
            missing_evidence=[] if self.completed else ["missing exact result"],
            evidence_refs=["evidence:evi_demo"] if self.completed else [],
        )


class _OneTurnAgent:
    async def next_turn(self, request_state, conversation_state):
        return ReActTurn(
            thought="try query first",
            action="sql_query",
            action_input={"message": request_state.message, "database_context": request_state.database_context.model_dump(mode="json")},
        )


class _PlanEvaluator:
    last_step_verdict = None

    async def evaluate_plan_requirement(self, **kwargs):
        return PlanRequirementVerdict(
            requires_plan=True,
            reason="multiple deliverables",
            deliverables=["count", "earliest rows"],
        )


class _UnusedExecutor:
    async def execute(self, *args, **kwargs):
        raise AssertionError("tool executor should not run when plan is required")


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


def test_policy_rejects_repeated_todowrite_but_allows_next_action():
    request_state = RequestStateModel(
        request_id="req-policy-todo",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "提炼趋势事实", "task_type": "insight", "status": "in_progress", "priority": 1}
        ],
    )

    allowed, reason = validate_action(request_state, "todowrite")

    assert allowed is False
    assert "already exists" in (reason or "")
    assert validate_action(request_state, "insight") == (True, None)
    forecast_allowed, forecast_reason = validate_action(request_state, "forecast")
    assert forecast_allowed is False
    assert "active plan step" in (forecast_reason or "")


def test_policy_blocks_unrelated_sql_when_active_analysis_step_has_evidence():
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

    allowed, reason = validate_action(request_state, "sql_query")

    assert allowed is False
    assert "Current task_type is 'anomaly'" in (reason or "")
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


def test_todo_evidence_needed_ignores_domain_field_names():
    todo = normalize_todo_for_completion(
        {
            "content": "查询 appliances_energy_wh 的原始时间序列",
            "task_type": "query",
            "evidence_needed": ["timestamp", "appliances_energy_wh", "time_series"],
        }
    )

    assert todo["evidence_needed"] == ["time_series"]


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
            {"content": "分析趋势", "task_type": "insight", "status": "pending", "priority": 2},
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
async def test_llm_completion_verdict_false_keeps_todo_in_progress():
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
        "query": "from(...)",
        "summary": "Loaded 5 rows.",
        "data": {"rows": [{"value": 1}]},
        "columns": ["value"],
        "metadata": {},
        "diagnostics": {},
    }

    await apply_observation_async(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="rows", payload=payload),
        payload,
        _EvidenceSpec(),
        completion_evaluator=_CompletionEvaluator(completed=False),
    )

    assert request_state.todo_list[0]["status"] == "in_progress"
    assert request_state.todo_list[1]["status"] == "pending"
    assert request_state.completion_state["latest_step"]["completed"] is False
    assert request_state.completion_state["latest_step"]["missing_evidence"] == ["missing exact result"]


@pytest.mark.asyncio
async def test_llm_completion_verdict_true_advances_todo():
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
        completion_evaluator=_CompletionEvaluator(completed=True),
    )

    assert request_state.todo_list[0]["status"] == "completed"
    assert request_state.todo_list[1]["status"] == "in_progress"
    assert request_state.completion_state["latest_step"]["completed"] is True


@pytest.mark.asyncio
async def test_runtime_blocks_direct_query_when_llm_requires_plan():
    request_state = RequestStateModel(
        request_id="req-plan-gate",
        message="请完成以下任务: 1.返回总数; 2.返回最早5条。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        max_iterations=1,
    )
    loop = ReActLoop(
        data_agent=_OneTurnAgent(),
        tool_executor=_UnusedExecutor(),
        settings=type("_Settings", (), {"conversation_log_enabled": False, "resolved_conversation_log_dir": "."})(),
        runtime_evaluator=_PlanEvaluator(),
    )

    events = [event async for event in loop._iterate(request_state, ConversationStateModel(conversation_id="conv"))]
    observation = next(event for event in events if event.event_type == "observation")

    assert observation.payload["success"] is False
    assert "todo plan is required" in observation.payload["summary"].lower()
    assert request_state.tool_history == []


def test_policy_blocks_format_answer_until_database_goal_has_evidence():
    request_state = RequestStateModel(
        request_id="req-policy-format-block",
        message="总共有多少条数据？",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
    )

    allowed, reason = validate_action(request_state, "format_answer")

    assert allowed is False
    assert "not complete" in (reason or "")
    assert request_state.completion_state["latest_goal"]["missing_evidence"] == ["database_evidence"]


def test_runtime_does_not_complete_count_todo_with_schema_only():
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
    assert request_state.completion_state["latest_step"]["missing_evidence"] == ["count"]


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

    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]
    assert request_state.completion_state["latest_step"]["missing_evidence"] == []


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
            {"content": "分析趋势", "task_type": "insight", "status": "pending", "priority": 2},
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

    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress"]


def test_runtime_advances_todo_after_successful_actions():
    request_state = RequestStateModel(
        request_id="req-policy-progress",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "查询时序数据", "task_type": "query", "status": "in_progress", "priority": 1},
            {"content": "提炼趋势事实", "task_type": "insight", "status": "pending", "priority": 2},
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

    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "in_progress", "pending"]

    insight = {
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
        ToolObservation(tool_name="insight", success=True, summary="ok", payload={}),
        insight,
        _AnalysisSpec(),
    )

    assert [todo["status"] for todo in request_state.todo_list] == ["completed", "completed", "in_progress"]
