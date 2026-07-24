from __future__ import annotations

import pytest

from core.completion import normalize_todo_for_completion
from core.runtime_evaluator import GoalVerificationResult, RuntimeLLMEvaluator
from runtime.react_loop import ReActLoop
from runtime.action_policy import validate_action
from runtime.request_state import apply_observation, apply_observation_async
from schemas.agent_turn import ReActTurn
from schemas.state import ConversationStateModel
from schemas.database_context import DatabaseContext
from schemas.state import RequestStateModel
from schemas.tool import ToolCall, ToolObservation
from tools.todowrite import TodoWriteInput, TodoWriteTool


class _EvidenceSpec:
    result_target = "evidence"
    produces_terminal_payload = False


class _AnalysisSpec:
    result_target = "analysis"
    produces_terminal_payload = False


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


class _RejectingVerifier:
    async def verify_final_answer(self, *, request_state, candidate_answer):
        return GoalVerificationResult(
            can_answer=False,
            reason="missing requested maximum value",
            missing_items=["maximum value"],
            unsupported_claims=[],
            next_action_hint="Run sql_query for the maximum value.",
            confidence=0.9,
        )


def test_runtime_evaluator_marks_complete_query_preview_as_full_fidelity():
    evaluator = RuntimeLLMEvaluator(llm=None)
    summary = evaluator._summarize_payload(
        {
            "evidence_id": "evi_extrema",
            "result_type": "timeseries",
            "query": "from(...) |> sort(columns: [\"_value\"], desc: true) |> limit(n: 1)",
            "data": {
                "rows": [{"timestamp": "2023-01-04T23:04:00Z", "value": 168249475888010.0}],
                "points": [{"timestamp": "2023-01-04T23:04:00Z", "value": 168249475888010.0}],
            },
            "diagnostics": {
                "is_full_fidelity": True,
                "summary_stats": {"rows_count": 1, "points_count": 1, "series_count": 1},
                "prompt_sampling": {
                    "sampled_for_prompt": False,
                    "full_counts": {"rows_count": 1, "points_count": 1, "series_count": 1},
                    "visible_counts": {"rows_count": 1, "points_count": 1, "series_count": 1},
                    "full_artifact_ref": "evidence:evi_extrema",
                },
            },
        }
    )

    assert "sample_rows" not in summary
    assert summary["result_rows_preview"] == [{"timestamp": "2023-01-04T23:04:00Z", "value": 168249475888010.0}]
    assert summary["data_completeness"]["is_full_fidelity"] is True
    assert summary["data_completeness"]["sampled_for_prompt"] is False
    assert summary["data_completeness"]["full_row_count"] == 1


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
            {"content": "分析趋势", "task_type": "insight", "status": "pending", "priority": 2},
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
async def test_goal_verifier_rejects_terminal_candidate_and_continues_loop():
    request_state = RequestStateModel(
        request_id="req-verifier-reject",
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
        goal_verifier=_RejectingVerifier(),
    )

    events = [event async for event in loop._iterate(request_state, ConversationStateModel(conversation_id="conv"))]

    verifier_observation = next(
        event for event in events
        if event.event_type == "observation" and event.payload["tool_name"] == "goal_verifier"
    )
    assert verifier_observation.payload["success"] is False
    assert verifier_observation.payload["payload"]["missing_items"] == ["maximum value"]
    assert request_state.completion_state["semantic_repair_directive"]["missing_items"] == ["maximum value"]
    assert request_state.completion_state["semantic_repair_directive"]["next_action_hint"] == "Run sql_query for the maximum value."
    assert request_state.final_answer_draft is None
    assert "final_answer" not in [event.event_type for event in events]


@pytest.mark.asyncio
async def test_goal_verifier_becomes_advisory_after_rejection_limit():
    request_state = RequestStateModel(
        request_id="req-verifier-bypass",
        message="返回最大值。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        max_iterations=2,
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
        settings=type(
            "_Settings",
            (),
            {
                "conversation_log_enabled": False,
                "resolved_conversation_log_dir": ".",
                "goal_verifier_max_rejections": 1,
            },
        )(),
        goal_verifier=_RejectingVerifier(),
    )

    events = [event async for event in loop._iterate(request_state, ConversationStateModel(conversation_id="conv"))]

    verifier_failures = [
        event for event in events
        if event.event_type == "observation" and event.payload["tool_name"] == "goal_verifier"
    ]
    assert len(verifier_failures) == 1
    assert request_state.completion_state["goal_verifier_bypassed"]["missing_items"] == ["maximum value"]
    assert request_state.final_answer_draft is not None
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
                    "runtime_missing_or_uncertain": [
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
