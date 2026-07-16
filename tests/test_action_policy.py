from __future__ import annotations

from runtime.action_policy import validate_action
from runtime.request_state import apply_observation
from schemas.database_context import DatabaseContext
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation


class _EvidenceSpec:
    result_target = "evidence"


class _AnalysisSpec:
    result_target = "analysis"


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
    assert validate_action(request_state, "forecast") == (True, None)


def test_policy_rejects_unknown_action_only():
    request_state = RequestStateModel(
        request_id="req-policy-unknown",
        message="分析趋势。",
        status="running",
    )

    allowed, reason = validate_action(request_state, "ask_user")

    assert allowed is False
    assert "runtime contract" in (reason or "")


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
