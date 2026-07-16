from __future__ import annotations

from runtime.action_policy import validate_action
from schemas.database_context import DatabaseContext
from schemas.state import RequestStateModel


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

    allowed, reason = validate_action(request_state, "query_database")

    assert allowed is True
    assert reason is None


def test_policy_allows_repeated_todowrite_and_todo_mismatch():
    request_state = RequestStateModel(
        request_id="req-policy-todo",
        message="分析趋势和异常。",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        status="running",
        todo_list=[
            {"content": "提炼趋势事实", "task_type": "insight", "status": "in_progress", "priority": 1}
        ],
    )

    assert validate_action(request_state, "todowrite") == (True, None)
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
