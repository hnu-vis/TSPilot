from __future__ import annotations

import app.deps as deps
import app.routes.chat as chat_route
from fastapi.testclient import TestClient

from app.server import app
from app.settings import get_settings
from tests.fakes import ComplexReActLLM, FakeLLM, RepeatingTodoLLM, TodoScopeLLM


def _build_client(llm, *, max_iterations: int | None = None) -> TestClient:
    settings = get_settings()
    if max_iterations is not None:
        settings.max_iterations = max_iterations

    deps.get_llm = lambda: llm
    deps.get_data_agent.cache_clear()
    deps.get_tool_registry.cache_clear()
    deps.get_tool_executor.cache_clear()
    chat_route.get_react_loop = deps.get_react_loop
    return TestClient(app)


def test_chat_json_path_returns_final_answer():
    client = _build_client(FakeLLM())
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "请分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "time_range": {
                "start": "2016-01-11T17:00:00",
                "end": "2016-01-12T23:00:00",
            },
            "constraints": {"max_points": 48},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["response_kind"] == "final_answer"
    assert payload["used_tools"] == ["query_database", "insight", "format_answer"]
    assert payload["answer"]["summary"]


def test_chat_sse_path_returns_event_stream():
    client = _build_client(FakeLLM())
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "message": "请分析 appliances_energy_wh 的趋势",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "stream": True,
        },
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: conversation_id" in body
    assert "event: agent_step" in body
    assert "event: tool_call" in body
    assert "event: tool_result" in body
    assert "event: final_answer" in body
    assert "event: terminate" in body
    assert "event: thought" not in body
    assert "event: action" not in body
    assert "event: observation" not in body


def test_chat_json_path_supports_complex_multi_step_react():
    client = _build_client(ComplexReActLLM(), max_iterations=8)
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "先规划一下，然后分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势，检查异常，再给一个短期预测，最后总结结论。",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "time_range": {
                "start": "2016-01-11T17:00:00",
                "end": "2016-01-12T23:00:00",
            },
            "constraints": {"max_points": 48},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["response_kind"] == "final_answer"
    assert payload["used_tools"] == [
        "todowrite",
        "query_database",
        "insight",
        "anomaly",
        "forecast",
        "format_answer",
    ]
    assert payload["answer"]["summary"]
    section_types = [section["section_type"] for section in payload["answer"]["sections"]]
    assert "facts" in section_types
    assert "anomaly" in section_types
    assert "forecast" in section_types
    reference_types = [reference["source_type"] for reference in payload["answer"]["references"]]
    assert "forecast" in reference_types
    assert "anomaly" in reference_types
    trace_event_types = [event["event_type"] for event in payload["trace"]]
    assert trace_event_types.count("action") == 6
    assert trace_event_types.count("observation") == 6
    assert "final_answer" in trace_event_types
    assert "terminate" in trace_event_types


def test_chat_sse_path_supports_complex_multi_step_react():
    client = _build_client(ComplexReActLLM(), max_iterations=8)
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "message": "先规划一下，然后分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势，检查异常，再给一个短期预测，最后总结结论。",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "time_range": {
                "start": "2016-01-11T17:00:00",
                "end": "2016-01-12T23:00:00",
            },
            "constraints": {"max_points": 48},
            "stream": True,
        },
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body.count("event: tool_call") == 6
    assert body.count("event: tool_result") == 6
    assert '"tool": "todowrite"' in body
    assert '"tool": "query_database"' in body
    assert '"tool": "insight"' in body
    assert '"tool": "anomaly"' in body
    assert '"tool": "forecast"' in body
    assert '"tool": "format_answer"' in body
    assert '"phase": "intent"' in body
    assert '"phase": "analysis"' in body
    assert '"phase": "answer_assembly"' in body
    assert "event: final_answer" in body
    assert "event: terminate" in body
    assert "event: thought" not in body
    assert "event: action" not in body
    assert "event: observation" not in body


def test_runtime_rejects_redundant_todowrite_and_recovers():
    client = _build_client(RepeatingTodoLLM(), max_iterations=8)
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "先规划，再分析 appliances_energy_wh 的趋势和异常，最后总结。",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "time_range": {
                "start": "2016-01-11T17:00:00",
                "end": "2016-01-12T23:00:00",
            },
            "constraints": {"max_points": 48},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["used_tools"] == ["todowrite", "query_database", "insight", "anomaly", "format_answer"]
    observations = [event for event in payload["trace"] if event["event_type"] == "observation"]
    rejected = [
        event for event in observations
        if event["payload"]["tool_name"] == "todowrite" and event["payload"]["success"] is False
    ]
    assert rejected
    assert "does not match the current state" in rejected[-1]["payload"]["summary"]


def test_runtime_enforces_current_todo_step_before_forecast():
    client = _build_client(TodoScopeLLM(), max_iterations=4)
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "分析 appliances_energy_wh 的趋势和异常。",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "time_range": {
                "start": "2016-01-11T17:00:00",
                "end": "2016-01-12T23:00:00",
            },
            "constraints": {"max_points": 48},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    observations = [event for event in payload["trace"] if event["event_type"] == "observation"]
    mismatches = [
        event for event in observations
        if event["payload"]["tool_name"] == "forecast" and event["payload"]["success"] is False
    ]
    assert mismatches
    assert "current todo step" in mismatches[-1]["payload"]["summary"]
