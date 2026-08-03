from __future__ import annotations

import json

import app.deps as deps
import app.routes.chat as chat_route
from fastapi.testclient import TestClient

from app.server import app
from app.settings import get_settings
from runtime.react_loop import ReActLoop
from runtime.request_state import build_conversation_state, build_request_state
from schemas.api import ChatRequest
from tests.fakes import ( 
    BitcoinMultiQueryLLM,
    CasualLLM,
    CodeRequiredRepairLLM,
    ComplexReActLLM,
    FakeLLM,
    RepeatingTodoLLM,
    SandboxAnalysisLLM,
    TodoScopeLLM,
)


def _build_client(llm, *, max_iterations: int | None = None) -> TestClient:
    settings = get_settings()
    if max_iterations is not None:
        settings.max_iterations = max_iterations

    deps.get_llm = lambda: llm
    deps.get_data_agent_llm = lambda: llm
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
    assert payload["used_tools"][:2] == ["sql_query", "code_interpreter"]
    assert payload["answer"]["summary"]
    assert payload["token_usage"]["totals"]["total_tokens"] > 0
    assert payload["token_usage"]["totals"]["call_count"] >= 1
    assert payload["token_usage"]["totals"]["counting_method"] == "tiktoken_estimate"


def test_chat_json_path_uses_code_interpreter_tool(tmp_path):
    settings = get_settings()
    old_log_dir = settings.conversation_log_dir
    old_enabled = settings.conversation_log_enabled
    settings.conversation_log_dir = str(tmp_path)
    settings.conversation_log_enabled = True
    try:
        client = _build_client(SandboxAnalysisLLM(), max_iterations=5)
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "用 code interpreter 计算相邻点差值。",
                "database_context": {
                    "database_id": "influxdb2-energydata",
                    "database_type": "influxdb",
                },
                "time_range": {
                    "start": "2016-01-11T17:00:00",
                    "end": "2016-01-12T23:00:00",
                },
                "constraints": {"max_points": 12},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["used_tools"] == ["sql_query", "code_interpreter"]

        code_observation = next(
            event
            for event in payload["trace"]
            if event["event_type"] == "tool_result"
            and event["payload"]["tool"] == "code_interpreter"
        )
        code_payload = code_observation["payload"]["payload_preview"]
        assert code_payload["code_type"] == "code_interpreter_v1"
        assert code_payload["code_preview"]
        assert code_payload["metrics_preview"]["value_count"] > 1
        assert code_payload["metrics_preview"]["delta_count"] == code_payload["metrics_preview"]["value_count"] - 1
        assert "Code interpreter computed" in code_observation["payload"]["summary"]
        assert "Code interpreter computed" in code_payload["summary"]

        section_types = [section["section_type"] for section in payload["answer"]["sections"]]
        assert "analysis" in section_types
        assert "Code interpreter computed 180 pairwise deltas" in payload["answer"]["summary"]
        conclusion = next(section for section in payload["answer"]["sections"] if section["section_type"] == "conclusion")
        assert "Code interpreter computed 180 pairwise deltas" in conclusion["content"]
        assert any(reference["source_type"] == "analysis" for reference in payload["answer"]["references"])

        request_dir = next(tmp_path.glob(f"*_{payload['conversation_id']}/requests/{payload['request_id']}"))
        code_outputs = list((request_dir / "artifacts" / "code_interpreter").glob("*/output.json"))
        assert len(code_outputs) == 1
        code_output = json.loads(code_outputs[0].read_text(encoding="utf-8"))
        assert code_output["status"] == "succeeded"
        assert code_output["result"]["metrics"]["delta_count"] == code_payload["metrics_preview"]["delta_count"]
    finally:
        settings.conversation_log_dir = old_log_dir
        settings.conversation_log_enabled = old_enabled


def test_code_required_metrics_repair_generates_code():
    client = _build_client(CodeRequiredRepairLLM(), max_iterations=6)
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "计算收益率、波动率和最大回撤。",
            "database_context": {
                "database_id": "influxdb2-energydata",
                "database_type": "influxdb",
            },
            "time_range": {
                "start": "2016-01-11T17:00:00",
                "end": "2016-01-12T23:00:00",
            },
            "constraints": {"max_points": 24},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["used_tools"].count("code_interpreter") == 2
    failures = [
        event
        for event in payload["trace"]
        if event["event_type"] == "tool_result"
        and event["payload"].get("tool") == "code_interpreter"
        and event["payload"].get("success") is False
    ]
    assert failures
    assert failures[-1]["payload"]["payload_preview"]["error_type"] == "code_required_for_metrics"
    successful_code = [
        event["payload"]["payload_preview"]
        for event in payload["trace"]
        if event["event_type"] == "tool_result"
        and event["payload"].get("tool") == "code_interpreter"
        and event["payload"].get("success") is True
    ][-1]
    assert successful_code["code_type"] == "code_interpreter_v1"
    assert successful_code["code_preview"]
    metrics = successful_code["metrics_preview"]
    assert set(metrics) == {"total_return", "volatility", "max_drawdown"}


def test_first_visible_action_does_not_wait_for_separate_intent_llm_call():
    llm = FakeLLM()
    client = _build_client(llm)
    react_loop = deps.get_react_loop()
    request = ChatRequest(
        message="请分析 appliances_energy_wh 的趋势",
        database_context={"database_id": "influxdb2-energydata", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, get_settings())
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    async def first_action_event():
        async for event in react_loop._iterate(request_state, conversation_state):
            if event.event_type == "action":
                return event
        return None

    import asyncio

    event = asyncio.run(first_action_event())

    assert client
    assert event is not None
    assert event.payload["action"] == "sql_query"
    assert llm.calls == 1


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
    assert "event: step.start" in body
    assert "event: step.meta" in body
    assert "event: step.done" in body
    assert "event: final_answer" in body
    assert "event: terminate" in body
    assert "event: thought" in body
    assert '"action_input"' in body
    assert '"observation"' in body
    assert body.index('"placeholder": true') < body.index("event: tool_call")
    assert "event: action" not in body
    assert "event: observation" not in body


def test_chat_json_path_persists_complete_trace_log(tmp_path):
    settings = get_settings()
    old_log_dir = settings.conversation_log_dir
    old_enabled = settings.conversation_log_enabled
    settings.conversation_log_dir = str(tmp_path)
    settings.conversation_log_enabled = True
    try:
        client = _build_client(FakeLLM())
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "请分析 appliances_energy_wh 的趋势",
                "database_context": {
                    "database_id": "influxdb2-energydata",
                    "database_type": "influxdb",
                },
            },
        )
        payload = response.json()
        run_dir = next(tmp_path.glob(f"*_{payload['conversation_id']}"))
        request_dir = run_dir / "requests" / payload["request_id"]
        log_path = request_dir / "conversation_trace.json"
        index_path = tmp_path / "index.jsonl"

        assert response.status_code == 200
        assert log_path.exists()
        assert (request_dir / "request.json").exists()
        assert (request_dir / "response.json").exists()
        assert (request_dir / "state.json").exists()
        assert (request_dir / "trace_internal.jsonl").exists()
        assert (request_dir / "tool_calls.jsonl").exists()
        assert index_path.exists()

        log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        assert log_payload["schema_version"] == "conversation_trace_v1"
        assert log_payload["mode"] == "json"
        assert log_payload["status"] == "completed"
        assert log_payload["request"]["message"] == "请分析 appliances_energy_wh 的趋势"
        assert [event["event_type"] for event in log_payload["trace"]["internal"]].count("action") == 3
        assert log_payload["summary"]["used_tools"] == ["sql_query", "code_interpreter"]
        assert log_payload["state"]["tool_history"]
        assert log_payload["state"]["react_transcript"]
        first_step = log_payload["state"]["react_transcript"][0]
        assert first_step["thought"]
        assert first_step["action"]
        assert first_step["action_input"] is not None
        assert first_step["observation"]["tool_name"] == first_step["action"]

        index_entry = json.loads(index_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert index_entry["request_id"] == payload["request_id"]
        assert index_entry["log_path"] == str(log_path)
    finally:
        settings.conversation_log_dir = old_log_dir
        settings.conversation_log_enabled = old_enabled


def test_chat_sse_path_persists_internal_and_public_trace_logs(tmp_path):
    settings = get_settings()
    old_log_dir = settings.conversation_log_dir
    old_enabled = settings.conversation_log_enabled
    settings.conversation_log_dir = str(tmp_path)
    settings.conversation_log_enabled = True
    try:
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

        request_id_line = next(line for line in body.splitlines() if line.startswith("data: {"))
        request_id = json.loads(request_id_line.removeprefix("data: "))["request_id"]
        conversation_id = json.loads(request_id_line.removeprefix("data: "))["conversation_id"]
        run_dir = next(tmp_path.glob(f"*_{conversation_id}"))
        request_dir = run_dir / "requests" / request_id
        log_path = request_dir / "conversation_trace.json"

        assert response.status_code == 200
        assert log_path.exists()
        assert (request_dir / "trace_public.jsonl").exists()
        log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        assert log_payload["mode"] == "sse"
        assert "thought" in [event["event_type"] for event in log_payload["trace"]["internal"]]
        public_event_types = [event["event_type"] for event in log_payload["trace"]["public"]]
        assert "thought" in public_event_types
        assert "step.start" in public_event_types
        assert "step.meta" in public_event_types
        assert "step.done" in public_event_types
        assert "tool_call" in public_event_types
        assert "tool_result" in public_event_types
        assert "final_answer" in public_event_types
    finally:
        settings.conversation_log_dir = old_log_dir
        settings.conversation_log_enabled = old_enabled


def test_sql_tool_result_preview_exposes_query_and_samples():
    loop = ReActLoop.__new__(ReActLoop)
    preview = loop._payload_preview(
        {
            "tool_name": "sql_query",
            "success": True,
            "summary": "ok",
            "payload_truncated": False,
            "payload": {
                "evidence_id": "evi_sql",
                "query_language": "sql",
                "query": "SELECT value FROM metrics",
                "columns": ["value"],
                "data": {
                    "rows": [{"value": 12.3}, {"value": 13.4}],
                    "points": [{"timestamp": "t0", "value": 12.3}],
                },
                "diagnostics": {
                    "summary_stats": {"rows_count": 2, "points_count": 10},
                    "prompt_sampling": {
                        "policy": "head_tail_edges",
                        "sampled_for_prompt": True,
                        "full_counts": {"points_count": 10},
                        "visible_counts": {"points_count": 1},
                        "full_artifact_ref": "evidence:evi_sql",
                    },
                    "task_coverage": {
                        "satisfied": ["已返回价格值"],
                        "missing_or_uncertain": ["尚未返回时间戳"],
                        "next_action_hint": "继续查询 timestamp 和 value",
                        "requires_followup": True,
                    },
                    "query_trace": {
                        "logical_plan": {
                            "filters": [
                                {"source": "m1", "column": "code", "operator": "=", "value": "USD"},
                                {"source": "m1", "column": "crypto", "operator": "=", "value": "bitcoin"},
                            ],
                            "schema_linking": {
                                "confidence": "high",
                                "sources": [
                                    {
                                        "name": "coindesk",
                                        "kind": "measurement",
                                        "time_column": "_time",
                                        "value_columns": ["price"],
                                        "dimension_columns": ["code", "crypto"],
                                    }
                                ],
                                "time_columns": ["_time"],
                                "value_columns": ["price"],
                                "evidence": ["sources linked from schema names"],
                            },
                        },
                        "field_mappings": [
                            {
                                "source_name": "coindesk",
                                "field_name": "price",
                                "role": "value",
                                "confidence": 0.8,
                            }
                        ],
                    },
                },
            },
        }
    )

    assert preview["query_language"] == "sql"
    assert preview["query"] == "SELECT value FROM metrics"
    assert preview["columns"] == ["value"]
    assert preview["row_count"] == 2
    assert preview["point_count"] == 10
    assert preview["sample_rows"] == [{"value": 12.3}, {"value": 13.4}]
    assert preview["sample_points"] == [{"timestamp": "t0", "value": 12.3}]
    assert preview["sampling"]["sampled_for_prompt"] is True
    assert preview["sampling"]["full_counts"]["points_count"] == 10
    assert preview["task_coverage"]["requires_followup"] is True
    assert preview["task_coverage"]["missing"] == ["尚未返回时间戳"]
    assert "missing_or_uncertain" not in preview["task_coverage"]
    assert preview["schema_linking"]["confidence"] == "high"
    assert preview["schema_linking"]["sources"][0]["name"] == "coindesk"
    assert preview["schema_linking"]["field_mappings"][0]["field_name"] == "price"
    assert preview["schema_linking"]["required_filters"] == [
        {"source": "m1", "column": "code", "operator": "=", "value": "USD"},
        {"source": "m1", "column": "crypto", "operator": "=", "value": "bitcoin"},
    ]


def test_code_interpreter_trace_preview_exposes_code_and_result():
    loop = ReActLoop.__new__(ReActLoop)
    input_preview = loop._input_preview(
        "code_interpreter",
        {
            "database_evidence": "latest",
            "analysis_goal": "compute stats",
            "code": "result = {'summary': 'ok', 'metrics': {'mean': 1.2}, 'details': {'n': 3}}",
        },
    )
    payload_preview = loop._payload_preview(
        {
            "tool_name": "code_interpreter",
            "success": True,
            "summary": "ok",
            "payload": {
                "analysis_id": "ana_stats",
                "analysis_goal": "compute stats",
                "code_type": "code_interpreter_v1",
                "code_hash": "sha256:abc",
                "input_evidence_id": "evi_sql",
                "input_row_count": 3,
                "status": "succeeded",
                "summary": "ok",
                "result": {
                    "summary": "ok",
                    "metrics": {"mean": 1.2},
                    "details": {"n": 3},
                },
                "diagnostics": {
                    "runtime_ms": 12.4,
                    "input_columns": ["timestamp", "value"],
                },
            },
        }
    )

    assert input_preview["code_preview"].startswith("result =")
    assert input_preview["analysis_code_chars"] > 0
    assert payload_preview["analysis_metrics"] == {"mean": 1.2}
    assert payload_preview["analysis_details"] == {"n": 3}
    assert payload_preview["runtime_ms"] == 12.4
    assert payload_preview["input_columns"] == ["timestamp", "value"]


def test_forecast_and_anomaly_trace_previews_expose_tool_specific_outputs():
    loop = ReActLoop.__new__(ReActLoop)
    forecast_preview = loop._payload_preview(
        {
            "tool_name": "forecast",
            "success": True,
            "summary": "ok",
            "payload": {
                "forecast_id": "forecast_evi",
                "model_name": "linear_regression",
                "horizon": 2,
                "status": "succeeded",
                "forecast_plan": {"mode": "direct", "requested_steps": 2},
                "forecast_points": [
                    {"timestamp": "t1", "value": 1.0},
                    {"timestamp": "t2", "value": 2.0},
                ],
            },
        }
    )
    anomaly_preview = loop._payload_preview(
        {
            "tool_name": "anomaly",
            "success": True,
            "summary": "ok",
            "payload": {
                "anomaly_id": "anomaly_evi",
                "detector_name": "zscore",
                "anomaly_points": [{"timestamp": "t1", "value": 99.0, "score": 3.1}],
                "scores": [{"timestamp": "t1", "score": 3.1}],
                "anomaly_spans": [],
            },
        }
    )

    assert forecast_preview["forecast_status"] == "succeeded"
    assert forecast_preview["forecast_plan"] == {"mode": "direct", "requested_steps": 2}
    assert forecast_preview["forecast_points"] == [
        {"timestamp": "t1", "value": 1.0},
        {"timestamp": "t2", "value": 2.0},
    ]
    assert anomaly_preview["detector_name"] == "zscore"
    assert anomaly_preview["anomaly_points"] == [{"timestamp": "t1", "value": 99.0, "score": 3.1}]
    assert anomaly_preview["anomaly_scores"] == [{"timestamp": "t1", "score": 3.1}]


def test_chat_json_path_can_answer_without_database_context():
    client = _build_client(CasualLLM())
    response = client.post("/api/v1/chat", json={"message": "你好"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["response_kind"] == "final_answer"
    assert payload["used_tools"] == []
    assert payload["trace"] == []
    assert "TSPilot" in payload["answer"]["summary"]
    assert payload["answer"]["title"] is None
    assert payload["answer"]["sections"] == []
    assert payload["answer"]["references"] == []
    assert payload["token_usage"]["totals"]["call_count"] == 1


def test_chat_sse_path_can_answer_without_database_context():
    client = _build_client(CasualLLM())
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"message": "你好", "stream": True},
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert "event: final_answer" in body
    assert "event: tool_call" not in body
    assert "event: tool_result" not in body
    assert "event: step.start" not in body
    assert '"tool": "terminate"' not in body
    assert "TSPilot" in body


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
        "sql_query",
        "code_interpreter",
        "anomaly",
        "forecast",
    ]
    assert payload["answer"]["summary"]
    section_types = [section["section_type"] for section in payload["answer"]["sections"]]
    assert "analysis" in section_types
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
    assert body.count("event: thought") == 6
    assert body.count("event: tool_call") == 6
    assert body.count("event: tool_result") == 6
    assert '"tool": "todowrite"' in body
    assert '"tool": "sql_query"' in body
    assert '"tool": "code_interpreter"' in body
    assert '"tool": "anomaly"' in body
    assert '"tool": "forecast"' in body
    assert '"tool": "terminate"' in body
    assert '"action_input"' in body
    assert '"observation"' in body
    assert '"phase": "intent"' in body
    assert '"phase": "analysis"' in body
    assert '"phase": "answer_assembly"' in body
    assert "event: final_answer" in body
    assert "event: terminate" in body
    assert "event: action" not in body
    assert "event: observation" not in body


def test_runtime_advances_plan_without_repeated_todowrite():
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
    assert payload["used_tools"].count("todowrite") == 1
    assert "sql_query" in payload["used_tools"]
    assert "code_interpreter" in payload["used_tools"]
    assert "anomaly" in payload["used_tools"]
    assert any(event["event_type"] == "todo_updated" for event in payload["trace"])
    assert all(
        "Required actions" not in str(event["payload"].get("summary") or "")
        for event in payload["trace"]
        if event["event_type"] == "tool_result"
    )
    assert any(event["event_type"] == "policy_decision" for event in payload["trace"])
    tool_results = [event for event in payload["trace"] if event["event_type"] == "tool_result"]
    todo_updates = [
        event for event in tool_results
        if isinstance(event["payload"].get("payload_preview"), dict)
        and event["payload"]["payload_preview"].get("todos")
    ]
    assert len(todo_updates) >= 2
    initial_todos = todo_updates[0]["payload"]["payload_preview"]["todos"]
    final_todos = todo_updates[-1]["payload"]["payload_preview"]["todos"]
    assert initial_todos[0]["status"] == "in_progress"
    assert initial_todos[0]["content"] != "1"
    assert sum(1 for todo in final_todos if todo["status"] == "completed") >= 2
    assert final_todos[-1]["status"] in {"in_progress", "completed"}


def test_tool_failure_returns_observation_and_model_can_recover():
    client = _build_client(TodoScopeLLM(), max_iterations=6)
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
    assert payload["status"] == "completed"
    assert payload["used_tools"] == ["todowrite", "forecast", "sql_query", "code_interpreter", "anomaly"]
    observations = [event for event in payload["trace"] if event["event_type"] == "observation"]
    failures = [
        event for event in observations
        if event["payload"]["tool_name"] == "forecast" and event["payload"]["success"] is False
    ]
    assert failures
    assert "forecast" in failures[-1]["payload"]["summary"].lower()


def test_chat_json_path_preserves_multi_query_results_in_final_answer():
    client = _build_client(BitcoinMultiQueryLLM(), max_iterations=10)
    response = client.post(
        "/api/v1/chat",
        json={
            "message": _bitcoin_multi_query_message(),
            "database_context": {
                "database_id": "influxdb2-bitcoin-sample",
                "database_type": "influxdb",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["used_tools"] == ["todowrite", "sql_query", "sql_query", "sql_query", "sql_query"]

    query_results = next(
        section
        for section in payload["answer"]["sections"]
        if section["section_type"] == "query_results"
    )
    content = query_results["content"]
    assert content.count("查询目的：") == 4
    assert content.count("实际返回行数：") == 4
    assert content.count("```flux") == 4
    assert "结果值：count = 2680" in content
    assert "| timestamp | price | code | crypto | description | symbol |" in content
    assert "| timestamp | price | bound |" in content
    assert "| 2023-01-04T23:04:00+00:00 | 168249475888010.0 | earliest |" in content
    assert "| 2023-02-03T22:47:00+00:00 | 23428.6802 | latest |" in content

    observations = [
        event
        for event in payload["trace"]
        if event["event_type"] == "observation"
        and event["payload"]["tool_name"] == "sql_query"
    ]
    assert [
        event["payload"]["payload"]["diagnostics"]["row_count_total"]
        for event in observations
    ] == [1, 5, 5, 2]
    assert len(payload["answer"]["references"]) == 4


def test_chat_sse_path_preserves_multi_query_results_in_final_answer():
    client = _build_client(BitcoinMultiQueryLLM(), max_iterations=10)
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "message": _bitcoin_multi_query_message(),
            "database_context": {
                "database_id": "influxdb2-bitcoin-sample",
                "database_type": "influxdb",
            },
            "stream": True,
        },
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body.count("event: tool_call") == 6
    assert body.count("event: tool_result") == 6
    assert "event: final_answer" in body
    assert "event: terminate" in body
    assert "query_results" in body
    assert "结果值：count = 2680" in body
    assert "实际返回行数：5" in body
    assert "2023-01-04T23:04:00+00:00" in body
    assert "2023-02-03T22:47:00+00:00" in body
    assert "```flux" in body


def _bitcoin_multi_query_message() -> str:
    return (
        "请查询当前数据源中的比特币USD价格数据，并完成以下任务： "
        "1.返回USD价格数据的总记录数； "
        "2.返回按时间升序排列的最早5条原始记录； "
        "3.返回按时间降序排列的最晚5条原始记录； "
        "4.返回整个数据集的最早时间和最晚时间，精确到秒； "
        "5.展示每项结果对应的完整Flux查询语句和实际返回行数。"
    )
