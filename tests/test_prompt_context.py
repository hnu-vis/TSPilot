from __future__ import annotations

from prompts.data_agent import DataAgentPromptBuilder
from core.harness.observation_view import model_observation_view
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.action_output import ActionOutput
from schemas.task_contract import TaskContract
from schemas.tool import ToolObservation
from app.settings import get_settings


class _EvidenceSpec:
    result_target = "evidence"


def test_sql_query_prompt_keeps_schema_linking_inside_tool():
    builder = DataAgentPromptBuilder()
    system_prompt = builder.build_system_prompt()

    assert "Allowed actions: todowrite, sql_query, code_interpreter, forecast, anomaly, visualization, rag, skill, terminate." in system_prompt
    assert "For terminate" in system_prompt
    assert "For format_answer" not in system_prompt
    assert "For sql_query, provide only natural-language message" in system_prompt
    assert "outer ReAct agent must not write SQL" in system_prompt
    assert "Tool-internal rules live inside tools" in system_prompt


def test_prompt_uses_dbgpt_style_todowrite_planning_rule():
    settings = get_settings()
    request = ChatRequest(
        message=(
            "请查询当前数据源中的比特币USD价格数据，并完成以下任务："
            "1.返回总记录数；2.返回最早5条；3.返回最晚5条；"
            "4.返回最早和最晚时间；5.展示每项查询语句和实际返回行数。"
        ),
        database_context={"database_id": "demo", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    builder = DataAgentPromptBuilder()
    system_prompt = builder.build_system_prompt()
    context = builder.build_context(request_state, conversation_state)
    todowrite_action = next(action for action in context["tools"] if action["action"] == "todowrite")

    assert "3 or more independently verifiable user-visible steps" in todowrite_action["use_when"]
    assert todowrite_action["parameters"] == ["message", "todos", "task_contract?"]


def test_prompt_keeps_anomaly_audit_out_of_code_interpreter_output():
    system_prompt = DataAgentPromptBuilder().build_system_prompt()

    assert "Anomaly Artifact is authoritative" in system_prompt
    assert "independent LLM Insight Binder owns statements" in system_prompt
    assert "metrics/details containers" in system_prompt
    assert "details.outlier_rule" not in system_prompt
    assert "details.excluded_rows" not in system_prompt


def test_prompt_keeps_insight_content_atomic_and_visual_refs_explicit():
    system_prompt = DataAgentPromptBuilder().build_system_prompt()

    assert "Do not create separate method, basis, provenance, input-count, or display-context Insights" in system_prompt
    assert "Put only visually inspectable conclusion Insights in source_refs" in system_prompt
    assert "prefer insight:<exact insight_key>" in system_prompt
    assert "never reconstruct or abbreviate an opaque identifier" in system_prompt


def test_prompt_context_keeps_runtime_decision_state_out_of_model_context():
    settings = get_settings()
    request = ChatRequest(
        message="找出最高价和最低价。",
        database_context={"database_id": "demo", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert "decision_frame" not in context["state"]
    assert "completion_state" not in context["state"]
    assert "requested_capabilities" not in context["state"]
    assert "semantic_repair_directive" not in context["state"]


def test_outer_react_context_omits_internal_presentation_inventory_and_compacts_history():
    settings = get_settings()
    request = ChatRequest(message="展示完整趋势", database_context={"database_id": "demo", "database_type": "unit"})
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    request_state.memory_fragments = [{
        "iteration": 1,
        "action": "sql_query",
        "action_input": {"message": "repeat the full user request"},
        "observation": {
            "tool": "sql_query",
            "success": True,
            "summary": "Loaded 2680 rows.",
            "resource_ref": "evidence:evi_full",
            "data_preview": {"rows": [{"timestamp": "t0", "value": 1.0}]},
            "coverage_delta": {"can_answer": False, "missing_outputs": ["visualization"]},
        },
        "resource_ref": "evidence:evi_full",
        "status": "succeeded",
    }]
    request_state.latest_action_output = ActionOutput(
        tool_name="visualization",
        success=True,
        content="Created visualization.",
        observations={"tool": "visualization", "success": True, "visualization_ids": ["viz_1"]},
        meta={"iteration": 2},
    )

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert set(context["artifacts"]) == {"refs", "facts"}
    history = context["recent_trajectory"][0]
    assert history == {
        "iteration": 1,
        "action": "sql_query",
        "status": "succeeded",
        "resource_ref": "evidence:evi_full",
        "summary": "Loaded 2680 rows.",
        "coverage_delta": {"can_answer": False, "missing_outputs": ["visualization"]},
    }
    assert "action_input" not in history
    assert "data_preview" not in str(history)


def test_prompt_context_exposes_task_contract_as_state():
    settings = get_settings()
    request = ChatRequest(
        message="返回总数和最早 3 条。",
        database_context={"database_id": "demo", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    request_state.task_contract = TaskContract.model_validate(
        {
            "source": "llm",
            "goal": "返回总数和最早 3 条",
            "required_outputs": [
                {"id": "total_count", "description": "总记录数"},
                {"id": "earliest_rows", "description": "最早 3 条记录"},
            ],
            "constraints": {},
            "assumptions": [],
            "evidence_quality_notes": [],
        }
    )

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    system_prompt = DataAgentPromptBuilder().build_system_prompt()

    assert context["state"]["task_contract"]["required_outputs"][0]["id"] == "total_count"
    assert "task_contract" in system_prompt
    assert "Task-contract coverage is semantic" in system_prompt


def test_prompt_context_contains_only_the_compact_outer_contract():
    settings = get_settings()
    request = ChatRequest(message="analysis", database_context={"database_id": "demo", "database_type": "unit"})
    state = build_request_state(request, settings)
    conversation = build_conversation_state(request, state.conversation_id or "conv")

    context = DataAgentPromptBuilder().build_context(state, conversation)

    assert set(context) == {"task", "tools", "state", "artifacts", "last_observation", "recent_trajectory"}
    assert set(context["state"]) == {
        "execution", "next_action_constraints", "todo_progress", "task_contract", "insight_state",
    }
    assert set(context["state"]["execution"]) == {"iteration", "max_iterations"}


def test_prompt_context_does_not_expose_sql_query_task_coverage_as_global_guidance():
    settings = get_settings()
    request = ChatRequest(
        message="返回总数和最早 3 条记录。",
        database_context={"database_id": "demo", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    payload = {
        "evidence_id": "evi_count",
        "result_type": "table",
        "database": "demo",
        "query_language": "flux",
        "query": "from(...) |> count()",
        "summary": "Loaded 1 row.",
        "data": {"rows": [{"count": 10}]},
        "columns": ["count"],
        "metadata": {},
        "diagnostics": {
            "task_coverage": {
                "satisfied": ["已返回总数"],
                "missing_or_uncertain": ["尚未返回最早 3 条记录"],
                "next_action_hint": "继续按时间升序查询 3 条原始记录",
                "requires_followup": True,
            }
        },
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="Loaded 1 row.", payload=payload),
        payload,
        _EvidenceSpec(),
    )

    observation = model_observation_view(request_state.observations[-1])
    request_state.latest_action_output = ActionOutput(
        tool_name="sql_query", success=True, content="Loaded 1 row.", observations=observation or {},
    )
    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert "diagnostics.task_coverage" not in DataAgentPromptBuilder().build_system_prompt()
    assert "task_coverage" not in str(context["last_observation"])
    assert "missing_or_uncertain" not in str(context)


def test_prompt_builder_summarizes_heavy_context():
    settings = get_settings()
    request = ChatRequest(
        message="分析季节性",
        database_context={"database_id": "demo", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    full_payload = {
        "evidence_id": "evi_demo",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo",
        "summary": "Loaded 100 points.",
        "data": {
            "points": [{"timestamp": f"2023-01-01T00:{i:02d}:00Z", "value": float(i)} for i in range(100)],
            "rows": [{"timestamp": f"2023-01-01T00:{i:02d}:00Z", "value": float(i)} for i in range(100)],
            "series": [],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {},
    }
    apply_observation(request_state, observation, full_payload, _EvidenceSpec())
    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    sql_action = next(action for action in context["tools"] if action["action"] == "sql_query")

    assert any(action["action"] == "todowrite" for action in context["tools"])
    assert any(action["action"] == "code_interpreter" for action in context["tools"])
    assert any(action["action"] == "terminate" for action in context["tools"])
    assert sql_action["parameters"] == ["message", "purpose?", "insight_requests?"]
    assert context["artifacts"]["refs"]["database_evidence"] == ["evidence:evi_demo"]
    fact = context["artifacts"]["facts"][0]
    assert fact["source_ref"] == "evidence:evi_demo"
    assert fact["row_count"] == 100
    assert len(fact["records"]) == 12
    assert "data" not in fact
    assert "rows" not in str(context)


def test_prompt_hides_presentation_point_budget_from_outer_reasoning():
    settings = get_settings()
    request = ChatRequest(
        message="分析完整趋势并可视验证",
        database_context={"database_id": "demo", "database_type": "unit"},
        constraints={"max_points": 48, "timezone": "UTC"},
    )
    state = build_request_state(request, settings)
    conversation = build_conversation_state(request, state.conversation_id or "conv")

    context = DataAgentPromptBuilder().build_context(state, conversation)

    assert context["task"]["constraints"] == {"timezone": "UTC"}
    assert "max_points" not in str(context["task"])


def test_prompt_builder_exposes_compact_sql_observation_receipt():
    settings = get_settings()
    request = ChatRequest(message="算平均值")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    raw_observation = ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="ok",
            payload={
                "evidence_id": "evi_sql",
                "query_language": "sql",
                "query": "SELECT AVG(value) AS avg_value FROM metrics",
                "columns": ["avg_value"],
                "data": {"rows": [{"avg_value": 12.3}, {"avg_value": 13.4}]},
                "diagnostics": {
                    "summary_stats": {"rows_count": 2},
                    "sql_query": {"execution_time_ms": 8},
                    "irrelevant": "hidden",
                },
            },
        )
    request_state.observations.append(raw_observation)
    request_state.latest_action_output = ActionOutput(
        tool_name="sql_query",
        success=True,
        content="ok",
        observations=model_observation_view(raw_observation) or {},
    )

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    payload = context["last_observation"]["payload"]

    assert payload["columns"] == ["avg_value"]
    assert payload["row_count"] == 2
    assert "query" not in payload
    assert "query_language" not in payload
    assert "data_preview" not in payload
    assert "diagnostics" not in payload


def test_prompt_builder_uses_compact_action_output_trajectory():
    settings = get_settings()
    request = ChatRequest(message="算平均值")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    request_state.memory_fragments.append({
        "iteration": 1,
        "action": "sql_query",
        "action_input": {"message": "查询平均值", "history": [{"role": "user"}]},
        "observation": {
            "summary": "Loaded 1 row.",
            "resource_ref": "evidence:evi_avg",
            "coverage_delta": {"can_answer": False, "missing_outputs": ["analysis"]},
            "data": {"rows": [{"avg": 12.3}]},
        },
        "resource_ref": "evidence:evi_avg",
        "status": "succeeded",
    })

    builder = DataAgentPromptBuilder()
    prompt = builder.build_user_prompt(request_state, conversation_state)
    context = builder.build_context(request_state, conversation_state)

    trajectory = context["recent_trajectory"][0]
    assert trajectory == {
        "iteration": 1,
        "action": "sql_query",
        "status": "succeeded",
        "resource_ref": "evidence:evi_avg",
        "summary": "Loaded 1 row.",
        "coverage_delta": {"can_answer": False, "missing_outputs": ["analysis"]},
    }
    assert '"action": "sql_query"' in prompt
    assert '"summary": "Loaded 1 row."' in prompt
    assert "action_intention" not in prompt
    assert "action_reason" not in prompt
    assert '"history"' not in prompt
    assert '"data"' not in prompt


def test_prompt_builder_bounds_long_query_context():
    settings = get_settings()
    request = ChatRequest(message="复杂查询")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    long_query = "SELECT " + ", ".join(f"{i} AS c{i}" for i in range(2000))
    full_payload = {
        "evidence_id": "evi_long",
        "result_type": "table",
        "database": "demo",
        "query_language": "sql",
        "query": long_query,
        "summary": "x" * 5000,
        "data": {"rows": [{"value": 1.0}]},
        "columns": ["value"],
        "metadata": {"raw_schema": "y" * 5000},
        "diagnostics": {
            "query_trace": {
                "rendered_query": {"query_text": long_query},
                "large": ["z" * 1000 for _ in range(20)],
            }
        },
    }
    apply_observation(request_state, observation, full_payload, _EvidenceSpec())

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert long_query not in str(context)
    assert "raw_schema" not in str(context)
    assert "query_trace" not in str(context)
    assert context["artifacts"]["refs"]["database_evidence"] == ["evidence:evi_long"]
