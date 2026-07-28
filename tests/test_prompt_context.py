from __future__ import annotations

from core.database.connector import ColumnSchema, DatabaseSchema, TableSchema
from core.database.schema import schema_preview
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.task_contract import TaskContract
from schemas.tool import ReActTranscriptStep, ToolObservation
from schemas.visualization import VisualizationPayload
from app.settings import get_settings


class _EvidenceSpec:
    result_target = "evidence"


def test_sql_query_prompt_keeps_schema_linking_inside_tool():
    builder = DataAgentPromptBuilder()
    system_prompt = builder.build_system_prompt()

    assert "Allowed actions: todowrite, sql_query, code_interpreter, forecast, anomaly, rag, skill, terminate." in system_prompt
    assert "For terminate" in system_prompt
    assert "For format_answer" not in system_prompt
    assert "Use automatic message-based input for normal database requests" in system_prompt
    assert "field confirmation, query generation, or query planning" in system_prompt
    assert "schema linking" not in system_prompt.lower()
    assert "schema linking participates as auxiliary grounding" not in system_prompt
    assert "Do not write an explicit database query from user-facing names" not in system_prompt
    assert "Context is budgeted" in system_prompt
    assert "diagnostics.prompt_sampling" in system_prompt


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
    todowrite_action = next(action for action in context["available_actions"] if action["action"] == "todowrite")

    assert "3 or more independently verifiable user-visible steps" in system_prompt
    assert "BEFORE starting work" in system_prompt
    assert "query text, row counts" in system_prompt
    assert "one grounded action can fully cover every deliverable" not in system_prompt
    assert "3 or more independently verifiable user-visible steps" in todowrite_action["use_when"]


def test_prompt_requires_transparent_outlier_treatment_in_code_analysis():
    system_prompt = DataAgentPromptBuilder().build_system_prompt()

    assert "Do not silently replace raw metrics with adjusted metrics" in system_prompt
    assert "do not use a first-difference/spike detector to clean level metrics" in system_prompt
    assert "excluded_rows must be the row list, not only a count" in system_prompt
    assert "do not concatenate aliases or double-count duplicate timestamp/value records" in system_prompt
    assert "details.outlier_rule" in system_prompt
    assert "details.threshold_or_formula" in system_prompt
    assert "details.excluded_rows" in system_prompt
    assert "details.raw_metrics" in system_prompt
    assert "details.adjusted_metrics" in system_prompt


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
    assert "required_outputs" in system_prompt


def test_prompt_context_handles_diagnostics_without_data_preview():
    payload = {
        "summary": "analysis ok",
        "diagnostics": {"runtime_ms": 12, "summary_stats": {"rows_count": 0}},
    }

    summarized = DataAgentPromptBuilder()._summarize_observation_payload(payload)

    assert "runtime_ms" not in summarized["diagnostics"]
    assert summarized["diagnostics"]["prompt_sampling"]["visible_counts"] == {}


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

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert "diagnostics.task_coverage" not in DataAgentPromptBuilder().build_system_prompt()
    assert "task_coverage" not in context["evidence"]["latest"]["diagnostics"]
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
    request_state.visualizations = [
        VisualizationPayload(
            visualization_id="viz_demo",
            visualization_type="chart",
            visualization_kind="line",
            renderer="linechart",
            title="Demo",
            summary="Demo chart",
            chart={
                "x_axis_data": [f"t{i}" for i in range(100)],
                "series_data": [{"name": "value", "data": list(range(100))}],
            },
        )
    ]

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    sql_action = next(action for action in context["available_actions"] if action["action"] == "sql_query")

    assert any(action["action"] == "todowrite" for action in context["available_actions"])
    assert any(action["action"] == "code_interpreter" for action in context["available_actions"])
    assert any(action["action"] == "terminate" for action in context["available_actions"])
    assert not any(action["action"] == "format_answer" for action in context["available_actions"])
    assert "prefer message" in sql_action["input"]
    assert "schema-linked generation" not in sql_action["input"]
    assert "automatic database querying" in sql_action["input"]
    assert context["state"]["execution"]["artifacts"]["has_database_evidence"] is True
    assert context["state"]["execution"]["last_successful_tool"] == "sql_query"
    assert len(context["evidence"]["latest"]["data"]["points"]) <= 8
    sampling = context["evidence"]["latest"]["diagnostics"]["prompt_sampling"]
    assert sampling["sampled_for_prompt"] is True
    assert sampling["full_counts"]["points_count"] == 100
    assert sampling["visible_counts"]["points_count"] == 8
    assert context["evidence"]["prior_queries"] == []
    assert context["outputs"]["visualizations"][0]["chart_summary"]["x_axis_count"] == 100
    assert "chart" not in context["outputs"]["visualizations"][0]
    assert "latest_database_evidence" not in context
    assert "visualizations" not in context


def test_initial_context_includes_bounded_schema_hint_for_reference_dataset():
    settings = get_settings()
    request = ChatRequest(
        message="总共有多少条数据？",
        database_context={"database_id": "influxdb2-energydata", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    schema_hint = context["task"]["database_context"]["schema_hint"]

    assert schema_hint["query_language"] == "flux"
    assert schema_hint["tables_or_measurements"][0]["name"] == "home_energy_environment"
    assert schema_hint["tables_or_measurements"][0]["row_count"] == 19735
    assert "appliances_energy_wh" in schema_hint["tables_or_measurements"][0]["field_columns"]
    assert len(schema_hint["tables_or_measurements"][0]["sample_rows"]) == 3


def test_schema_preview_promotes_reference_dataset_metadata_to_table_preview():
    schema = DatabaseSchema(
        database="demo",
        tables=[
            TableSchema(
                name="metrics",
                columns=[
                    ColumnSchema(name="_time", data_type="datetime"),
                    ColumnSchema(name="value", data_type="float"),
                    ColumnSchema(name="host", data_type="string"),
                ],
            )
        ],
        metadata={
            "reference_dataset": {
                "measurement": "metrics",
                "row_count": 42,
                "time_range": {"start": "2024-01-01T00:00:00Z", "stop": "2024-01-02T00:00:00Z"},
                "sample_rows": [{"timestamp": "2024-01-01 00:00:00", "value": "1.0"}],
            },
            "value_domains": {"metrics": {"_field": ["value"], "host": ["a", "b"]}},
        },
    )

    preview = schema_preview(schema)
    table = preview["tables_or_measurements"][0]

    assert table["row_count"] == 42
    assert table["field_values"] == ["value"]
    assert table["sample_rows"] == [{"timestamp": "2024-01-01 00:00:00", "value": "1.0"}]
    assert preview["labels_or_tags"] == [{"table": "metrics", "name": "host", "values": ["a", "b"]}]


def test_prompt_builder_exposes_sql_observation_details():
    settings = get_settings()
    request = ChatRequest(message="算平均值")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    request_state.observations.append(
        ToolObservation(
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
    )

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    payload = context["recent_observations"][0]["payload"]

    assert payload["query"] == "SELECT AVG(value) AS avg_value FROM metrics"
    assert payload["query_language"] == "sql"
    assert payload["columns"] == ["avg_value"]
    assert payload["data_preview"]["rows"] == [{"avg_value": 12.3}, {"avg_value": 13.4}]
    assert payload["diagnostics"]["summary_stats"] == {"rows_count": 2}
    assert "irrelevant" not in payload["diagnostics"]


def test_prompt_builder_prefers_structured_react_transcript():
    settings = get_settings()
    request = ChatRequest(message="算平均值")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    request_state.react_transcript.append(
        ReActTranscriptStep(
            iteration=1,
            question="算平均值",
            thought="需要查询数据库。",
            action_intention="查询均值",
            action_reason="当前没有证据",
            action="sql_query",
            action_input={"query": "SELECT AVG(value) FROM metrics"},
            observation=ToolObservation(
                tool_name="sql_query",
                success=True,
                summary="Loaded 1 row.",
                payload={
                    "query": "SELECT AVG(value) FROM metrics",
                    "query_language": "sql",
                    "data": {"rows": [{"avg": 12.3}]},
                },
            ),
        )
    )

    builder = DataAgentPromptBuilder()
    prompt = builder.build_user_prompt(request_state, conversation_state)
    context = builder.build_context(request_state, conversation_state)

    assert "Question: 算平均值" in prompt
    assert "Thought: 需要查询数据库。" in prompt
    assert "Action Intention: 查询均值" in prompt
    assert "Action Reason: 当前没有证据" in prompt
    assert "Action: sql_query" in prompt
    assert "Observation:" in prompt
    assert context["recent_react_transcript"][0]["action"] == "sql_query"


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
    evidence = context["evidence"]["latest"]

    assert len(evidence["query"]) < len(long_query)
    assert "truncated" in evidence["query"]
    assert context["evidence"]["prior_queries"] == []
    assert "truncated" in evidence["metadata"]["raw_schema"]
    assert "truncated_items" in evidence["diagnostics"]["query_trace"]["large"][-1]
