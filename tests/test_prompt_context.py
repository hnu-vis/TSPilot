from __future__ import annotations

from core.database.connector import ColumnSchema, DatabaseSchema, TableSchema
from core.database.schema import schema_preview
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.insight import InsightResult
from schemas.tool import ToolObservation
from schemas.visualization import VisualizationPayload
from app.settings import get_settings


class _EvidenceSpec:
    result_target = "evidence"


def test_sql_query_prompt_prefers_schema_linked_automatic_generation():
    builder = DataAgentPromptBuilder()
    system_prompt = builder.build_system_prompt()

    assert "Default to message-only automatic planning" in system_prompt
    assert "schema linking participates as auxiliary grounding" in system_prompt
    assert "Do not write an explicit database query from user-facing names" in system_prompt
    assert "Context is budgeted" in system_prompt
    assert "diagnostics.prompt_sampling" in system_prompt


def test_prompt_context_exposes_context_budget_rule():
    settings = get_settings()
    request = ChatRequest(
        message="找出最高价和最低价。",
        database_context={"database_id": "demo", "database_type": "influxdb"},
    )
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert "semantic_repair_directive" not in context["state"]["decision_frame"]
    assert "Prompt context contains bounded previews only" in context["state"]["decision_frame"]["context_budget_rule"]


def test_prompt_context_handles_diagnostics_without_data_preview():
    payload = {
        "summary": "analysis ok",
        "diagnostics": {"runtime_ms": 12, "summary_stats": {"rows_count": 0}},
    }

    summarized = DataAgentPromptBuilder()._summarize_observation_payload(payload)

    assert summarized["diagnostics"]["runtime_ms"] == 12
    assert summarized["diagnostics"]["prompt_sampling"]["visible_counts"] == {}


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
    request_state.latest_insight = InsightResult(
        insight_id="ins_demo",
        verified_facts=[],
        visualizations=[
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
        ],
    )
    request_state.visualizations = request_state.latest_insight.visualizations

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    sql_action = next(action for action in context["available_actions"] if action["action"] == "sql_query")

    assert any(action["action"] == "todowrite" for action in context["available_actions"])
    assert "prefer message" in sql_action["input"]
    assert "schema-linked generation" in sql_action["input"]
    assert context["state"]["execution"]["artifacts"]["has_database_evidence"] is True
    assert context["state"]["execution"]["last_successful_tool"] == "sql_query"
    guidance = " ".join(context["state"]["decision_frame"]["recommended_next_action_types"])
    assert "continue with explicit read-only queries" in guidance
    assert "A prior sql_query does not force insight" in context["state"]["decision_frame"]["sql_loop_rule"]
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
