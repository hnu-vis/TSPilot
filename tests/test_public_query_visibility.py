from __future__ import annotations

from core.harness.observation_view import model_observation_view, public_observation_view
from core.harness.action_output import ActionOutputBuilder, ActionOutputBuildInput
from runtime.request_state import public_final_answer
from runtime.request_state import _compact_analysis_value
from schemas.output import AnswerReference, AnswerSection, FinalAnswer
from schemas.tool import ToolObservation


def _database_observation() -> ToolObservation:
    return ToolObservation(
        tool_name="sql_query",
        success=True,
        summary="Loaded 1 row for query 'from(bucket:\"bitcoin\") |> range(start: -30d) |> max()'.",
        payload={
            "evidence_id": "evi_demo",
            "result_type": "timeseries",
            "database": "influxdb2-bitcoin-sample",
            "query_language": "flux",
            "query": 'from(bucket:"bitcoin")\n  |> range(start: -30d)\n  |> max()',
            "summary": "Loaded 1 row.",
            "data": {
                "rows": [{"timestamp": "2023-01-01T00:00:00Z", "value": 12.3}],
                "points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 12.3}],
            },
            "columns": ["timestamp", "value"],
            "diagnostics": {
                "summary_stats": {"rows_count": 1, "points_count": 1, "series_count": 1},
                "query_trace": {"raw": "internal"},
            },
        },
    )


def test_public_database_observation_keeps_query_and_preview_rows():
    public_view = public_observation_view(_database_observation())

    assert public_view is not None
    payload = public_view["payload"]
    assert payload["query_language"] == "flux"
    assert 'from(bucket:"bitcoin")' in payload["query"]
    assert payload["data_preview"]["rows"] == [{"timestamp": "2023-01-01T00:00:00Z", "value": 12.3}]
    assert "query_trace" not in payload["diagnostics"]
    assert "for query" not in public_view["summary"]


def test_public_database_observation_keeps_complete_long_query():
    observation = _database_observation()
    long_query = 'from(bucket:"bitcoin")\n' + "\n".join(
        f'  |> filter(fn: (r) => r["field_{index}"] != "")' for index in range(150)
    )
    assert len(long_query) > 5000
    observation.payload["query"] = long_query

    public_view = public_observation_view(observation)

    assert public_view is not None
    assert public_view["payload"]["query"] == long_query


def test_model_database_observation_still_hides_query_code():
    model_view = model_observation_view(_database_observation())

    assert model_view is not None
    payload = model_view["payload"]
    assert "query" not in payload
    assert "query_language" not in payload
    assert "for query" not in model_view["summary"]
    assert "data_preview" not in payload
    assert "raw_available_in_artifact" not in payload
    assert payload["row_count"] == 1
    assert payload["point_count"] == 1


def test_prompt_safe_evidence_drops_snapshot_paths_and_planning_diagnostics():
    from runtime.request_state import _build_prompt_safe_evidence
    from schemas.database import DatabaseEvidence

    safe = _build_prompt_safe_evidence(
        DatabaseEvidence(
            evidence_id="evi_safe",
            result_type="timeseries",
            database="demo",
            summary="Loaded rows.",
            data={"rows": [{"timestamp": "t0", "value": 1.0}]},
            diagnostics={
                "is_full_fidelity": True,
                "snapshot_ref": {"uri": "/private/request/artifact.json"},
                "schema_linking_generation": {"internal": "large"},
                "task_coverage": {
                    "missing": ["maximum"],
                    "next_action_hint": "code_interpreter",
                    "query_task_contract": {"downstream_action": "code_interpreter"},
                    "executed_query": "select secret",
                },
            },
        )
    )

    assert safe.diagnostics["is_full_fidelity"] is True
    assert safe.diagnostics["task_coverage"]["missing"] == ["maximum"]
    assert "snapshot_ref" not in safe.diagnostics
    assert "schema_linking_generation" not in safe.diagnostics
    assert "executed_query" not in safe.diagnostics["task_coverage"]


def test_terminal_observation_keeps_receipt_not_full_visualization_descriptor():
    from runtime.request_state import enrich_observation_payload

    class _PresentationSpec:
        result_target = "presentation"

    observation = ToolObservation(tool_name="terminate", success=True, summary="done", payload={})
    enriched = enrich_observation_payload(
        type("State", (), {"completion_state": {}})(),
        observation,
        {
            "title": "Result",
            "summary": "done",
            "visualizations": [
                {
                    "visualization_id": "viz_demo",
                    "source_refs": ["view:evidence:evi_demo:default"],
                    "datasets": [{"row_count": 2680}],
                }
            ],
        },
        _PresentationSpec(),
    )

    assert enriched.payload == {
        "title": "Result",
        "visualization_ids": ["viz_demo"],
    }


def test_action_output_keeps_only_canonical_resource_receipt():
    payload = _database_observation().payload
    output = ActionOutputBuilder().build(ActionOutputBuildInput(
        tool_name="sql_query",
        success=True,
        summary="Loaded 1 row.",
        full_payload=payload,
        result_target="evidence",
        action_input={"message": "load data"},
        iteration=1,
        request_id="req_receipt",
    ))

    assert output.resource_ref == "evidence:evi_demo"
    assert output.resource_value == {"resource_ref": "evidence:evi_demo"}
    assert output.observations["resource_ref"] == "evidence:evi_demo"
    assert "artifact_ref" not in output.observations
    assert "data_preview" not in output.observations


def test_analysis_observation_compacts_long_timeseries_but_keeps_small_extreme_locator():
    details = {
        "timeseries_data": [
            {"timestamp": f"2023-01-01T00:{index:02d}:00Z", "value": index}
            for index in range(20)
        ],
        "max_value_points": [
            {"timestamp": "2023-01-01T00:19:00Z", "value": 19}
        ],
    }

    compact = _compact_analysis_value(details)

    assert compact["timeseries_data"] == {
        "item_count": 20,
        "available_in_artifact": True,
    }
    assert compact["max_value_points"] == [
        {"timestamp": "2023-01-01T00:19:00Z", "value": 19}
    ]


def test_public_final_answer_preserves_query_evidence_fields():
    answer = FinalAnswer(
        summary="最大值是 12.3。",
        sections=[
            AnswerSection(
                section_type="query_results",
                heading="查询结果",
                content="查询结果。",
                structured_payload={
                    "items": [
                        {
                            "query_language": "flux",
                            "query": 'from(bucket:"bitcoin") |> max()',
                            "rows_preview": [{"value": 12.3}],
                            "query_trace": {"raw": "internal"},
                        }
                    ]
                },
            )
        ],
        references=[
            AnswerReference(
                source_type="query",
                source_id="evi_demo",
                label="Bitcoin USD maximum",
                evidence={
                    "query_language": "flux",
                    "query": 'from(bucket:"bitcoin") |> max()',
                    "row_count": 1,
                    "query_trace": {"raw": "internal"},
                },
            ),
            AnswerReference(
                source_type="analysis",
                source_id="analysis_demo",
                label="Analysis",
                evidence={"query": "internal helper query", "result": {"value": 12.3}},
            ),
        ],
    )

    public_answer = public_final_answer(answer)

    item = public_answer.sections[0].structured_payload["items"][0]
    assert item["query_language"] == "flux"
    assert 'from(bucket:"bitcoin")' in item["query"]
    assert "query_trace" not in item
    query_reference = public_answer.references[0].evidence
    assert query_reference["query_language"] == "flux"
    assert 'from(bucket:"bitcoin")' in query_reference["query"]
    assert "query_trace" not in query_reference
    assert "query" not in public_answer.references[1].evidence


def test_public_final_answer_preserves_user_visible_query_sections():
    query = 'from(bucket:"bitcoin") |> range(start: -30d) |> max()'
    answer = FinalAnswer(
        summary=f"最大值是 12.3。Query statement:\n```flux\n{query}\n```",
        sections=[
            AnswerSection(
                section_type="query",
                heading="查询",
                content=f"```flux\n{query}\n```",
                structured_payload={"query_language": "flux", "database": "bitcoin"},
            ),
            AnswerSection(
                section_type="query_results",
                heading="查询结果",
                content=f"查询语句：\n```flux\n{query}\n```",
                structured_payload={
                    "items": [
                        {
                            "query_language": "flux",
                            "query": query,
                            "rows_preview": [{"value": 12.3}],
                        }
                    ]
                },
            ),
            AnswerSection(
                section_type="analysis",
                heading="分析",
                content=f"分析基于内部查询：\n```flux\n{query}\n```",
            ),
        ],
    )

    public_answer = public_final_answer(answer)

    assert "[query omitted]" not in public_answer.sections[0].content
    assert "[query omitted]" not in public_answer.sections[1].content
    assert query in public_answer.sections[0].content
    assert query in public_answer.sections[1].content
    assert public_answer.sections[1].structured_payload["items"][0]["query"] == query
    assert public_answer.summary == "最大值是 12.3。"
    assert public_answer.sections[2].content == "分析基于内部查询："


def test_public_code_interpreter_action_output_keeps_bounded_code_preview():
    action_output = ActionOutputBuilder().build(
        ActionOutputBuildInput(
            tool_name="code_interpreter",
            success=True,
            summary="computed",
            full_payload={
                "analysis_id": "ana_demo",
                "analysis_goal": "compute stats",
                "code_type": "code_interpreter_v2",
                "code_hash": "sha256:abc",
                "input_evidence_id": "evi_demo",
                "input_row_count": 2,
                "status": "succeeded",
                "summary": "computed",
                "computed_insights": [{
                    "insight_key": "mean", "value": 1.5,
                    "calculation_trace": {"operation": "mean", "n": 2},
                }],
                "derived_evidence": [],
                "diagnostics": {
                    "runtime_ms": 12.5,
                    "sandbox": "subprocess_code_interpreter_v2",
                    "executed_code": "result = {'computed_insights': [], 'derived_evidence': []}",
                    "executed_code_preview": {
                        "line_count": 1,
                        "char_count": 87,
                        "preview": "result = {'computed_insights': [], 'derived_evidence': []}",
                    },
                },
            },
            result_target="analysis",
            action_input={
                "database_evidence": "latest",
                "analysis_goal": "compute stats",
                "analysis_code": "result = {'computed_insights': [], 'derived_evidence': []}",
            },
            iteration=1,
            request_id="req_demo",
        )
    )

    payload = action_output.view["payload"]
    assert payload["code_preview"].startswith("result =")
    assert payload["analysis_code_chars"] > 0
    assert payload["computed_insights"][0]["value"] == 1.5
    assert payload["runtime_ms"] == 12.5
    assert payload["code_preview"].startswith("result =")
    assert payload["analysis_code_chars"] == len(payload["code_preview"])
    assert action_output.memory_fragment["action_input"]["analysis_goal"] == "compute stats"
    assert "analysis_code" not in action_output.memory_fragment["action_input"]


def test_refresh_after_transition_exposes_registered_code_insights_and_keeps_artifact():
    builder = ActionOutputBuilder()
    initial = builder.build(
        ActionOutputBuildInput(
            tool_name="code_interpreter",
            success=True,
            summary="computed",
            full_payload={"analysis_id": "ana_demo", "status": "succeeded", "result": {}},
            result_target="analysis",
            action_input={"analysis_code": "result = {}"},
            iteration=2,
            request_id="req_demo",
        )
    )
    observation = ToolObservation(
        tool_name="code_interpreter",
        success=True,
        summary="computed",
        payload={
            "analysis_id": "ana_demo",
            "status": "succeeded",
            "result": {},
            "produced_insights": [
                {
                    "insight_id": "insight_demo",
                    "insight_key": "mean_value",
                    "name": "Mean value",
                    "insight_type": "aggregate",
                    "statement": "Mean value is 1.5.",
                    "value": 1.5,
                    "method": "code_interpreter",
                    "status": "verified",
                    "evidence_refs": [],
                    "calculation_trace": {},
                    "derived_from": [],
                }
            ],
            "insight_coverage": {
                "requested": ["Mean value"],
                "verified": ["Mean value"],
                "missing": [],
                "unavailable": [],
                "rejected": [],
                "partial": [],
            },
        },
    )

    refreshed = builder.refresh_after_transition(
        initial,
        observation,
        action_input={},
        request_id="req_demo",
    )

    payload = refreshed.view["payload"]
    assert payload["produced_insight_count"] == 1
    assert payload["produced_insights_preview"][0]["insight_key"] == "mean_value"
    assert payload["insight_coverage"]["verified"] == ["Mean value"]
    assert payload["code_preview"] == "result = {}"
    assert refreshed.resource_value == initial.resource_value
    assert refreshed.resource_ref == initial.resource_ref
