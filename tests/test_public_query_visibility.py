from __future__ import annotations

from core.harness.observation_view import model_observation_view, public_observation_view
from core.harness.action_output import ActionOutputBuilder, ActionOutputBuildInput
from runtime.request_state import public_final_answer
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


def test_model_database_observation_still_hides_query_code():
    model_view = model_observation_view(_database_observation())

    assert model_view is not None
    payload = model_view["payload"]
    assert "query" not in payload
    assert "query_language" not in payload
    assert "for query" not in model_view["summary"]


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
                "code_type": "code_interpreter_v1",
                "code_hash": "sha256:abc",
                "input_evidence_id": "evi_demo",
                "input_row_count": 2,
                "status": "succeeded",
                "summary": "computed",
                "result": {
                    "summary": "computed",
                    "metrics": {"mean": 1.5},
                    "details": {"n": 2},
                },
                "diagnostics": {
                    "runtime_ms": 12.5,
                    "sandbox": "subprocess_code_interpreter_v1",
                    "executed_code": "result = {'summary': 'computed', 'metrics': {'mean': 1.5}, 'details': {'n': 2}}",
                    "executed_code_preview": {
                        "line_count": 1,
                        "char_count": 87,
                        "preview": "result = {'summary': 'computed', 'metrics': {'mean': 1.5}, 'details': {'n': 2}}",
                    },
                },
            },
            result_target="analysis",
            action_input={
                "database_evidence": "latest",
                "analysis_goal": "compute stats",
                "analysis_code": "result = {'summary': 'computed', 'metrics': {'mean': 1.5}, 'details': {'n': 2}}",
            },
            iteration=1,
            request_id="req_demo",
        )
    )

    payload = action_output.view["payload"]
    assert payload["code_preview"].startswith("result =")
    assert payload["analysis_code_chars"] > 0
    assert payload["metrics_preview"] == {"mean": 1.5}
    assert payload["details_preview"] == {"n": 2}
    assert payload["runtime_ms"] == 12.5
    assert payload["code_preview"].startswith("result =")
    assert payload["analysis_code_chars"] == len(payload["code_preview"])
    assert action_output.memory_fragment["action_input"]["analysis_goal"] == "compute stats"
    assert "analysis_code" not in action_output.memory_fragment["action_input"]


def test_refresh_after_transition_exposes_registered_code_facts_and_keeps_artifact():
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
            "produced_facts": [
                {
                    "fact_id": "fact_demo",
                    "fact_key": "mean_value",
                    "name": "Mean value",
                    "fact_type": "aggregate",
                    "statement": "Mean value is 1.5.",
                    "value": 1.5,
                    "method": "code_interpreter",
                    "status": "verified",
                    "evidence_refs": [],
                    "calculation_trace": {},
                    "derived_from": [],
                }
            ],
            "fact_coverage": {
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
    assert payload["produced_fact_count"] == 1
    assert payload["produced_facts_preview"][0]["fact_key"] == "mean_value"
    assert payload["fact_coverage"]["verified"] == ["Mean value"]
    assert payload["code_preview"] == "result = {}"
    assert refreshed.resource_value == initial.resource_value
    assert refreshed.resource_ref == initial.resource_ref
