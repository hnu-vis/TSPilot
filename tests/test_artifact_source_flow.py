from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.artifact_sources import primary_analysis_input, resolve_artifact_sources, source_prompt_manifest
from core.harness import build_action_space, build_observation_frame
from core.completion import apply_previous_observation_assessment
from core.visualization import PresentationCatalog, VisualizationArtifactStore
from schemas.database import DatabaseEvidence
from schemas.key_insight import InsightEvidenceRef, InsightItem, KeyInsight, KeyInsightRequest
from schemas.agent_turn import PreviousObservationAssessment
from schemas.state import RequestStateModel
from schemas.timeseries import AnomalyResult, ForecastResult
from schemas.tool import ToolObservation
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool
from tools.base import StructuredToolError
from tools.forecast import ForecastInput, ForecastTool
from tools.visualization import VisualizationEvidenceRequest, VisualizationInput, VisualizationTool
from runtime.request_state import apply_observation
from runtime.tool_executor import ToolExecutor
from runtime.action_policy import _latest_visualization_source_request


class _BinderLlm:
    async def ainvoke(self, _messages):
        return SimpleNamespace(content=json.dumps({
            "bindings": [{
                "insight_key": "forecast_change",
                "supported": True,
                "unsupported_reason": None,
                "statement": "Forecast rises by 30%.",
                "derived_from": [],
            }],
        }))


class _NeedsForecastCalculationLlm:
    async def ainvoke(self, messages):
        requirement = {
            "required_action": "code_interpreter",
            "purpose": "Calculate forecast direction and percentage change",
            "message": None,
            "required_shape": "scalar",
            "required_fields": ["direction", "change_pct"],
            "required_properties": ["derived from forecast endpoints"],
            "input_evidence": "view:forecast:forecast_demo:points",
            "input_source_refs": ["view:forecast:forecast_demo:points"],
            "insight_requests": [{
                "insight_key": "forecast_change",
                "name": "Forecast change",
                "insight_type": "change",
            }],
        }
        payload = {
            "visual_question": None,
            "interpretation": None,
            "target_insight_ids": [],
            "charts": [],
            "required_data_request": requirement,
        }
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


class _UnsafeForecastInputLlm:
    async def ainvoke(self, _messages):
        return SimpleNamespace(content=json.dumps({
            "safe_to_forecast": False,
            "reason": "A few edge values have a corruption-like scale discontinuity.",
            "quality_issues": ["scale discontinuity", "dominant edge points"],
        }))


class _CapturingSafeForecastInputLlm:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=json.dumps({
            "safe_to_forecast": True,
            "reason": "Both local edge windows are coherent.",
            "quality_issues": [],
        }))


class _MultiSourceAnalysisLlm:
    async def ainvoke(self, messages):
        system = str(messages[0][1])
        if "Choose the one grounded artifact source" in system:
            return SimpleNamespace(content=json.dumps({
                "source_ref": "forecast:forecast_demo",
                "reason": "The requested endpoint change is owned by the forecast artifact.",
            }))
        if "computation-only Code Interpreter" in system:
            return SimpleNamespace(content=json.dumps({
                "code": (
                    "start = float(df.iloc[0][value_col])\n"
                    "end = float(df.iloc[-1][value_col])\n"
                    "result = {'computed_insights': [{'insight_key': 'forecast_change', "
                    "'value': (end - start) / start * 100, "
                    "'calculation_trace': {'start': start, 'end': end}}], 'derived_evidence': []}"
                ),
            }))
        return SimpleNamespace(content=json.dumps({
            "bindings": [{
                "insight_key": "forecast_change",
                "supported": True,
                "unsupported_reason": None,
                "statement": "Forecast rises by 30%.",
                "derived_from": [],
            }],
        }))


def _state() -> RequestStateModel:
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 100.0},
        {"timestamp": "2026-01-02T00:00:00Z", "value": 110.0},
    ]
    evidence = DatabaseEvidence(
        evidence_id="evi_demo", result_type="timeseries", database="demo",
        query_language="unit", query="unit:test", summary="history",
        data={"rows": rows, "points": rows}, columns=["timestamp", "value"],
    )
    forecast = ForecastResult(
        forecast_id="forecast_demo", model_name="unit", horizon=2,
        forecast_points=[
            {"timestamp": "2026-01-03T00:00:00Z", "value": 100.0},
            {"timestamp": "2026-01-04T00:00:00Z", "value": 130.0},
        ],
        confidence_interval=[
            {"timestamp": "2026-01-03T00:00:00Z", "lower": 95.0, "upper": 105.0},
            {"timestamp": "2026-01-04T00:00:00Z", "lower": 120.0, "upper": 140.0},
        ],
        diagnostics={"coverage": {"input_evidence_refs": ["evi_demo"]}},
    )
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_demo", detector_name="unit",
        anomaly_points=[{"timestamp": "2026-01-02T00:00:00Z", "value": 110.0}],
        anomaly_spans=[{"start": "2026-01-02T00:00:00Z", "end": "2026-01-02T01:00:00Z"}],
        scores=[{"timestamp": "2026-01-02T00:00:00Z", "score": 3.2}],
        diagnostics={"resolved_evidence_id": "evi_demo"},
    )
    return RequestStateModel(
        request_id="req_artifact_flow", message="forecast and visualize", status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        latest_forecast=forecast, forecast_artifacts={forecast.forecast_id: forecast},
        latest_anomaly=anomaly, anomaly_artifacts={anomaly.anomaly_id: anomaly},
    )


def test_resolver_exposes_complete_forecast_and_anomaly_datasets():
    sources = resolve_artifact_sources(
        _state(), ["forecast:forecast_demo", "anomaly:anomaly_evi_demo"],
    )

    by_ref = {source["source_ref"]: source for source in sources}
    forecast_names = {item["name"] for item in by_ref["forecast:forecast_demo"]["datasets"]}
    anomaly_names = {item["name"] for item in by_ref["anomaly:anomaly_evi_demo"]["datasets"]}
    assert forecast_names == {"forecast_points", "confidence_intervals"}
    assert anomaly_names == {"anomaly_points", "anomaly_status", "anomaly_spans", "anomaly_scores"}
    assert by_ref["forecast:forecast_demo"]["lineage"] == ["forecast:forecast_demo", "evidence:evi_demo"]
    assert by_ref["forecast:forecast_demo"]["forecast_points"] == [
        {"timestamp": "2026-01-03T00:00:00Z", "value": 100.0},
        {"timestamp": "2026-01-04T00:00:00Z", "value": 130.0},
    ]
    assert by_ref["anomaly:anomaly_evi_demo"]["anomaly_status"] == {"detected_count": 1}


def test_source_prompt_manifest_exposes_complete_schema_but_no_records():
    state = _state()
    rows = [
        {
            "timestamp": f"2026-01-{index // 24 + 1:02d}T{index % 24:02d}:00:00Z",
            "value": float(index),
        }
        for index in range(40)
    ]
    rows.append({
        "timestamp": "2026-02-10T00:00:00Z",
        "value": 40.0,
        "late_dimension": "appears after earlier records",
    })
    state.latest_database_evidence.data = {"rows": rows}
    state.database_evidence_artifacts["evi_demo"] = state.latest_database_evidence

    manifest = source_prompt_manifest(resolve_artifact_sources(state, ["evidence:evi_demo"]))

    dataset = manifest[0]["datasets"][0]
    assert {field["name"] for field in dataset["schema_fields"]} == {
        "timestamp", "value", "late_dimension",
    }
    assert "preview" not in dataset
    assert "rows" not in dataset
    assert "appears after earlier records" not in str(manifest)


def test_explicit_forecast_is_the_primary_analysis_table_not_its_database_ancestor():
    sources = resolve_artifact_sources(
        _state(), ["forecast:forecast_demo", "evidence:evi_demo"],
    )

    primary = primary_analysis_input(sources)

    assert primary["source_ref"] == "forecast:forecast_demo"
    assert primary["dataset_name"] == "forecast_points"
    assert primary["rows"] == [
        {"timestamp": "2026-01-03T00:00:00Z", "value": 100.0},
        {"timestamp": "2026-01-04T00:00:00Z", "value": 130.0},
    ]


def test_tool_executor_preserves_code_source_refs_without_injecting_latest_database():
    executor = ToolExecutor.__new__(ToolExecutor)

    normalized = executor._normalize_action_input(
        "code_interpreter",
        {
            "source_refs": ["forecast:forecast_demo"],
            "analysis_goal": "Calculate forecast change",
            "insight_requests": [],
        },
        _state(),
    )

    assert normalized["source_refs"] == ["forecast:forecast_demo"]
    assert "database_evidence" not in normalized


def test_tool_executor_unwraps_single_evidence_ref_for_specialized_tools():
    executor = ToolExecutor.__new__(ToolExecutor)

    normalized = executor._normalize_action_input(
        "anomaly",
        {"database_evidence": ["evidence:evi_demo"]},
        _state(),
    )

    assert normalized["database_evidence"] == "evidence:evi_demo"


def test_visualization_runtime_injects_presentation_budget_without_outer_reasoning_field():
    executor = ToolExecutor.__new__(ToolExecutor)
    state = _state()
    state.constraints = {"max_points": 48, "timezone": "UTC"}

    normalized = executor._normalize_action_input(
        "visualization",
        {"message": "Verify the series.", "constraints": {"theme": "dark"}},
        state,
    )

    assert normalized["constraints"] == {
        "max_points": 48,
        "timezone": "UTC",
        "theme": "dark",
    }


def test_visualization_runtime_separates_repair_control_from_semantic_constraints():
    executor = ToolExecutor.__new__(ToolExecutor)

    normalized = executor._normalize_action_input(
        "visualization",
        {
            "message": "Repair the visual.",
            "constraints": {
                "theme": "dark",
                "mode": "repair",
                "repair_contract": {"execution_error": "rejected candidate payload"},
                "_validation_failure": {"message": "internal diagnostic"},
            },
        },
        _state(),
    )

    assert normalized["constraints"] == {"theme": "dark"}


@pytest.mark.asyncio
async def test_code_interpreter_calculates_directly_from_forecast_source():
    result = await CodeInterpreterTool(llm=_BinderLlm()).execute(
        CodeInterpreterInput(
            source_refs=["forecast:forecast_demo", "anomaly:anomaly_evi_demo"],
            analysis_goal="Calculate forecast endpoint change",
            code=(
                "forecast_rows = source_by_ref['forecast:forecast_demo']['datasets'][0]['rows']\n"
                "start = forecast_rows[0]['value']\n"
                "end = forecast_rows[-1]['value']\n"
                "change_pct = (end - start) / start * 100\n"
                    "result = {'computed_insights': [{'insight_key': 'forecast_change', "
                    "'value': {'direction': 'up', 'change_pct': change_pct}, "
                    "'calculation_trace': {'source_ref': 'forecast:forecast_demo', 'operation': 'endpoint_change'}, "
                    "'derived_evidence_names': ['forecast_change_summary']}], "
                "'derived_evidence': [{'name': 'forecast_change_summary', "
                "'scalar': {'direction': 'up', 'change_pct': change_pct}, "
                "'transform_summary': 'Endpoint change from forecast points'}]}"
            ),
            insight_requests=[KeyInsightRequest(
                insight_key="forecast_change", name="Forecast change", insight_type="change",
            )],
        ),
        request_state=_state(),
    )

    assert result["computed_insights"][0]["value"]["change_pct"] == 30.0
    assert result["input_source_refs"] == ["forecast:forecast_demo", "anomaly:anomaly_evi_demo"]
    assert result["derived_evidence"][0]["lineage"] == [
        "forecast:forecast_demo", "anomaly:anomaly_evi_demo",
    ]


@pytest.mark.asyncio
async def test_generated_code_uses_llm_selected_owning_source_as_canonical_df():
    result = await CodeInterpreterTool(llm=_MultiSourceAnalysisLlm()).execute(
        CodeInterpreterInput(
            source_refs=["evidence:evi_demo", "forecast:forecast_demo"],
            analysis_goal="Calculate change across the existing forecast endpoints",
            insight_requests=[KeyInsightRequest(
                insight_key="forecast_change", name="Forecast change", insight_type="change",
            )],
        ),
        request_state=_state(),
    )

    assert result["input_source_refs"][0] == "forecast:forecast_demo"
    assert result["input_row_count"] == 2
    assert result["computed_insights"][0]["value"] == 30.0
    assert result["diagnostics"]["primary_source"] == {
        "source_ref": "forecast:forecast_demo",
        "dataset_name": "forecast_points",
        "shape": "timeseries",
    }


def test_presentation_catalog_publishes_all_specialized_views():
    refs = PresentationCatalog(_state()).projection_refs()
    assert "view:forecast:forecast_demo:points" in refs
    assert "view:forecast:forecast_demo:interval" in refs
    assert "view:anomaly:anomaly_evi_demo:status" in refs
    assert "view:anomaly:anomaly_evi_demo:points" in refs
    assert "view:anomaly:anomaly_evi_demo:spans" in refs
    assert "view:anomaly:anomaly_evi_demo:scores" in refs


def test_planner_inventory_exposes_bounded_grounded_facts_for_small_specialized_views():
    inventory = PresentationCatalog(_state()).planner_inventory()
    by_ref = {item["source_ref"]: item for item in inventory["sources"]}

    assert by_ref["view:anomaly:anomaly_evi_demo:points"]["grounded_preview"] == [{
        "timestamp": "2026-01-02T00:00:00Z",
        "value": 110.0,
    }]
    assert by_ref["view:anomaly:anomaly_evi_demo:points"]["semantic_contract"] == {
        "data_role": "anomaly_detection_output",
        "materializes_input_transformation": False,
        "operation_description": "Detected anomaly points, scores, spans, or detector status.",
        "supported_visual_uses": ["anomaly_markers", "exclusion_markers", "anomaly_scores"],
        "limitations": [
            "Does not contain the retained/cleaned input records after exclusions.",
            "Does not apply anomaly exclusions to the input series.",
        ],
    }
    assert by_ref["view:evidence:evi_demo:default"]["semantic_contract"]["data_role"] == "raw_observations"
    assert "grounded_preview" not in by_ref["view:evidence:evi_demo:default"]


def test_planner_inventory_does_not_expand_every_insight_item_into_a_source():
    state = _state()
    state.insight_set.insights = [KeyInsight(
        insight_id="ins_many", insight_key="many", name="Many points", insight_type="series",
        statement="Many calculated points.", method="code_interpreter", status="verified",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_demo")],
        items=[
            InsightItem(item_id=f"item_{index}", timestamp=f"2026-01-01T00:{index:02d}:00Z", value=float(index))
            for index in range(60)
        ],
    )]

    catalog = PresentationCatalog(state)
    inventory = catalog.planner_inventory()
    refs = [item["source_ref"] for item in inventory["sources"]]

    assert "insight:ins_many" in refs
    assert not any(ref.startswith("insight:ins_many#") for ref in refs)
    preferred = catalog.planner_inventory({"insight:ins_many#item_4"})
    assert "insight:ins_many#item_4" in [item["source_ref"] for item in preferred["sources"]]


@pytest.mark.asyncio
async def test_visualization_needs_sources_preserves_forecast_ref_and_drives_code(tmp_path):
    state = _state()
    result = await VisualizationTool(
        llm=_NeedsForecastCalculationLlm(),
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Show forecast direction and change",
            source_refs=["forecast:forecast_demo"],
        ),
        request_state=state,
    )

    assert result["status"] == "needs_sources"
    request = result["required_data_request"]
    assert request["input_source_refs"] == ["forecast:forecast_demo", "evidence:evi_demo"]

    state.observations.append(ToolObservation(
        tool_name="visualization", success=True, summary=result["summary"], payload=result,
    ))
    action_space = build_action_space(build_observation_frame(
        state, pending_source_request=request,
    )).model_view()
    assert action_space["required_actions"][0]["action"] == "code_interpreter"
    assert action_space["required_actions"][0]["input_guidance"]["source_refs"] == [
        "forecast:forecast_demo", "evidence:evi_demo",
    ]


def test_visualization_dependency_does_not_advance_active_todo():
    state = _state()
    state.todo_list = [
        {"content": "Create forecast visualization", "task_type": "visualization", "status": "in_progress", "priority": 1},
        {"content": "Answer", "task_type": "answer", "status": "pending", "priority": 2},
    ]
    dependency = {
        "required_action": "code_interpreter",
        "purpose": "Calculate forecast change",
        "required_shape": "scalar",
        "input_source_refs": ["forecast:forecast_demo"],
        "insight_requests": [{"insight_key": "forecast_change", "name": "Forecast change", "insight_type": "change"}],
    }
    payload = {
        "status": "needs_sources",
        "summary": "Additional source required.",
        "visualization_ids": [],
        "visualizations": [],
        "source_refs": [],
        "required_data_request": dependency,
    }

    safe = apply_observation(
        state,
        ToolObservation(tool_name="visualization", success=True, summary=payload["summary"], payload=payload),
        payload,
        type("_VisualizationSpec", (), {"result_target": "visualization"})(),
    )

    assert [todo["status"] for todo in state.todo_list] == ["in_progress", "pending"]
    assert safe.payload["status"] == "needs_sources"
    assessment = apply_previous_observation_assessment(
        state,
        PreviousObservationAssessment(
            completed_active_todo=False,
            reason="Visualization still needs a forecast-derived change insight.",
            missing=["forecast_change", "visualization"],
            can_answer=False,
        ),
    )
    assert assessment.completed is False
    assert [todo["status"] for todo in state.todo_list] == ["in_progress", "pending"]


def test_completed_active_todo_advances_current_step_after_hard_gate():
    state = _state()
    state.todo_list = [
        {"content": "Query history", "task_type": "query", "status": "in_progress", "priority": 1},
        {"content": "Detect anomalies", "task_type": "anomaly", "status": "pending", "priority": 2},
    ]
    state.observations.append(ToolObservation(
        tool_name="sql_query",
        success=True,
        summary="Loaded history",
        payload={"artifact_ref": "evidence:evi_demo"},
    ))

    result = apply_previous_observation_assessment(
        state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason="The history query is covered by the cited SQL artifact.",
            evidence_refs=["evidence:evi_demo"],
            missing=["Detect anomalies"],
            can_answer=False,
        ),
    )

    assert result.completed is True
    assert [todo["status"] for todo in state.todo_list] == ["completed", "in_progress"]


def test_completed_todo_result_ref_uses_owning_artifact_type():
    state = _state()
    state.todo_list = [
        {"content": "Detect anomalies", "task_type": "anomaly", "status": "in_progress", "priority": 1},
        {"content": "Answer", "task_type": "answer", "status": "pending", "priority": 2},
    ]
    state.observations.append(ToolObservation(
        tool_name="anomaly",
        success=True,
        summary="Detected anomalies",
        payload={"artifact_ref": "anomaly:anomaly_evi_demo"},
    ))

    result = apply_previous_observation_assessment(
        state,
        PreviousObservationAssessment(
            completed_active_todo=True,
            reason="The anomaly artifact satisfies the active Todo.",
            evidence_refs=["evidence:evi_demo", "anomaly:anomaly_evi_demo"],
            missing=["Answer"],
            can_answer=False,
        ),
    )

    assert result.completed is True
    assert state.todo_list[0]["result_ref"] == "anomaly:anomaly_evi_demo"


def test_specialized_visualization_dependency_drops_code_only_insight_requests():
    request = VisualizationEvidenceRequest(
        required_action="forecast",
        purpose="Regenerate forecast after anomaly handling",
        required_shape="timeseries",
        input_evidence="evidence:evi_demo",
        insight_requests=[KeyInsightRequest(
            insight_key="invalid_for_forecast",
            name="Invalid for forecast",
            insight_type="forecast",
        )],
    )

    assert request.insight_requests == []


def test_failed_visualization_dependency_remains_pending_until_success():
    state = _state()
    dependency = {
        "required_action": "forecast",
        "purpose": "Regenerate forecast",
        "required_shape": "timeseries",
        "input_evidence": "evi_demo",
        "input_source_refs": ["evidence:evi_demo"],
        "insight_requests": [],
    }
    state.observations.extend([
        ToolObservation(
            tool_name="visualization", success=True, summary="Needs forecast",
            payload={"status": "needs_sources", "required_data_request": dependency},
        ),
        ToolObservation(
            tool_name="forecast", success=False, summary="Forecast failed", payload={}, error="failed",
        ),
    ])

    assert _latest_visualization_source_request(state) == dependency

    state.observations.append(ToolObservation(
        tool_name="forecast", success=True, summary="Forecast succeeded", payload={},
    ))
    retry = _latest_visualization_source_request(state)
    assert retry is not None
    assert retry["required_action"] == "visualization"
    assert retry["originating_request"] == dependency


@pytest.mark.asyncio
async def test_forecast_llm_quality_gate_requests_anomaly_instead_of_modeling_contaminated_input():
    with pytest.raises(StructuredToolError) as captured:
        await ForecastTool(llm=_UnsafeForecastInputLlm()).execute(
            ForecastInput(database_evidence="latest", horizon=2),
            request_state=_state().model_copy(update={"latest_anomaly": None, "anomaly_artifacts": {}}),
        )

    assert captured.value.error_type == "forecast_input_semantic_quality"
    assert captured.value.recommended_next_action == "anomaly"


@pytest.mark.asyncio
async def test_forecast_does_not_repeat_semantic_gate_after_matching_anomaly_is_applied():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 100.0},
        {"timestamp": "2026-01-02T00:00:00Z", "value": 900.0},
        {"timestamp": "2026-01-03T00:00:00Z", "value": 110.0},
    ]
    evidence = DatabaseEvidence(
        evidence_id="evi_filtered", result_type="timeseries", database="demo",
        query_language="unit", query="unit:test", summary="history",
        data={"rows": rows, "points": rows}, columns=["timestamp", "value"],
    )
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_filtered", detector_name="unit",
        anomaly_points=[rows[1]], diagnostics={"resolved_evidence_id": "evi_filtered"},
    )
    state = RequestStateModel(
        request_id="req_filtered", message="forecast", status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        latest_anomaly=anomaly,
        anomaly_artifacts={anomaly.anomaly_id: anomaly},
    )

    result = await ForecastTool(llm=_UnsafeForecastInputLlm()).execute(
        ForecastInput(database_evidence="latest", horizon=1), request_state=state,
    )

    assert result["status"] == "succeeded"
    assert result["diagnostics"]["source_anomaly_id"] == "anomaly_evi_filtered"
    assert result["diagnostics"]["excluded_anomaly_count"] == 1


@pytest.mark.asyncio
async def test_forecast_quality_assessment_marks_distant_edge_windows_as_non_adjacent():
    rows = [
        {"timestamp": f"2026-01-{day:02d}T00:00:00Z", "value": 100.0 + day}
        for day in range(1, 21)
    ]
    evidence = DatabaseEvidence(
        evidence_id="evi_long", result_type="timeseries", database="demo",
        query_language="unit", query="unit:test", summary="long history",
        data={"rows": rows, "points": rows}, columns=["timestamp", "value"],
    )
    llm = _CapturingSafeForecastInputLlm()

    result = await ForecastTool(llm=llm).execute(
        ForecastInput(database_evidence=evidence, horizon=2),
    )

    assert result["status"] == "succeeded"
    system_prompt = str(llm.messages[0][1])
    payload = json.loads(llm.messages[1][1])
    assert set(payload["ordered_samples"]) == {"start_window", "end_window"}
    assert "not adjacent" in payload["sample_layout"]
    assert "never treat the gap" in system_prompt
