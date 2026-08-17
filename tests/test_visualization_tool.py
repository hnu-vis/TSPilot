from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from app.settings import get_settings
from core.completion import evaluate_goal_completion
from core.harness.observation_view import model_observation_view
from core.visualization import PresentationCatalog, VisualizationArtifactStore
from core.harness import build_action_space, build_observation_frame
from runtime.request_state import apply_observation, build_request_state
from schemas.analysis import AnalysisResult, ComputedInsight, DerivedEvidence
from runtime.action_policy import validate_action
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsight, InsightEvidenceRef
from schemas.task_contract import TaskContract, TaskContractOutput
from schemas.visualization import VisualizationPayload
from schemas.tool import ToolObservation
from tools.base import StructuredToolError
from tools.visualization import VisualizationInput, VisualizationTool, _expand_source_preferences, _semantic_error
from tools.registry import build_tool_registry


class _PlannerLlm:
    def __init__(self, payload: str, audit_payload: str | None = None):
        self.payload = payload
        self.audit_payload = audit_payload or '{"decision":"approve","revised_visual_goals":[],"required_data_request":null}'
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if "independently audit" in str(_messages[0][1]):
            return SimpleNamespace(
                content=self.audit_payload,
                response_metadata={},
            )
        return SimpleNamespace(content=self.payload, response_metadata={})


class _SequencePlannerLlm:
    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls = 0

    async def ainvoke(self, _messages):
        if "independently audit" in str(_messages[0][1]):
            self.calls += 1
            return SimpleNamespace(
                content='{"decision":"approve","revised_visual_goals":[],"required_data_request":null}',
                response_metadata={},
            )
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return SimpleNamespace(content=payload, response_metadata={})


class _WorkflowPlannerLlm:
    def __init__(self, *, plans: list[str], audits: list[str]):
        self.plans = list(plans)
        self.audits = list(audits)
        self.plan_calls = 0
        self.audit_calls = 0

    async def ainvoke(self, messages):
        if "independently audit" in str(messages[0][1]):
            payload = self.audits[min(self.audit_calls, len(self.audits) - 1)]
            self.audit_calls += 1
        else:
            payload = self.plans[min(self.plan_calls, len(self.plans) - 1)]
            self.plan_calls += 1
        return SimpleNamespace(content=payload, response_metadata={})


def _state(point_count: int = 500):
    state = build_request_state(
        ChatRequest(
            message="Show the complete series and its maximum point.",
            database_context={"database_id": "demo", "database_type": "unit"},
        ),
        get_settings(),
    )
    rows = [
        {"timestamp": f"2026-01-{index // 24 + 1:02d}T{index % 24:02d}:00:00Z", "value": float(index)}
        for index in range(point_count)
    ]
    evidence = DatabaseEvidence(
        evidence_id="evi_full",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:full",
        summary=f"Loaded {point_count} points.",
        data={"rows": rows, "time_field": "timestamp", "value_field": "value"},
        columns=["timestamp", "value"],
        diagnostics={"is_full_fidelity": True},
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    return state


def test_planner_inventory_distinguishes_materialization_from_interval_coverage():
    inventory = PresentationCatalog(_state(25)).planner_inventory()
    source = next(item for item in inventory["sources"] if item["source_ref"] == "view:evidence:evi_full:default")

    assert source["materialization_complete"] is True
    assert source["time_range"] == {
        "field": "timestamp",
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-02T00:00:00Z",
    }
    assert source["query_context"][0]["query"] == "unit:full"
    assert "full_fidelity" not in source


def _state_with_analysis_views():
    state = _state(25)
    derived = DerivedEvidence(
        evidence_id="dev_endpoints",
        name="interval endpoints",
        shape="timeseries",
        rows=[
            {"role": "start", "timestamp": "2026-01-01T00:00:00Z", "value": 0.0},
            {"role": "end", "timestamp": "2026-01-02T00:00:00Z", "value": 24.0},
        ],
        lineage=["evidence:evi_full"],
        transform_summary="First and last observations.",
    )
    analysis = AnalysisResult(
        analysis_id="ana_demo",
        analysis_goal="calculate interval change",
        code_hash="abc",
        input_evidence_id="evi_full",
        input_row_count=25,
        status="succeeded",
        summary="Computed interval change.",
        computed_insights=[
            ComputedInsight(
                insight_key="absolute_change",
                value=24.0,
                calculation_trace="last minus first",
                derived_evidence_ids=[derived.evidence_id],
            )
        ],
        derived_evidence=[derived],
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    state.derived_evidence_artifacts[derived.evidence_id] = derived
    return state


def test_planner_inventory_exposes_renderable_sources_not_storage_artifacts():
    inventory = PresentationCatalog(_state_with_analysis_views()).planner_inventory()
    refs = {item["source_ref"] for item in inventory["sources"]}

    assert "analysis:ana_demo" not in refs
    assert "derived_evidence:dev_endpoints" not in refs
    assert "evidence:evi_full" not in refs
    assert "view:evidence:evi_full:default" in refs
    assert "view:derived_evidence:dev_endpoints" in refs


def test_scalar_insight_inventory_cannot_be_used_as_a_timestamped_layer():
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_change",
            insight_key="absolute_change",
            name="absolute change",
            insight_type="change",
            statement="Absolute change is 24.",
            value=24.0,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "last-first"},
        )
    ]
    inventory = PresentationCatalog(state).planner_inventory()
    source = next(item for item in inventory["sources"] if item["source_ref"] == "insight:insight_change")

    assert source["render_capabilities"]["scalar_only"] is True
    assert source["render_capabilities"]["timestamped_numeric"] is False
    assert source["render_capabilities"]["supported_marks"] == ["text", "table"]


@pytest.mark.asyncio
async def test_audit_can_verify_scalar_conclusion_with_context_series_without_new_analysis(tmp_path):
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_change_rate",
            insight_key="change_rate",
            name="change rate",
            insight_type="change_rate",
            statement="The interval change rate is 24%.",
            value=0.24,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "(end-start)/start"},
        )
    ]
    initial_plan = (
        '{"visual_goals":[{"purpose":"verify interval change","title":"Change",'
        '"priority":"primary","summary":null,"required_roles":["series","change_rate"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"change_rate","source_ref":"insight:insight_change_rate","mark":"point",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    revised_goals = (
        '[{"purpose":"verify interval change","title":"Change","priority":"primary",'
        '"summary":"The full interval and computed rate are shown together.",'
        '"required_roles":["series","change_rate"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"change_rate","source_ref":"insight:insight_change_rate","mark":"text",'
        '"encoding":{"text":"value"},"label":"24%"}]}]'
    )
    audit = (
        '{"decision":"revise","revised_visual_goals":' + revised_goals
        + ',"required_data_request":null}'
    )
    llm = _WorkflowPlannerLlm(plans=[initial_plan], audits=[audit])
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(
        llm=llm, artifact_store=store,
    ).execute(
        VisualizationInput(
            message="Analyze and visually verify the interval change rate.",
            source_refs=["evidence:evi_full", "insight:insight_change_rate"],
        ),
        request_state=state,
    )

    visualization = result["visualizations"][0]
    assert llm.plan_calls == 1
    assert llm.audit_calls == 1
    assert [layer["mark"] for layer in visualization["layers"]] == ["line", "text"]
    complete = store.get(visualization["visualization_id"])
    assert complete is not None
    assert complete.datasets[1].metric["value"] == 0.24


@pytest.mark.asyncio
async def test_structured_insight_supports_located_and_metric_projections(tmp_path):
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_trade",
            insight_key="max_trade_return",
            name="maximum single-trade return",
            insight_type="optimization",
            statement="Buy at 10 and sell at 25 for a profit of 15.",
            value={
                "max_profit_amount": 15.0,
                "max_profit_ratio": 1.5,
                "buy_time": "2026-01-01T00:00:00Z",
                "sell_time": "2026-01-02T00:00:00Z",
                "buy_price": 10.0,
                "sell_price": 25.0,
            },
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "sell_price-buy_price"},
        )
    ]
    plan = (
        '{"visual_goals":[{"purpose":"verify the optimal trade","title":"Optimal trade",'
        '"priority":"primary","summary":null,"required_roles":["series","buy","sell","return"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"buy","source_ref":"insight:insight_trade","mark":"point",'
        '"encoding":{"x":"buy_time","y":"buy_price"},"label":"Buy"},'
        '{"role":"sell","source_ref":"insight:insight_trade","mark":"point",'
        '"encoding":{"x":"sell_time","y":"sell_price"},"label":"Sell"},'
        '{"role":"return","source_ref":"insight:insight_trade","mark":"text",'
        '"encoding":{},"label":"Maximum return"}]}],"required_data_request":null}'
    )
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(
        llm=_PlannerLlm(plan), artifact_store=store,
    ).execute(
        VisualizationInput(
            message="Exclude anomalies and verify the maximum single-trade return.",
            source_refs=["evidence:evi_full", "insight:insight_trade"],
        ),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert complete.datasets[1].series[0].points[0].x == "2026-01-01T00:00:00Z"
    assert complete.datasets[2].series[0].points[0].x == "2026-01-02T00:00:00Z"
    assert complete.datasets[3].metric["max_profit_amount"] == 15.0


def test_direct_timestamp_value_insight_is_a_renderable_locator():
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_start",
            insight_key="start_value",
            name="start value",
            insight_type="point_value",
            statement="Start value is 0.",
            value={"timestamp": "2026-01-01T00:00:00Z", "value": 0.0},
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "first observation"},
        )
    ]
    inventory = PresentationCatalog(state).planner_inventory()
    source = next(item for item in inventory["sources"] if item["source_ref"] == "insight:insight_start")

    assert source["locator_fields"] == ["timestamp", "value"]
    assert source["render_capabilities"]["timestamped_numeric"] is True
    assert source["render_capabilities"]["scalar_only"] is False


@pytest.mark.asyncio
async def test_visualization_tool_persists_every_timeseries_point_and_returns_descriptor(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"show complete pattern","title":"Complete series",'
        '"priority":"primary","summary":null,"required_roles":["base_series"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":{"field":"timestamp","type":"temporal"},"y":{"field":"value","type":"quantitative"}},"label":"Value"}]}],"required_data_request":null}'
    )
    store = VisualizationArtifactStore(tmp_path)
    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(message="Show the complete series."),
        request_state=_state(),
    )

    descriptor = result["visualizations"][0]
    assert descriptor["data_ref"].endswith("/data")
    assert descriptor["datasets"][0]["row_count"] == 500
    assert descriptor["datasets"][0]["series"] == []
    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert complete.datasets[0].row_count == 500
    assert len(complete.datasets[0].series[0].points) == 500


@pytest.mark.asyncio
async def test_visualization_tool_repairs_invalid_llm_plan_inside_tool_boundary(tmp_path):
    llm = _SequencePlannerLlm(
        [
            '{"visual_goals":[{"purpose":"show complete pattern","title":"Complete series",'
            '"priority":"primary","summary":null,"required_roles":["base_series"],"layers":['
            '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
            '"encoding":{"color":{"value":"blue"}},"label":"Value"}]}],"required_data_request":null}',
            '{"visual_goals":[{"purpose":"show complete pattern","title":"Complete series",'
            '"priority":"primary","summary":null,"required_roles":["base_series"],"layers":['
            '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
            '"encoding":{"x":"timestamp","y":"value"},"label":"Value"}]}],"required_data_request":null}',
        ]
    )

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the complete series."), request_state=_state(25))

    assert llm.calls == 3
    assert result["visualizations"][0]["datasets"][0]["row_count"] == 25


@pytest.mark.asyncio
async def test_materialization_repair_reenters_independent_audit(tmp_path):
    state = _state_with_analysis_views()
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_change",
            insight_key="absolute_change",
            name="absolute change",
            insight_type="change",
            statement="Absolute change is 24.",
            value=24.0,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "last-first"},
        )
    ]
    invalid = (
        '{"visual_goals":[{"purpose":"verify change","title":"Change","priority":"primary",'
        '"summary":null,"required_roles":["base_series","change"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"change","source_ref":"insight:insight_change","mark":"point",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    repaired = (
        '{"visual_goals":[{"purpose":"verify change","title":"Change","priority":"primary",'
        '"summary":null,"required_roles":["base_series","endpoints","change"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"endpoints","source_ref":"view:derived_evidence:dev_endpoints","mark":"point",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"change","source_ref":"insight:insight_change","mark":"text",'
        '"encoding":{"y":"value"},"label":null}]}],"required_data_request":null}'
    )
    approved = '{"decision":"approve","revised_visual_goals":[],"required_data_request":null}'
    llm = _WorkflowPlannerLlm(plans=[invalid, repaired], audits=[approved, approved])

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Verify interval change.",
            source_refs=["evidence:evi_full", "analysis:ana_demo"],
        ),
        request_state=state,
    )

    assert llm.plan_calls == 2
    assert llm.audit_calls == 2
    assert result["visualizations"][0]["required_roles"] == ["base_series", "endpoints", "change"]


def test_visualization_failure_receipt_does_not_expose_internal_inventory():
    error = _semantic_error(
        ValueError("layer uses unavailable field"),
        {"sources": [{"source_ref": "view:evidence:evi_full:default"}]},
    )

    payload = error.validation_failure
    assert "presentation_inventory" not in payload["repair_contract"]
    assert "view:evidence:" not in str(payload)


@pytest.mark.asyncio
async def test_visualization_tool_requests_full_sql_evidence_instead_of_falling_back(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[],"required_data_request":{"required_action":"sql_query","purpose":"show complete series with max point",'
        '"required_shape":"full_timeseries","required_fields":["timestamp","value"],'
        '"required_properties":["complete requested time range","maximum row with timestamp"],"insight_requests":[]}}'
    )
    tool = VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path))

    with pytest.raises(StructuredToolError) as caught:
        await tool.execute(VisualizationInput(message="Show all points and max."), request_state=_state(1))

    failure = caught.value.validation_failure
    assert caught.value.recommended_next_action == "sql_query"
    assert failure["retry_policy"]["required_action"] == "sql_query"
    assert failure["repair_contract"]["constraints"]["full_fidelity"] is True


@pytest.mark.asyncio
async def test_visualization_routes_missing_calculated_layer_to_code_interpreter(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[],"required_data_request":{"required_action":"code_interpreter",'
        '"purpose":"calculate the optimal single-trade decision points",'
        '"required_shape":"decision_points","required_fields":["timestamp","value"],'
        '"required_properties":["buy precedes sell","exclude authoritative anomalies"],'
        '"insight_requests":[{"insight_key":"optimal_trade","name":"Optimal single trade",'
        '"insight_type":"optimization"}]}}'
    )
    tool = VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path))

    with pytest.raises(StructuredToolError) as caught:
        await tool.execute(VisualizationInput(message="Show the full series and optimal trade."), request_state=_state(25))

    failure = caught.value.validation_failure
    assert caught.value.recommended_next_action == "code_interpreter"
    assert failure["retry_policy"]["required_action"] == "code_interpreter"
    assert failure["repair_contract"]["insight_requests"][0]["insight_key"] == "optimal_trade"


@pytest.mark.asyncio
async def test_visualization_audit_rejects_plan_that_omits_requested_decision_points(tmp_path):
    plan = (
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["base_series"],"layers":[{"role":"base_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    audit = (
        '{"decision":"need_data","revised_visual_goals":[],"required_data_request":{'
        '"required_action":"code_interpreter","purpose":"calculate requested buy and sell points",'
        '"required_shape":"decision_points","required_fields":["timestamp","value"],'
        '"required_properties":["buy precedes sell"],"insight_requests":[{'
        '"insight_key":"optimal_trade","name":"Optimal single trade","insight_type":"optimization"}]}}'
    )
    tool = VisualizationTool(
        llm=_PlannerLlm(plan, audit_payload=audit), artifact_store=VisualizationArtifactStore(tmp_path),
    )

    with pytest.raises(StructuredToolError) as caught:
        await tool.execute(
            VisualizationInput(message="Show the complete series with buy and sell points."),
            request_state=_state(25),
        )

    assert caught.value.recommended_next_action == "code_interpreter"


def test_visualization_data_endpoint_returns_complete_artifact(tmp_path, monkeypatch):
    store = VisualizationArtifactStore(tmp_path)
    state = _state(25)
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )

    import asyncio
    result = asyncio.run(VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(message="Show trend."), request_state=state,
    ))
    monkeypatch.setattr("app.routes.visualizations.get_visualization_artifact_store", lambda: store)
    response = TestClient(create_app()).get(result["visualizations"][0]["data_ref"])

    assert response.status_code == 200
    assert len(response.json()["datasets"][0]["series"][0]["points"]) == 25


def test_visualization_is_a_nonterminal_required_action_when_requested(tmp_path):
    state = _state(10)
    state.requested_capabilities = ["query", "visualization"]
    action_space = build_action_space(build_observation_frame(state)).model_view()
    spec = build_tool_registry(
        get_settings(),
        llm=_PlannerLlm("{}"),
        visualization_artifact_store=VisualizationArtifactStore(tmp_path),
    ).resolve("visualization")

    assert action_space["required_actions"][0]["action"] == "visualization"
    assert spec.produces_terminal_payload is False
    assert spec.result_target == "visualization"


@pytest.mark.asyncio
async def test_completion_distinguishes_full_timeseries_evidence_from_visual_delivery(tmp_path):
    state = _state(25)
    state.task_contract = TaskContract(
        goal="Show the complete series.",
        required_outputs=[
            TaskContractOutput(
                id="complete_timeseries",
                description="complete time-series data",
                output_type="evidence",
                evidence_kind="time_series",
            ),
            TaskContractOutput(
                id="visualization",
                description="chart of all time-series points",
                output_type="visualization",
                evidence_kind="analysis",
            ),
        ],
    )

    before = evaluate_goal_completion(state)
    assert before.can_answer is False
    assert before.missing_evidence == ["visualization"]

    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show trend."), request_state=state)
    state.visualizations = [VisualizationPayload.model_validate(item) for item in result["visualizations"]]
    state.completion_state["latest_gap_assessment"] = {
        "can_answer": False,
        "covered": ["complete time-series data"],
        "missing": ["可视化验证通过"],
    }

    after = evaluate_goal_completion(state)
    assert after.can_answer is True
    assert after.missing_evidence == []


def test_terminate_accepts_sql_verified_derived_insight_without_analysis_artifact():
    state = _state(25)
    state.task_contract = TaskContract(
        goal="Show the complete series and maximum.",
        required_outputs=[
            TaskContractOutput(
                id="max_value",
                description="maximum value",
                output_type="insight",
                evidence_kind="derived_insight",
                measures=["max"],
            ),
            TaskContractOutput(
                id="chart",
                description="complete series with maximum point",
                output_type="visualization",
                evidence_kind="visualization",
            ),
        ],
    )
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_max",
            insight_key="max_value",
            name="max_value",
            insight_type="extreme",
            statement="max_value is 24",
            value=24,
            method="sql_query",
            evidence_refs=[
                InsightEvidenceRef(
                    source_type="query",
                    source_id="evi_full",
                    label="complete series",
                )
            ],
            calculation_trace={"operator": "max", "value_key": "value"},
        )
    ]
    state.visualizations = [
        VisualizationPayload(
            visualization_id="viz_complete",
            data_ref="/api/v1/visualizations/viz_complete/data",
            purpose="show complete pattern and maximum",
            title="Complete series",
            source_refs=["evidence:evi_full", "insight:insight_max"],
            required_roles=["base_series", "max_value_highlight"],
            accessibility={"description": "Complete series with maximum point."},
        )
    ]

    allowed, reason = validate_action(
        state,
        "terminate",
        {"direct_answer": "The complete series and maximum are shown."},
    )

    assert allowed is True
    assert reason is None


@pytest.mark.asyncio
async def test_visualization_observation_is_a_react_receipt_not_a_render_payload(tmp_path):
    state = _state(25)
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["base_series"],"layers":[{"role":"base_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the complete trend."), request_state=state)
    spec = SimpleNamespace(result_target="visualization")
    observation = apply_observation(
        state,
        ToolObservation(
            tool_name="visualization",
            success=True,
            summary=result["summary"],
            payload=result,
        ),
        result,
        spec,
    )

    payload = model_observation_view(observation)["payload"]
    assert payload["visualization_ids"] == result["visualization_ids"]
    assert payload["grounded_by"] == ["evidence:evi_full"]
    assert payload["verification"][0]["full_fidelity"] is True
    assert payload["verification"][0]["datasets"][0]["row_count"] == 25
    assert payload["verification"][0]["materialized_roles"] == ["base_series"]
    assert "visualizations" not in payload
    assert "data_ref" not in str(payload)
    assert "visualization_count" not in payload
    assert "view:evidence:" not in str(payload)


def test_visualization_tool_expands_outer_artifact_refs_to_internal_views():
    catalog = PresentationCatalog(_state_with_analysis_views())

    expanded, unknown = _expand_source_preferences(
        ["analysis:ana_demo", "derived_evidence:dev_endpoints"],
        catalog,
    )

    assert unknown == set()
    assert expanded == {
        "view:evidence:evi_full:default",
        "view:derived_evidence:dev_endpoints",
    }
    assert "analysis:ana_demo" not in expanded
    assert "derived_evidence:dev_endpoints" not in expanded


@pytest.mark.asyncio
async def test_react_policy_reuses_current_visualization_instead_of_regenerating(tmp_path):
    state = _state(25)
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["base_series"],"layers":[{"role":"base_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show trend."), request_state=state)
    state.visualizations = [VisualizationPayload.model_validate(item) for item in result["visualizations"]]

    allowed, reason = validate_action(
        state,
        "visualization",
        {"message": "Show trend.", "source_refs": ["evidence:evi_full"]},
    )

    assert allowed is False
    assert "Reuse its visualization_id" in reason
