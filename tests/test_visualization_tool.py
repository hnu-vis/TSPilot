from __future__ import annotations

import json
import re
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
from schemas.key_insight import KeyInsight, InsightEvidenceRef, InsightItem
from schemas.task_contract import TaskContract, TaskContractOutput
from schemas.visualization import VisualizationPayload
from schemas.tool import ToolObservation
from tools.base import StructuredToolError
from tools.visualization import VisualizationInput, VisualizationTool, _expand_source_preferences, _semantic_error
from tools.registry import build_tool_registry


class _PlannerLlm:
    def __init__(
        self,
        payload: str,
        audit_payload: str | list[str] | None = None,
        projection_payload: str | None = None,
    ):
        self.payload = payload
        self.projection_payload = projection_payload
        self.audit_payloads = (
            list(audit_payload)
            if isinstance(audit_payload, list)
            else [audit_payload or '{"decision":"approve","issues":[],"required_data_request":null}']
        )
        self.calls = 0
        self.projection_prompts = []
        self.audit_prompts = []

    async def ainvoke(self, _messages):
        if _is_verification_prompt(_messages):
            return SimpleNamespace(content=_verification_payload(_messages), response_metadata={})
        if _is_projection_prompt(_messages):
            self.projection_prompts.append(_messages)
            return SimpleNamespace(
                content=self.projection_payload or _projection_for_chart_payload(self.payload),
                response_metadata={},
            )
        self.calls += 1
        if "independently audit" in str(_messages[0][1]):
            self.audit_prompts.append(_messages)
            return SimpleNamespace(
                content=self.audit_payloads[min(len(self.audit_prompts) - 1, len(self.audit_payloads) - 1)],
                response_metadata={},
            )
        return SimpleNamespace(content=self.payload, response_metadata={})


class _SequencePlannerLlm:
    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls = 0

    async def ainvoke(self, _messages):
        if _is_verification_prompt(_messages):
            return SimpleNamespace(content=_verification_payload(_messages), response_metadata={})
        if _is_projection_prompt(_messages):
            return SimpleNamespace(
                content=_projection_for_chart_payload(self.payloads[-1]),
                response_metadata={},
            )
        if "independently audit" in str(_messages[0][1]):
            self.calls += 1
            return SimpleNamespace(
                content='{"decision":"approve","issues":[],"required_data_request":null}',
                response_metadata={},
            )
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return SimpleNamespace(content=payload, response_metadata={})


def _is_projection_prompt(messages) -> bool:
    return "You are the semantic projection stage" in str(messages[0][1])


def _is_verification_prompt(messages) -> bool:
    return "You are the visual verification planner inside" in str(messages[0][1])


def _verification_payload(messages) -> str:
    prompt = str(messages[0][1])
    insight_ids = list(dict.fromkeys(re.findall(r'"insight_id":\s*"([^"]+)"', prompt)))
    return json.dumps({
        "decision": "visualize",
        "target_insight_ids": insight_ids,
        "verification_question": "Does the grounded visual evidence support the requested relationship?",
        "interpretation": "Inspect the complete contextual data and the highlighted analytical relationship.",
        "visual_relation": "grounded_comparison",
        "required_context": ["complete contextual evidence"],
        "non_visual_insight_ids": [],
        "required_data_request": None,
    })


def _projection_for_chart_payload(payload: str) -> str:
    decoded = json.loads(payload)
    requirement = decoded.get("required_data_request")
    if requirement:
        return json.dumps({"semantic_views": [], "required_data_request": requirement})
    layers = [
        layer
        for goal in decoded.get("visual_goals", [])
        for layer in goal.get("layers", [])
    ]
    source_ref = next(
        (layer.get("source_ref") for layer in layers if layer.get("source_ref")),
        "view:evidence:evi_full:default",
    )
    fields = []
    seen = set()
    for layer in layers:
        if layer.get("source_ref") != source_ref:
            continue
        for value in (layer.get("encoding") or {}).values():
            values = value if isinstance(value, list) else [value]
            for item in values:
                field_name = item if isinstance(item, str) else item.get("field") if isinstance(item, dict) else None
                if field_name and field_name not in seen:
                    seen.add(field_name)
                    fields.append({"name": field_name, "semantic_role": field_name, "source_path": f"$.{field_name}"})
    if not fields:
        fields = [{"name": "value", "semantic_role": "measure", "source_path": "$.value"}]
    view_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_ref))
    return json.dumps({
        "semantic_views": [{
            "view_id": f"test_{view_id}",
            "name": "Test semantic view",
            "purpose": "Support the chart plan under test",
            "grain": "records",
            "source_ref": source_ref,
            "fields": fields,
        }],
        "required_data_request": None,
    })


class _WorkflowPlannerLlm:
    def __init__(self, *, plans: list[str], audits: list[str]):
        self.plans = list(plans)
        self.audits = list(audits)
        self.plan_calls = 0
        self.audit_calls = 0

    async def ainvoke(self, messages):
        if _is_verification_prompt(messages):
            payload = _verification_payload(messages)
        elif "independently audit" in str(messages[0][1]):
            payload = self.audits[min(self.audit_calls, len(self.audits) - 1)]
            self.audit_calls += 1
        else:
            payload = self.plans[min(self.plan_calls, len(self.plans) - 1)]
            self.plan_calls += 1
        return SimpleNamespace(content=payload, response_metadata={})


class _TwoStagePlannerLlm:
    def __init__(self, *, projection: dict, chart: dict):
        self.projection = projection
        self.chart = chart
        self.projection_calls = 0
        self.chart_calls = 0
        self.verification_prompts = []
        self.projection_prompts = []
        self.chart_prompts = []
        self.audit_prompts = []

    async def ainvoke(self, messages):
        if _is_verification_prompt(messages):
            self.verification_prompts.append(messages)
            return SimpleNamespace(content=_verification_payload(messages), response_metadata={})
        if "independently audit" in str(messages[0][1]):
            self.audit_prompts.append(messages)
            return SimpleNamespace(
                content='{"decision":"approve","issues":[],"required_data_request":null}',
                response_metadata={},
            )
        if _is_projection_prompt(messages):
            self.projection_calls += 1
            self.projection_prompts.append(messages)
            payload = self.projection
        else:
            self.chart_calls += 1
            self.chart_prompts.append(messages)
            payload = self.chart
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


class _RepairingTwoStagePlannerLlm:
    def __init__(self, *, projections: list[dict], chart: dict | list[dict]):
        self.projections = projections
        self.charts = chart if isinstance(chart, list) else [chart]
        self.projection_prompts = []
        self.chart_calls = 0

    async def ainvoke(self, messages):
        if _is_verification_prompt(messages):
            return SimpleNamespace(content=_verification_payload(messages), response_metadata={})
        if "independently audit" in str(messages[0][1]):
            return SimpleNamespace(
                content='{"decision":"approve","issues":[],"required_data_request":null}',
                response_metadata={},
            )
        if _is_projection_prompt(messages):
            index = min(len(self.projection_prompts), len(self.projections) - 1)
            self.projection_prompts.append(messages)
            payload = self.projections[index]
        else:
            payload = self.charts[min(self.chart_calls, len(self.charts) - 1)]
            self.chart_calls += 1
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


class _VerificationOnlyLlm:
    def __init__(self, payload: dict):
        self.payload = payload

    async def ainvoke(self, messages):
        if not _is_verification_prompt(messages):
            raise AssertionError("visualization must stop after the verification decision")
        return SimpleNamespace(content=json.dumps(self.payload), response_metadata={})


class _RenderAuditStub:
    def __init__(self, decisions: list[dict]):
        self.decisions = list(decisions)
        self.calls: list[list[VisualizationPayload]] = []

    async def audit(self, *, visualizations, **_kwargs):
        self.calls.append(list(visualizations))
        return self.decisions[min(len(self.calls) - 1, len(self.decisions) - 1)]


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


def test_planner_inventory_hides_partial_insights_but_keeps_verified_ones():
    state = _state(25)
    partial = KeyInsight(
        insight_id="ins_partial",
        insight_key="partial_point",
        name="Partial point",
        insight_type="point_value",
        statement="Unverified point.",
        method="code_interpreter",
        status="partial",
        items=[InsightItem(item_id="p1", timestamp="2026-01-01T01:00:00Z", value=1.0)],
    )
    verified = partial.model_copy(update={
        "insight_id": "ins_verified",
        "insight_key": "verified_point",
        "name": "Verified point",
        "status": "verified",
    })
    state.insight_set.insights = [partial, verified]

    refs = {
        item["source_ref"]
        for item in PresentationCatalog(state).planner_inventory()["sources"]
    }

    assert "insight:ins_partial" not in refs
    assert "insight:ins_partial#p1" not in refs
    assert "insight:ins_verified" in refs
    assert "insight:ins_verified#p1" not in refs
    verified_source = next(
        item
        for item in PresentationCatalog(state).planner_inventory()["sources"]
        if item["source_ref"] == "insight:ins_verified"
    )
    assert verified_source["insight_key"] == "verified_point"
    assert verified_source["name"] == "Verified point"
    assert verified_source["items"][0]["value"] == 1.0


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


def test_verified_numeric_scalar_inventory_supports_only_lossless_graphical_marks():
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
    assert source["render_capabilities"]["renderable"] is True
    assert source["render_capabilities"]["renderer_series_type"] == "open"


@pytest.mark.asyncio
async def test_temporal_context_is_preferred_over_isolated_scalar_layer(tmp_path):
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
    plan = (
        '{"visual_goals":[{"purpose":"verify interval change","title":"Change","priority":"primary",'
        '"summary":"The scalar rate remains in the answer while the chart preserves temporal context.",'
        '"required_roles":["series"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    llm = _PlannerLlm(plan)
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
    assert llm.calls == 1  # chart planning publishes after grounded materialization
    assert [layer["mark"] for layer in visualization["layers"]] == ["line"]
    complete = store.get(visualization["visualization_id"])
    assert complete is not None
    assert len(complete.datasets) == 1
    assert state.insight_set.insights[0].value == 0.24


@pytest.mark.asyncio
async def test_structured_insight_supports_multiple_located_projections(tmp_path):
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
        '"priority":"primary","summary":null,"required_roles":["series","buy","sell"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"buy","source_ref":"insight:insight_trade","mark":"point",'
        '"encoding":{"x":"buy_time","y":"buy_price"},"label":"Buy"},'
        '{"role":"sell","source_ref":"insight:insight_trade","mark":"point",'
        '"encoding":{"x":"sell_time","y":"sell_price"},"label":"Sell"}]}],"required_data_request":null}'
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


@pytest.mark.asyncio
async def test_structured_insight_accepts_semantic_encoding_channel_aliases(tmp_path):
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_trade_aliases",
            insight_key="max_trade_return_aliases",
            name="maximum single-trade return",
            insight_type="optimization",
            statement="Buy at 10 and sell at 25 for a profit of 15.",
            value={
                "max_profit": 15.0,
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
        '{"visual_goals":[{"purpose":"verify trade aliases","title":"Optimal trade",'
        '"priority":"primary","summary":null,"required_roles":["buy","sell"],"layers":['
        '{"role":"buy","source_ref":"insight:insight_trade_aliases","mark":"point",'
        '"encoding":{"timestamp":"buy_time","value":"buy_price"},"label":"Buy"},'
        '{"role":"sell","source_ref":"insight:insight_trade_aliases","mark":"point",'
        '"encoding":{"timestamp":"sell_time","value":"sell_price"},"label":"Sell"}]}],'
        '"required_data_request":null}'
    )
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(
        llm=_PlannerLlm(plan), artifact_store=store,
    ).execute(
        VisualizationInput(message="Verify the trade."),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    buy_points = complete.datasets[0].series[0].points
    sell_points = complete.datasets[1].series[0].points
    assert [(point.x, point.y) for point in buy_points] == [("2026-01-01T00:00:00Z", 10.0)]
    assert [(point.x, point.y) for point in sell_points] == [("2026-01-02T00:00:00Z", 25.0)]
    assert complete.layers[0].encoding == {"x": "buy_time", "y": "buy_price"}
    assert complete.layers[1].encoding == {"x": "sell_time", "y": "sell_price"}


@pytest.mark.asyncio
async def test_two_stage_llm_projects_nested_forecast_insight_and_composes_interval(tmp_path):
    state = _state(25)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_forecast_nested",
        insight_key="week_ahead_forecast",
        name="week ahead forecast",
        insight_type="series",
        statement="Daily forecast points with central price and lower/upper uncertainty bounds.",
        value={"direction": "上涨", "change_pct": 8.0},
        value_shape="collection",
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
        items=[
            InsightItem(
                item_id="f1", label="forecast_point", timestamp="2026-01-03T00:00:00Z",
                value={"predicted_price": 25.0, "lower_price": 22.0, "upper_price": 28.0},
            ),
            InsightItem(
                item_id="f2", label="forecast_point", timestamp="2026-01-04T00:00:00Z",
                value={"predicted_price": 27.0, "lower_price": 23.0, "upper_price": 31.0},
            ),
        ],
    )]
    projection = {
        "semantic_views": [
            {
                "view_id": "history",
                "name": "Observed price history",
                "purpose": "Provide temporal context for the forecast",
                "grain": "observation",
                "source_ref": "view:evidence:evi_full:default",
                "record_path": "$.records",
                "fields": [
                    {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                    {"name": "observed_price", "semantic_role": "observed_price", "source_path": "$.value"},
                ],
            },
            {
                "view_id": "forecast",
                "name": "Forecast path and interval",
                "purpose": "Expose the central forecast and uncertainty interval",
                "grain": "forecast_point",
                "source_ref": "insight:insight_forecast_nested",
                "record_path": "$.items",
                "fields": [
                    {"name": "time", "semantic_role": "forecast_time", "source_path": "$.timestamp"},
                    {"name": "central", "semantic_role": "forecast_central", "source_path": "$.value.predicted_price"},
                    {"name": "lower", "semantic_role": "forecast_lower", "source_path": "$.value.lower_price"},
                    {"name": "upper", "semantic_role": "forecast_upper", "source_path": "$.value.upper_price"},
                ],
            },
            {
                "view_id": "forecast_summary",
                "name": "Forecast direction summary",
                "purpose": "Expose the overall direction and magnitude at summary grain",
                "grain": "forecast_summary",
                "source_ref": "insight:insight_forecast_nested",
                "record_path": "$.value",
                "fields": [
                    {"name": "direction", "semantic_role": "forecast_direction", "source_path": "$.direction"},
                    {"name": "change_pct", "semantic_role": "forecast_change_percent", "source_path": "$.change_pct"},
                ],
            },
        ],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Verify the forecast in historical context",
            "title": "Observed and forecast price",
            "priority": "primary",
            "summary": "History, central forecast, and uncertainty interval.",
            "required_roles": ["history", "forecast", "uncertainty"],
            "layers": [
                {"role": "history", "source_ref": "semantic:history", "mark": "line", "encoding": {"x": "time", "y": "observed_price"}},
                {"role": "forecast", "source_ref": "semantic:forecast", "mark": "line", "encoding": {"x": "time", "y": "central"}},
                {"role": "uncertainty", "source_ref": "semantic:forecast", "mark": "band", "encoding": {"x": "time", "lower": "lower", "upper": "upper"}},
            ],
        }],
        "required_data_request": None,
    }
    llm = _TwoStagePlannerLlm(projection=projection, chart=chart)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(
            message="Show the history, week-ahead forecast, and its interval.",
            source_refs=["evidence:evi_full", "insight:insight_forecast_nested"],
        ),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert llm.projection_calls == 1
    assert llm.chart_calls == 1
    assert [layer.source_ref for layer in complete.layers] == [
        "semantic:history", "semantic:forecast", "semantic:forecast",
    ]
    forecast_points = complete.datasets[1].series[0].points
    interval_points = complete.datasets[2].series[0].points
    assert [point.y for point in forecast_points] == [25.0, 27.0]
    assert [(point.lower, point.upper) for point in interval_points] == [(22.0, 28.0), (23.0, 31.0)]
    assert {binding.item_id for binding in complete.bindings if binding.item_id} == {"f1", "f2"}
    assert complete.layout == "overlay"
    assert complete.source_refs == [
        "view:evidence:evi_full:default",
        "insight:insight_forecast_nested",
    ]
    fresh_catalog = PresentationCatalog(state)
    assert all(fresh_catalog.resolve(ref) for ref in complete.source_refs)


@pytest.mark.asyncio
async def test_visualization_tool_projects_nested_insight_records_with_wildcard_path(tmp_path):
    state = _state(6)
    state.latest_database_evidence.data["rows"][-1]["value"] = 987654.321
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_nested_reversal",
        insight_key="nested_reversal",
        name="Nested reversal interval",
        insight_type="interval",
        statement="The selected interval declines into a turning point and then rises.",
        value={"count": 1},
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
        items=[InsightItem(
            item_id="interval_1",
            value={"summary": "decline then rise"},
            dimensions={"items": [
                {"role": "decline_end", "timestamp": "2026-01-01T01:00:00Z", "value": 1.0},
                {"role": "turn", "timestamp": "2026-01-01T02:00:00Z", "value": 2.0},
                {"role": "rise_start", "timestamp": "2026-01-01T03:00:00Z", "value": 3.0},
            ]},
        )],
    )]
    projection = {
        "semantic_views": [
            {
                "view_id": "full_series",
                "name": "Full observed series",
                "purpose": "Provide the complete temporal context",
                "grain": "observation",
                "source_ref": "view:evidence:evi_full:default",
                "record_path": "$.records",
                "fields": [
                    {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                    {"name": "price", "semantic_role": "observed_price", "source_path": "$.value"},
                ],
            },
            {
                "view_id": "reversal_events",
                "name": "Reversal events",
                "purpose": "Expose each located boundary in the nested analytical result",
                "grain": "event",
                "source_ref": "insight:insight_nested_reversal",
                "record_path": "$.items[*].items",
                "fields": [
                    {"name": "event_role", "semantic_role": "event_role", "source_path": "$.role"},
                    {"name": "time", "semantic_role": "event_time", "source_path": "$.timestamp"},
                    {"name": "price", "semantic_role": "event_price", "source_path": "$.value"},
                ],
            },
        ],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Verify the decline-to-rise interval in context",
            "title": "Reversal in context",
            "priority": "primary",
            "summary": "The full series provides context and the nested interval records mark its boundaries.",
            "required_roles": ["complete_series", "reversal_events"],
            "layers": [
                {
                    "role": "complete_series",
                    "source_ref": "semantic:full_series",
                    "mark": "line",
                    "encoding": {"x": "time", "y": "price"},
                },
                {
                    "role": "reversal_events",
                    "source_ref": "semantic:reversal_events",
                    "mark": "point",
                    "encoding": {"x": "time", "y": "price", "series": "event_role"},
                },
            ],
        }],
        "required_data_request": None,
    }
    llm = _TwoStagePlannerLlm(projection=projection, chart=chart)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(
            message="Show the full series and the exact decline-to-rise interval.",
            source_refs=["insight:nested_reversal"],
        ),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert result["status"] == "created"
    assert len(complete.datasets[0].series[0].points) == 6
    assert sum(len(series.points) for series in complete.datasets[1].series) == 3
    assert {series.name.rsplit(": ", 1)[-1] for series in complete.datasets[1].series} == {
        "decline_end", "turn", "rise_start",
    }
    prompts = [
        *llm.verification_prompts,
        *llm.projection_prompts,
        *llm.chart_prompts,
        *llm.audit_prompts,
    ]
    assert prompts
    assert all("987654.321" not in str(messages) for messages in prompts)
    assert "The selected interval declines into a turning point and then rises." in str(
        llm.projection_prompts[0]
    )
    assert '"source_ref": "view:evidence:evi_full:default"' in str(llm.projection_prompts[0])
    assert '"source_ref": "insight:insight_nested_reversal"' in str(llm.chart_prompts[0])


@pytest.mark.asyncio
async def test_semantic_projection_repairs_from_path_execution_feedback_without_chart_fallback(tmp_path):
    invalid_projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose the requested temporal measure",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.missing.measure"},
            ],
        }],
        "required_data_request": None,
    }
    repaired_projection = {
        **invalid_projection,
        "semantic_views": [{
            **invalid_projection["semantic_views"][0],
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
    }
    chart = {
        "visual_goals": [{
            "purpose": "Show the observed series",
            "title": "Observed series",
            "priority": "primary",
            "summary": None,
            "required_roles": ["observed_series"],
            "layers": [{
                "role": "observed_series",
                "source_ref": "semantic:observations",
                "mark": "line",
                "encoding": {"x": "time", "y": "measure"},
            }],
        }],
        "required_data_request": None,
    }
    llm = _RepairingTwoStagePlannerLlm(
        projections=[invalid_projection, repaired_projection], chart=chart,
    )

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed series."), request_state=_state(5))

    assert len(llm.projection_prompts) == 2
    assert "semantic source path '$.missing.measure' is unavailable in every record" in llm.projection_prompts[1][0][1]
    assert llm.chart_calls == 1
    assert result["visualizations"][0]["datasets"][0]["row_count"] == 5


@pytest.mark.asyncio
async def test_chart_requirement_repairs_ungrounded_input_evidence_with_llm(tmp_path):
    projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose observations",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": "$.records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    requirement = {
        "required_action": "code_interpreter",
        "purpose": "Calculate a missing interval",
        "message": None,
        "required_shape": "intervals",
        "required_fields": ["time", "lower", "upper"],
        "required_properties": ["aligned with observations"],
        "insight_requests": [{
            "name": "prediction interval",
            "insight_type": "prediction_interval",
            "insight_key": "prediction_interval",
        }],
    }
    invalid = {"visual_goals": [], "required_data_request": {**requirement, "input_evidence": "the observed data"}}
    repaired = {"visual_goals": [], "required_data_request": {**requirement, "input_evidence": "semantic:observations"}}
    llm = _RepairingTwoStagePlannerLlm(
        projections=[projection], chart=[invalid, repaired],
    )

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show a real interval."), request_state=_state(5))

    assert llm.chart_calls == 2
    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "code_interpreter"
    assert result["required_data_request"]["input_source_refs"] == ["evidence:evi_full"]


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

    assert llm.calls == 2  # invalid materialization plus repaired chart plan
    assert result["visualizations"][0]["datasets"][0]["row_count"] == 25


@pytest.mark.asyncio
async def test_incompatible_chart_domains_are_replanned_inside_the_tool_boundary(tmp_path):
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
        '"summary":null,"required_roles":["base_series","endpoints"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"endpoints","source_ref":"view:derived_evidence:dev_endpoints","mark":"point",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    llm = _SequencePlannerLlm([invalid, repaired])

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Verify interval change.",
            source_refs=["evidence:evi_full", "analysis:ana_demo"],
        ),
        request_state=state,
    )

    assert llm.calls == 2  # incompatible domains plus repaired chart plan
    assert result["visualizations"][0]["required_roles"] == ["base_series", "endpoints"]


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

    result = await tool.execute(VisualizationInput(message="Show all points and max."), request_state=_state(1))

    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "sql_query"


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

    result = await tool.execute(VisualizationInput(message="Show the full series and optimal trade."), request_state=_state(25))

    request = result["required_data_request"]
    assert result["status"] == "needs_sources"
    assert request["required_action"] == "code_interpreter"
    assert request["insight_requests"][0]["insight_key"] == "optimal_trade"


@pytest.mark.asyncio
async def test_visualization_normalizes_view_ref_for_code_interpreter_repair_contract(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[],"required_data_request":{"required_action":"code_interpreter",'
        '"purpose":"derive renderer-ready dimensions","required_shape":"records",'
        '"required_fields":["timestamp","value"],"required_properties":["preserve values"],'
        '"input_evidence":"view:evidence:evi_full:default","insight_requests":[{'
        '"insight_key":"renderer_dimensions","name":"Renderer dimensions","insight_type":"series"}]}}'
    )
    tool = VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path))

    result = await tool.execute(VisualizationInput(message="Show a specialized chart."), request_state=_state(25))

    assert result["required_data_request"]["input_source_refs"] == ["evidence:evi_full"]


@pytest.mark.asyncio
async def test_visualization_planner_requests_missing_decision_points(tmp_path):
    plan = (
        '{"visual_goals":[],"required_data_request":{'
        '"required_action":"code_interpreter","purpose":"calculate requested buy and sell points",'
        '"required_shape":"decision_points","required_fields":["timestamp","value"],'
        '"required_properties":["buy precedes sell"],"insight_requests":[{'
        '"insight_key":"optimal_trade","name":"Optimal single trade","insight_type":"optimization"}]}}'
    )
    tool = VisualizationTool(
        llm=_PlannerLlm(plan), artifact_store=VisualizationArtifactStore(tmp_path),
    )

    result = await tool.execute(
        VisualizationInput(message="Show the complete series with buy and sell points."),
        request_state=_state(25),
    )

    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "code_interpreter"


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
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}',
        audit_payload=(
            '{"decision":"approve","issues":[],"required_data_request":null}'
        ),
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


@pytest.mark.asyncio
async def test_visualization_carries_key_insight_verification_through_projection_and_artifact(tmp_path):
    state = _state(25)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_max",
        insight_key="maximum_value",
        name="Maximum value",
        insight_type="extreme",
        statement="The maximum value is 24 at the end of the interval.",
        value=24.0,
        method="sql_query",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
    )]
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"verify the maximum in context","title":"Maximum in full context",'
        '"priority":"primary","summary":"The full interval makes the maximum inspectable.",'
        '"required_roles":["complete_series"],"layers":[{"role":"complete_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Observed value"}]}],'
        '"required_data_request":null}'
    )

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Show the maximum in its full interval.",
            source_refs=["insight:maximum_value"],
        ),
        request_state=state,
    )

    verification = result["visualizations"][0]["verification"]
    assert verification["target_insight_ids"] == ["insight_max"]
    assert verification["verification_question"]
    assert verification["interpretation"]
    assert '"target_insight_ids": ["insight_max"]' in str(llm.projection_prompts[0][0][1])
    assert '"source_ref": "view:evidence:evi_full:default"' in str(llm.projection_prompts[0][0][1])
    assert '"preferred_by_caller": true' in str(llm.projection_prompts[0][0][1])


@pytest.mark.asyncio
async def test_visualization_preserves_full_series_and_highlights_interval_inside_broader_viewport(tmp_path):
    state = _state(100)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_reversal",
        insight_key="reversal_interval",
        name="Reversal interval",
        insight_type="interval",
        statement="The series falls and then rises inside the located interval.",
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
        items=[
            InsightItem(item_id="start", timestamp="2026-01-02T12:00:00Z", value=36.0, label="Start"),
            InsightItem(item_id="end", timestamp="2026-01-03T12:00:00Z", value=60.0, label="End"),
        ],
    )]
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"verify the located reversal interval",'
        '"title":"Reversal in monthly context","priority":"primary",'
        '"summary":"The full series remains scrollable while the located interval is emphasized.",'
        '"required_roles":["complete_series","highlighted_interval"],'
        '"presentation":{"dataZoom":[{"type":"inside",'
        '"startValue":"2026-01-02T00:00:00Z","endValue":"2026-01-04T00:00:00Z"}]},'
        '"layers":[{"role":"complete_series","source_ref":"view:evidence:evi_full:default",'
        '"mark":"line","encoding":{"x":"timestamp","y":"value"}},'
        '{"role":"highlighted_interval","source_ref":"view:evidence:evi_full:default",'
        '"mark":"line","encoding":{"x":"timestamp","y":"value"},'
        '"transform":[{"type":"filter","field":"timestamp","operator":"between",'
        '"value":["2026-01-02T12:00:00Z","2026-01-03T12:00:00Z"]}],'
        '"presentation":{"lineStyle":{"width":5}}}]}],"required_data_request":null}'
    )

    store = VisualizationArtifactStore(tmp_path)
    result = await VisualizationTool(
        llm=llm,
        artifact_store=store,
    ).execute(
        VisualizationInput(
            message="Show the full series and highlight the located reversal interval.",
            source_refs=["insight:reversal_interval"],
        ),
        request_state=state,
    )

    visualization = store.get(result["visualization_ids"][0])
    assert visualization is not None
    assert len(visualization.datasets[0].series[0].points) == 100
    assert len(visualization.datasets[1].series[0].points) == 25
    assert visualization.presentation["dataZoom"][0] == {
        "type": "inside",
        "startValue": "2026-01-02T00:00:00Z",
        "endValue": "2026-01-04T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_not_visualizable_conclusion_is_closed_without_publishing_fallback(tmp_path):
    llm = _VerificationOnlyLlm({
        "decision": "not_visualizable",
        "target_insight_ids": [],
        "verification_question": None,
        "interpretation": "The requested causal explanation cannot be verified by an observational chart.",
        "visual_relation": None,
        "required_context": [],
        "non_visual_insight_ids": [],
        "required_data_request": None,
    })

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Prove what caused the change."), request_state=_state(25))

    assert result["status"] == "unavailable"
    assert "causal explanation" in result["unavailable_reason"]
    assert result["visualization_ids"] == []
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_completed_visual_dependency_becomes_unavailable_instead_of_looping(tmp_path):
    requirement = {
        "required_action": "sql_query",
        "purpose": "load the missing complete comparison set",
        "message": "Load every comparison candidate.",
        "required_shape": "complete_records",
        "required_fields": ["category", "value"],
        "required_properties": ["complete comparison set"],
        "input_evidence": None,
        "input_source_refs": [],
        "insight_requests": [],
    }
    state = _state(25)
    state.observations = [
        ToolObservation(
            tool_name="visualization",
            success=True,
            summary="Additional evidence is required.",
            payload={"status": "needs_sources", "required_data_request": requirement},
        ),
        ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="The source owner completed.",
            payload={"evidence_id": "evi_full"},
        ),
    ]
    llm = _VerificationOnlyLlm({
        "decision": "needs_sources",
        "target_insight_ids": [],
        "verification_question": None,
        "interpretation": None,
        "visual_relation": "complete comparison",
        "required_context": ["every comparison candidate"],
        "non_visual_insight_ids": [],
        "required_data_request": requirement,
    })

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Compare every candidate."), request_state=state)

    assert result["status"] == "unavailable"
    assert "remained unavailable" in result["unavailable_reason"]
    assert result["required_data_request"] is None


@pytest.mark.asyncio
async def test_disabled_render_audit_does_not_block_candidate_publication(tmp_path):
    initial = (
        '{"visual_goals":[{"purpose":"trend","title":"Ambiguous trend","priority":"primary",'
        '"summary":null,"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Value"}]}],"required_data_request":null}'
    )
    repaired = (
        '{"visual_goals":[{"purpose":"trend","title":"Observed value over time","priority":"primary",'
        '"summary":"Read the observed series from the first timestamp to the last.",'
        '"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Observed value"}]}],"required_data_request":null}'
    )
    render_auditor = _RenderAuditStub([
        {"decision": "revise", "issues": ["The title does not state what is observed."]},
        {"decision": "approve", "issues": []},
    ])

    result = await VisualizationTool(
        llm=_SequencePlannerLlm([initial, repaired]),
        artifact_store=VisualizationArtifactStore(tmp_path),
        render_auditor=render_auditor,
    ).execute(VisualizationInput(message="Show the observed trend."), request_state=_state(25))

    assert result["status"] == "created"
    assert result["visualizations"][0]["title"] == "Ambiguous trend"
    assert render_auditor.calls == []
    assert [item.name for item in tmp_path.glob("*.json")] == [
        f'{result["visualization_ids"][0]}.json'
    ]


@pytest.mark.asyncio
async def test_disabled_candidate_semantic_audit_is_not_invoked(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Observed trend","priority":"primary",'
        '"summary":null,"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Observed value"}]}],"required_data_request":null}',
        audit_payload=(
            '{"decision":"unavailable","issues":["Do not publish this candidate"],'
            '"required_data_request":null}'
        ),
    )

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed trend."), request_state=_state(25))

    assert result["status"] == "created"
    assert llm.audit_prompts == []
