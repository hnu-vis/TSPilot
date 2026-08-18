from __future__ import annotations

import asyncio

import pytest

from core.key_insight.runtime import register_key_insights_from_payload
from core.visualization import VisualizationMaterializer
from runtime.request_state import build_request_state, public_final_answer
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsight, InsightItem
from schemas.database_context import DatabaseContext
from schemas.output import AnswerClaim, FinalAnswer, FinalResponsePlan, PlannedAnswerSection, VisualGoal, VisualLayerPlan
from schemas.state import RequestStateModel
from schemas.tool import ToolCall
from tools.anomaly import AnomalyInput
from tools.forecast import ForecastInput


def _evidence() -> DatabaseEvidence:
    return DatabaseEvidence(
        evidence_id="evi_observation_viz",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:test",
        summary="Three daily values.",
        data={
            "rows": [
                {"timestamp": "2023-01-01", "value": 10.0},
                {"timestamp": "2023-01-02", "value": 12.0},
                {"timestamp": "2023-01-03", "value": 15.0},
            ]
        },
        columns=["timestamp", "value"],
        metadata={},
        diagnostics={},
    )


def test_collection_insight_preserves_item_identity_for_visualization():
    request_state = RequestStateModel(
        request_id="req_observation_viz",
        message="最近三天的价格",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="unit"),
        latest_database_evidence=_evidence(),
        database_evidence_artifacts={"evi_observation_viz": _evidence()},
    )
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {
                        "insight_key": "price.recent_3_days",
                        "name": "recent three days",
                        "insight_type": "value",
                        "semantic_class": "measurement_series",
                        "derivation": "select_recent",
                        "result_shape": "series",
                        "expected_item_count": 3,
                        "selection": {"order_by": "timestamp", "direction": "asc"},
                    }
                ]
            },
        )
    )
    coverage = register_key_insights_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_recent_three_days",
            "analysis_goal": "recent three days",
            "input_evidence_id": "evi_observation_viz",
            "produced_insights": [{
                "insight_id": "ins_recent_three_days",
                "insight_key": "price.recent_3_days",
                "name": "recent three days",
                "insight_type": "value",
                "semantic_class": "measurement_series",
                "derivation": "select_recent",
                "value_shape": "series",
                "statement": "The recent three daily prices are 10, 12, and 15.",
                "items": [
                    {"item_id": "day-1", "timestamp": "2023-01-01", "value": 10.0},
                    {"item_id": "day-2", "timestamp": "2023-01-02", "value": 12.0},
                    {"item_id": "day-3", "timestamp": "2023-01-03", "value": 15.0},
                ],
                "method": "code_interpreter",
                "evidence_refs": [
                    {"source_type": "analysis", "source_id": "ana_recent_three_days"},
                    {"source_type": "query", "source_id": "evi_observation_viz"},
                ],
                "calculation_trace": {"selection_rule": "recent three observations"},
            }],
        },
    )

    insight = request_state.insight_set.insights[0]
    assert coverage.verified == ["recent three days"]
    assert insight.value_shape == "series"
    assert [item.item_id for item in insight.items] == ["day-1", "day-2", "day-3"]
    assert insight.evidence_refs

    visualization = VisualizationMaterializer(request_state).materialize(VisualGoal(
        purpose="show recent observations", title="Recent three days",
        required_roles=["observations"],
        layers=[VisualLayerPlan(
            role="observations", source_ref=f"insight:{insight.insight_id}", mark="line",
            encoding={"x": "timestamp", "y": "value"},
        )],
    ))
    assert visualization.schema_version == "3"
    assert len(visualization.bindings) == 3
    assert {binding.item_id for binding in visualization.bindings} == {"day-1", "day-2", "day-3"}


def test_pairwise_insight_items_can_point_to_source_items():
    insight = KeyInsight(
        insight_id="insight_pairwise",
        insight_key="price.top3.pairwise_difference",
        name="Top 3 pairwise difference",
        insight_type="difference",
        semantic_class="comparison_set",
        derivation="pairwise_difference",
        value_shape="pairwise_set",
        statement="Pairwise differences for the Top 3 values.",
        value=None,
        items=[
            InsightItem(item_id="p1-p2", value=20, source_item_ids=["p1", "p2"]),
            InsightItem(item_id="p1-p3", value=40, source_item_ids=["p1", "p3"]),
            InsightItem(item_id="p2-p3", value=20, source_item_ids=["p2", "p3"]),
        ],
        method="code_interpreter",
        calculation_trace={"formula": "a - b"},
    )

    assert len(insight.items) == 3
    assert insight.items[1].source_item_ids == ["p1", "p3"]


def test_forecast_and_anomaly_inputs_do_not_accept_insight_contracts():
    with pytest.raises(ValueError, match="analysis artifact"):
        ForecastInput(insight_requests=[{"name": "forecast value", "insight_type": "value"}])
    with pytest.raises(ValueError, match="analysis artifact"):
        AnomalyInput(insight_requests=[{"name": "anomaly count", "insight_type": "count"}])


def test_public_final_answer_preserves_claims():
    answer = FinalAnswer(
        summary="A grounded result.",
        claims=[AnswerClaim(claim_id="claim-1", text="A grounded result.", insight_ids=["insight-1"])],
    )
    public = public_final_answer(answer)
    assert public.claims[0].insight_ids == ["insight-1"]
