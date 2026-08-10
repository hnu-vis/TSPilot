from __future__ import annotations

import asyncio

import pytest

from core.data_fact.runtime import register_data_facts_from_payload
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.data_fact import DataFact, FactItem
from schemas.database_context import DatabaseContext
from schemas.output import AnswerClaim, FinalAnswer
from schemas.state import RequestStateModel
from schemas.tool import ToolCall
from tools.anomaly import AnomalyInput
from tools.forecast import ForecastInput
from tools.format_answer import FormatAnswerInput, FormatAnswerTool
from runtime.request_state import public_final_answer


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


def test_collection_fact_preserves_item_identity_and_generates_answer_visualization():
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
                "fact_requests": [
                    {
                        "fact_key": "price.recent_3_days",
                        "name": "recent three days",
                        "fact_type": "value",
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
    coverage = register_data_facts_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_recent_three_days",
            "analysis_goal": "recent three days",
            "input_evidence_id": "evi_observation_viz",
            "result": {
                "facts": [
                    {
                        "fact_key": "price.recent_3_days",
                        "name": "recent three days",
                        "fact_type": "value",
                        "semantic_class": "measurement_series",
                        "derivation": "select_recent",
                        "value_shape": "series",
                        "statement": "The recent three daily prices are 10, 12, and 15.",
                        "value": [
                            {"item_id": "day-1", "timestamp": "2023-01-01", "value": 10.0},
                            {"item_id": "day-2", "timestamp": "2023-01-02", "value": 12.0},
                            {"item_id": "day-3", "timestamp": "2023-01-03", "value": 15.0},
                        ],
                        "calculation_trace": {"selection_rule": "recent three observations"},
                    }
                ],
                "metrics": {},
                "details": {},
            },
        },
    )

    fact = request_state.fact_set.facts[0]
    assert coverage.verified == ["recent three days"]
    assert fact.value_shape == "series"
    assert [item.item_id for item in fact.items] == ["day-1", "day-2", "day-3"]
    assert all(item.evidence_refs for item in fact.items)

    answer = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(include_fact_ids=[fact.fact_id]),
            request_state=request_state,
        )
    )
    visualization = next(item for item in answer["visualizations"] if item["visualization_id"] == f"viz_fact_{fact.fact_id}")
    assert visualization["visualization_type"] == "chart"
    assert len(visualization["bindings"]) == 3
    assert answer["claims"][0]["item_ids"] == ["day-1", "day-2", "day-3"]


def test_pairwise_fact_items_can_point_to_source_items():
    fact = DataFact(
        fact_id="fact_pairwise",
        fact_key="price.top3.pairwise_difference",
        name="Top 3 pairwise difference",
        fact_type="difference",
        semantic_class="comparison_set",
        derivation="pairwise_difference",
        value_shape="pairwise_set",
        statement="Pairwise differences for the Top 3 values.",
        value=None,
        items=[
            FactItem(item_id="p1-p2", value=20, source_item_ids=["p1", "p2"]),
            FactItem(item_id="p1-p3", value=40, source_item_ids=["p1", "p3"]),
            FactItem(item_id="p2-p3", value=20, source_item_ids=["p2", "p3"]),
        ],
        method="code_interpreter",
        calculation_trace={"formula": "a - b"},
    )

    assert len(fact.items) == 3
    assert fact.items[1].source_item_ids == ["p1", "p3"]


def test_forecast_and_anomaly_inputs_do_not_accept_fact_contracts():
    with pytest.raises(ValueError, match="analysis artifact"):
        ForecastInput(fact_requests=[{"name": "forecast value", "fact_type": "value"}])
    with pytest.raises(ValueError, match="analysis artifact"):
        AnomalyInput(fact_requests=[{"name": "anomaly count", "fact_type": "count"}])


def test_public_final_answer_preserves_claims():
    answer = FinalAnswer(
        summary="A grounded result.",
        claims=[AnswerClaim(claim_id="claim-1", text="A grounded result.", fact_ids=["fact-1"])],
    )
    public = public_final_answer(answer)
    assert public.claims[0].fact_ids == ["fact-1"]
