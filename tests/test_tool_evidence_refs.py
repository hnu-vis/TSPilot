from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsightRequest
from schemas.state import RequestStateModel
from tools.anomaly import AnomalyInput, AnomalyTool
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool
from tools.forecast import ForecastInput, ForecastTool


class _BinderLLM:
    async def ainvoke(self, _messages):
        return SimpleNamespace(content=json.dumps({
            "bindings": [{"insight_key": "point_count", "statement": "There are six points."}]
        }))


def _evidence():
    points = [
        {"timestamp": f"2023-01-01T0{index}:00:00Z", "value": float(index + 1)}
        for index in range(6)
    ]
    return DatabaseEvidence(
        evidence_id="evi_ref", result_type="timeseries", database="demo",
        query_language="unit", query="unit:test", summary="six points",
        data={"points": points, "rows": points}, columns=["timestamp", "value"],
    )


def _state():
    evidence = _evidence()
    return RequestStateModel(
        request_id="req_refs", message="analyze", status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )


@pytest.mark.asyncio
async def test_code_interpreter_resolves_lightweight_evidence_ref_from_state():
    result = await CodeInterpreterTool(llm=_BinderLLM()).execute(
        CodeInterpreterInput(
            database_evidence="evidence:evi_ref",
            analysis_goal="count points",
            code=(
                "result = {'computed_insights': [{"
                "'insight_key': 'point_count', 'value': len(rows), "
                "'calculation_trace': {'operation': 'len(rows)'}}], 'derived_evidence': []}"
            ),
            insight_requests=[KeyInsightRequest(
                insight_key="point_count", name="Point count", insight_type="count",
            )],
        ),
        request_state=_state(),
    )
    assert result["input_evidence_id"] == "evi_ref"
    assert result["computed_insights"][0]["value"] == 6


@pytest.mark.asyncio
async def test_specialized_timeseries_tools_resolve_evidence_refs():
    state = _state()
    forecast = await ForecastTool().execute(
        ForecastInput(database_evidence="evidence:evi_ref", horizon=2), request_state=state,
    )
    anomaly = await AnomalyTool().execute(
        AnomalyInput(database_evidence="evidence:evi_ref"), request_state=state,
    )
    assert forecast["diagnostics"]["selected_evidence_id"] == "evi_ref"
    assert anomaly["diagnostics"]["resolved_evidence_id"] == "evi_ref"


@pytest.mark.asyncio
async def test_analysis_tools_reject_unknown_evidence_refs_instead_of_using_latest():
    with pytest.raises(ValueError, match="grounded database_evidence"):
        await CodeInterpreterTool(llm=_BinderLLM()).execute(
            CodeInterpreterInput(
                database_evidence="evidence:missing", analysis_goal="bad ref",
                code="result = {'computed_insights': [{'insight_key': 'point_count', 'value': 1, 'calculation_trace': {'operation': 'count'}}], 'derived_evidence': []}",
                insight_requests=[KeyInsightRequest(
                    insight_key="point_count", name="Point count", insight_type="count",
                )],
            ),
            request_state=_state(),
        )
    with pytest.raises(ValueError, match="could not resolve database_evidence"):
        await ForecastTool().execute(
            ForecastInput(database_evidence="evidence:missing", horizon=2), request_state=_state(),
        )
