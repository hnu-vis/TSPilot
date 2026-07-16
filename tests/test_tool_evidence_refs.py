from __future__ import annotations

import pytest

from schemas.database import DatabaseEvidence
from schemas.state import RequestStateModel
from tools.anomaly import AnomalyInput, AnomalyTool
from tools.forecast import ForecastInput, ForecastTool
from tools.insight import InsightInput, InsightTool


def _evidence() -> DatabaseEvidence:
    points = [
        {"timestamp": f"2023-01-01T0{index}:00:00Z", "value": float(index * 10 + 1)}
        for index in range(6)
    ]
    return DatabaseEvidence(
        evidence_id="evi_ref",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:test",
        summary="demo evidence",
        data={
            "points": points,
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        columns=["timestamp", "value"],
    )


@pytest.mark.asyncio
async def test_analysis_tools_resolve_lightweight_evidence_refs_from_state():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_refs",
        message="分析趋势、异常和预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )
    evidence_ref = {"evidence_id": evidence.evidence_id}

    insight = await InsightTool().execute(
        InsightInput(database_evidence=evidence_ref, requested_fact_types=["trend"]),
        request_state=request_state,
    )
    anomaly = await AnomalyTool().execute(
        AnomalyInput(database_evidence=evidence_ref),
        request_state=request_state,
    )
    forecast = await ForecastTool().execute(
        ForecastInput(database_evidence=evidence_ref, horizon=2),
        request_state=request_state,
    )

    assert insight["insight_id"].startswith("ins_evi_ref")
    assert anomaly["anomaly_id"] == "anomaly_evi_ref"
    assert forecast["forecast_id"] == "forecast_evi_ref"
