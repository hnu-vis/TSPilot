from __future__ import annotations

import pytest

from schemas.database import DatabaseEvidence
from schemas.state import RequestStateModel
from tools.anomaly import AnomalyInput, AnomalyTool
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool
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
        InsightInput(
            database_evidence=f"evidence:{evidence.evidence_id}",
            analysis_goal="compute point count",
            analysis_code="result = {'summary': f'{len(rows)} rows available', 'metrics': {'row_count': len(rows)}, 'details': {}}",
        ),
        request_state=request_state,
    )
    anomaly = await AnomalyTool().execute(
        AnomalyInput(database_evidence="latest_database_evidence"),
        request_state=request_state,
    )
    forecast = await ForecastTool().execute(
        ForecastInput(database_evidence=evidence_ref, horizon={"steps": "next 2 points"}),
        request_state=request_state,
    )

    assert insight["analysis_id"].startswith("ana_compute_point_count_")
    assert anomaly["anomaly_id"] == "anomaly_evi_ref"
    assert forecast["forecast_id"] == "forecast_evi_ref"
    assert forecast["horizon"] == 2


@pytest.mark.asyncio
async def test_analysis_tools_reject_unknown_evidence_refs_instead_of_using_latest():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_refs",
        message="分析趋势、异常和预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    with pytest.raises(ValueError, match="could not resolve database_evidence"):
        await InsightTool().execute(
            InsightInput(
                database_evidence="evidence:missing",
                analysis_goal="bad ref",
                analysis_code="result = {'summary': 'ok', 'metrics': {}, 'details': {}}",
            ),
            request_state=request_state,
        )
    with pytest.raises(ValueError, match="could not resolve database_evidence"):
        await CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence="evidence:missing",
                analysis_goal="bad ref",
                code="result = {'summary': 'ok', 'metrics': {}, 'details': {}}",
            ),
            request_state=request_state,
        )
    with pytest.raises(ValueError, match="could not resolve database_evidence"):
        await AnomalyTool().execute(
            AnomalyInput(database_evidence="evidence:missing"),
            request_state=request_state,
        )
    with pytest.raises(ValueError, match="could not resolve database_evidence"):
        await ForecastTool().execute(
            ForecastInput(database_evidence="evidence:missing"),
            request_state=request_state,
        )
