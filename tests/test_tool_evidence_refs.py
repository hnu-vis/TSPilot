from __future__ import annotations

import pytest

from core.timeseries.anomaly_registry import AnomalyDetectorOutput, register_anomaly_detector
from core.timeseries.forecast_registry import ForecastModelOutput, register_forecast_model
from schemas.database import DatabaseEvidence
from schemas.timeseries import TimeSeriesPoint
from schemas.state import RequestStateModel
from tools.anomaly import AnomalyInput, AnomalyTool
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool
from tools.forecast import ForecastInput, ForecastTool


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

    code_interpreter = await CodeInterpreterTool().execute(
        CodeInterpreterInput(
            database_evidence=f"evidence:{evidence.evidence_id}",
            analysis_goal="compute point count",
            code="result = {'summary': f'{len(rows)} rows available', 'metrics': {'row_count': len(rows)}, 'details': {}}",
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

    assert code_interpreter["analysis_id"].startswith("ana_compute_point_count_")
    assert anomaly["anomaly_id"] == "anomaly_evi_ref"
    assert anomaly["detector_name"] == "zscore"
    assert anomaly["diagnostics"]["detector_registry_name"] == "zscore"
    assert forecast["forecast_id"] == "forecast_evi_ref"
    assert forecast["model_name"] == "linear_regression"
    assert forecast["diagnostics"]["model_registry_name"] == "linear_regression"
    assert forecast["horizon"] == 2


@pytest.mark.asyncio
async def test_forecast_and_anomaly_tools_use_registered_models_from_input():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_registered_models",
        message="分析异常和预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    class UnitForecastModel:
        name = "unit_forecast"

        def forecast(self, series, *, horizon: int, params: dict):
            return ForecastModelOutput(
                forecast_points=[
                    TimeSeriesPoint(timestamp=series.points[-1].timestamp, value=float(params["constant_value"]))
                    for _ in range(horizon)
                ],
                diagnostics={"constant_value": params["constant_value"]},
            )

    class UnitAnomalyDetector:
        name = "unit_detector"

        def detect(self, series, *, params: dict):
            point = series.points[int(params.get("index", 0))]
            return AnomalyDetectorOutput(
                anomaly_points=[{"timestamp": point.timestamp, "value": point.value, "score": 99.0}],
                scores=[{"timestamp": point.timestamp, "score": 99.0}],
                diagnostics={"index": int(params.get("index", 0))},
            )

    register_forecast_model(UnitForecastModel())
    register_anomaly_detector(UnitAnomalyDetector())

    forecast = await ForecastTool().execute(
        ForecastInput(
            database_evidence="latest",
            horizon=3,
            model_name="unit_forecast",
            constraints={"constant_value": 123.0},
        ),
        request_state=request_state,
    )
    anomaly = await AnomalyTool().execute(
        AnomalyInput(
            database_evidence="latest",
            detector_name="unit_detector",
            constraints={"index": 2},
        ),
        request_state=request_state,
    )

    assert forecast["model_name"] == "unit_forecast"
    assert [point["value"] for point in forecast["forecast_points"]] == [123.0, 123.0, 123.0]
    assert forecast["diagnostics"]["model_registry_name"] == "unit_forecast"
    assert anomaly["detector_name"] == "unit_detector"
    assert anomaly["anomaly_points"][0]["value"] == 21.0
    assert anomaly["diagnostics"]["detector_registry_name"] == "unit_detector"


@pytest.mark.asyncio
async def test_forecast_and_anomaly_tools_reject_unknown_registered_model_names():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_unknown_models",
        message="分析异常和预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    with pytest.raises(ValueError, match="Unknown forecast model .*linear_regression"):
        await ForecastTool().execute(
            ForecastInput(database_evidence="latest", model_name="missing_forecast"),
            request_state=request_state,
        )
    with pytest.raises(ValueError, match="Unknown anomaly detector .*zscore"):
        await AnomalyTool().execute(
            AnomalyInput(database_evidence="latest", detector_name="missing_detector"),
            request_state=request_state,
        )


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
