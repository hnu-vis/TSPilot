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
async def test_analysis_request_returns_only_requested_metrics():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_requested_metrics",
        message="最大值和最小值的差异是多少？",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    result = await CodeInterpreterTool().execute(
        CodeInterpreterInput(
            database_evidence="latest",
            analysis_goal="最大值和最小值的差异是多少？",
            analysis_request={
                "required_outputs": ["最大值", "最小值", "最大值和最小值的差异"],
            },
        ),
        request_state=request_state,
    )

    metrics = result["result"]["metrics"]
    assert metrics == {
        "max_value": 51.0,
        "min_value": 1.0,
        "max_min_difference": 50.0,
    }
    details = result["result"]["details"]
    assert set(details) == {"columns", "requested_outputs", "max_time", "max_point", "min_time", "min_point"}
    assert "record_count" not in result["summary"]
    assert "start_end_change" not in metrics


@pytest.mark.asyncio
async def test_analysis_request_requires_code_for_uncovered_metrics():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_code_required_metrics",
        message="计算收益率、波动率和最大回撤",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    with pytest.raises(Exception) as exc_info:
        await CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence="latest",
                analysis_goal="计算收益率、波动率和最大回撤",
                analysis_request={"required_outputs": ["returns_volatility_max_drawdown"]},
            ),
            request_state=request_state,
        )

    exc = exc_info.value
    assert getattr(exc, "error_type", None) == "code_required_for_metrics"
    payload = exc.to_observation_payload()
    missing = payload["diagnostics"]["missing_metrics"]
    assert missing == ["total_return", "volatility", "max_drawdown"]
    assert payload["recommended_next_action"] == "code_interpreter"
    assert payload["validation_failure"]["repair_contract"]["mode"] == "generated_code_required"


@pytest.mark.asyncio
async def test_generated_code_computes_uncovered_metrics():
    evidence = _evidence()
    request_state = RequestStateModel(
        request_id="req_generated_code_metrics",
        message="计算收益率、波动率和最大回撤",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    result = await CodeInterpreterTool().execute(
        CodeInterpreterInput(
            database_evidence="latest",
            analysis_goal="计算收益率、波动率和最大回撤",
            required_outputs=["total_return", "volatility", "max_drawdown"],
            code=(
                "import statistics\n"
                "values = [float(row.get('value')) for row in rows if row.get('value') is not None]\n"
                "returns = [(right / left) - 1 for left, right in zip(values, values[1:]) if left != 0]\n"
                "peak = values[0]\n"
                "max_drawdown = 0.0\n"
                "for value in values:\n"
                "    peak = max(peak, value)\n"
                "    if peak:\n"
                "        max_drawdown = min(max_drawdown, value / peak - 1)\n"
                "result = {\n"
                "    'summary': 'computed financial metrics',\n"
                "    'metrics': {\n"
                "        'total_return': values[-1] / values[0] - 1,\n"
                "        'volatility': statistics.stdev(returns) if len(returns) > 1 else 0.0,\n"
                "        'max_drawdown': max_drawdown,\n"
                "    },\n"
                "    'details': {'return_count': len(returns)},\n"
                "}\n"
            ),
        ),
        request_state=request_state,
    )

    assert result["code_type"] == "code_interpreter_v1"
    assert set(result["result"]["metrics"]) == {"total_return", "volatility", "max_drawdown"}
    assert result["result"]["details"]["return_count"] == 5


@pytest.mark.asyncio
async def test_generated_code_execution_error_has_repair_contract():
    evidence = _evidence()
    with pytest.raises(Exception) as exc_info:
        await CodeInterpreterTool().execute(
            CodeInterpreterInput(
                database_evidence=evidence,
                analysis_goal="计算收益率、波动率和最大回撤",
                required_outputs=["total_return", "volatility", "max_drawdown"],
                code="df['returns'] = df['price'].pct_change()\nresult = {}",
            )
        )

    exc = exc_info.value
    assert getattr(exc, "error_type", None) == "analysis_code_execution_failed"
    validation_failure = getattr(exc, "validation_failure", {})
    repair_contract = validation_failure["repair_contract"]
    assert repair_contract["mode"] == "code_execution_repair"
    assert repair_contract["available_inputs"]["rows"] == "list[dict]"
    assert repair_contract["available_inputs"]["points"] == "list[dict]"
    assert "columns" in repair_contract["available_inputs"]
    assert "df, data" in repair_contract["instruction"]


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
async def test_anomaly_defaults_to_prior_raw_evidence_when_latest_is_outlier_filtered():
    raw = _evidence()
    cleaned = raw.model_copy(
        update={
            "evidence_id": "evi_cleaned",
            "query": "unit:test |> filter outliers",
            "summary": "cleaned evidence after excluding outliers",
            "data": {
                **raw.data,
                "points": raw.data["points"][1:],
            },
            "metadata": {"purpose": "过滤异常点后的清洗序列"},
        }
    )
    request_state = RequestStateModel(
        request_id="req_anomaly_raw_policy",
        message="检测异常并预测。",
        status="running",
        latest_database_evidence=cleaned,
        database_evidence_artifacts={
            raw.evidence_id: raw,
            cleaned.evidence_id: cleaned,
        },
    )

    class CountDetector:
        name = "count_detector"

        def detect(self, series, *, params: dict):
            return AnomalyDetectorOutput(
                anomaly_points=[],
                scores=[],
                diagnostics={"input_point_count": len(series.points)},
            )

    register_anomaly_detector(CountDetector())

    result = await AnomalyTool().execute(
        AnomalyInput(database_evidence="latest", detector_name="count_detector"),
        request_state=request_state,
    )

    assert result["diagnostics"]["selected_evidence_id"] == "evi_cleaned"
    assert result["diagnostics"]["resolved_evidence_id"] == "evi_ref"
    assert result["diagnostics"]["input_point_count"] == 6


@pytest.mark.asyncio
async def test_anomaly_can_use_selected_cleaned_evidence_when_requested():
    raw = _evidence()
    cleaned = raw.model_copy(
        update={
            "evidence_id": "evi_cleaned",
            "query": "unit:test |> filter outliers",
            "summary": "cleaned evidence after excluding outliers",
            "data": {
                **raw.data,
                "points": raw.data["points"][1:],
            },
            "metadata": {"purpose": "过滤异常点后的清洗序列"},
        }
    )
    request_state = RequestStateModel(
        request_id="req_anomaly_selected_policy",
        message="在清洗序列上检测异常。",
        status="running",
        latest_database_evidence=cleaned,
        database_evidence_artifacts={
            raw.evidence_id: raw,
            cleaned.evidence_id: cleaned,
        },
    )

    class SelectedCountDetector:
        name = "selected_count_detector"

        def detect(self, series, *, params: dict):
            return AnomalyDetectorOutput(
                anomaly_points=[],
                scores=[],
                diagnostics={"input_point_count": len(series.points)},
            )

    register_anomaly_detector(SelectedCountDetector())

    result = await AnomalyTool().execute(
        AnomalyInput(
            database_evidence="latest",
            detector_name="selected_count_detector",
            constraints={"input_policy": "selected"},
        ),
        request_state=request_state,
    )

    assert result["diagnostics"]["selected_evidence_id"] == "evi_cleaned"
    assert result["diagnostics"]["resolved_evidence_id"] == "evi_cleaned"
    assert result["diagnostics"]["input_point_count"] == 5


@pytest.mark.asyncio
async def test_anomaly_keeps_raw_flux_evidence_for_anomaly_detection_purpose():
    evidence = _evidence().model_copy(
        update={
            "query_language": "flux",
            "query": 'from(bucket: "bitcoin") |> filter(fn: (r) => r._field == "price")',
            "summary": "Loaded 2679 raw rows across 1 series",
            "metadata": {"purpose": "原始历史价格时序证据，用于后续异常检测。"},
        }
    )
    request_state = RequestStateModel(
        request_id="req_anomaly_flux_raw_policy",
        message="查询原始价格并检测异常点。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
    )

    class FluxRawDetector:
        name = "flux_raw_detector"

        def detect(self, series, *, params: dict):
            return AnomalyDetectorOutput(
                anomaly_points=[],
                scores=[],
                diagnostics={"input_point_count": len(series.points)},
            )

    register_anomaly_detector(FluxRawDetector())

    result = await AnomalyTool().execute(
        AnomalyInput(database_evidence="latest", detector_name="flux_raw_detector"),
        request_state=request_state,
    )

    assert result["diagnostics"]["selected_evidence_id"] == "evi_ref"
    assert result["diagnostics"]["resolved_evidence_id"] == "evi_ref"
    assert "input_policy_note" not in result["diagnostics"]


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
