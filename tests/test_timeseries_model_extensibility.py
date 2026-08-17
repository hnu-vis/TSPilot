from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from core.timeseries.anomaly_registry import register_api_anomaly_detector
from core.timeseries.forecast_registry import register_api_forecast_model
from schemas.database import DatabaseEvidence
from schemas.state import RequestStateModel
from schemas.timeseries import AnomalyResult
from tools.anomaly import AnomalyInput, AnomalyTool
from tools.forecast import ForecastInput, ForecastTool
from tools.sql_query import _ExplicitQueryExecutor


def _bitcoin_evidence(
    *,
    evidence_id: str = "btc_window",
    query: str = 'from(bucket:"crypto") |> range(start: 2023-01-04T00:00:00Z, stop: 2023-02-03T00:00:00Z) |> aggregateWindow(every: 3h, fn: last)',
    count: int = 20,
) -> DatabaseEvidence:
    points = [
        {"timestamp": f"2023-01-{4 + index // 8:02d}T{(index % 8) * 3:02d}:00:00Z", "value": 16800.0 + index * 72.5}
        for index in range(count)
    ]
    return DatabaseEvidence(
        evidence_id=evidence_id,
        result_type="timeseries",
        database="crypto",
        query_language="flux",
        query=query,
        summary="Bitcoin USD 3h window",
        data={
            "points": points,
            "time_field": "timestamp",
            "value_field": "BTC_USD",
            "series_name": "BTC_USD",
            "labels": {},
        },
        columns=["timestamp", "BTC_USD"],
    )


@pytest.mark.asyncio
async def test_forecast_resolves_explicit_steps_and_sampling_interval():
    result = await ForecastTool().execute(
        ForecastInput(database_evidence=_bitcoin_evidence(), horizon={"steps": "next 6 points"})
    )

    assert result["status"] == "succeeded"
    assert result["horizon"] == 6
    assert len(result["forecast_points"]) == 6
    assert result["forecast_plan"]["horizon_source"] == "explicit_steps"
    assert result["forecast_plan"]["sampling_interval_seconds"] == 10800
    assert "visualizations" not in result


@pytest.mark.asyncio
async def test_forecast_resolves_duration_to_steps_from_input_sampling():
    result = await ForecastTool().execute(
        ForecastInput(database_evidence=_bitcoin_evidence(), horizon="1 day")
    )

    assert result["status"] == "succeeded"
    assert result["horizon"] == 8
    assert len(result["forecast_points"]) == 8
    assert result["forecast_plan"]["horizon_source"] == "duration_from_user"
    assert result["forecast_plan"]["forecast_duration_seconds"] == 86400


@pytest.mark.asyncio
async def test_forecast_accepts_semantic_forecast_horizon_constraint():
    result = await ForecastTool().execute(
        ForecastInput(database_evidence=_bitcoin_evidence(), constraints={"forecast_horizon": 7})
    )

    assert result["horizon"] == 7
    assert len(result["forecast_points"]) == 7


@pytest.mark.asyncio
async def test_forecast_uses_short_term_default_for_fuzzy_horizon():
    result = await ForecastTool().execute(ForecastInput(database_evidence=_bitcoin_evidence(count=30)))

    assert result["status"] == "succeeded"
    assert result["horizon"] == 3
    assert len(result["forecast_points"]) == 3
    assert result["forecast_plan"]["horizon_source"] == "inferred_short_term_default"


@pytest.mark.asyncio
async def test_forecast_runs_rolling_chunks_when_horizon_exceeds_direct_window():
    result = await ForecastTool().execute(
        ForecastInput(
            database_evidence=_bitcoin_evidence(),
            horizon="1 year",
            constraints={"max_direct_steps": 48},
        )
    )

    assert result["status"] == "succeeded"
    assert len(result["forecast_points"]) == 2920
    assert result["forecast_plan"]["mode"] == "rolling"
    assert result["forecast_plan"]["requested_steps"] == 2920
    assert result["forecast_plan"]["recommended_chunk_steps"] == 48
    assert result["diagnostics"]["rolling_chunk_count"] == 61


@pytest.mark.asyncio
async def test_forecast_excludes_points_from_latest_anomaly_by_default():
    evidence = _bitcoin_evidence(count=6)
    anomaly = AnomalyResult(
        anomaly_id=f"anomaly_{evidence.evidence_id}",
        detector_name="zscore",
        anomaly_points=[{"timestamp": evidence.data["points"][0]["timestamp"], "value": evidence.data["points"][0]["value"], "score": 9.0}],
        diagnostics={"resolved_evidence_id": evidence.evidence_id},
    )
    request_state = RequestStateModel(
        request_id="req_forecast_exclude_anomaly",
        message="检测异常后预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        latest_anomaly=anomaly,
        anomaly_artifacts={anomaly.anomaly_id: anomaly},
    )

    result = await ForecastTool().execute(
        ForecastInput(database_evidence="latest", horizon=2),
        request_state=request_state,
    )

    assert result["diagnostics"]["input_policy"] == "exclude_detected_anomalies"
    assert result["diagnostics"]["excluded_anomaly_count"] == 1
    assert result["diagnostics"]["training_point_count_before_policy"] == 6
    assert result["diagnostics"]["training_point_count_after_policy"] == 5
    assert result["diagnostics"]["source_anomaly_id"] == anomaly.anomaly_id


@pytest.mark.asyncio
async def test_forecast_can_keep_selected_raw_points_when_requested():
    evidence = _bitcoin_evidence(count=6)
    anomaly = AnomalyResult(
        anomaly_id=f"anomaly_{evidence.evidence_id}",
        detector_name="zscore",
        anomaly_points=[{"timestamp": evidence.data["points"][0]["timestamp"], "value": evidence.data["points"][0]["value"], "score": 9.0}],
        diagnostics={"resolved_evidence_id": evidence.evidence_id},
    )
    request_state = RequestStateModel(
        request_id="req_forecast_keep_raw",
        message="按原始数据预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        latest_anomaly=anomaly,
        anomaly_artifacts={anomaly.anomaly_id: anomaly},
    )

    result = await ForecastTool().execute(
        ForecastInput(database_evidence="latest", horizon=2, constraints={"input_policy": "raw"}),
        request_state=request_state,
    )

    assert result["diagnostics"]["input_policy"] == "raw"
    assert result["diagnostics"]["excluded_anomaly_count"] == 0
    assert result["diagnostics"]["training_point_count_before_policy"] == 6
    assert result["diagnostics"]["training_point_count_after_policy"] == 6


@pytest.mark.asyncio
async def test_forecast_does_not_use_code_interpreter_as_anomaly_fallback():
    evidence = _bitcoin_evidence(count=6)
    unrelated_anomaly = AnomalyResult(
        anomaly_id="anomaly_other_evidence",
        detector_name="zscore",
        anomaly_points=[],
        diagnostics={"resolved_evidence_id": "other_evidence"},
    )
    request_state = RequestStateModel(
        request_id="req_forecast_exclude_analysis_outliers",
        message="先分析异常再预测。",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        latest_anomaly=unrelated_anomaly,
        anomaly_artifacts={unrelated_anomaly.anomaly_id: unrelated_anomaly},
    )

    result = await ForecastTool().execute(
        ForecastInput(database_evidence="latest", horizon=2),
        request_state=request_state,
    )

    assert result["diagnostics"]["excluded_anomaly_count"] == 0
    assert result["diagnostics"]["training_point_count_before_policy"] == 6
    assert result["diagnostics"]["training_point_count_after_policy"] == 6


@pytest.mark.asyncio
async def test_forecast_rejects_raw_limit_evidence_that_does_not_cover_requested_range():
    evidence = _bitcoin_evidence(
        query='from(bucket:"crypto") |> range(start: 2023-01-04T00:00:00Z, stop: 2023-02-03T00:00:00Z) |> sort(columns:["_time"]) |> limit(n: 20)',
        count=20,
    )
    request_state = RequestStateModel(
        request_id="req_raw_limit",
        message="预测 Bitcoin 接下来 6 个点",
        status="running",
        latest_database_evidence=evidence,
        database_evidence_artifacts={evidence.evidence_id: evidence},
        time_range={"start": "2023-01-04T00:00:00Z", "end": "2023-02-03T00:00:00Z"},
    )

    with pytest.raises(ValueError, match="raw limit"):
        await ForecastTool().execute(
            ForecastInput(database_evidence="latest", horizon=6),
            request_state=request_state,
        )


def test_sql_query_diagnostics_flag_raw_limit_for_timeseries_forecast_evidence():
    executor = _ExplicitQueryExecutor.__new__(_ExplicitQueryExecutor)
    evidence = _bitcoin_evidence(
        query='from(bucket:"crypto") |> range(start: 2023-01-04T00:00:00Z, stop: 2023-02-03T00:00:00Z) |> limit(n: 240)'
    ).model_dump(mode="json")

    diagnostics = executor._task_coverage_diagnostics(
        validated_input=type("Input", (), {"purpose": "查询并预测 Bitcoin USD 趋势"})(),
        evidence=evidence,
        query=evidence["query"],
        query_language="flux",
        base=None,
        selected_fields=None,
    )

    assert diagnostics["requires_followup"] is True
    assert any("raw LIMIT" in item for item in diagnostics["runtime_missing"])


class _ModelApiHandler(BaseHTTPRequestHandler):
    forecast_requests: list[dict] = []
    anomaly_requests: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if payload["task"] == "forecast":
            self.__class__.forecast_requests.append(payload)
            last_timestamp = payload["series"]["points"][-1]["timestamp"]
            response = {
                "forecast_points": [
                    {"timestamp": f"{last_timestamp}+{index}", "value": 20000.0 + index}
                    for index in range(1, payload["horizon"] + 1)
                ],
                "diagnostics": {"remote_model": payload["model_name"]},
            }
        else:
            self.__class__.anomaly_requests.append(payload)
            response = {
                "anomaly_points": [{"timestamp": payload["series"]["points"][1]["timestamp"], "value": 16872.5, "score": 3.2}],
                "scores": [{"timestamp": payload["series"]["points"][1]["timestamp"], "score": 3.2}],
                "diagnostics": {"remote_detector": payload["detector_name"]},
            }
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):  # noqa: A003
        return


@pytest.fixture()
def model_api_server():
    _ModelApiHandler.forecast_requests = []
    _ModelApiHandler.anomaly_requests = []
    server = HTTPServer(("127.0.0.1", 0), _ModelApiHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_forecast_and_anomaly_registries_support_api_models(model_api_server):
    register_api_forecast_model("api_forecast_unit", endpoint=model_api_server)
    register_api_anomaly_detector("api_anomaly_unit", endpoint=model_api_server)
    evidence = _bitcoin_evidence()

    forecast = await ForecastTool().execute(
        ForecastInput(database_evidence=evidence, horizon=4, model_name="api_forecast_unit")
    )
    anomaly = await AnomalyTool().execute(
        AnomalyInput(database_evidence=evidence, detector_name="api_anomaly_unit")
    )

    assert [point["value"] for point in forecast["forecast_points"]] == [20001.0, 20002.0, 20003.0, 20004.0]
    assert forecast["diagnostics"]["model_family"] == "api"
    assert forecast["diagnostics"]["model_registry_name"] == "api_forecast_unit"
    assert anomaly["anomaly_points"][0]["score"] == 3.2
    assert anomaly["diagnostics"]["model_family"] == "api"
    assert anomaly["diagnostics"]["detector_registry_name"] == "api_anomaly_unit"
    assert _ModelApiHandler.forecast_requests[0]["horizon"] == 4
    assert _ModelApiHandler.anomaly_requests[0]["task"] == "anomaly"
