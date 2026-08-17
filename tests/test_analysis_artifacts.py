from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.settings import get_settings
from core.harness.observation_view import public_observation_view
from runtime.request_state import apply_observation, build_request_state
from runtime.request_state import enrich_observation_payload
from schemas.api import ChatRequest
from schemas.database_context import DatabaseContext
from schemas.tool import ToolObservation


def _request_state():
    return build_request_state(
        ChatRequest(
            message="分析 Bitcoin USD 的周期性和异常",
            database_context=DatabaseContext(
                database_id="influxdb2-bitcoin-sample",
                database_type="influxdb",
            ),
        ),
        get_settings(),
    )


def test_analysis_payloads_are_snapshotted_and_summarized():
    request_state = _request_state()
    tool_spec = SimpleNamespace(result_target="analysis")

    analysis_payload = {
        "analysis_id": "ana_demo",
        "analysis_goal": "seasonality",
        "code_type": "code_interpreter_v2",
        "code_hash": "sha256:demo",
        "input_evidence_id": "evi_demo",
        "input_row_count": 20,
        "status": "succeeded",
        "summary": "analysis summary",
        "computed_insights": [{
            "insight_key": "row_count", "value": 20,
            "calculation_trace": {"operation": "len(rows)"},
        }],
        "derived_evidence": [],
        "diagnostics": {},
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}, error=None),
        analysis_payload,
        tool_spec,
    )

    enriched_analysis = enrich_observation_payload(
        request_state,
        ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}, error=None),
        analysis_payload,
        tool_spec,
    )
    snapshot_ref = enriched_analysis.payload["diagnostics"]["snapshot_ref"]
    assert "ana_demo" in request_state.analysis_artifacts
    assert Path(snapshot_ref["uri"]).exists()

    anomaly_payload = {
        "anomaly_id": "an_demo",
        "detector_name": "zscore",
        "anomaly_points": [{"timestamp": f"t{i}", "value": i, "score": i} for i in range(20)],
        "anomaly_spans": [],
        "scores": [{"timestamp": f"t{i}", "score": i} for i in range(20)],
        "diagnostics": {"threshold": 2.5},
        "visualizations": [],
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="anomaly", success=True, summary="ok", payload={}, error=None),
        anomaly_payload,
        tool_spec,
    )
    assert "an_demo" in request_state.anomaly_artifacts
    assert len(request_state.latest_anomaly.anomaly_points) == 20
    assert Path(request_state.latest_anomaly.diagnostics["snapshot_ref"]["uri"]).exists()
    enriched_anomaly = enrich_observation_payload(
        request_state,
        ToolObservation(tool_name="anomaly", success=True, summary="ok", payload={}, error=None),
        anomaly_payload,
        tool_spec,
    )
    assert len(enriched_anomaly.payload["anomaly_points"]) == 12
    assert enriched_anomaly.payload["diagnostics"]["snapshot_ref"]["artifact_kind"] == "anomaly_result"

    forecast_payload = {
        "forecast_id": "fc_demo",
        "model_name": "linear",
        "horizon": 24,
        "forecast_points": [{"timestamp": f"t{i}", "value": float(i)} for i in range(20)],
        "confidence_interval": [],
        "diagnostics": {"series_name": "price"},
        "visualizations": [],
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="forecast", success=True, summary="ok", payload={}, error=None),
        forecast_payload,
        tool_spec,
    )
    assert "fc_demo" in request_state.forecast_artifacts
    assert len(request_state.latest_forecast.forecast_points) == 20
    assert Path(request_state.latest_forecast.diagnostics["snapshot_ref"]["uri"]).exists()
    enriched_forecast = enrich_observation_payload(
        request_state,
        ToolObservation(tool_name="forecast", success=True, summary="ok", payload={}, error=None),
        forecast_payload,
        tool_spec,
    )
    assert len(enriched_forecast.payload["forecast_points"]) == 12
    assert enriched_forecast.payload["diagnostics"]["forecast_point_count"] == 20
    assert enriched_forecast.payload["diagnostics"]["snapshot_ref"]["artifact_kind"] == "forecast_result"
    public_forecast = public_observation_view(enriched_forecast)
    assert public_forecast["payload"]["forecast_point_count"] == 20
    assert len(public_forecast["payload"]["forecast_points_preview"]) == 6
