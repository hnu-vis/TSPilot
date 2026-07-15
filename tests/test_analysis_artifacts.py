from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.settings import get_settings
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

    insight_payload = {
        "insight_id": "ins_demo",
        "requested_fact_types": ["seasonality"],
        "supported_fact_types": ["seasonality"],
        "fact_candidates": [{"fact_id": "f1", "fact_type": "seasonality", "statement": "candidate"}] * 10,
        "completed_facts": [{"fact_id": "f1", "fact_type": "seasonality", "statement": "candidate", "focus": "btc"}] * 10,
        "verified_facts": [{"fact_id": "f1", "fact_type": "seasonality", "statement": "no seasonality", "confidence": 0.4, "evidence": {}, "verification_rule": "rule"}] * 10,
        "rejected_facts": [],
        "summary_blocks": [{"text": "summary"}] * 10,
        "visualizations": [
            {
                "visualization_id": "viz1",
                "visualization_type": "chart",
                "visualization_kind": "line",
                "renderer": "linechart",
                "title": "trend",
                "summary": "trend",
                "chart": {
                    "x_axis_data": [f"2023-01-01T00:{i:02d}:00Z" for i in range(20)],
                    "series_data": [{"name": "price", "data": list(range(20))}],
                },
            }
        ],
        "diagnostics": {},
    }
    apply_observation(
        request_state,
        ToolObservation(tool_name="insight", success=True, summary="ok", payload={}, error=None),
        insight_payload,
        tool_spec,
    )

    assert "ins_demo" in request_state.insight_artifacts
    snapshot_ref = request_state.latest_insight.diagnostics["snapshot_ref"]
    assert Path(snapshot_ref["uri"]).exists()
    assert request_state.latest_insight.visualizations[0].chart["x_axis_count"] == 20
    enriched_insight = enrich_observation_payload(
        request_state,
        ToolObservation(tool_name="insight", success=True, summary="ok", payload={}, error=None),
        insight_payload,
        tool_spec,
    )
    assert enriched_insight.payload["diagnostics"]["snapshot_ref"]["uri"] == snapshot_ref["uri"]

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
    assert len(request_state.latest_anomaly.anomaly_points) == 12
    assert Path(request_state.latest_anomaly.diagnostics["snapshot_ref"]["uri"]).exists()
    enriched_anomaly = enrich_observation_payload(
        request_state,
        ToolObservation(tool_name="anomaly", success=True, summary="ok", payload={}, error=None),
        anomaly_payload,
        tool_spec,
    )
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
    assert len(request_state.latest_forecast.forecast_points) == 12
    assert Path(request_state.latest_forecast.diagnostics["snapshot_ref"]["uri"]).exists()
    enriched_forecast = enrich_observation_payload(
        request_state,
        ToolObservation(tool_name="forecast", success=True, summary="ok", payload={}, error=None),
        forecast_payload,
        tool_spec,
    )
    assert enriched_forecast.payload["diagnostics"]["snapshot_ref"]["artifact_kind"] == "forecast_result"
