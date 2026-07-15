from __future__ import annotations

import asyncio

from app.settings import get_settings
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.tool import ToolObservation
from tools.insight import InsightInput, InsightTool


class _ToolSpec:
    result_target = "evidence"


def _build_full_evidence_payload():
    points = [
        {"timestamp": f"2023-01-01T00:{index:02d}:00Z", "value": float(index)}
        for index in range(40)
    ]
    return {
        "evidence_id": "evi_demo_full",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo",
        "summary": "Loaded 40 points.",
        "data": {
            "points": points,
            "rows": [{"timestamp": item["timestamp"], "value": item["value"]} for item in points],
            "series": [
                {
                    "series_name": "value",
                    "value_field": "value",
                    "time_field": "timestamp",
                    "points": points,
                    "labels": {},
                }
            ],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {"query_trace": {"raw_result_summary": {"row_count": 40, "columns": ["timestamp", "value"]}}},
    }


def test_request_state_keeps_summary_evidence_and_full_artifact():
    settings = get_settings()
    request = ChatRequest(message="分析趋势")
    request_state = build_request_state(request, settings)
    observation = ToolObservation(tool_name="query_database", success=True, summary="ok", payload={})

    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    latest = request_state.latest_database_evidence
    assert latest is not None
    assert latest.evidence_id == "evi_demo_full"
    assert len(latest.data["points"]) == 24
    assert latest.diagnostics["artifact_ref"] == "evidence:evi_demo_full"
    assert request_state.database_evidence_artifacts["evi_demo_full"].data["points"][0]["timestamp"] == "2023-01-01T00:00:00Z"
    assert len(request_state.database_evidence_artifacts["evi_demo_full"].data["points"]) == 40


def test_insight_uses_full_evidence_artifact_from_request_state():
    settings = get_settings()
    request = ChatRequest(message="分析趋势")
    request_state = build_request_state(request, settings)
    request_state.latest_database_evidence = None
    request_state.database_evidence_artifacts = {}
    observation = ToolObservation(tool_name="query_database", success=True, summary="ok", payload={})
    apply_observation(request_state, observation, _build_full_evidence_payload(), _ToolSpec())

    result = asyncio.run(
        InsightTool().execute(
            InsightInput(requested_fact_types=["trend"]),
            request_state=request_state,
        )
    )

    assert result["verified_facts"]
