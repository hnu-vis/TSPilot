from __future__ import annotations

from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.insight import InsightResult
from schemas.tool import ToolObservation
from schemas.visualization import VisualizationPayload
from app.settings import get_settings


class _EvidenceSpec:
    result_target = "evidence"


def test_prompt_builder_summarizes_heavy_context():
    settings = get_settings()
    request = ChatRequest(message="分析季节性")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    observation = ToolObservation(tool_name="query_database", success=True, summary="ok", payload={})
    full_payload = {
        "evidence_id": "evi_demo",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo",
        "summary": "Loaded 100 points.",
        "data": {
            "points": [{"timestamp": f"2023-01-01T00:{i:02d}:00Z", "value": float(i)} for i in range(100)],
            "rows": [{"timestamp": f"2023-01-01T00:{i:02d}:00Z", "value": float(i)} for i in range(100)],
            "series": [],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {},
    }
    apply_observation(request_state, observation, full_payload, _EvidenceSpec())
    request_state.latest_insight = InsightResult(
        insight_id="ins_demo",
        verified_facts=[],
        visualizations=[
            VisualizationPayload(
                visualization_id="viz_demo",
                visualization_type="chart",
                visualization_kind="line",
                renderer="linechart",
                title="Demo",
                summary="Demo chart",
                chart={
                    "x_axis_data": [f"t{i}" for i in range(100)],
                    "series_data": [{"name": "value", "data": list(range(100))}],
                },
            )
        ],
    )
    request_state.visualizations = request_state.latest_insight.visualizations

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert len(context["latest_database_evidence"]["data"]["points"]) <= 8
    assert context["visualizations"][0]["chart_summary"]["x_axis_count"] == 100
    assert "chart" not in context["visualizations"][0]
