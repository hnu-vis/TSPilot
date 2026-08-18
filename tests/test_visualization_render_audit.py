from __future__ import annotations

import base64
import os
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.visualization.render_audit import PlaywrightEChartsRenderAuditor
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.visual_verification import VisualizationVerification
from schemas.visualization import VisualizationPayload
from tools.visualization import VisualizationInput


class _ImageInspectingLlm:
    def __init__(self):
        self.png_sizes: list[int] = []

    async def ainvoke(self, messages):
        content = messages[-1].content
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = item["image_url"]["url"]
            encoded = url.split(",", 1)[1]
            screenshot = base64.b64decode(encoded)
            assert screenshot.startswith(b"\x89PNG\r\n\x1a\n")
            self.png_sizes.append(len(screenshot))
        return SimpleNamespace(
            content='{"decision":"approve","issues":[]}',
            response_metadata={},
        )


def _visualization() -> VisualizationPayload:
    return VisualizationPayload.model_validate({
        "visualization_id": "viz_render_audit",
        "purpose": "verify the observed trend",
        "title": "Observed value over time",
        "summary": "The complete observed interval is shown.",
        "verification": {
            "target_insight_ids": ["insight_trend"],
            "verification_question": "Does the complete observed series rise across the interval?",
            "interpretation": "Read the complete series from its first point to its last point.",
        },
        "source_refs": ["evidence:evi_render"],
        "required_roles": ["complete_series"],
        "datasets": [{
            "dataset_id": "dataset_0",
            "source_ref": "semantic:render_series",
            "dimensions": [
                {"name": "timestamp", "data_type": "time", "role": "x"},
                {"name": "value", "data_type": "number", "role": "y"},
            ],
            "series": [{
                "series_id": "series_0",
                "name": "Observed value",
                "role": "complete_series",
                "points": [
                    {"x": "2026-01-01T00:00:00Z", "y": 10.0},
                    {"x": "2026-01-02T00:00:00Z", "y": 12.0},
                    {"x": "2026-01-03T00:00:00Z", "y": 15.0},
                ],
            }],
        }],
        "layers": [{
            "layer_id": "layer_0",
            "mark": "line",
            "role": "complete_series",
            "source_ref": "semantic:render_series",
            "encoding": {"x": "timestamp", "y": "value"},
            "dataset_id": "dataset_0",
            "series_id": "series_0",
        }],
        "bindings": [],
        "accessibility": {"description": "A complete observed time series."},
    })


@pytest.mark.asyncio
async def test_real_echarts_desktop_and_mobile_render_pass_through_multimodal_gate():
    audit_url = os.environ.get("TSPILOT_E2E_FRONTEND_URL")
    if not audit_url:
        pytest.skip("set TSPILOT_E2E_FRONTEND_URL to run the real ECharts browser audit")
    llm = _ImageInspectingLlm()
    auditor = PlaywrightEChartsRenderAuditor(llm=llm, audit_url=audit_url)
    state = build_request_state(
        ChatRequest(
            message="Show the observed trend.",
            database_context={"database_id": "demo", "database_type": "unit"},
        ),
        get_settings(),
    )
    verification = VisualizationVerification(
        target_insight_ids=["insight_trend"],
        verification_question="Does the complete observed series rise across the interval?",
        interpretation="Read the complete series from its first point to its last point.",
    )
    try:
        result = await auditor.audit(
            visualizations=[_visualization()],
            verification=verification,
            request=VisualizationInput(message="Show the observed trend."),
            request_state=state,
        )
    finally:
        await auditor.close()

    assert result == {"decision": "approve", "issues": []}
    assert len(llm.png_sizes) == 2
    assert all(size > 1_000 for size in llm.png_sizes)
