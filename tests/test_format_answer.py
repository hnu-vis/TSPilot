from __future__ import annotations

import asyncio
import json

import pytest

from app.settings import get_settings
from core.visualization import EChartsCompiler, PresentationCatalog, VisualizationArtifactStore
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.echarts_plan import EChartsChartPlan, EChartsPlan
from schemas.output import FinalResponsePlan, PlannedAnswerSection
from tools.base import StructuredToolError
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


class _ForbiddenLlm:
    async def ainvoke(self, _messages):
        raise AssertionError("format_answer must not invoke an LLM")


def _state():
    state = build_request_state(
        ChatRequest(message="show recent prices", database_context={"database_id": "demo", "database_type": "unit"}),
        get_settings(),
    )
    evidence = DatabaseEvidence(
        evidence_id="evi_prices", result_type="timeseries", database="demo", summary="Loaded three prices.",
        data={"rows": [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
            {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0},
        ]}, columns=["timestamp", "value"],
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    return state


def _attach_visualization(state, tmp_path):
    source = "view:evidence:evi_prices:default"
    plan = EChartsPlan(
        visual_question="Did prices rise?", interpretation="Read the complete line.",
        charts=[EChartsChartPlan(
            chart_id="trend", purpose="show the trend", title="Price trend", priority="primary",
            accessibility_description="Three daily prices.",
            option_json=json.dumps({
                "dataset": {"source": {"$dataset": source}},
                "xAxis": {"type": "time"}, "yAxis": {"type": "value", "scale": True},
                "series": {"type": "line", "encode": {"x": "timestamp", "y": "value"}},
            }),
        )],
    )
    complete = EChartsCompiler(PresentationCatalog(state)).compile(plan)[0]
    descriptor = VisualizationArtifactStore(tmp_path).put(complete)
    state.visualizations = [descriptor]
    return descriptor


def test_format_answer_reuses_v5_echarts_without_another_llm_call(tmp_path):
    state = _state()
    descriptor = _attach_visualization(state, tmp_path)
    plan = FinalResponsePlan(
        title="Recent prices", summary="The latest value is 15.",
        sections=[PlannedAnswerSection(section_type="analysis", heading="Trend", content="Prices rose.", source_refs=["evidence:evi_prices"])],
        visualization_ids=[descriptor.visualization_id],
    )
    result = asyncio.run(FormatAnswerTool(llm=_ForbiddenLlm()).execute(
        FormatAnswerInput(response_plan=plan), request_state=state,
    ))
    chart = result["visualizations"][0]
    assert chart["schema_version"] == "5"
    assert chart["chart_type"] == "echarts"
    assert chart["option"]["dataset"]["source"] == []
    assert result["claims"][0]["visualization_ids"] == [descriptor.visualization_id]


def test_format_answer_routes_unknown_visualization_id_back_to_visualization():
    state = _state()
    with pytest.raises(StructuredToolError) as exc_info:
        asyncio.run(FormatAnswerTool(llm=_ForbiddenLlm()).execute(
            FormatAnswerInput(response_plan=FinalResponsePlan(summary="Done", visualization_ids=["viz_missing"])),
            request_state=state,
        ))
    assert exc_info.value.recommended_next_action == "visualization"


def test_format_answer_preserves_visualization_verification(tmp_path):
    state = _state()
    descriptor = _attach_visualization(state, tmp_path)
    result = asyncio.run(FormatAnswerTool(llm=_ForbiddenLlm()).execute(
        FormatAnswerInput(response_plan=FinalResponsePlan(summary="Prices rose.", visualization_ids=[descriptor.visualization_id])),
        request_state=state,
    ))
    assert result["visualizations"][0]["verification"]["verification_question"] == "Did prices rise?"
