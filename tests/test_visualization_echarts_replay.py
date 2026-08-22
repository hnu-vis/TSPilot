from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.visualization import VisualizationArtifactStore
from schemas.echarts_plan import EChartsChartPlan, EChartsPlan
from schemas.state import RequestStateModel
from tools.visualization import VisualizationInput, VisualizationTool


HISTORICAL_STATE = Path(
    "cache_data/conversation_logs/2026-08-22_16-40-32_conv_h41gkpzxw5d/"
    "requests/req_6785ae586caa/state.json"
)
AMOUNT_ID = "ins_ana_f502f7e9ce7dbd7d_rebound_amount_807008ca5e51"
LOW_ID = "ins_ana_f502f7e9ce7dbd7d_rebound_low_point_a2acf63d52b0"
PEAK_ID = "ins_ana_f502f7e9ce7dbd7d_rebound_peak_point_4db54b022117"


class _HistoricalReplayLlm:
    async def ainvoke(self, _messages):
        low_ref, peak_ref, amount_ref = f"insight:{LOW_ID}", f"insight:{PEAK_ID}", f"insight:{AMOUNT_ID}"
        option = {
            "useUTC": True,
            "legend": {"show": True},
            "tooltip": {"trigger": "axis"},
            "dataset": [{
                "id": "prices",
                "source": {"$dataset": "view:evidence:evi_influxdb2-bitcoin-sample_timeseries_d73eaf462ca6:default"},
            }],
            "xAxis": [{"type": "time", "name": "Time (UTC)"}],
            "yAxis": [{"type": "value", "name": "USD", "scale": True}],
            "series": [{
                "name": "USD price", "type": "line", "datasetId": "prices",
                "encode": {"x": "timestamp", "y": "value"}, "showSymbol": False,
                "markPoint": {"symbol": "circle", "symbolSize": 10, "label": {"show": False}, "data": [
                    {"name": "Monthly low", "coord": [
                        {"$value": {"source_ref": low_ref, "field": "timestamp"}},
                        {"$value": {"source_ref": low_ref, "field": "value"}},
                    ]},
                    {"name": "Subsequent peak", "coord": [
                        {"$value": {"source_ref": peak_ref, "field": "timestamp"}},
                        {"$value": {"source_ref": peak_ref, "field": "value"}},
                    ], "value": {"$value": {"source_ref": amount_ref, "field": "value"}}},
                ]},
                "markArea": {"data": [[
                    {"name": "Rebound interval", "xAxis": {"$value": {"source_ref": low_ref, "field": "timestamp"}}},
                    {"xAxis": {"$value": {"source_ref": peak_ref, "field": "timestamp"}}},
                ]]},
            }],
        }
        plan = EChartsPlan(
            visual_question="What was the largest rebound from the monthly low after excluding outliers?",
            interpretation="Read one complete price line, the two endpoints, and the highlighted interval.",
            target_insight_ids=[AMOUNT_ID, LOW_ID, PEAK_ID],
            charts=[EChartsChartPlan(
                chart_id="monthly_rebound", purpose="verify rebound", priority="primary",
                title="USD price rebound from the monthly low",
                summary="The rebound amount is attached to the peak mark, not plotted as a price series.",
                accessibility_description="One USD price line with the monthly low, subsequent peak, and rebound interval.",
                option_json=json.dumps(option),
            )],
        )
        return SimpleNamespace(content=plan.model_dump_json(), response_metadata={})


@pytest.mark.asyncio
async def test_historical_monthly_rebound_tool_replay_publishes_clean_native_option(tmp_path):
    if not HISTORICAL_STATE.is_file():
        pytest.skip("historical tool state is not available in this checkout")
    state = RequestStateModel.model_validate_json(HISTORICAL_STATE.read_text(encoding="utf-8"))
    refs = [f"insight:{AMOUNT_ID}", f"insight:{LOW_ID}", f"insight:{PEAK_ID}"]
    store = VisualizationArtifactStore(tmp_path)
    result = await VisualizationTool(llm=_HistoricalReplayLlm(), artifact_store=store).execute(
        VisualizationInput(message=state.message, source_refs=refs), request_state=state,
    )
    assert result["status"] == "created"
    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    option = complete.option
    assert len(option["series"]) == 1
    assert option["series"][0]["type"] == "line"
    assert len(option["series"][0]["markPoint"]["data"]) == 2
    assert len(option["series"][0]["markArea"]["data"]) == 1
    assert len(option["yAxis"]) == 1
    assert option["series"][0]["markPoint"]["data"][1]["value"] == pytest.approx(1285.1743)
    assert len(option["dataset"][0]["source"]) == 258
