from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.visualization import VisualizationArtifactStore
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
        payload = {
            "visual_question": "What was the largest rebound from the monthly low after excluding outliers?",
            "interpretation": "Read one complete price line, the two endpoints, and the highlighted interval.",
            "target_insight_ids": [],
            "charts": [{
                "chart_id": "monthly_rebound",
                "purpose": "verify rebound",
                "priority": "primary",
                "title": "USD price rebound from the monthly low",
                "summary": f"The grounded rebound amount is provided by {amount_ref}.",
                "accessibility_description": "One USD price line with the monthly low, subsequent peak, and rebound interval.",
                "accessibility_table_columns": ["timestamp", "value"],
                "series": [{
                    "series_id": "prices",
                    "name": "USD price",
                    "source_ref": "view:evidence:evi_influxdb2-bitcoin-sample_timeseries_d73eaf462ca6:default",
                    "x_field": "timestamp",
                    "y_field": "value",
                }],
                "point_annotations": [
                    {
                        "series_id": "prices", "name": "Monthly low",
                        "time": {"source_ref": low_ref, "value_id": "time_1"},
                        "value": {"source_ref": low_ref, "value_id": "number_1"},
                    },
                    {
                        "series_id": "prices", "name": "Subsequent peak",
                        "time": {"source_ref": peak_ref, "value_id": "time_1"},
                        "value": {"source_ref": peak_ref, "value_id": "number_1"},
                    },
                ],
                "interval_annotations": [{
                    "series_id": "prices", "name": "Rebound interval",
                    "start": {"source_ref": low_ref, "value_id": "time_1"},
                    "end": {"source_ref": peak_ref, "value_id": "time_1"},
                }],
                "reference_lines": [],
                "y_axis_name": "USD",
            }],
            "required_data_request": None,
        }
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


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
    assert option["yAxis"] == {"type": "value", "scale": True, "name": "USD"}
    assert len(option["dataset"][0]["source"]) == 258
