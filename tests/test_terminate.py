from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.output import FinalResponsePlan
from schemas.state import RequestStateModel
from tools.terminate import TerminateInput, TerminateTool


@pytest.mark.asyncio
async def test_terminate_assembles_the_supplied_response_plan():
    state = RequestStateModel(request_id="req", message="hello", status="running")
    payload = await TerminateTool().execute(
        TerminateInput(response_plan=FinalResponsePlan(summary="Hello.")),
        request_state=state,
    )
    assert payload["summary"] == "Hello."
    assert payload["visualizations"] == []


def test_terminate_requires_structured_response_plan():
    with pytest.raises(ValidationError, match="response_plan"):
        TerminateInput.model_validate({"result": "legacy prose"})


def test_terminate_rejects_renderer_level_visualization_fields():
    with pytest.raises(ValidationError):
        TerminateInput.model_validate({
            "response_plan": {
                "summary": "Answer",
                "visual_goals": [{
                    "purpose": "trend",
                    "title": "Trend",
                    "required_roles": ["series"],
                    "layers": [{
                        "role": "series",
                        "source_ref": "evidence:evi",
                        "mark": "line",
                        "encoding": {"x": "timestamp", "y": "value"},
                        "echarts_option": {"series": []},
                    }],
                    "echarts_option": {"series": []},
                }],
            }
        })
