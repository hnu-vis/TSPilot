from __future__ import annotations

import pytest

from schemas.state import RequestStateModel
from tools.terminate import TerminateInput, TerminateTool


@pytest.mark.asyncio
async def test_terminate_uses_result_as_direct_answer_without_datasource():
    request_state = RequestStateModel(
        request_id="req-terminate",
        message="你好",
        status="running",
    )
    tool = TerminateTool()

    payload = await tool.execute(
        TerminateInput(result="你好！我是 TSPilot。"),
        request_state=request_state,
    )

    assert payload["summary"] == "你好！我是 TSPilot。"
    assert payload["sections"] == []
    assert payload["references"] == []
