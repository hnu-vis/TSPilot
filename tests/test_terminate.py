from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.data_fact import DataFact, FactCoverage, FactEvent, FactEvidenceRef
from schemas.database_context import DatabaseContext
from schemas.state import ConversationStateModel, RequestStateModel
from runtime.tool_executor import ToolExecutor
from tools.registry import build_tool_registry
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


def test_terminate_rejects_structured_result_wrappers():
    with pytest.raises(ValidationError, match="natural-language string"):
        TerminateInput.model_validate(
            {
                "result": {
                    "summary": "已完成对2023-01-29起BTC/USD价格水平的短期预测。",
                    "prediction": "接下来几天价格大致在 24283 到 24368 美元附近。",
                    "basis": {"forecast_model": "linear regression", "forecast_horizon": 24},
                }
            }
        )


def test_terminate_accepts_natural_language_answer_with_colons():
    payload = TerminateInput.model_validate(
        {
            "result": "最低点: 2023-01-01T00:00:00+00:00，价格: 16500 美元。"
        }
    )

    assert payload.direct_answer == "最低点: 2023-01-01T00:00:00+00:00，价格: 16500 美元。"


@pytest.mark.asyncio
async def test_executor_does_not_stringify_structured_terminate_answer():
    request_state = RequestStateModel(
        request_id="req-structured-answer",
        message="预测 BTC/USD",
        status="running",
    )
    executor = ToolExecutor(build_tool_registry(settings=None))

    with pytest.raises(ValidationError, match="natural-language string"):
        await executor.execute(
            "terminate",
            {
                "result": {
                    "summary": "已完成预测。",
                    "prediction": "价格小幅上行。",
                    "basis": {"forecast_horizon": 24},
                }
            },
            request_state,
            ConversationStateModel(conversation_id="conv-structured-answer"),
        )


@pytest.mark.asyncio
async def test_terminate_defaults_to_process_fact_ids_for_formatter():
    request_state = RequestStateModel(
        request_id="req-facts",
        message="最大值和最小值的差异是多少？",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
    )
    max_fact = DataFact(
        fact_id="fact_max",
        name="max_value",
        fact_type="extreme",
        statement="max_value is 51.0.",
        value=51.0,
        method="code_interpreter",
        evidence_refs=[FactEvidenceRef(source_type="analysis", source_id="ana")],
    )
    stale_fact = DataFact(
        fact_id="fact_count",
        name="record_count",
        fact_type="count",
        statement="record_count is 6.",
        value=6,
        method="sql_query",
        evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi")],
    )
    request_state.fact_set.facts = [stale_fact, max_fact]
    request_state.fact_events.append(
        FactEvent(
            iteration=2,
            tool_name="code_interpreter",
            produced_fact_ids=[max_fact.fact_id],
            coverage=FactCoverage(requested=["max_value"], verified=["max_value"]),
        )
    )

    payload = await TerminateTool().execute(
        TerminateInput(result="最大值是 51。"),
        request_state=request_state,
    )

    facts_section = next(section for section in payload["sections"] if section["section_type"] == "facts")
    assert "max_value is 51.0" in facts_section["content"]
    assert "record_count" not in facts_section["content"]


@pytest.mark.asyncio
async def test_terminate_resolves_semantic_fact_keys_and_names():
    request_state = RequestStateModel(
        request_id="req-semantic-facts",
        message="change",
        status="running",
        database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
    )
    fact = DataFact(
        fact_id="fact_change",
        fact_key="price.percentage_change",
        name="percentage_change",
        fact_type="difference",
        statement="Price increased by 20%.",
        value=20.0,
        method="code_interpreter",
        evidence_refs=[FactEvidenceRef(source_type="analysis", source_id="ana_change")],
    )
    request_state.fact_set.facts = [fact]

    payload = await TerminateTool().execute(
        TerminateInput(result="Price increased by 20%.", include_fact_ids=["percentage_change"]),
        request_state=request_state,
    )

    facts_section = next(section for section in payload["sections"] if section["section_type"] == "facts")
    assert "Price increased by 20%." in facts_section["content"]
