from __future__ import annotations

import pytest

from schemas.data_fact import DataFact, FactCoverage, FactEvent, FactEvidenceRef
from schemas.database_context import DatabaseContext
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
