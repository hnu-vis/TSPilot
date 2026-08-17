from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.settings import get_settings
from core.analysis.python_runner import AnalysisCodeError, validate_analysis_result_payload
from core.visualization.materializer import PresentationCatalog
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsightRequest
from schemas.tool import ToolObservation
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool, _preflight_analysis_code


class _AnalysisSpec:
    result_target = "analysis"


class _QueueLLM:
    def __init__(self, *payloads: dict):
        self.payloads = list(payloads)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self.payloads:
            raise AssertionError("unexpected LLM call")
        return SimpleNamespace(content=json.dumps(self.payloads.pop(0), ensure_ascii=False))


def _evidence() -> DatabaseEvidence:
    rows = [
        {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
        {"timestamp": "2023-01-01T01:00:00Z", "value": 15.0},
        {"timestamp": "2023-01-01T02:00:00Z", "value": 25.0},
    ]
    return DatabaseEvidence(
        evidence_id="evi_generated",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:test",
        summary="three points",
        columns=["timestamp", "value"],
        data={"rows": rows, "points": rows},
    )


def _request(key: str = "period_change") -> KeyInsightRequest:
    return KeyInsightRequest(name="Period change", insight_type="change", insight_key=key)


def _binding(key: str = "period_change", statement: str = "The period change is 15.") -> dict:
    return {"bindings": [{"insight_key": key, "statement": statement}]}


def _state():
    state = build_request_state(ChatRequest(message="calculate period change"), get_settings())
    evidence = _evidence()
    state.latest_database_evidence = evidence
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    return state


def test_sandbox_contract_accepts_only_computation_outputs():
    payload = validate_analysis_result_payload({
        "computed_insights": [{
            "insight_key": "period_change",
            "value": 15.0,
            "calculation_trace": {"formula": "last-first"},
        }],
        "derived_evidence": [],
    })
    assert payload["computed_insights"][0]["value"] == 15.0
    with pytest.raises(AnalysisCodeError, match="computed_insights"):
        validate_analysis_result_payload({"summary": "legacy", "metrics": {}, "details": {}})


def test_computation_trace_accepts_natural_text_without_semantic_rewriting():
    code = """
result = {
    'computed_insights': [{
        'insight_key': 'period_change',
        'value': float(value.iloc[-1] - value.iloc[0]),
        'calculation_trace': 'Subtract the first observed value from the last observed value.',
    }],
    'derived_evidence': [],
}
"""
    result = asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding())).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(), analysis_goal="calculate period change",
            code=code, insight_requests=[_request()],
        )
    ))
    trace = result["computed_insights"][0]["calculation_trace"]
    assert trace == "Subtract the first observed value from the last observed value."
    assert result["produced_insights"][0]["calculation_trace"] == trace


def test_input_requires_explicit_insight_contract():
    with pytest.raises(ValidationError, match="insight_requests"):
        CodeInterpreterInput(database_evidence=_evidence(), analysis_goal="calculate", insight_requests=[])


def test_code_interpreter_computes_then_llm_binds_without_value_mutation():
    code = """
result = {
    'computed_insights': [{
        'insight_key': 'period_change',
        'value': float(value.iloc[-1] - value.iloc[0]),
        'calculation_trace': {'formula': 'last - first', 'first': float(value.iloc[0]), 'last': float(value.iloc[-1])},
    }],
    'derived_evidence': [],
}
"""
    result = asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding())).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(), analysis_goal="calculate period change",
            code=code, insight_requests=[_request()],
        )
    ))
    assert result["code_type"] == "code_interpreter_v2"
    assert "result" not in result
    assert "data_views" not in result
    assert result["computed_insights"][0]["value"] == 15.0
    assert result["produced_insights"][0]["value"] == 15.0
    assert result["produced_insights"][0]["statement"] == "The period change is 15."


def test_computed_keys_must_exactly_match_requests_in_order():
    code = """
result = {
    'computed_insights': [{'insight_key': 'other', 'value': 1, 'calculation_trace': {'formula': 'x'}}],
    'derived_evidence': [],
}
"""
    with pytest.raises(AnalysisCodeError, match="exactly match requested keys"):
        asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding())).execute(
            CodeInterpreterInput(
                database_evidence=_evidence(), analysis_goal="calculate",
                code=code, insight_requests=[_request()],
            )
        ))


def test_unavailable_computation_never_requires_placeholder_value():
    code = """
result = {
    'computed_insights': [{
        'insight_key': 'period_change',
        'unavailable_reason': 'The evidence contains no usable values.',
        'calculation_trace': {'input_rows': 0},
    }],
    'derived_evidence': [],
}
"""
    result = asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding(statement="Period change is unavailable."))).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(), analysis_goal="calculate",
            code=code, insight_requests=[_request()],
        )
    ))
    insight = result["produced_insights"][0]
    assert insight["status"] == "unavailable"
    assert insight["value"] is None
    assert insight["unavailable_reason"]


def test_derived_series_is_registered_as_independent_evidence():
    code = """
derived_rows = [{'timestamp': str(time.iloc[i]), 'value': float(value.iloc[i] * 2)} for i in range(len(value))]
result = {
    'computed_insights': [{
        'insight_key': 'period_change', 'value': 30.0,
        'calculation_trace': {'formula': '2 * (last-first)'},
        'derived_evidence_names': ['doubled_series'],
    }],
    'derived_evidence': [{
        'name': 'doubled_series', 'shape': 'timeseries', 'rows': derived_rows,
        'transform_summary': 'Each source value multiplied by two.',
    }],
}
"""
    state = _state()
    result = asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding(statement="The doubled change is 30."))).execute(
        CodeInterpreterInput(
            database_evidence="latest", analysis_goal="double series",
            code=code, insight_requests=[_request()],
        ), request_state=state,
    ))
    apply_observation(
        state,
        ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}),
        result,
        _AnalysisSpec(),
    )
    derived_id = result["derived_evidence"][0]["evidence_id"]
    assert derived_id in state.derived_evidence_artifacts
    assert result["computed_insights"][0]["derived_evidence_ids"] == [derived_id]
    assert f"view:derived_evidence:{derived_id}" in PresentationCatalog(state).canonical_refs()


def test_code_generation_and_binding_are_separate_llm_calls():
    code = "result = {'computed_insights': [{'insight_key': 'period_change', 'value': 15.0, 'calculation_trace': {'formula': 'last-first'}}], 'derived_evidence': []}"
    llm = _QueueLLM({"code": code}, _binding())
    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(database_evidence=_evidence(), analysis_goal="calculate", insight_requests=[_request()])
    ))
    assert len(llm.calls) == 2
    assert "Do not produce statements" in llm.calls[0][0][1]
    assert "Do not calculate" in llm.calls[1][0][1]
    assert result["computed_insights"][0]["value"] == 15.0


def test_preflight_blocks_imports_and_requires_result_assignment():
    assert "blocked syntax" in (_preflight_analysis_code("import os\nresult = {}") or "")
    assert "assign" in (_preflight_analysis_code("x = 1") or "")
    assert _preflight_analysis_code("result = {'computed_insights': [], 'derived_evidence': []}") is None


def test_analysis_workspace_exposes_latest_analysis_and_computed_receipts():
    state = _state()
    result = asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding())).execute(
        CodeInterpreterInput(
            database_evidence="latest", analysis_goal="calculate",
            code="result = {'computed_insights': [{'insight_key': 'period_change', 'value': 15.0, 'calculation_trace': {'formula': 'last-first'}}], 'derived_evidence': []}",
            insight_requests=[_request()],
        ), request_state=state,
    ))
    apply_observation(state, ToolObservation(tool_name="code_interpreter", success=True, summary="ok", payload={}), result, _AnalysisSpec())
    context = DataAgentPromptBuilder().build_context(
        state, build_conversation_state(ChatRequest(message="x"), "conv")
    )
    assert context["artifacts"]["refs"]["latest_analysis"] == f"analysis:{result['analysis_id']}"
    workspace = DataAgentPromptBuilder()._analysis_workspace(state)
    assert workspace["analyses"][0]["computed_insights"][0]["insight_key"] == "period_change"
