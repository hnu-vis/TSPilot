from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.analysis.python_runner import AnalysisCodeError
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsightRequest
from schemas.timeseries import AnomalyResult
from tools.code_interpreter import CodeInterpreterInput, CodeInterpreterTool


class _BinderLLM:
    def __init__(self, key="row_count"):
        self.key = key
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=json.dumps({
            "bindings": [{"insight_key": self.key, "statement": f"Computed {self.key}."}]
        }))


def _request(key="row_count"):
    return KeyInsightRequest(name=key.replace("_", " "), insight_type="custom", insight_key=key)


def _evidence(row_count=40):
    rows = [
        {"timestamp": f"2023-01-01T00:{index:02d}:00Z", "price": float(index + 1)}
        for index in range(row_count)
    ]
    return DatabaseEvidence(
        evidence_id="evi_full",
        result_type="table",
        database="demo",
        query_language="unit",
        query="unit:test",
        summary="full rows",
        columns=["timestamp", "price"],
        data={"rows": rows},
    )


def _state():
    state = build_request_state(ChatRequest(message="计算记录数"), get_settings())
    evidence = _evidence()
    state.latest_database_evidence = evidence
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    return state


def _count_code(key="row_count"):
    return (
        "result = {'computed_insights': [{"
        f"'insight_key': '{key}', 'value': len(rows), "
        "'calculation_trace': {'operation': 'len(rows)', 'input_row_count': len(rows)}}], "
        "'derived_evidence': []}"
    )


def test_code_interpreter_resolves_full_request_scoped_evidence():
    result = asyncio.run(CodeInterpreterTool(llm=_BinderLLM()).execute(
        CodeInterpreterInput(
            database_evidence="latest", analysis_goal="count rows", code=_count_code(),
            insight_requests=[_request()],
        ),
        request_state=_state(),
    ))
    assert result["input_row_count"] == 40
    assert result["computed_insights"][0]["value"] == 40


def test_code_interpreter_accepts_analysis_code_alias():
    payload = CodeInterpreterInput.model_validate({
        "database_evidence": _evidence(),
        "analysis_goal": "count rows",
        "analysis_code": _count_code(),
        "insight_requests": [_request().model_dump(mode="json")],
    })
    assert payload.code == _count_code()


def test_code_interpreter_rejects_unknown_evidence_reference():
    with pytest.raises(ValueError, match="grounded database_evidence"):
        asyncio.run(CodeInterpreterTool(llm=_BinderLLM()).execute(
            CodeInterpreterInput(
                database_evidence="evidence:missing", analysis_goal="count rows", code=_count_code(),
                insight_requests=[_request()],
            ),
            request_state=_state(),
        ))


def test_rows_only_table_remains_available_without_guessing_value_column():
    result = asyncio.run(CodeInterpreterTool(llm=_BinderLLM()).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(3), analysis_goal="count rows", code=_count_code(),
            insight_requests=[_request()],
        )
    ))
    assert result["computed_insights"][0]["value"] == 3


def test_invalid_empty_derived_evidence_is_rejected_before_registration():
    code = """
result = {
    'computed_insights': [{'insight_key': 'row_count', 'value': len(rows), 'calculation_trace': {'operation': 'count'}}],
    'derived_evidence': [{'name': 'empty', 'shape': 'records', 'rows': [], 'transform_summary': 'empty output'}],
}
"""
    with pytest.raises(AnalysisCodeError, match="invalid derived evidence"):
        asyncio.run(CodeInterpreterTool(llm=_BinderLLM()).execute(
            CodeInterpreterInput(
                database_evidence=_evidence(), analysis_goal="count rows", code=code,
                insight_requests=[_request()],
            )
        ))


def test_authoritative_anomaly_ref_must_be_carried_in_computation_trace():
    state = _state()
    anomaly = AnomalyResult(
        anomaly_id="ano_evi_full",
        detector_name="test",
        status="succeeded",
        summary="one anomaly",
        anomaly_points=[{"timestamp": "2023-01-01T00:39:00Z", "value": 40.0}],
        diagnostics={"resolved_evidence_id": "evi_full"},
    )
    state.latest_anomaly = anomaly
    state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
    code = """
result = {
    'computed_insights': [{
        'insight_key': 'row_count', 'value': len(rows) - len(analysis_context['anomaly_context']['anomaly_points']),
        'calculation_trace': {'operation': 'exclude authoritative anomalies', 'source_ref': analysis_context['anomaly_context']['source_ref']},
    }],
    'derived_evidence': [],
}
"""
    result = asyncio.run(CodeInterpreterTool(llm=_BinderLLM()).execute(
        CodeInterpreterInput(
            database_evidence="latest", analysis_goal="count clean rows", code=code,
            insight_requests=[_request()],
        ), request_state=state,
    ))
    assert result["computed_insights"][0]["calculation_trace"]["source_ref"] == "anomaly:ano_evi_full"
