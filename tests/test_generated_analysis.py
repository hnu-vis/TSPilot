from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.settings import get_settings
from core.analysis.python_runner import AnalysisCodeError, validate_analysis_result_payload
from core.key_insight.binder import LLMInsightBinder
from core.visualization.materializer import PresentationCatalog
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from sandbox import execute_python_sandbox_v1
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.analysis import ComputedInsight
from schemas.key_insight import KeyInsightRequest
from schemas.timeseries import AnomalyResult
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


def _binding(
    key: str = "period_change",
    statement: str = "The period change is 15.",
    derived_from: list[str] | None = None,
) -> dict:
    return {"bindings": [{
        "insight_key": key,
        "supported": True,
        "unsupported_reason": None,
        "statement": statement,
        "derived_from": derived_from or [],
    }]}


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


def test_input_normalizes_a_single_evidence_reference_list():
    payload = CodeInterpreterInput.model_validate({
        "database_evidence": ["evidence:evi_generated"],
        "analysis_goal": "calculate",
        "insight_requests": [_request().model_dump(mode="json")],
    })
    assert payload.database_evidence == "evidence:evi_generated"


def test_embedded_prompt_preview_resolves_to_full_state_artifact_before_execution():
    state = _state()
    preview = _evidence().model_copy(update={"data": {"rows": [_evidence().data["rows"][0]]}})
    code = (
        "result = {'computed_insights': [{'insight_key': 'row_count', 'value': int(len(df)), "
        "'calculation_trace': 'len(df)'}], 'derived_evidence': []}"
    )

    result = asyncio.run(CodeInterpreterTool(llm=_QueueLLM(_binding("row_count", "Three rows."))).execute(
        CodeInterpreterInput(
            database_evidence=preview.model_dump(mode="json"), analysis_goal="count full rows",
            code=code, insight_requests=[_request("row_count")],
        ),
        request_state=state,
    ))

    assert result["input_row_count"] == 3
    assert result["computed_insights"][0]["value"] == 3


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


def test_binder_removes_consistency_only_dependency_and_preserves_endpoint_claim():
    llm = _QueueLLM({
        "bindings": [{
            "insight_key": "endpoint_change",
            "supported": True,
            "unsupported_reason": None,
            "statement": "The last observed value is 15 higher than the first observed value.",
            "derived_from": [],
        }]
    })
    request = KeyInsightRequest(
        insight_key="endpoint_change",
        name="Endpoint change",
        insight_type="difference",
        derived_from=["maximum_value"],
    )

    insights = asyncio.run(LLMInsightBinder(llm).bind(
        requests=[request],
        computed=[ComputedInsight(
            insight_key="endpoint_change",
            value=15.0,
            calculation_trace={"formula": "last row - first row", "first": 10.0, "last": 25.0},
        )],
        analysis_id="ana_endpoint",
        analysis_goal="Compare the interval endpoints",
        input_evidence_id="evi_generated",
        computation_code="child = float(df['value'].iloc[-1] - df['value'].iloc[0])",
        response_language="en",
    ))

    assert insights[0].derived_from == []
    assert "last observed value" in insights[0].statement
    binder_prompt = llm.calls[0][0][1]
    assert "consistency checks" in binder_prompt
    assert "endpoint-change claim" in binder_prompt
    assert "executed_computation_code" in llm.calls[0][1][1]
    assert "input_insights" in binder_prompt


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
        'name': 'doubled_series', 'rows': derived_rows,
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
    assert result["derived_evidence"][0]["shape"] == "timeseries"
    assert result["computed_insights"][0]["derived_evidence_ids"] == [derived_id]
    assert f"view:derived_evidence:{derived_id}" in PresentationCatalog(state).canonical_refs()


def test_code_generation_and_binding_are_separate_llm_calls():
    code = "result = {'computed_insights': [{'insight_key': 'period_change', 'value': float(value.iloc[-1] - value.iloc[0]), 'calculation_trace': {'formula': 'last-first'}}], 'derived_evidence': []}"
    llm = _QueueLLM({"code": code}, _binding())
    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(database_evidence=_evidence(), analysis_goal="calculate", insight_requests=[_request()])
    ))
    assert len(llm.calls) == 2
    assert "Do not produce statements" in llm.calls[0][0][1]
    assert "Do not calculate" in llm.calls[1][0][1]
    assert result["computed_insights"][0]["value"] == 15.0


def test_code_generation_prompt_uses_source_contracts_without_record_previews():
    code = (
        "result = {'computed_insights': [{'insight_key': 'row_count', 'value': int(len(rows)), "
        "'calculation_trace': 'len(rows)'}], 'derived_evidence': []}"
    )
    llm = _QueueLLM({"code": code}, _binding("row_count", "There are three rows."))
    state = _state()

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(
            database_evidence="latest", analysis_goal="count observations",
            insight_requests=[_request("row_count")],
        ),
        request_state=state,
    ))

    prompt_payload = json.loads(llm.calls[0][1][1])
    assert result["computed_insights"][0]["value"] == 3
    assert not any(key.startswith("sample_") for key in prompt_payload["canonical_inputs"])
    assert prompt_payload["artifact_sources"][0]["datasets"] == [{
        "name": "records",
        "shape": "timeseries",
        "row_count": 3,
        "schema_fields": [
            {"name": "timestamp", "type": "str"},
            {"name": "value", "type": "float"},
        ],
    }]


def test_code_generation_prompt_references_authoritative_anomalies_without_copying_points():
    state = _state()
    anomaly = AnomalyResult(
        anomaly_id="anomaly_generated",
        detector_name="unit",
        anomaly_points=[{
            "timestamp": "2023-01-01T01:00:00Z",
            "value": 987654.321,
            "score": 4.2,
        }],
        diagnostics={"resolved_evidence_id": "evi_generated"},
    )
    state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
    state.latest_anomaly = anomaly
    code = (
        "result = {'computed_insights': [{'insight_key': 'anomaly_count', "
        "'value': int(len(anomaly_context['anomaly_points'])), "
        "'calculation_trace': {'source_ref': anomaly_context['source_ref']}}], "
        "'derived_evidence': []}"
    )
    llm = _QueueLLM({"code": code}, _binding("anomaly_count", "There is one anomaly."))

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(
            database_evidence="latest", analysis_goal="count authoritative anomalies",
            insight_requests=[_request("anomaly_count")],
        ),
        request_state=state,
    ))

    prompt_payload = json.loads(llm.calls[0][1][1])
    assert result["computed_insights"][0]["value"] == 1
    assert prompt_payload["anomaly_context"] == {
        "source_ref": "anomaly:anomaly_generated",
        "point_count": 1,
        "schema_fields": [
            {"name": "timestamp", "type": "str"},
            {"name": "value", "type": "float"},
            {"name": "score", "type": "float"},
        ],
        "runtime_variable": "anomaly_context",
    }
    assert "987654.321" not in llm.calls[0][1][1]


def test_generated_code_preflight_failure_is_repaired_by_llm_before_execution():
    invalid = {"code": "import pandas as pd\nresult = {}"}
    repaired_code = "result = {'computed_insights': [{'insight_key': 'period_change', 'value': float(value.iloc[-1] - value.iloc[0]), 'calculation_trace': 'last minus first'}], 'derived_evidence': []}"
    llm = _QueueLLM(invalid, {"code": repaired_code}, _binding())

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(database_evidence=_evidence(), analysis_goal="calculate", insight_requests=[_request()])
    ))

    assert len(llm.calls) == 3
    assert result["computed_insights"][0]["value"] == 15.0


def test_binder_semantic_rejection_regenerates_code_before_publishing_insight():
    endpoint_code = (
        "direction = 'up' if float(value.iloc[-1]) > float(value.iloc[0]) else 'down'\n"
        "result = {'computed_insights': [{'insight_key': 'overall_trend', 'value': direction, "
        "'calculation_trace': {'method': 'endpoint comparison', 'n': int(len(value))}}], "
        "'derived_evidence': []}"
    )
    regression_code = (
        "slope = float(np.polyfit(np.arange(len(value)), value.astype(float), 1)[0])\n"
        "direction = 'up' if slope > 0 else 'down' if slope < 0 else 'flat'\n"
        "result = {'computed_insights': [{'insight_key': 'overall_trend', 'value': direction, "
        "'calculation_trace': {'method': 'least-squares slope', 'n': int(len(value)), 'slope': slope}}], "
        "'derived_evidence': []}"
    )
    rejected_binding = {
        "bindings": [{
            "insight_key": "overall_trend",
            "supported": False,
            "unsupported_reason": "Endpoint comparison does not support an overall trend claim.",
            "statement": "The endpoint is higher than the start.",
            "derived_from": [],
        }]
    }
    accepted_binding = {
        "bindings": [{
            "insight_key": "overall_trend",
            "supported": True,
            "unsupported_reason": None,
            "statement": "The least-squares slope across three observations is positive.",
            "derived_from": [],
        }]
    }
    llm = _QueueLLM(
        {"code": endpoint_code}, rejected_binding,
        {"code": regression_code}, accepted_binding,
    )

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(),
            analysis_goal="calculate the overall trend across the interval",
            insight_requests=[_request("overall_trend")],
        )
    ))

    assert len(llm.calls) == 4
    assert result["computed_insights"][0]["value"] == "up"
    assert result["computed_insights"][0]["calculation_trace"]["method"] == "least-squares slope"
    assert "Insight Binder rejected computed semantics" in llm.calls[2][-1][1]


def test_binder_schema_failure_is_repaired_by_llm_without_recomputing_value():
    code = "result = {'computed_insights': [{'insight_key': 'period_change', 'value': 15.0, 'calculation_trace': 'last minus first'}], 'derived_evidence': []}"
    llm = _QueueLLM(
        {"bindings": [{"insight_key": "period_change", "statement": "", "derived_from": []}]},
        _binding(),
    )

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(), analysis_goal="calculate", code=code,
            insight_requests=[_request()],
        )
    ))

    assert len(llm.calls) == 2
    assert result["produced_insights"][0]["value"] == 15.0


def test_generated_derived_evidence_validation_failure_is_repaired_by_llm():
    invalid_code = (
        "result = {'computed_insights': [{'insight_key': 'period_change', 'value': float(value.iloc[-1] - value.iloc[0]), "
        "'calculation_trace': 'last minus first'}], 'derived_evidence': [{"
        "'name': 'decision_points', 'rows': [['buy', 10.0]], 'transform_summary': 'bad rows'}]}"
    )
    repaired_code = (
        "result = {'computed_insights': [{'insight_key': 'period_change', 'value': float(value.iloc[-1] - value.iloc[0]), "
        "'calculation_trace': 'last minus first', 'derived_evidence_names': ['decision_points']}], "
        "'derived_evidence': [{'name': 'decision_points', 'rows': [{"
        "'timestamp': str(time.iloc[0]), 'value': 10.0, 'role': 'buy'}], "
        "'transform_summary': 'one decision point'}]}"
    )
    llm = _QueueLLM({"code": invalid_code}, {"code": repaired_code}, _binding())

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(database_evidence=_evidence(), analysis_goal="calculate", insight_requests=[_request()])
    ))

    assert len(llm.calls) == 3
    assert result["derived_evidence"][0]["shape"] == "timeseries"


def test_generated_unreferenced_derived_evidence_is_repaired_by_llm():
    unreferenced_code = (
        "derived_rows = [{'timestamp': str(time.iloc[0]), 'value': float(value.iloc[0]), 'role': 'start'}]\n"
        "result = {'computed_insights': [{'insight_key': 'period_change', "
        "'value': float(value.iloc[-1] - value.iloc[0]), 'calculation_trace': 'last minus first'}], "
        "'derived_evidence': [{'name': 'turning_boundaries', 'rows': derived_rows, "
        "'transform_summary': 'calculated boundary rows'}]}"
    )
    linked_code = unreferenced_code.replace(
        "'calculation_trace': 'last minus first'",
        "'calculation_trace': 'last minus first', 'derived_evidence_names': ['turning_boundaries']",
    )
    llm = _QueueLLM({"code": unreferenced_code}, {"code": linked_code}, _binding())

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(), analysis_goal="calculate", insight_requests=[_request()],
        )
    ))

    derived_id = result["derived_evidence"][0]["evidence_id"]
    assert len(llm.calls) == 3
    assert result["computed_insights"][0]["derived_evidence_ids"] == [derived_id]
    assert "unreferenced" in llm.calls[1][-1][1]


def test_preflight_allows_runtime_modules_but_blocks_unsafe_imports():
    assert _preflight_analysis_code("import math\nresult = {}") is None
    assert _preflight_analysis_code("import pandas as pd\nimport numpy as np\nresult = {}") is None
    assert "unsupported module" in (_preflight_analysis_code("import os\nresult = {}") or "")
    assert "top-level" in (
        _preflight_analysis_code("if True:\n    import math\nresult = {}") or ""
    )


def test_preflight_requires_result_assignment_and_grounded_computation():
    assert "assign" in (_preflight_analysis_code("x = 1") or "")
    assert _preflight_analysis_code("result = {'computed_insights': [], 'derived_evidence': []}") is None
    assert "grounded sandbox inputs" in (
        _preflight_analysis_code(
            "result = {'computed_insights': [], 'derived_evidence': []}",
            require_grounded_computation=True,
        )
        or ""
    )
    assert _preflight_analysis_code(
        "result = {'computed_insights': [{'value': float(value.iloc[0])}], 'derived_evidence': []}",
        require_grounded_computation=True,
    ) is None


def test_code_interpreter_executes_redundant_safe_dataframe_imports_without_repair():
    code = """
import pandas as pd
import numpy as np
result = {
    'computed_insights': [{
        'insight_key': 'period_change',
        'value': float(np.asarray(value)[-1] - np.asarray(value)[0]),
        'calculation_trace': {'formula': 'last-first'},
    }],
    'derived_evidence': [],
}
"""
    llm = _QueueLLM(_binding())

    result = asyncio.run(CodeInterpreterTool(llm=llm).execute(
        CodeInterpreterInput(
            database_evidence=_evidence(), analysis_goal="calculate period change",
            code=code, insight_requests=[_request()],
        )
    ))

    assert result["computed_insights"][0]["value"] == 15.0
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "code,error",
    [
        ("import os\nresult = {}", "unsupported module"),
        ("result = open('/tmp/blocked')", "blocked function"),
        ("result = __builtins__", "blocked name"),
    ],
)
def test_subprocess_worker_enforces_the_shared_policy_without_tool_preflight(code, error):
    with pytest.raises(AnalysisCodeError, match=error):
        execute_python_sandbox_v1(
            code=code, rows=[], points=[], columns=[], metadata={}, diagnostics={},
        )


def test_subprocess_worker_exposes_authoritative_anomaly_context_as_promised():
    code = """
result = {
    'computed_insights': [{
        'insight_key': 'anomaly_count',
        'value': len(anomaly_context['anomaly_points']),
        'calculation_trace': {'source_ref': anomaly_context['source_ref']},
    }],
    'derived_evidence': [],
}
"""

    output = execute_python_sandbox_v1(
        code=code, rows=[], points=[], columns=[], metadata={}, diagnostics={},
        analysis_context={
            "anomaly_context": {
                "source_ref": "anomaly:demo",
                "anomaly_points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 99.0}],
            },
        },
    )

    assert output.result["computed_insights"][0]["value"] == 1


def test_prompt_context_exposes_analysis_ref_and_bound_insight_receipt():
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
    recent = context["state"]["insight_state"]["recent_insights"]
    assert recent[0]["insight_key"] == "period_change"
    assert recent[0]["status"] == "verified"
    assert recent[0]["value"] == 15.0
    assert recent[0]["evidence_refs"] == [
        f"analysis:{result['analysis_id']}",
        "evidence:evi_generated",
    ]
