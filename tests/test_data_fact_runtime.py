from __future__ import annotations

from core.data_fact import register_data_facts_from_payload
from runtime.conversation_state import sync_from_request
from schemas.state import ConversationStateModel, RequestStateModel
from schemas.tool import ToolCall


def _request_state() -> RequestStateModel:
    return RequestStateModel(
        request_id="req_fact",
        conversation_id="conv_fact",
        message="Bitcoin USD 2023-01-04 到 2023-02-03，计算起始值、结束值、涨跌幅、最高最低",
        status="running",
    )


def test_sql_evidence_registers_boundary_and_extreme_facts():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {"name": "start_value", "fact_type": "point_value", "requirements": {"time_position": "start"}},
                    {"name": "end_value", "fact_type": "point_value", "requirements": {"time_position": "end"}},
                    {"name": "highest_value", "fact_type": "extreme", "requirements": {"operator": "max"}},
                    {"name": "lowest_value", "fact_type": "extreme", "requirements": {"operator": "min"}},
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_btc",
        "summary": "Loaded Bitcoin prices.",
        "data": {
            "rows": [
                {"timestamp": "2023-01-04", "price": 10.0},
                {"timestamp": "2023-01-05", "price": 8.0},
                {"timestamp": "2023-02-03", "price": 12.0},
            ]
        },
    }

    coverage = register_data_facts_from_payload(request_state, "sql_query", payload)

    facts_by_name = {fact.name: fact for fact in request_state.fact_set.facts}
    assert facts_by_name["start_value"].value == 10.0
    assert facts_by_name["end_value"].value == 12.0
    assert facts_by_name["highest_value"].value == 12.0
    assert facts_by_name["lowest_value"].value == 8.0
    assert set(coverage.verified) >= {"start_value", "end_value", "highest_value", "lowest_value"}


def test_empty_sql_evidence_marks_requested_facts_unavailable():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={"fact_requests": [{"name": "start_value", "fact_type": "point_value"}]},
        )
    )

    coverage = register_data_facts_from_payload(
        request_state,
        "sql_query",
        {"evidence_id": "evi_empty", "summary": "No rows.", "data": {"rows": []}},
    )

    assert coverage.unavailable == ["start_value"]
    fact = next(item for item in request_state.fact_set.facts if item.name == "start_value")
    assert fact.status == "unavailable"
    assert "no row-like records" in fact.unavailable_reason.lower()


def test_code_interpreter_registers_result_facts_and_conversation_memory():
    request_state = _request_state()
    request_state.iteration = 2
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=2,
            tool_input={"fact_requests": [{"name": "percentage_change", "fact_type": "change"}]},
        )
    )
    payload = {
        "analysis_id": "ana_btc",
        "analysis_goal": "Calculate change.",
        "input_evidence_id": "evi_btc",
        "code_hash": "abc123",
        "result": {
            "facts": [
                {
                    "name": "percentage_change",
                    "fact_type": "change",
                    "statement": "Bitcoin USD changed by 20%.",
                    "value": 20.0,
                }
            ],
            "metrics": {},
            "details": {},
        },
    }

    coverage = register_data_facts_from_payload(request_state, "code_interpreter", payload)
    assert coverage.verified == ["percentage_change"]
    fact = next(item for item in request_state.fact_set.facts if item.name == "percentage_change")
    assert fact.method == "code_interpreter"
    assert {ref.source_type for ref in fact.evidence_refs} == {"analysis", "query"}

    conversation_state = ConversationStateModel(conversation_id="conv_fact")
    sync_from_request(request_state, conversation_state)
    assert conversation_state.recent_fact_memory[-1].name == "percentage_change"
    assert "percentage_change: verified" in conversation_state.fact_memory_summary

