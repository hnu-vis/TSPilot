from __future__ import annotations

from core.data_fact import register_data_facts_from_payload
from runtime.conversation_state import sync_from_request
from runtime.tool_executor import ToolExecutor
from schemas.database import DatabaseEvidence
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
    assert "record_count" not in facts_by_name
    assert "data_coverage" not in facts_by_name


def test_sql_evidence_without_fact_requests_does_not_register_default_facts():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={},
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

    assert coverage.requested == []
    assert request_state.fact_set.facts == []


def test_sql_fact_uses_query_selected_measure_when_rows_have_multiple_numeric_columns():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {"name": "latest_price", "fact_type": "point_value", "requirements": {"time_position": "end"}}
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_multi_numeric",
        "summary": "Loaded rows.",
        "data": {
            "rows": [
                {"timestamp": "2023-01-01", "sequence": 1, "price": 10.0},
                {"timestamp": "2023-01-02", "sequence": 2, "price": 12.0},
            ]
        },
        "diagnostics": {"llm_query_generation": {"selected_fields": ["price"]}},
    }

    coverage = register_data_facts_from_payload(request_state, "sql_query", payload)

    assert coverage.verified == ["latest_price"]
    assert request_state.fact_set.facts[-1].value == 12.0
    assert request_state.fact_set.facts[-1].calculation_trace["value_key"] == "price"


def test_sql_facts_bind_each_request_to_its_result_dimensions():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {
                        "name": "series a latest",
                        "fact_type": "point_value",
                        "requirements": {"time_position": "end", "metric_name": "series_a"},
                    },
                    {
                        "name": "series b latest",
                        "fact_type": "point_value",
                        "dimensions": {"metric_name": "series_b"},
                        "requirements": {"time_position": "end"},
                    },
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_multi_series",
        "summary": "Loaded two series.",
        "data": {
            "rows": [
                {"metric_name": "series_a", "timestamp": "2024-01-01", "value": 1.0},
                {"metric_name": "series_b", "timestamp": "2024-01-01", "value": 9.0},
            ]
        },
    }

    coverage = register_data_facts_from_payload(request_state, "sql_query", payload)

    facts = {fact.name: fact for fact in request_state.fact_set.facts}
    assert coverage.verified == ["series a latest", "series b latest"]
    assert facts["series a latest"].value == 1.0
    assert facts["series b latest"].value == 9.0
    assert facts["series a latest"].calculation_trace["row_selectors"] == {"metric_name": "series_a"}


def test_sql_fact_marks_unmatched_result_dimensions_unavailable():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {
                        "name": "missing series latest",
                        "fact_type": "point_value",
                        "requirements": {"time_position": "end", "metric_name": "series_missing"},
                    }
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_one_series",
        "summary": "Loaded one series.",
        "data": {
            "rows": [
                {"metric_name": "series_a", "timestamp": "2024-01-01", "value": 1.0},
            ]
        },
    }

    coverage = register_data_facts_from_payload(request_state, "sql_query", payload)

    fact = request_state.fact_set.facts[-1]
    assert coverage.unavailable == ["missing series latest"]
    assert fact.status == "unavailable"
    assert "metric_name='series_missing'" in (fact.unavailable_reason or "")


def test_sql_fact_does_not_guess_between_ambiguous_numeric_columns():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {"name": "latest_value", "fact_type": "point_value", "requirements": {"time_position": "end"}}
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_ambiguous",
        "summary": "Loaded rows.",
        "data": {"rows": [{"timestamp": "2023-01-01", "left": 1.0, "right": 2.0}]},
    }

    coverage = register_data_facts_from_payload(request_state, "sql_query", payload)

    assert coverage.verified == []
    assert coverage.missing == ["latest_value"]


def test_sql_time_boundary_fact_does_not_require_numeric_measure():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {"name": "latest_time", "fact_type": "time_boundary", "requirements": {"time_position": "end"}}
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_time_only",
        "summary": "Loaded timestamps.",
        "data": {"rows": [{"timestamp": "2023-01-01"}, {"timestamp": "2023-01-02"}]},
    }

    coverage = register_data_facts_from_payload(request_state, "sql_query", payload)

    assert coverage.verified == ["latest_time"]
    assert request_state.fact_set.facts[-1].value == "2023-01-02"


def test_code_analysis_fact_dependencies_drop_evidence_refs_but_keep_fact_keys():
    request_state = _request_state()
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_series",
        result_type="timeseries",
        database="demo",
        summary="Loaded rows.",
        data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
    )
    executor = ToolExecutor.__new__(ToolExecutor)

    normalized = executor._remove_evidence_refs_from_fact_dependencies(
        [
            {
                "fact_key": "trend",
                "name": "trend",
                "fact_type": "analysis",
                "derived_from": ["evi_series", "evidence:evi_series", "verified_parent"],
            }
        ],
        request_state,
    )

    assert normalized[0]["derived_from"] == ["verified_parent"]


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
                    "calculation_trace": {"formula": "(end - start) / start * 100"},
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


def test_code_interpreter_ignores_unrequested_analysis_metrics():
    request_state = _request_state()
    request_state.message = "最大值和最小值的差异是多少？"
    request_state.iteration = 2
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=2,
            tool_input={"fact_requests": [{"name": "max_min_difference", "fact_type": "difference"}]},
        )
    )
    payload = {
        "analysis_id": "ana_btc",
        "analysis_goal": "Calculate max-min difference.",
        "input_evidence_id": "evi_btc",
        "code_hash": "abc123",
        "result": {
            "facts": [],
            "metrics": {
                "record_count": 3,
                "max_value": 12.0,
                "min_value": 8.0,
                "max_min_difference": 4.0,
                "start_end_change": 2.0,
            },
            "details": {},
        },
    }

    coverage = register_data_facts_from_payload(request_state, "code_interpreter", payload)

    facts_by_name = {fact.name: fact for fact in request_state.fact_set.facts}
    assert coverage.verified == ["max_min_difference"]
    assert facts_by_name["max_min_difference"].value == 4.0
    assert set(facts_by_name) == {"max_min_difference"}


def test_fact_coverage_accumulates_across_sql_and_composite_analysis():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {
                        "fact_key": "price.start",
                        "name": "start_price",
                        "fact_type": "point_value",
                        "requirements": {"time_position": "start"},
                    },
                    {
                        "fact_key": "price.end",
                        "name": "end_price",
                        "fact_type": "point_value",
                        "requirements": {"time_position": "end"},
                    },
                ]
            },
        )
    )
    register_data_facts_from_payload(
        request_state,
        "sql_query",
        {
            "evidence_id": "evi_prices",
            "summary": "Price evidence.",
            "data": {
                "rows": [
                    {"timestamp": "2023-01-01", "price": 10.0},
                    {"timestamp": "2023-01-02", "price": 12.0},
                ]
            },
        },
    )

    request_state.iteration = 2
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=2,
            tool_input={
                "fact_requests": [
                    {
                        "fact_key": "price.percentage_change",
                        "name": "percentage_change",
                        "fact_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ]
            },
        )
    )
    coverage = register_data_facts_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_change",
            "analysis_goal": "Calculate percentage change.",
            "input_evidence_id": "evi_prices",
            "code_hash": "change123",
            "result": {
                "summary": "Price increased by 20%.",
                "metrics": {"percentage_change": 20.0},
                "details": {},
                "facts": [
                    {
                        "fact_key": "price.percentage_change",
                        "name": "percentage_change",
                        "fact_type": "difference",
                        "statement": "Price increased by 20%.",
                        "value": 20.0,
                        "derived_from": ["price.start", "price.end"],
                        "calculation_trace": {"formula": "(end - start) / start * 100"},
                    }
                ],
            },
        },
    )

    assert coverage.missing == []
    assert coverage.verified == ["start_price", "end_price", "percentage_change"]
    composite = next(fact for fact in request_state.fact_set.facts if fact.fact_key == "price.percentage_change")
    assert composite.status == "verified"
    assert composite.derived_from == ["price.start", "price.end"]
    assert {ref.source_id for ref in composite.evidence_refs} >= {"evi_prices", "ana_change"}


def test_composite_fact_with_missing_parent_is_partial():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {
                        "fact_key": "price.change",
                        "name": "change",
                        "fact_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ]
            },
        )
    )

    coverage = register_data_facts_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_partial",
            "input_evidence_id": "evi_prices",
            "result": {
                "facts": [
                    {
                        "fact_key": "price.change",
                        "name": "change",
                        "fact_type": "difference",
                        "statement": "Change is 2.",
                        "value": 2.0,
                        "derived_from": ["price.start", "price.end"],
                        "calculation_trace": {"formula": "end - start"},
                    }
                ],
                "metrics": {},
                "details": {},
            },
        },
    )

    assert coverage.partial == ["change"]
    fact = request_state.fact_set.facts[0]
    assert fact.status == "partial"
    assert "unverified_dependencies" in fact.quality_flags


def test_code_fact_directly_computed_from_rows_does_not_inherit_planned_dependencies():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=1,
            tool_input={
                "fact_requests": [
                    {
                        "fact_key": "price.change",
                        "name": "change",
                        "fact_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ]
            },
        )
    )

    coverage = register_data_facts_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_direct_change",
            "input_evidence_id": "evi_prices",
            "result": {
                "facts": [
                    {
                        "fact_key": "price.change",
                        "name": "change",
                        "fact_type": "difference",
                        "statement": "Change is 2.",
                        "value": 2.0,
                        "derived_from": [],
                        "calculation_trace": {"formula": "last row - first row"},
                    }
                ],
                "metrics": {},
                "details": {},
            },
        },
    )

    fact = request_state.fact_set.facts[0]
    assert coverage.verified == ["change"]
    assert fact.derived_from == []
    assert fact.status == "verified"
