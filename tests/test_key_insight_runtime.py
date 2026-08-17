from __future__ import annotations

from core.key_insight import register_key_insights_from_payload
from runtime.conversation_state import sync_from_request
from runtime.tool_executor import ToolExecutor
from schemas.database import DatabaseEvidence
from schemas.state import ConversationStateModel, RequestStateModel
from schemas.tool import ToolCall


def _request_state() -> RequestStateModel:
    return RequestStateModel(
        request_id="req_insight",
        conversation_id="conv_insight",
        message="Bitcoin USD 2023-01-04 到 2023-02-03，计算起始值、结束值、涨跌幅、最高最低",
        status="running",
    )


def test_sql_evidence_registers_boundary_and_extreme_insights():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {"name": "start_value", "insight_type": "point_value", "requirements": {"time_position": "start"}},
                    {"name": "end_value", "insight_type": "point_value", "requirements": {"time_position": "end"}},
                    {"name": "highest_value", "insight_type": "extreme", "requirements": {"operator": "max"}},
                    {"name": "lowest_value", "insight_type": "extreme", "requirements": {"operator": "min"}},
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

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    insights_by_name = {insight.name: insight for insight in request_state.insight_set.insights}
    assert insights_by_name["start_value"].value == 10.0
    assert insights_by_name["end_value"].value == 12.0
    assert insights_by_name["highest_value"].value == 12.0
    assert insights_by_name["lowest_value"].value == 8.0
    assert set(coverage.verified) >= {"start_value", "end_value", "highest_value", "lowest_value"}
    assert "record_count" not in insights_by_name
    assert "data_coverage" not in insights_by_name


def test_sql_evidence_without_insight_requests_does_not_register_default_insights():
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

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    assert coverage.requested == []
    assert request_state.insight_set.insights == []


def test_sql_insight_uses_query_selected_measure_when_rows_have_multiple_numeric_columns():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {"name": "latest_price", "insight_type": "point_value", "requirements": {"time_position": "end"}}
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

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    assert coverage.verified == ["latest_price"]
    assert request_state.insight_set.insights[-1].value == 12.0
    assert request_state.insight_set.insights[-1].calculation_trace["value_key"] == "price"


def test_sql_insights_bind_each_request_to_its_result_dimensions():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {
                        "name": "series a latest",
                        "insight_type": "point_value",
                        "requirements": {"time_position": "end", "metric_name": "series_a"},
                    },
                    {
                        "name": "series b latest",
                        "insight_type": "point_value",
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

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    insights = {insight.name: insight for insight in request_state.insight_set.insights}
    assert coverage.verified == ["series a latest", "series b latest"]
    assert insights["series a latest"].value == 1.0
    assert insights["series b latest"].value == 9.0
    assert insights["series a latest"].calculation_trace["row_selectors"] == {"metric_name": "series_a"}


def test_sql_insight_marks_unmatched_result_dimensions_unavailable():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {
                        "name": "missing series latest",
                        "insight_type": "point_value",
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

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    insight = request_state.insight_set.insights[-1]
    assert coverage.unavailable == ["missing series latest"]
    assert insight.status == "unavailable"
    assert "metric_name='series_missing'" in (insight.unavailable_reason or "")


def test_sql_insight_does_not_guess_between_ambiguous_numeric_columns():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {"name": "latest_value", "insight_type": "point_value", "requirements": {"time_position": "end"}}
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_ambiguous",
        "summary": "Loaded rows.",
        "data": {"rows": [{"timestamp": "2023-01-01", "left": 1.0, "right": 2.0}]},
    }

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    assert coverage.verified == []
    assert coverage.missing == ["latest_value"]


def test_sql_time_boundary_insight_does_not_require_numeric_measure():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {"name": "latest_time", "insight_type": "time_boundary", "requirements": {"time_position": "end"}}
                ]
            },
        )
    )
    payload = {
        "evidence_id": "evi_time_only",
        "summary": "Loaded timestamps.",
        "data": {"rows": [{"timestamp": "2023-01-01"}, {"timestamp": "2023-01-02"}]},
    }

    coverage = register_key_insights_from_payload(request_state, "sql_query", payload)

    assert coverage.verified == ["latest_time"]
    assert request_state.insight_set.insights[-1].value == "2023-01-02"


def test_code_analysis_insight_dependencies_drop_evidence_refs_but_keep_insight_keys():
    request_state = _request_state()
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_series",
        result_type="timeseries",
        database="demo",
        summary="Loaded rows.",
        data={"points": [{"timestamp": "2023-01-01", "value": 1.0}]},
    )
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "constraints": {
                    "unsupported_insight_requests": [
                        {
                            "insight_key": "raw_series_window",
                            "insight_type": "series",
                            "contract_error": "SQL does not materialize row collections as Insights.",
                        }
                    ]
                }
            },
        )
    )
    executor = ToolExecutor.__new__(ToolExecutor)

    normalized = executor._remove_evidence_refs_from_insight_dependencies(
        [
            {
                "insight_key": "trend",
                "name": "trend",
                "insight_type": "analysis",
                "derived_from": [
                    "evi_series",
                    "evidence:evi_series",
                    "raw_series_window",
                    "verified_parent",
                ],
            }
        ],
        request_state,
    )

    assert normalized[0]["derived_from"] == ["verified_parent"]


def test_empty_sql_evidence_marks_requested_insights_unavailable():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={"insight_requests": [{"name": "start_value", "insight_type": "point_value"}]},
        )
    )

    coverage = register_key_insights_from_payload(
        request_state,
        "sql_query",
        {"evidence_id": "evi_empty", "summary": "No rows.", "data": {"rows": []}},
    )

    assert coverage.unavailable == ["start_value"]
    insight = next(item for item in request_state.insight_set.insights if item.name == "start_value")
    assert insight.status == "unavailable"
    assert "no row-like records" in insight.unavailable_reason.lower()


def test_code_interpreter_registers_result_insights_and_conversation_memory():
    request_state = _request_state()
    request_state.iteration = 2
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=2,
            tool_input={"insight_requests": [{"name": "percentage_change", "insight_type": "change"}]},
        )
    )
    payload = {
        "analysis_id": "ana_btc",
        "analysis_goal": "Calculate change.",
        "input_evidence_id": "evi_btc",
        "code_hash": "abc123",
        "produced_insights": [
                {
                    "name": "percentage_change",
                    "insight_type": "change",
                    "statement": "Bitcoin USD changed by 20%.",
                    "value": 20.0,
                    "calculation_trace": {"formula": "(end - start) / start * 100"},
                    "evidence_refs": [{"source_type": "analysis", "source_id": "ana_btc"}, {"source_type": "query", "source_id": "evi_btc"}],
                }
            ],
    }

    coverage = register_key_insights_from_payload(request_state, "code_interpreter", payload)
    assert coverage.verified == ["percentage_change"]
    insight = next(item for item in request_state.insight_set.insights if item.name == "percentage_change")
    assert insight.method == "code_interpreter"
    assert {ref.source_type for ref in insight.evidence_refs} == {"analysis", "query"}

    conversation_state = ConversationStateModel(conversation_id="conv_insight")
    sync_from_request(request_state, conversation_state)
    assert conversation_state.recent_insight_memory[-1].name == "percentage_change"
    assert "percentage_change: verified" in conversation_state.insight_memory_summary


def test_code_interpreter_does_not_promote_generic_metrics_to_insights():
    request_state = _request_state()
    request_state.message = "最大值和最小值的差异是多少？"
    request_state.iteration = 2
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=2,
            tool_input={"insight_requests": [{"name": "max_min_difference", "insight_type": "difference"}]},
        )
    )
    payload = {
        "analysis_id": "ana_btc",
        "analysis_goal": "Calculate max-min difference.",
        "input_evidence_id": "evi_btc",
        "code_hash": "abc123",
        "result": {
            "insights": [],
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

    coverage = register_key_insights_from_payload(request_state, "code_interpreter", payload)

    insights_by_name = {insight.name: insight for insight in request_state.insight_set.insights}
    assert coverage.missing == ["max_min_difference"]
    assert insights_by_name == {}


def test_insight_coverage_accumulates_across_sql_and_composite_analysis():
    request_state = _request_state()
    request_state.iteration = 1
    request_state.tool_history.append(
        ToolCall(
            tool_name="sql_query",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {
                        "insight_key": "price.start",
                        "name": "start_price",
                        "insight_type": "point_value",
                        "requirements": {"time_position": "start"},
                    },
                    {
                        "insight_key": "price.end",
                        "name": "end_price",
                        "insight_type": "point_value",
                        "requirements": {"time_position": "end"},
                    },
                ]
            },
        )
    )
    register_key_insights_from_payload(
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
                "insight_requests": [
                    {
                        "insight_key": "price.percentage_change",
                        "name": "percentage_change",
                        "insight_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ]
            },
        )
    )
    coverage = register_key_insights_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_change",
            "analysis_goal": "Calculate percentage change.",
            "input_evidence_id": "evi_prices",
            "code_hash": "change123",
            "produced_insights": [
                    {
                        "insight_key": "price.percentage_change",
                        "name": "percentage_change",
                        "insight_type": "difference",
                        "statement": "Price increased by 20%.",
                        "value": 20.0,
                        "derived_from": ["price.start", "price.end"],
                        "calculation_trace": {"formula": "(end - start) / start * 100"},
                        "evidence_refs": [{"source_type": "analysis", "source_id": "ana_change"}, {"source_type": "query", "source_id": "evi_prices"}],
                    }
                ],
        },
    )

    assert coverage.missing == []
    assert coverage.verified == ["start_price", "end_price", "percentage_change"]
    composite = next(insight for insight in request_state.insight_set.insights if insight.insight_key == "price.percentage_change")
    assert composite.status == "verified"
    assert composite.derived_from == ["price.start", "price.end"]
    assert {ref.source_id for ref in composite.evidence_refs} >= {"evi_prices", "ana_change"}


def test_composite_insight_with_missing_parent_is_partial():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {
                        "insight_key": "price.change",
                        "name": "change",
                        "insight_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ]
            },
        )
    )

    coverage = register_key_insights_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_partial",
            "input_evidence_id": "evi_prices",
            "produced_insights": [
                    {
                        "insight_key": "price.change",
                        "name": "change",
                        "insight_type": "difference",
                        "statement": "Change is 2.",
                        "value": 2.0,
                        "derived_from": ["price.start", "price.end"],
                        "calculation_trace": {"formula": "end - start"},
                        "evidence_refs": [{"source_type": "analysis", "source_id": "ana_partial"}],
                    }
                ],
        },
    )

    assert coverage.partial == ["change"]
    insight = request_state.insight_set.insights[0]
    assert insight.status == "partial"
    assert "unverified_dependencies" in insight.quality_flags


def test_bound_code_insight_preserves_planned_dependencies():
    request_state = _request_state()
    request_state.tool_history.append(
        ToolCall(
            tool_name="code_interpreter",
            iteration=1,
            tool_input={
                "insight_requests": [
                    {
                        "insight_key": "price.change",
                        "name": "change",
                        "insight_type": "difference",
                        "derived_from": ["price.start", "price.end"],
                    }
                ]
            },
        )
    )

    coverage = register_key_insights_from_payload(
        request_state,
        "code_interpreter",
        {
            "analysis_id": "ana_direct_change",
            "input_evidence_id": "evi_prices",
            "produced_insights": [
                    {
                        "insight_key": "price.change",
                        "name": "change",
                        "insight_type": "difference",
                        "statement": "Change is 2.",
                        "value": 2.0,
                        "derived_from": [],
                        "calculation_trace": {"formula": "last row - first row"},
                        "evidence_refs": [{"source_type": "analysis", "source_id": "ana_direct_change"}, {"source_type": "query", "source_id": "evi_prices"}],
                    }
                ],
        },
    )

    insight = request_state.insight_set.insights[0]
    assert coverage.partial == ["change"]
    assert insight.derived_from == ["price.start", "price.end"]
    assert insight.status == "partial"
