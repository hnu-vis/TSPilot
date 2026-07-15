from __future__ import annotations

from app.settings import get_settings
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence

from core.insight import build_insight_result


def _rich_timeseries_evidence() -> DatabaseEvidence:
    rows = []
    points = []
    for index in range(24):
        timestamp = f"2016-01-11T{index:02d}:00:00"
        base = [10.0, 20.0, 12.0, 22.0][index % 4]
        value = base + index * 0.4
        if index == 12:
            value += 35.0
        aux = value * 1.5 + 3.0
        segment = "A" if index % 2 == 0 else "B"
        row = {
            "timestamp": timestamp,
            "value": value,
            "aux": aux,
            "segment": segment,
        }
        rows.append(row)
        points.append({"timestamp": timestamp, "value": value})

    return DatabaseEvidence(
        evidence_id="evi_rich_series",
        result_type="timeseries",
        database="test-db",
        query_language="unit",
        query="unit:test",
        summary="synthetic",
        data={
            "points": points,
            "rows": rows,
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        columns=["timestamp", "value", "aux", "segment"],
        metadata={},
        diagnostics={},
    )


def test_build_insight_result_supports_eleven_fact_families():
    evidence = _rich_timeseries_evidence()
    requested = [
        "aggregation",
        "extreme",
        "trend",
        "difference",
        "rank",
        "distribution",
        "association",
        "outlier",
        "seasonality",
        "proportion",
        "categorization",
    ]

    result = build_insight_result(evidence, requested, "full fact coverage")

    assert result.requested_fact_types == requested
    verified_types = {fact.fact_type for fact in result.verified_facts}
    assert verified_types == set(requested)
    assert not result.rejected_facts
    assert result.diagnostics["point_count"] == 24
    assert result.diagnostics["multi_series_evidence_detected"] is True


def test_request_state_infers_richer_requested_fact_types():
    request = ChatRequest(
        message="分析趋势、极值、分布、异常、周期、排名、占比和相关性，并比较变化幅度。",
        database_context=None,
        selected_database=None,
        selected_database_type=None,
        time_range=None,
        constraints={},
        history=[],
        conversation_id=None,
        stream=False,
    )

    state = build_request_state(request, get_settings())

    inferred = set(state.requested_fact_types)
    assert {
        "trend",
        "extreme",
        "distribution",
        "outlier",
        "seasonality",
        "rank",
        "proportion",
        "association",
        "difference",
    }.issubset(inferred)
