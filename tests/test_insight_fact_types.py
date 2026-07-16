from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_request_state_does_not_infer_outer_intent_requirements():
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

    assert state.intent_profile == {}
    assert state.requested_fact_types == []
    assert state.answer_requirements == ["conclusion"]


def test_seasonality_uses_daily_weekly_time_profiles_with_outlier_context():
    rows = []
    points = []
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for index in range(30 * 24):
        timestamp = start + timedelta(hours=index)
        value = 100.0 + index * 0.2
        if index in {0, 1}:
            value = 1_000_000.0
        row = {"timestamp": timestamp.isoformat(), "value": value}
        rows.append(row)
        points.append(row.copy())
    evidence = DatabaseEvidence(
        evidence_id="evi_no_strong_calendar_cycle",
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
        },
        columns=["timestamp", "value"],
        metadata={},
        diagnostics={},
    )

    result = build_insight_result(evidence, ["seasonality"], "check daily and weekly repetition")
    fact = result.verified_facts[0]

    assert fact.verification_rule == "deterministic_time_aware_daily_weekly_periodicity"
    assert fact.evidence["outlier_count"] == 2
    assert fact.evidence["daily"]["bucket_count"] == 24
    assert fact.evidence["weekly"]["bucket_count"] == 7
    assert fact.evidence["has_seasonality"] is False
    assert "每天或每周" in fact.statement


def test_proportion_uses_explicit_threshold_from_focus():
    evidence = _rich_timeseries_evidence()

    result = build_insight_result(evidence, ["proportion"], "统计 value 高于 20 的记录比例")
    fact = result.verified_facts[0]

    assert fact.fact_type == "proportion"
    assert fact.evidence["threshold"] == 20
    assert fact.evidence["operator"] == ">"
    assert fact.evidence["count"] == 14
    assert fact.evidence["total"] == 24
    assert "高于 20.00" in fact.statement


def test_categorization_builds_low_middle_high_quartile_buckets_with_outlier_context():
    rows = []
    points = []
    for index, value in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 1000]):
        row = {"timestamp": f"2023-01-01T00:{index:02d}:00Z", "value": float(value)}
        rows.append(row)
        points.append(row.copy())
    evidence = DatabaseEvidence(
        evidence_id="evi_bucket",
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
        },
        columns=["timestamp", "value"],
        metadata={},
        diagnostics={},
    )

    result = build_insight_result(evidence, ["categorization"], "分成高位、低位和中间区间")
    fact = result.verified_facts[0]

    assert fact.fact_type == "categorization"
    assert fact.verification_rule == "deterministic_quartile_bucket_from_points"
    assert fact.evidence["outlier_count"] == 1
    assert fact.evidence["low_count"] == 2
    assert fact.evidence["middle_count"] == 4
    assert fact.evidence["high_count"] == 2
    assert "低位" in fact.statement
    assert "中间区间" in fact.statement
    assert "高位" in fact.statement
