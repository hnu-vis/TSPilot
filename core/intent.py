"""Structured request intent profile helpers."""
from __future__ import annotations

from typing import Any

from core.insight.fact_engine import normalize_requested_fact_types


AGGREGATION_FACT_TYPES = {"aggregation", "extreme", "rank", "distribution", "proportion", "categorization"}


def build_intent_profile_fallback(message: str) -> dict[str, Any]:
    """Build a conservative structured intent profile without backend-specific query rules."""

    requested_fact_types = _infer_requested_fact_types(message)
    has_exact_stat = any(item in AGGREGATION_FACT_TYPES for item in requested_fact_types)
    has_outlier = "outlier" in requested_fact_types
    analysis_kind = "statistical_summary" if has_exact_stat else "anomaly_detection" if has_outlier else "timeseries_analysis"
    requested_metrics = _requested_metrics_from_facts(requested_fact_types)
    data_policy = {
        "preserve_raw_values": has_exact_stat,
        "filter_outliers": False if has_exact_stat else None,
    }
    required_outputs = _answer_requirements_from_facts(requested_fact_types)
    return {
        "source": "fallback",
        "primary_goal": message,
        "analysis_kind": analysis_kind,
        "requested_fact_types": requested_fact_types,
        "requested_metrics": requested_metrics,
        "data_policy": data_policy,
        "required_outputs": required_outputs,
        "needs_plan": _needs_plan(requested_fact_types, message),
    }


def normalize_intent_profile(raw: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return fallback
    requested_fact_types = raw.get("requested_fact_types")
    if not isinstance(requested_fact_types, list):
        requested_fact_types = fallback.get("requested_fact_types", [])
    requested_fact_types = normalize_requested_fact_types([str(item) for item in requested_fact_types])
    data_policy = raw.get("data_policy") if isinstance(raw.get("data_policy"), dict) else {}
    required_outputs = raw.get("required_outputs")
    if not isinstance(required_outputs, list):
        required_outputs = _answer_requirements_from_facts(requested_fact_types)
    requested_metrics = raw.get("requested_metrics")
    if not isinstance(requested_metrics, list):
        requested_metrics = _requested_metrics_from_facts(requested_fact_types)
    profile = {
        "source": str(raw.get("source") or "llm"),
        "primary_goal": str(raw.get("primary_goal") or fallback.get("primary_goal") or ""),
        "analysis_kind": str(raw.get("analysis_kind") or fallback.get("analysis_kind") or "timeseries_analysis"),
        "requested_fact_types": requested_fact_types,
        "requested_metrics": [str(item) for item in requested_metrics],
        "data_policy": {
            "preserve_raw_values": data_policy.get("preserve_raw_values", fallback.get("data_policy", {}).get("preserve_raw_values")),
            "filter_outliers": data_policy.get("filter_outliers", fallback.get("data_policy", {}).get("filter_outliers")),
        },
        "required_outputs": [str(item) for item in required_outputs],
        "needs_plan": bool(raw.get("needs_plan", fallback.get("needs_plan", False))),
    }
    return profile


def apply_intent_profile_to_state(request_state, profile: dict[str, Any]) -> None:
    request_state.intent_profile = profile
    request_state.requested_fact_types = list(profile.get("requested_fact_types") or request_state.requested_fact_types)
    if request_state.database_context is not None:
        requirements = ["conclusion", *[item for item in profile.get("required_outputs", []) if item != "conclusion"]]
        request_state.answer_requirements = _dedupe(requirements)
        request_state.answer_coverage = {requirement: request_state.answer_coverage.get(requirement, False) for requirement in request_state.answer_requirements}


def _infer_requested_fact_types(message: str) -> list[str]:
    normalized = message.lower()
    requested: list[str] = []
    keyword_map = [
        ("aggregation", ("平均", "均值", "总和", "sum", "count", "aggregate", "aggregation", "统计")),
        ("extreme", ("最高", "最低", "最大", "最小", "peak", "trough", "extreme", "extrema", "极值")),
        ("trend", ("趋势", "走势", "trend", "movement", "upward", "downward")),
        ("difference", ("差值", "差异", "变化", "变化幅度", "change", "difference", "delta", "compare", "comparison")),
        ("rank", ("排名", "排行", "top", "bottom", "rank")),
        ("distribution", ("分布", "中位数", "四分位", "distribution", "median", "quartile")),
        ("association", ("相关", "关联", "correlation", "association", "同步")),
        ("outlier", ("异常", "离群", "尖峰", "异常点", "outlier", "anomaly", "spike", "dip")),
        ("seasonality", ("周期", "季节性", "重复", "seasonality", "seasonal", "periodic", "cycle")),
        ("proportion", ("占比", "比例", "percent", "percentage", "ratio", "share")),
        ("categorization", ("分类", "分桶", "分成", "高位", "低位", "中间区间", "bucket", "categorization", "category")),
    ]
    for fact_type, keywords in keyword_map:
        if any(keyword in normalized for keyword in keywords):
            requested.append(fact_type)
    if not requested:
        requested = ["trend", "difference", "extreme"]
    return normalize_requested_fact_types(requested)


def _requested_metrics_from_facts(fact_types: list[str]) -> list[str]:
    metrics = []
    if "extreme" in fact_types:
        metrics.append("max_or_min")
    if "aggregation" in fact_types:
        metrics.append("aggregate")
    if "rank" in fact_types:
        metrics.append("rank")
    if "distribution" in fact_types:
        metrics.append("distribution")
    return metrics


def _answer_requirements_from_facts(fact_types: list[str]) -> list[str]:
    requirements = ["conclusion"]
    for fact_type in fact_types:
        if fact_type == "outlier":
            requirements.append("anomaly")
        elif fact_type in {"aggregation", "extreme", "rank", "distribution", "proportion", "categorization"}:
            requirements.append("analysis")
        else:
            requirements.append(fact_type)
    return _dedupe(requirements)


def _needs_plan(fact_types: list[str], message: str) -> bool:
    normalized = message.lower()
    return (
        len(fact_types) > 1
        or any(token in normalized for token in ("步骤", "过程", "规划", "plan", "todo", "执行过程"))
        or any(fact in fact_types for fact in ("seasonality", "association"))
    )


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
