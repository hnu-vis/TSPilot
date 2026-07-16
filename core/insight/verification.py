"""Verify completed facts against evidence."""
from __future__ import annotations

from datetime import datetime
import math
import re
import statistics

from core.insight.fact_engine import (
    detect_period,
    evidence_rows,
    numeric_columns,
    pearson,
    quantile,
    seasonal_strength,
    to_number,
)
from schemas.database import DatabaseEvidence
from schemas.insight import CompletedFact, RejectedFact, VerifiedFact


def verify_completed_facts(
    completed_facts: list[CompletedFact],
    evidence: DatabaseEvidence,
) -> tuple[list[VerifiedFact], list[RejectedFact]]:
    rows, columns, time_field, value_field = evidence_rows(evidence)
    numeric = numeric_columns(rows, columns)
    if evidence.result_type != "timeseries" or not rows or value_field not in numeric:
        return [], [
            RejectedFact(
                fact_id=completed.fact_id,
                fact_type=completed.fact_type,
                statement=completed.statement,
                reason="insight requires time-series evidence with at least two points",
                evidence={"evidence_id": evidence.evidence_id},
                verification_rule="timeseries_required",
            )
            for completed in completed_facts
        ]

    indexed = numeric[value_field]
    values = [value for _, value in indexed]
    first_idx, first = indexed[0]
    last_idx, last = indexed[-1]
    first_row = rows[first_idx]
    last_row = rows[last_idx]

    verified: list[VerifiedFact] = []
    rejected: list[RejectedFact] = []
    for completed in completed_facts:
        outcome = _verify_one(
            completed,
            evidence,
            rows,
            time_field,
            value_field,
            indexed,
            values,
            first_idx,
            first,
            first_row,
            last_idx,
            last,
            last_row,
            numeric,
            columns,
        )
        if isinstance(outcome, VerifiedFact):
            verified.append(outcome)
        else:
            rejected.append(outcome)
    return verified, rejected


def _verify_one(
    completed: CompletedFact,
    evidence: DatabaseEvidence,
    rows: list[dict],
    time_field: str,
    value_field: str,
    indexed: list[tuple[int, float]],
    values: list[float],
    first_idx: int,
    first: float,
    first_row: dict,
    last_idx: int,
    last: float,
    last_row: dict,
    numeric: dict[str, list[tuple[int, float]]],
    columns: list[str],
) -> VerifiedFact | RejectedFact:
    fact_type = completed.fact_type
    if fact_type == "aggregation":
        avg_value = statistics.fmean(values)
        total = sum(values)
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="aggregation",
            statement=f"{value_field} 在该时间范围内共有 {len(values)} 个点，平均值为 {avg_value:.2f}，总和为 {total:.2f}。",
            confidence=0.88,
            evidence={
                "evidence_id": evidence.evidence_id,
                "count": len(values),
                "avg": round(avg_value, 2),
                "sum": round(total, 2),
            },
            verification_rule="deterministic_aggregate_from_points",
        )

    if fact_type == "extreme":
        max_idx, max_value = max(indexed, key=lambda item: item[1])
        min_idx, min_value = min(indexed, key=lambda item: item[1])
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="extreme",
            statement=(
                f"{value_field} 的最高值为 {max_value:.2f}（{rows[max_idx].get(time_field)}），"
                f"最低值为 {min_value:.2f}（{rows[min_idx].get(time_field)}）。"
            ),
            confidence=0.93,
            evidence={
                "evidence_id": evidence.evidence_id,
                "max_point": {"timestamp": rows[max_idx].get(time_field), "value": max_value},
                "min_point": {"timestamp": rows[min_idx].get(time_field), "value": min_value},
            },
            verification_rule="deterministic_min_max_from_points",
        )

    if fact_type == "trend":
        xs = list(range(len(values)))
        slope = _linear_slope(xs, values)
        delta = last - first
        percent = (delta / abs(first) * 100.0) if first else 0.0
        trend = "上升" if delta > 0 else "下降" if delta < 0 else "保持稳定"
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="trend",
            statement=f"{value_field} 在所选时间范围内整体{trend}，变化幅度约为 {percent:.2f}%。",
            confidence=0.95,
            evidence={
                "evidence_id": evidence.evidence_id,
                "start_value": first,
                "end_value": last,
                "change_percent": round(percent, 2),
                "slope": slope,
            },
            verification_rule="deterministic_delta_from_points",
        )

    if fact_type == "difference":
        delta = last - first
        percent = (delta / abs(first) * 100.0) if first else None
        direction = "增加" if delta > 0 else "减少" if delta < 0 else "保持不变"
        percent_text = f"（{percent:.2f}%）" if percent is not None else ""
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="difference",
            statement=(
                f"{value_field} 从 {first_row.get(time_field)} 的 {first:.2f} 到 "
                f"{last_row.get(time_field)} 的 {last:.2f}，净变化 {delta:.2f}{percent_text}，整体{direction}。"
            ),
            confidence=0.92,
            evidence={
                "evidence_id": evidence.evidence_id,
                "start_timestamp": first_row.get(time_field),
                "end_timestamp": last_row.get(time_field),
                "start_value": first,
                "end_value": last,
                "difference": round(delta, 2),
                "relative_difference": round(percent, 2) if percent is not None else None,
            },
            verification_rule="deterministic_difference_from_points",
        )

    if fact_type == "rank":
        ranked = sorted(
            ({"timestamp": rows[idx].get(time_field), "value": value} for idx, value in indexed),
            key=lambda item: item["value"],
            reverse=True,
        )
        top = ranked[0]
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="rank",
            statement=f"{value_field} 最高的时间点是 {top['timestamp']}，数值为 {top['value']:.2f}。",
            confidence=0.84,
            evidence={
                "evidence_id": evidence.evidence_id,
                "top_point": top,
                "ranked": ranked[:5],
            },
            verification_rule="deterministic_rank_from_points",
        )

    if fact_type == "distribution":
        sorted_values = sorted(values)
        q1 = quantile(sorted_values, 0.25)
        median = quantile(sorted_values, 0.5)
        q3 = quantile(sorted_values, 0.75)
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="distribution",
            statement=f"{value_field} 的中位数为 {median:.2f}，中间 50% 主要分布在 {q1:.2f} 到 {q3:.2f}。",
            confidence=0.82,
            evidence={
                "evidence_id": evidence.evidence_id,
                "q1": round(q1, 2),
                "median": round(median, 2),
                "q3": round(q3, 2),
            },
            verification_rule="deterministic_quantiles_from_points",
        )

    if fact_type == "association":
        numeric_columns = [column for column in numeric if column != value_field]
        if not numeric_columns:
            return _reject(completed, evidence, "association requires at least two numeric measures", "association_requires_multiple_measures")
        other = numeric_columns[0]
        pairs = []
        for row in rows:
            left = to_number(row.get(value_field))
            right = to_number(row.get(other))
            if left is not None and right is not None:
                pairs.append((left, right))
        if len(pairs) < 4:
            return _reject(completed, evidence, "association requires at least four aligned numeric pairs", "association_requires_pairs")
        corr = pearson([left for left, _ in pairs], [right for _, right in pairs])
        if corr is None:
            return _reject(completed, evidence, "association could not be computed", "association_not_computable")
        relation = "正相关" if corr > 0 else "负相关" if corr < 0 else "几乎无线性相关"
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="association",
            statement=f"{value_field} 与 {other} 呈{relation}，相关系数 r={corr:.2f}。",
            confidence=max(0.45, min(0.94, abs(corr))),
            evidence={
                "evidence_id": evidence.evidence_id,
                "left": value_field,
                "right": other,
                "correlation": round(corr, 4),
                "pair_count": len(pairs),
            },
            verification_rule="deterministic_pearson_from_rows",
        )

    if fact_type == "outlier":
        sorted_values = sorted(values)
        q1 = quantile(sorted_values, 0.25)
        q3 = quantile(sorted_values, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [
            {"timestamp": rows[idx].get(time_field), "value": value}
            for idx, value in indexed
            if value < lower or value > upper
        ]
        if outliers:
            preview = ", ".join(
                f"{point['timestamp']}={point['value']:.2f}"
                for point in outliers[:3]
            )
            statement = f"{value_field} 检测到 {len(outliers)} 个离群点，典型离群点包括 {preview}。"
        else:
            statement = f"{value_field} 未检测到显著离群点。"
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="outlier",
            statement=statement,
            confidence=0.86 if outliers else 0.72,
            evidence={
                "evidence_id": evidence.evidence_id,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": len(outliers),
                "outliers": outliers[:10],
            },
            verification_rule="deterministic_iqr_outlier_from_points",
        )

    if fact_type == "seasonality":
        seasonality = _time_aware_seasonality(rows, time_field, value_field, indexed, values)
        if seasonality is None:
            period, autocorrelation = detect_period(values)
            if period is None:
                statement = f"{value_field} 在该时间范围内没有明显周期性。"
                strength = 0.0
            else:
                strength = seasonal_strength(values, period)
                statement = f"{value_field} 呈现约每 {period} 个点重复一次的周期性，强度约为 {strength:.2f}。"
            evidence_payload = {
                "evidence_id": evidence.evidence_id,
                "period": period,
                "autocorrelation": round(autocorrelation, 4),
                "strength": round(strength, 4),
                "has_seasonality": period is not None and strength > 0.1,
            }
            confidence = max(0.45, min(0.93, max(autocorrelation, strength if period is not None else 0.45)))
            rule = "deterministic_autocorrelation_periodicity"
        else:
            has_daily = seasonality["daily"]["has_periodicity"]
            has_weekly = seasonality["weekly"]["has_periodicity"]
            if has_daily or has_weekly:
                parts = []
                if has_daily:
                    parts.append(
                        f"日内周期较明显（小时均值相对振幅 {seasonality['daily']['relative_amplitude']:.2%}，"
                        f"强度 {seasonality['daily']['strength']:.4f}）"
                    )
                if has_weekly:
                    parts.append(
                        f"周内周期较明显（星期均值相对振幅 {seasonality['weekly']['relative_amplitude']:.2%}，"
                        f"强度 {seasonality['weekly']['strength']:.4f}）"
                    )
                statement = f"{value_field} 在该时间范围内" + "，".join(parts) + "。"
            else:
                statement = (
                    f"{value_field} 在该时间范围内没有明显每天或每周重复的周期性波动；"
                    f"日内相对振幅 {seasonality['daily']['relative_amplitude']:.2%}、强度 "
                    f"{seasonality['daily']['strength']:.4f}，周内相对振幅 "
                    f"{seasonality['weekly']['relative_amplitude']:.2%}、强度 "
                    f"{seasonality['weekly']['strength']:.4f}。"
                )
            evidence_payload = {
                "evidence_id": evidence.evidence_id,
                **seasonality,
                "has_seasonality": has_daily or has_weekly,
            }
            confidence = 0.86 if has_daily or has_weekly else 0.78
            rule = "deterministic_time_aware_daily_weekly_periodicity"
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="seasonality",
            statement=statement,
            confidence=confidence,
            evidence=evidence_payload,
            verification_rule=rule,
        )

    if fact_type == "proportion":
        threshold, operator, threshold_source = _threshold_from_focus(completed.focus, completed.statement)
        if threshold is None:
            threshold = statistics.fmean(values)
            operator = ">="
            threshold_source = "mean_fallback"
        if operator == ">":
            selected = [value for value in values if value > threshold]
            comparator_text = "高于"
        elif operator == "<":
            selected = [value for value in values if value < threshold]
            comparator_text = "低于"
        elif operator == "<=":
            selected = [value for value in values if value <= threshold]
            comparator_text = "低于或等于"
        else:
            selected = [value for value in values if value >= threshold]
            comparator_text = "高于或等于"
        proportion = len(selected) / len(values)
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="proportion",
            statement=f"{len(selected)}/{len(values)} 个点（{proportion:.1%}）的 {value_field} {comparator_text} {threshold:.2f}。",
            confidence=min(0.9, 0.4 + proportion),
            evidence={
                "evidence_id": evidence.evidence_id,
                "threshold": round(threshold, 2),
                "operator": operator,
                "threshold_source": threshold_source,
                "count": len(selected),
                "total": len(values),
                "proportion": round(proportion, 4),
            },
            verification_rule="deterministic_threshold_share_from_points",
        )

    if fact_type == "categorization":
        clean_values, lower_bound, upper_bound, outlier_count = _iqr_clean_numeric_values(values)
        sorted_values = sorted(clean_values or values)
        q1 = quantile(sorted_values, 0.25)
        q3 = quantile(sorted_values, 0.75)
        low_count = sum(1 for value in sorted_values if value <= q1)
        high_count = sum(1 for value in sorted_values if value >= q3)
        mid_count = len(sorted_values) - low_count - high_count
        dominant_counts = {"low": low_count, "middle": mid_count, "high": high_count}
        dominant = max(dominant_counts, key=dominant_counts.get)
        statement = (
            f"按四分位阈值分类，{value_field} 低位为 <= {q1:.2f}，中间区间为 "
            f"{q1:.2f} 到 {q3:.2f}，高位为 >= {q3:.2f}；有效样本中低位 {low_count} 个，"
            f"中间区间 {mid_count} 个，高位 {high_count} 个。"
        )
        if outlier_count:
            statement += f" 另有 {outlier_count} 个 IQR 离群点未用于阈值计算。"
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="categorization",
            statement=statement,
            confidence=0.78,
            evidence={
                "evidence_id": evidence.evidence_id,
                "method": "iqr_cleaned_quartile_buckets",
                "low_max": round(q1, 2),
                "middle_min": round(q1, 2),
                "middle_max": round(q3, 2),
                "high_min": round(q3, 2),
                "high_count": high_count,
                "middle_count": mid_count,
                "low_count": low_count,
                "dominant_category": dominant,
                "effective_count": len(sorted_values),
                "outlier_count": outlier_count,
                "outlier_bounds": {"lower": round(lower_bound, 4), "upper": round(upper_bound, 4)},
            },
            verification_rule="deterministic_quartile_bucket_from_points",
        )

    return _reject(completed, evidence, "unsupported fact type", "fact_type_support")


def _reject(completed: CompletedFact, evidence: DatabaseEvidence, reason: str, rule: str) -> RejectedFact:
    return RejectedFact(
        fact_id=completed.fact_id,
        fact_type=completed.fact_type,
        statement=completed.statement,
        reason=reason,
        evidence={"evidence_id": evidence.evidence_id},
        verification_rule=rule,
    )


def _linear_slope(xs: list[int], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return numerator / denominator if denominator else 0.0


def _time_aware_seasonality(
    rows: list[dict],
    time_field: str,
    value_field: str,
    indexed: list[tuple[int, float]],
    values: list[float],
) -> dict | None:
    timed_values: list[tuple[datetime, float]] = []
    for index, value in indexed:
        timestamp = _parse_timestamp(rows[index].get(time_field))
        if timestamp is not None:
            timed_values.append((timestamp, value))
    if len(timed_values) < 24:
        return None

    clean_values, lower, upper, outlier_count = _iqr_clean_values(timed_values)
    if len(clean_values) < 24:
        return None
    ordered = sorted(clean_values, key=lambda item: item[0])
    span_days = max((ordered[-1][0] - ordered[0][0]).total_seconds() / 86400.0, 0.0)
    if span_days < 2:
        return None

    value_only = [value for _, value in ordered]
    overall_mean = statistics.fmean(value_only)
    total_variance = _population_variance(value_only)
    daily = _periodic_profile(
        ordered,
        key_fn=lambda timestamp: timestamp.hour,
        expected_buckets=24,
        overall_mean=overall_mean,
        total_variance=total_variance,
        min_bucket_count=2,
        label="hour",
    )
    weekly = _periodic_profile(
        ordered,
        key_fn=lambda timestamp: timestamp.weekday(),
        expected_buckets=7,
        overall_mean=overall_mean,
        total_variance=total_variance,
        min_bucket_count=2,
        label="weekday",
    )
    return {
        "method": "group_by_timestamp_profile",
        "period": None,
        "autocorrelation": None,
        "strength": max(daily["strength"], weekly["strength"]),
        "daily": daily,
        "weekly": weekly,
        "sample_count": len(timed_values),
        "clean_sample_count": len(ordered),
        "outlier_count": outlier_count,
        "outlier_bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
    }


def _periodic_profile(
    timed_values: list[tuple[datetime, float]],
    *,
    key_fn,
    expected_buckets: int,
    overall_mean: float,
    total_variance: float,
    min_bucket_count: int,
    label: str,
) -> dict:
    groups: dict[int, list[float]] = {}
    for timestamp, value in timed_values:
        groups.setdefault(int(key_fn(timestamp)), []).append(value)
    bucket_means = {bucket: statistics.fmean(items) for bucket, items in groups.items() if len(items) >= min_bucket_count}
    bucket_counts = {bucket: len(items) for bucket, items in groups.items()}
    if not bucket_means:
        return {
            "label": label,
            "bucket_count": 0,
            "strength": 0.0,
            "amplitude": 0.0,
            "relative_amplitude": 0.0,
            "has_periodicity": False,
            "bucket_means": {},
            "bucket_counts": bucket_counts,
        }
    means = list(bucket_means.values())
    amplitude = max(means) - min(means)
    relative_amplitude = amplitude / abs(overall_mean) if overall_mean else 0.0
    strength = _population_variance(means) / total_variance if total_variance > 0 else 0.0
    has_enough_coverage = len(bucket_means) >= max(2, math.ceil(expected_buckets * 0.8))
    has_periodicity = has_enough_coverage and strength >= 0.08 and relative_amplitude >= 0.05
    return {
        "label": label,
        "bucket_count": len(bucket_means),
        "strength": round(strength, 6),
        "amplitude": round(amplitude, 4),
        "relative_amplitude": round(relative_amplitude, 6),
        "has_periodicity": has_periodicity,
        "bucket_means": {str(bucket): round(mean, 4) for bucket, mean in sorted(bucket_means.items())},
        "bucket_counts": {str(bucket): count for bucket, count in sorted(bucket_counts.items())},
    }


def _iqr_clean_values(timed_values: list[tuple[datetime, float]]) -> tuple[list[tuple[datetime, float]], float, float, int]:
    values = sorted(value for _, value in timed_values)
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return timed_values, q1, q3, 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    cleaned = [(timestamp, value) for timestamp, value in timed_values if lower <= value <= upper]
    return cleaned, lower, upper, len(timed_values) - len(cleaned)


def _iqr_clean_numeric_values(values: list[float]) -> tuple[list[float], float, float, int]:
    sorted_values = sorted(values)
    q1 = quantile(sorted_values, 0.25)
    q3 = quantile(sorted_values, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return values, q1, q3, 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    cleaned = [value for value in values if lower <= value <= upper]
    return cleaned, lower, upper, len(values) - len(cleaned)


def _threshold_from_focus(focus: str | None, statement: str | None) -> tuple[float | None, str, str | None]:
    text = f"{focus or ''} {statement or ''}"
    patterns = [
        (r"(?:高于|大于|超过|above|greater than|over|>)\s*([0-9]+(?:\.[0-9]+)?)", ">", "focus_threshold"),
        (r"(?:不低于|至少|>=)\s*([0-9]+(?:\.[0-9]+)?)", ">=", "focus_threshold"),
        (r"(?:低于|小于|below|less than|under|<)\s*([0-9]+(?:\.[0-9]+)?)", "<", "focus_threshold"),
        (r"(?:不高于|至多|<=)\s*([0-9]+(?:\.[0-9]+)?)", "<=", "focus_threshold"),
    ]
    for pattern, operator, source in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), operator, source
    return None, ">=", None


def _population_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _parse_timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
