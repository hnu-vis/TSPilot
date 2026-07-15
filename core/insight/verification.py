"""Verify completed facts against evidence."""
from __future__ import annotations

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
        period, autocorrelation = detect_period(values)
        if period is None:
            statement = f"{value_field} 在该时间范围内没有明显周期性。"
            strength = 0.0
        else:
            strength = seasonal_strength(values, period)
            statement = f"{value_field} 呈现约每 {period} 个点重复一次的周期性，强度约为 {strength:.2f}。"
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="seasonality",
            statement=statement,
            confidence=max(0.45, min(0.93, max(autocorrelation, strength if period is not None else 0.45))),
            evidence={
                "evidence_id": evidence.evidence_id,
                "period": period,
                "autocorrelation": round(autocorrelation, 4),
                "strength": round(strength, 4),
                "has_seasonality": period is not None and strength > 0.1,
            },
            verification_rule="deterministic_autocorrelation_periodicity",
        )

    if fact_type == "proportion":
        avg_value = statistics.fmean(values)
        selected = [value for value in values if value >= avg_value]
        proportion = len(selected) / len(values)
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="proportion",
            statement=f"{len(selected)}/{len(values)} 个点（{proportion:.1%}）的 {value_field} 高于或等于该区间平均值 {avg_value:.2f}。",
            confidence=min(0.9, 0.4 + proportion),
            evidence={
                "evidence_id": evidence.evidence_id,
                "threshold": round(avg_value, 2),
                "count": len(selected),
                "total": len(values),
                "proportion": round(proportion, 4),
            },
            verification_rule="deterministic_threshold_share_from_points",
        )

    if fact_type == "categorization":
        avg_value = statistics.fmean(values)
        high_count = sum(1 for value in values if value >= avg_value)
        low_count = len(values) - high_count
        dominant = "high" if high_count >= low_count else "low"
        statement = (
            f"按是否高于均值 {avg_value:.2f} 分类，{value_field} 中高位点有 {high_count} 个，"
            f"低位点有 {low_count} 个，整体以{'高位' if dominant == 'high' else '低位'}为主。"
        )
        return VerifiedFact(
            fact_id=completed.fact_id,
            fact_type="categorization",
            statement=statement,
            confidence=0.78,
            evidence={
                "evidence_id": evidence.evidence_id,
                "threshold": round(avg_value, 2),
                "high_count": high_count,
                "low_count": low_count,
                "dominant_category": dominant,
            },
            verification_rule="deterministic_bucket_from_mean",
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
