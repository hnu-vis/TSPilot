"""Runtime registration for KeyInsights produced by tools."""
from __future__ import annotations

import json
from hashlib import sha1
from typing import Any

from schemas.key_insight import (
    KeyInsight,
    KeyInsightRequest,
    InsightCoverage,
    InsightEvent,
    InsightEvidenceRef,
    InsightItem,
    normalize_insight_key,
)


def register_key_insights_from_payload(request_state, tool_name: str, full_payload: dict) -> InsightCoverage:
    """Register prompt-safe insights from a completed tool payload.

    Coverage is diagnostic only. Missing insights should guide the next ReAct turn,
    not force the runtime to fail.
    """

    action_input = _latest_action_input(request_state, tool_name)
    requests = _insight_requests(action_input)
    produced: list[KeyInsight] = []
    rejected: list[KeyInsight] = []

    for raw_insight in full_payload.get("produced_insights") or []:
        insight = _bind_insight_contract(_validate_insight(raw_insight, default_method=tool_name), requests)
        if insight is not None:
            produced.append(insight)
    for raw_insight in full_payload.get("rejected_insights") or []:
        insight = _bind_insight_contract(
            _validate_insight(raw_insight, default_method=tool_name, default_status="rejected"),
            requests,
        )
        if insight is not None:
            rejected.append(insight)

    if tool_name == "sql_query":
        produced.extend(_insights_from_database_evidence(full_payload, requests))
    elif tool_name == "code_interpreter":
        # Code Interpreter outputs are already semantically bound by the LLM
        # binder. Re-deriving them from generic metrics would restore the old
        # computation/semantics coupling.
        pass
    # Forecast and anomaly outputs remain analysis artifacts. They can be
    # referenced by answers and visualizations, but are not Key Insights by default.

    produced = _verify_insight_dependencies(
        _dedupe_insights([insight for insight in produced if _insight_has_evidence_or_unavailable(insight)]),
        request_state.insight_set.insights,
        request_state=request_state,
    )
    rejected = _dedupe_insights(rejected)
    event_coverage = _coverage_for(requests, produced, rejected)
    coverage = _merge_insight_state(request_state, tool_name, requests, produced, rejected, event_coverage)
    _attach_insights_to_artifact(request_state, tool_name, full_payload, produced, rejected, event_coverage)
    return coverage


def key_insight_prompt_view(request_state) -> dict:
    insights = list(getattr(getattr(request_state, "insight_set", None), "insights", []) or [])
    requests = list(getattr(getattr(request_state, "insight_set", None), "requests", []) or [])
    coverage = getattr(getattr(request_state, "insight_set", None), "coverage", None)
    recent = insights[-12:]
    return {
        "summary": {
            "verified": [insight.name for insight in insights if insight.status == "verified"][-12:],
            "unavailable": [insight.name for insight in insights if insight.status == "unavailable"][-8:],
            "rejected": [insight.name for insight in insights if insight.status == "rejected"][-8:],
            "missing": list(getattr(coverage, "missing", []) or [])[-12:],
        },
        "plan": [
            {
                "insight_key": request.insight_key,
                "name": request.name,
                "insight_type": request.insight_type,
                "derived_from": request.derived_from,
            }
            for request in requests[-12:]
        ],
        "recent_insights": [
            _drop_empty({
                "insight_id": insight.insight_id,
                "insight_key": insight.insight_key,
                "name": insight.name,
                "insight_type": insight.insight_type,
                "status": insight.status,
                "statement": insight.statement,
                "value": _prompt_fact_value(insight.value),
                "items": [
                    _prompt_fact_value(item.model_dump(mode="json", exclude_none=True))
                    for item in insight.items[:12]
                ],
                "unit": insight.unit,
                "dimensions": _prompt_fact_value(insight.dimensions),
                "evidence_refs": [_canonical_prompt_ref(ref) for ref in insight.evidence_refs[:4]],
                "derived_from": insight.derived_from,
                "unavailable_reason": insight.unavailable_reason,
            })
            for insight in recent
        ],
    }


def _canonical_prompt_ref(ref: InsightEvidenceRef) -> str:
    source_id = str(ref.source_id or "").strip()
    if ":" in source_id:
        return source_id
    prefix = {
        "query": "evidence",
        "evidence": "evidence",
        "analysis": "analysis",
        "derived_evidence": "derived_evidence",
        "forecast": "forecast",
        "anomaly": "anomaly",
        "insight": "insight",
    }.get(str(ref.source_type or "").strip(), str(ref.source_type or "evidence").strip())
    return f"{prefix}:{source_id}"


def _prompt_fact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "[depth limit]"
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + f"... [truncated {len(value) - 500} chars]"
    if isinstance(value, list):
        selected = value if len(value) <= 12 else [*value[:6], *value[-6:]]
        return [_prompt_fact_value(item, depth=depth + 1) for item in selected]
    if isinstance(value, dict):
        return {
            str(key): _prompt_fact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:20]
        }
    return value


def _drop_empty(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _latest_action_input(request_state, tool_name: str) -> dict:
    for call in reversed(getattr(request_state, "tool_history", []) or []):
        if call.tool_name == tool_name:
            return dict(call.tool_input or {})
    return {}


def _insight_requests(action_input: dict) -> list[KeyInsightRequest]:
    items = action_input.get("insight_requests") or []
    requests: list[KeyInsightRequest] = []
    for item in items:
        if isinstance(item, dict):
            try:
                requests.append(KeyInsightRequest.model_validate(item))
            except Exception:
                continue
    return requests


def _validate_insight(raw_insight: Any, *, default_method: str, default_status: str = "verified") -> KeyInsight | None:
    if not isinstance(raw_insight, dict):
        return None
    payload = dict(raw_insight)
    payload.setdefault("method", default_method)
    payload.setdefault("status", default_status)
    payload.setdefault("name", payload.get("insight_id") or payload.get("insight_type") or "insight")
    payload.setdefault("insight_type", "custom")
    payload.setdefault("statement", str(payload.get("summary") or payload.get("name") or "Key insight."))
    payload.setdefault("insight_id", _insight_id(payload.get("method"), payload.get("name"), payload.get("value")))
    try:
        return KeyInsight.model_validate(payload)
    except Exception:
        return None


def _bind_insight_contract(
    insight: KeyInsight | None,
    requests: list[KeyInsightRequest],
    *,
    preserve_dependencies: bool = False,
) -> KeyInsight | None:
    if insight is None or not requests:
        return insight
    aliases = {insight.insight_key, normalize_insight_key(insight.name)}
    request = next(
        (
            item
            for item in requests
            if item.insight_key in aliases or normalize_insight_key(item.name) in aliases
        ),
        None,
    )
    if request is None:
        return None
    return insight.model_copy(
        update={
            "insight_key": request.insight_key,
            "name": request.name,
            "insight_type": request.insight_type,
            "subject": insight.subject or request.subject,
            "time_range": insight.time_range or request.time_range,
            "dimensions": insight.dimensions or request.dimensions,
            "derived_from": insight.derived_from if preserve_dependencies else insight.derived_from or request.derived_from,
            "value_shape": insight.value_shape or request.result_shape,
            "semantic_class": insight.semantic_class or request.semantic_class,
            "derivation": insight.derivation or request.derivation,
            "selection": insight.selection or request.selection,
        }
    )
def _insights_from_database_evidence(payload: dict, requests: list[KeyInsightRequest]) -> list[KeyInsight]:
    evidence_id = str(payload.get("evidence_id") or "")
    if not evidence_id:
        return []
    rows = _rows_from_evidence(payload)
    evidence_ref = InsightEvidenceRef(source_type="query", source_id=evidence_id, label=payload.get("summary"))
    insights: list[KeyInsight] = []
    if not requests:
        return insights
    if not rows:
        insights.extend(
            _unavailable_insight(request, evidence_ref, "Query returned no row-like records.")
            for request in requests
        )
        return insights

    time_key = _first_key(rows, ["timestamp", "time", "_time", "date"])
    value_key = _database_value_key(payload, rows, time_key)
    sorted_rows = sorted(rows, key=lambda row: str(row.get(time_key) or "")) if time_key else rows
    for request in requests:
        insight = _database_insight_for_request(request, sorted_rows, value_key, time_key, evidence_ref)
        if insight is not None:
            insights.append(insight)
    return insights


def _database_insight_for_request(
    request: KeyInsightRequest,
    rows: list[dict],
    value_key: str | None,
    time_key: str | None,
    evidence_ref: InsightEvidenceRef,
) -> KeyInsight | None:
    name = request.name
    insight_type = request.insight_type
    requirements = request.requirements or {}
    rows, row_selectors = _rows_for_insight_request(request, rows)
    if row_selectors and not rows:
        selector_text = ", ".join(f"{key}={value!r}" for key, value in row_selectors.items())
        return _unavailable_insight(
            request,
            evidence_ref,
            f"Query rows did not contain a record matching the requested dimensions: {selector_text}.",
        )
    normalized_name = str(name or "").strip().lower()
    if insight_type in {"count", "record_count", "row_count"} or normalized_name in {"record_count", "row_count", "count"}:
        value = len(rows)
        return KeyInsight(
            insight_id=_insight_id(evidence_ref.source_id, name, value),
            name=name,
            insight_type="count",
            insight_key=request.insight_key,
            statement=f"{name} is {value}.",
            value=value,
            subject=request.subject,
            time_range=request.time_range,
            method="sql_query",
            evidence_refs=[evidence_ref],
            calculation_trace={
                "source": "normalized_database_evidence",
                "count_target": requirements.get("count_target") or "rows",
                "row_selectors": row_selectors,
            },
            derived_from=request.derived_from,
        )
    if insight_type in {"time_boundary", "boundary_time"}:
        if not time_key:
            return None
        position = requirements.get("time_position")
        if position not in {"start", "end"}:
            return None
        row = rows[-1] if position == "end" else rows[0]
        value = row.get(time_key)
        return KeyInsight(
            insight_id=_insight_id(evidence_ref.source_id, name, value),
            name=name,
            insight_type="time_boundary",
            insight_key=request.insight_key,
            statement=f"{name} is {value}.",
            value=value,
            subject=request.subject,
            time_range=request.time_range,
            method="sql_query",
            evidence_refs=[evidence_ref],
            calculation_trace={
                "row": row,
                "time_key": time_key,
                "position": position,
                "row_selectors": row_selectors,
            },
            derived_from=request.derived_from,
        )
    if insight_type == "point_value":
        if not value_key:
            return None
        position = requirements.get("time_position")
        if position not in {"start", "end"}:
            return None
        row = rows[-1] if position == "end" else rows[0]
        value = row.get(value_key)
        timestamp = row.get(time_key) if time_key else None
        return KeyInsight(
            insight_id=_insight_id(evidence_ref.source_id, name, value),
            name=name,
            insight_type="point_value",
            insight_key=request.insight_key,
            statement=f"{name} is {value}" + (f" at {timestamp}." if timestamp else "."),
            value=value,
            subject=request.subject,
            time_range=request.time_range,
            method="sql_query",
            evidence_refs=[evidence_ref],
            calculation_trace={
                "row": row,
                "value_key": value_key,
                "time_key": time_key,
                "position": position,
                "row_selectors": row_selectors,
            },
            derived_from=request.derived_from,
        )
    if insight_type in {"extreme", "extrema", "extreme_time"}:
        if not value_key:
            return None
        operator = requirements.get("operator")
        if operator not in {"min", "max"}:
            return None
        numeric_rows = [row for row in rows if _number(row.get(value_key)) is not None]
        if not numeric_rows:
            return None
        row = min(numeric_rows, key=lambda item: _number(item.get(value_key)) or 0) if operator == "min" else max(numeric_rows, key=lambda item: _number(item.get(value_key)) or 0)
        value = row.get(value_key)
        timestamp = row.get(time_key) if time_key else None
        insight_value = timestamp if insight_type == "extreme_time" or name in {"max_time", "min_time"} else value
        located_item = InsightItem(
            item_id=_located_item_id(evidence_ref.source_id, request.insight_key, row),
            value=value,
            timestamp=str(timestamp) if timestamp is not None else None,
            dimensions={
                **(request.dimensions if isinstance(request.dimensions, dict) else {}),
                "operator": operator,
            },
            evidence_refs=[evidence_ref],
            locator={"row": row},
        )
        return KeyInsight(
            insight_id=_insight_id(evidence_ref.source_id, name, insight_value),
            name=name,
            insight_type="extreme_time" if insight_type == "extreme_time" or name in {"max_time", "min_time"} else "extreme",
            insight_key=request.insight_key,
            statement=(
                f"{name} is {timestamp} for {operator}_value {value}."
                if insight_type == "extreme_time" or name in {"max_time", "min_time"}
                else f"{name} is {value}" + (f" at {timestamp}." if timestamp else ".")
            ),
            value=insight_value,
            value_shape="collection",
            items=[located_item],
            subject=request.subject,
            time_range=request.time_range,
            method="sql_query",
            evidence_refs=[evidence_ref],
            calculation_trace={
                "row": row,
                "value_key": value_key,
                "time_key": time_key,
                "operator": operator,
                "row_selectors": row_selectors,
            },
            derived_from=request.derived_from,
        )
    return None


def _located_item_id(evidence_id: str, insight_key: str | None, row: dict) -> str:
    digest = sha1(
        json.dumps(
            {"evidence_id": evidence_id, "insight_key": insight_key, "row": row},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:12]
    return f"item_{digest}"


def _rows_for_insight_request(request: KeyInsightRequest, rows: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Bind a Key Insight contract to rows using dimensions grounded in result columns."""

    if not rows:
        return rows, {}
    available_keys = set().union(*(row.keys() for row in rows[:50]))
    requirements = request.requirements if isinstance(request.requirements, dict) else {}
    nested_filters = requirements.get("row_filters") if isinstance(requirements.get("row_filters"), dict) else {}
    candidates = {
        **(request.dimensions if isinstance(request.dimensions, dict) else {}),
        **nested_filters,
        **requirements,
    }
    selectors = {
        str(key): value
        for key, value in candidates.items()
        if str(key) in available_keys and not isinstance(value, (dict, list, tuple, set))
    }
    if not selectors:
        return rows, {}
    selected = [
        row
        for row in rows
        if all(_dimension_value_matches(row.get(key), expected) for key, expected in selectors.items())
    ]
    return selected, selectors


def _dimension_value_matches(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if actual is None or expected is None:
        return False
    return str(actual).strip() == str(expected).strip()


def _validate_registered_insight(insight: KeyInsight, request: KeyInsightRequest | None) -> KeyInsight:
    """Apply the Key Insight contract again at registration, the final trust boundary."""

    if request is None:
        return insight
    flags = list(insight.quality_flags)
    expected_count = request.expected_item_count
    requirements = request.requirements if isinstance(request.requirements, dict) else {}
    if expected_count is None:
        expected_count = requirements.get("expected_item_count", requirements.get("limit"))
    try:
        expected_count = int(expected_count) if expected_count is not None else None
    except (TypeError, ValueError):
        expected_count = None
    if expected_count is not None and len(insight.items) != expected_count and "item_count_mismatch" not in flags:
        flags.append("item_count_mismatch")
    selection = {**requirements, **(request.selection if isinstance(request.selection, dict) else {})}
    if request.result_shape in {"ranked_set", "ranking"}:
        ranks = [item.rank for item in insight.items]
        if ranks != list(range(1, len(insight.items) + 1)) and "invalid_rank_sequence" not in flags:
            flags.append("invalid_rank_sequence")
    order_by = str(selection.get("order_by") or "").strip()
    direction = str(selection.get("direction") or "asc").strip().lower()
    if order_by and len(insight.items) > 1:
        values = [getattr(item, order_by, None) for item in insight.items]
        values = [item.dimensions.get(order_by) if value is None else value for item, value in zip(insight.items, values)]
        if all(value is not None for value in values):
            try:
                if values != sorted(values, reverse=direction == "desc") and "invalid_item_order" not in flags:
                    flags.append("invalid_item_order")
            except TypeError:
                if "unorderable_item_values" not in flags:
                    flags.append("unorderable_item_values")
    distinct_by = str(selection.get("distinct_by") or "").strip()
    if distinct_by and len(insight.items) > 1:
        values = [item.dimensions.get(distinct_by, getattr(item, distinct_by, None)) for item in insight.items]
        if len(values) != len(set(map(str, values))) and "duplicate_distinct_key" not in flags:
            flags.append("duplicate_distinct_key")
    item_ids = {item.item_id for item in insight.items}
    source_ids = {source_id for item in insight.items for source_id in item.source_item_ids}
    if not source_ids.issubset(item_ids) and "unresolved_source_item" not in flags:
        flags.append("unresolved_source_item")
    return insight.model_copy(update={
        "value_shape": insight.value_shape or request.result_shape or ("collection" if insight.items else "scalar"),
        "semantic_class": insight.semantic_class or request.semantic_class,
        "derivation": insight.derivation or request.derivation,
        "selection": insight.selection or request.selection,
        "quality_flags": flags,
        "status": "partial" if flags and insight.status == "verified" else insight.status,
    })


def _unavailable_insight(request: KeyInsightRequest, evidence_ref: InsightEvidenceRef, reason: str) -> KeyInsight:
    return KeyInsight(
        insight_id=_insight_id(evidence_ref.source_id, request.name, "unavailable"),
        name=request.name,
        insight_type=request.insight_type,
        insight_key=request.insight_key,
        statement=f"{request.name} is unavailable: {reason}",
        subject=request.subject,
        time_range=request.time_range,
        method="sql_query",
        evidence_refs=[evidence_ref],
        status="unavailable",
        unavailable_reason=reason,
        derived_from=request.derived_from,
    )


def _coverage_for(requests: list[KeyInsightRequest], produced: list[KeyInsight], rejected: list[KeyInsight]) -> InsightCoverage:
    requested = list(dict.fromkeys(request.name for request in requests))
    requests_by_key = {request.insight_key: request for request in requests}
    by_key = {insight.insight_key: insight for insight in produced}
    rejected_keys = {insight.insight_key for insight in rejected}
    names_for_status = lambda status: [
        request.name
        for key, request in requests_by_key.items()
        if key in by_key and by_key[key].status == status
    ]
    return InsightCoverage(
        requested=requested,
        verified=names_for_status("verified"),
        unavailable=names_for_status("unavailable"),
        partial=names_for_status("partial"),
        rejected=[request.name for key, request in requests_by_key.items() if key in rejected_keys],
        missing=[
            request.name
            for key, request in requests_by_key.items()
            if key not in by_key and key not in rejected_keys
        ],
    )


def _merge_insight_state(
    request_state,
    tool_name: str,
    requests: list[KeyInsightRequest],
    produced: list[KeyInsight],
    rejected: list[KeyInsight],
    event_coverage: InsightCoverage,
) -> InsightCoverage:
    insight_set = request_state.insight_set
    planned = {request.insight_key: request for request in insight_set.requests}
    planned.update({request.insight_key: request for request in requests})
    insight_set.requests = list(planned.values())
    existing = {insight.insight_key: insight for insight in insight_set.insights}
    for insight in [*produced, *rejected]:
        current = existing.get(insight.insight_key)
        current_rank = _insight_status_rank(current.status) if current is not None else -1
        next_rank = _insight_status_rank(insight.status)
        preserves_located_detail = (
            current is not None
            and current_rank == next_rank
            and current.status == "verified"
            and insight.status == "verified"
            and (
                current.value == insight.value
                or (
                    isinstance(insight.value, dict)
                    and insight.value.get("value") == current.value
                )
            )
            and bool(current.items)
            and not insight.items
        )
        if not preserves_located_detail and (current is None or next_rank >= current_rank):
            existing[insight.insight_key] = insight
    insight_set.insights = list(existing.values())
    coverage = _coverage_for(insight_set.requests, insight_set.insights, [insight for insight in insight_set.insights if insight.status == "rejected"])
    insight_set.coverage = coverage
    request_state.insight_coverage = coverage
    request_state.insight_events.append(
        InsightEvent(
            iteration=request_state.iteration,
            tool_name=tool_name,
            produced_insight_ids=[insight.insight_id for insight in produced if insight.status == "verified"],
            unavailable_insight_ids=[insight.insight_id for insight in produced if insight.status == "unavailable"],
            rejected_insight_ids=[insight.insight_id for insight in rejected],
            coverage=event_coverage,
        )
    )
    return coverage


def _insight_status_rank(status: str) -> int:
    return {"rejected": 0, "unavailable": 1, "partial": 2, "verified": 3}.get(status, 0)


def _attach_insights_to_artifact(
    request_state,
    tool_name: str,
    payload: dict,
    produced: list[KeyInsight],
    rejected: list[KeyInsight],
    coverage: InsightCoverage,
) -> None:
    updates = {
        "produced_insights": produced,
        "rejected_insights": rejected,
        "insight_coverage": coverage,
    }
    if tool_name == "sql_query":
        evidence_id = str(payload.get("evidence_id") or "")
        artifact = getattr(request_state, "database_evidence_artifacts", {}).get(evidence_id)
        if artifact is not None:
            updated = artifact.model_copy(update=updates)
            request_state.database_evidence_artifacts[evidence_id] = updated
            if getattr(getattr(request_state, "latest_database_evidence", None), "evidence_id", None) == evidence_id:
                request_state.latest_database_evidence = request_state.latest_database_evidence.model_copy(update=updates)
    elif tool_name == "code_interpreter":
        analysis_id = str(payload.get("analysis_id") or "")
        artifact = getattr(request_state, "analysis_artifacts", {}).get(analysis_id)
        if artifact is not None:
            updated = artifact.model_copy(update=updates)
            request_state.analysis_artifacts[analysis_id] = updated


def _dedupe_insights(insights: list[KeyInsight]) -> list[KeyInsight]:
    result: dict[str, KeyInsight] = {}
    for insight in insights:
        result[insight.insight_key or insight.insight_id] = insight
    return list(result.values())


def _verify_insight_dependencies(
    produced: list[KeyInsight],
    existing: list[KeyInsight],
    *,
    request_state=None,
) -> list[KeyInsight]:
    """Verify an Insight DAG while canonicalizing traced artifact inputs as evidence.

    ``derived_from`` is the Insight-to-Insight DAG. Models occasionally place an
    artifact identifier there even though the computation consumed that artifact
    through its tool context. When the identifier resolves to a real request
    artifact *and* the calculation trace records that exact artifact, preserve the
    provenance as an evidence ref instead of treating it as a missing Insight.
    Unresolved or merely declared dependencies remain partial.
    """

    existing_index: dict[str, KeyInsight] = {}
    pending: dict[str, KeyInsight] = {}
    aliases: dict[str, str] = {}
    for insight in existing:
        existing_index[normalize_insight_key(insight.insight_id)] = insight
        existing_index[insight.insight_key] = insight
    for insight in produced:
        pending[insight.insight_key] = insight
        aliases[normalize_insight_key(insight.insight_id)] = insight.insight_key
    resolved: dict[str, KeyInsight] = {}
    artifact_refs = _artifact_dependency_index(request_state)

    def verify(insight_key: str, stack: set[str]) -> KeyInsight:
        if insight_key in resolved:
            return resolved[insight_key]
        insight = pending[insight_key]
        if insight.method == "code_interpreter" and not insight.calculation_trace:
            quality_flags = list(insight.quality_flags)
            if "missing_calculation_trace" not in quality_flags:
                quality_flags.append("missing_calculation_trace")
            insight = insight.model_copy(update={"status": "partial", "quality_flags": quality_flags})
        if not insight.derived_from:
            items = [
                item if item.evidence_refs else item.model_copy(update={"evidence_refs": insight.evidence_refs})
                for item in insight.items
            ]
            result = insight.model_copy(update={"items": items})
            resolved[insight_key] = result
            return result
        dependencies: list[KeyInsight | None] = []
        insight_dependency_keys: list[str] = []
        evidence_refs = list(insight.evidence_refs)
        seen_refs = {(ref.source_type, ref.source_id) for ref in evidence_refs}
        for reference in insight.derived_from:
            reference_key = normalize_insight_key(reference)
            dependency_key = reference_key if reference_key in pending else aliases.get(reference_key)
            if dependency_key in stack:
                dependencies.append(None)
            elif dependency_key in pending:
                dependencies.append(verify(dependency_key, {*stack, insight_key}))
                insight_dependency_keys.append(reference_key)
            elif reference_key in existing_index:
                dependencies.append(existing_index[reference_key])
                insight_dependency_keys.append(reference_key)
            else:
                artifact_ref = artifact_refs.get(reference_key)
                if artifact_ref is None:
                    traced_candidates = {
                        (candidate.source_type, candidate.source_id): candidate
                        for candidate in artifact_refs.values()
                        if (candidate.source_type, candidate.source_id) not in seen_refs
                        and _calculation_trace_uses_artifact(insight, candidate)
                    }
                    if len(traced_candidates) == 1:
                        artifact_ref = next(iter(traced_candidates.values()))
                if artifact_ref is not None and _calculation_trace_uses_artifact(insight, artifact_ref):
                    key = (artifact_ref.source_type, artifact_ref.source_id)
                    if key not in seen_refs:
                        evidence_refs.append(artifact_ref)
                        seen_refs.add(key)
                else:
                    dependencies.append(None)
        quality_flags = list(insight.quality_flags)
        if any(dependency is None or dependency.status != "verified" for dependency in dependencies):
            if "unverified_dependencies" not in quality_flags:
                quality_flags.append("unverified_dependencies")
            result = insight.model_copy(update={"status": "partial", "quality_flags": quality_flags})
            resolved[insight_key] = result
            return result
        for dependency in dependencies:
            for ref in dependency.evidence_refs:
                key = (ref.source_type, ref.source_id)
                if key not in seen_refs:
                    evidence_refs.append(ref)
                    seen_refs.add(key)
        if not insight.calculation_trace:
            if "missing_calculation_trace" not in quality_flags:
                quality_flags.append("missing_calculation_trace")
            result = insight.model_copy(
                update={
                    "status": "partial",
                    "quality_flags": quality_flags,
                    "evidence_refs": evidence_refs,
                    "derived_from": insight_dependency_keys,
                }
            )
            resolved[insight_key] = result
            return result
        items = [
            item if item.evidence_refs else item.model_copy(update={"evidence_refs": evidence_refs})
            for item in insight.items
        ]
        result = insight.model_copy(update={
            "evidence_refs": evidence_refs,
            "quality_flags": quality_flags,
            "derived_from": insight_dependency_keys,
            "items": items,
        })
        resolved[insight_key] = result
        return result

    return [verify(insight.insight_key, set()) for insight in produced]


def _artifact_dependency_index(request_state) -> dict[str, InsightEvidenceRef]:
    if request_state is None:
        return {}
    index: dict[str, InsightEvidenceRef] = {}
    collections = (
        ("query", "evidence", getattr(request_state, "database_evidence_artifacts", {})),
        ("derived_evidence", "derived_evidence", getattr(request_state, "derived_evidence_artifacts", {})),
        ("analysis", "analysis", getattr(request_state, "analysis_artifacts", {})),
        ("anomaly", "anomaly", getattr(request_state, "anomaly_artifacts", {})),
        ("forecast", "forecast", getattr(request_state, "forecast_artifacts", {})),
    )
    for source_type, ref_prefix, artifacts in collections:
        for source_id in artifacts or {}:
            evidence_ref = InsightEvidenceRef(source_type=source_type, source_id=str(source_id))
            index[normalize_insight_key(str(source_id))] = evidence_ref
            index[normalize_insight_key(f"{ref_prefix}:{source_id}")] = evidence_ref
    return index


def _calculation_trace_uses_artifact(insight: KeyInsight, artifact_ref: InsightEvidenceRef) -> bool:
    try:
        trace = json.dumps(insight.calculation_trace, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        trace = str(insight.calculation_trace or "")
    source_id = artifact_ref.source_id
    return source_id in trace or f"{artifact_ref.source_type}:{source_id}" in trace


def _insight_has_evidence_or_unavailable(insight: KeyInsight) -> bool:
    return insight.status == "unavailable" or bool(insight.evidence_refs) or bool(insight.derived_from)


def _rows_from_evidence(payload: dict) -> list[dict]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    rows = data.get("rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    points = data.get("points")
    if isinstance(points, list):
        return [dict(point) for point in points if isinstance(point, dict)]
    series = data.get("series")
    if isinstance(series, list):
        combined: list[dict] = []
        for item in series:
            if isinstance(item, dict) and isinstance(item.get("points"), list):
                combined.extend(dict(point) for point in item["points"] if isinstance(point, dict))
        return combined
    return []


def _first_key(rows: list[dict], candidates: list[str]) -> str | None:
    keys = set().union(*(row.keys() for row in rows[:20])) if rows else set()
    return next((key for key in candidates if key in keys), None)


def _database_value_key(payload: dict, rows: list[dict], time_key: str | None) -> str | None:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    generation = diagnostics.get("llm_query_generation") if isinstance(diagnostics.get("llm_query_generation"), dict) else {}
    candidates: list[str] = []

    for source in (metadata, diagnostics, generation):
        selected = source.get("selected_fields")
        if isinstance(selected, list):
            candidates.extend(str(item) for item in selected if str(item).strip())
        selected_one = source.get("selected_field")
        if isinstance(selected_one, str) and selected_one.strip():
            candidates.append(selected_one)

    schema_linking = diagnostics.get("schema_linking_generation")
    if isinstance(schema_linking, dict):
        for group_name in ("value_columns", "measures", "aggregate_targets"):
            group = schema_linking.get(group_name)
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    for key in ("physical_name", "column", "field", "name"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            candidates.append(value)

    available_keys = set().union(*(row.keys() for row in rows[:20])) if rows else set()
    for candidate in dict.fromkeys(candidates):
        if candidate == time_key or candidate not in available_keys:
            continue
        if any(_number(row.get(candidate)) is not None for row in rows[:20]):
            return candidate

    numeric_keys = [
        key
        for key in available_keys
        if key != time_key and any(_number(row.get(key)) is not None for row in rows[:20])
    ]
    return numeric_keys[0] if len(numeric_keys) == 1 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _insight_id(*parts: Any) -> str:
    base = ":".join(str(part) for part in parts if part is not None)
    return "insight_" + sha1(base.encode("utf-8")).hexdigest()[:16]
