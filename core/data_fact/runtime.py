"""Runtime registration for DataFacts produced by tools."""
from __future__ import annotations

from hashlib import sha1
import json
from typing import Any

from schemas.data_fact import (
    DataFact,
    DataFactRequest,
    FactCoverage,
    FactEvent,
    FactEvidenceRef,
    normalize_fact_key,
)


def register_data_facts_from_payload(request_state, tool_name: str, full_payload: dict) -> FactCoverage:
    """Register prompt-safe facts from a completed tool payload.

    Coverage is diagnostic only. Missing facts should guide the next ReAct turn,
    not force the runtime to fail.
    """

    action_input = _latest_action_input(request_state, tool_name)
    requests = _fact_requests(action_input)
    produced: list[DataFact] = []
    rejected: list[DataFact] = []

    for raw_fact in full_payload.get("produced_facts") or []:
        fact = _bind_fact_contract(_validate_fact(raw_fact, default_method=tool_name), requests)
        if fact is not None:
            produced.append(fact)
    for raw_fact in full_payload.get("rejected_facts") or []:
        fact = _bind_fact_contract(
            _validate_fact(raw_fact, default_method=tool_name, default_status="rejected"),
            requests,
        )
        if fact is not None:
            rejected.append(fact)

    if tool_name == "sql_query":
        produced.extend(_facts_from_database_evidence(full_payload, requests))
    elif tool_name == "code_interpreter":
        produced.extend(_facts_from_analysis(full_payload, requests))
    # Forecast and anomaly outputs remain analysis artifacts. They can be
    # referenced by answers and visualizations, but are not Facts by default.

    produced = _verify_fact_dependencies(
        _dedupe_facts([fact for fact in produced if _fact_has_evidence_or_unavailable(fact)]),
        request_state.fact_set.facts,
    )
    rejected = _dedupe_facts(rejected)
    event_coverage = _coverage_for(requests, produced, rejected)
    coverage = _merge_fact_state(request_state, tool_name, requests, produced, rejected, event_coverage)
    _attach_facts_to_artifact(request_state, tool_name, full_payload, produced, rejected, event_coverage)
    return coverage


def data_fact_prompt_view(request_state) -> dict:
    facts = list(getattr(getattr(request_state, "fact_set", None), "facts", []) or [])
    requests = list(getattr(getattr(request_state, "fact_set", None), "requests", []) or [])
    coverage = getattr(getattr(request_state, "fact_set", None), "coverage", None)
    recent = facts[-12:]
    return {
        "summary": {
            "verified": [fact.name for fact in facts if fact.status == "verified"][-12:],
            "unavailable": [fact.name for fact in facts if fact.status == "unavailable"][-8:],
            "rejected": [fact.name for fact in facts if fact.status == "rejected"][-8:],
            "missing": list(getattr(coverage, "missing", []) or [])[-12:],
        },
        "plan": [
            {
                "fact_key": request.fact_key,
                "name": request.name,
                "fact_type": request.fact_type,
                "derived_from": request.derived_from,
            }
            for request in requests[-12:]
        ],
        "recent_facts": [
            {
                "fact_id": fact.fact_id,
                "fact_key": fact.fact_key,
                "name": fact.name,
                "fact_type": fact.fact_type,
                "status": fact.status,
                "statement": fact.statement,
                "evidence_refs": [ref.source_id for ref in fact.evidence_refs[:4]],
                "derived_from": fact.derived_from,
                "unavailable_reason": fact.unavailable_reason,
            }
            for fact in recent
        ],
    }


def _latest_action_input(request_state, tool_name: str) -> dict:
    for call in reversed(getattr(request_state, "tool_history", []) or []):
        if call.tool_name == tool_name:
            return dict(call.tool_input or {})
    return {}


def _fact_requests(action_input: dict) -> list[DataFactRequest]:
    items = action_input.get("fact_requests") or []
    requests: list[DataFactRequest] = []
    for item in items:
        if isinstance(item, dict):
            try:
                requests.append(DataFactRequest.model_validate(item))
            except Exception:
                continue
    return requests


def _validate_fact(raw_fact: Any, *, default_method: str, default_status: str = "verified") -> DataFact | None:
    if not isinstance(raw_fact, dict):
        return None
    payload = dict(raw_fact)
    payload.setdefault("method", default_method)
    payload.setdefault("status", default_status)
    payload.setdefault("name", payload.get("fact_id") or payload.get("fact_type") or "fact")
    payload.setdefault("fact_type", "custom")
    payload.setdefault("statement", str(payload.get("summary") or payload.get("name") or "Data fact."))
    payload.setdefault("fact_id", _fact_id(payload.get("method"), payload.get("name"), payload.get("value")))
    try:
        return DataFact.model_validate(payload)
    except Exception:
        return None


def _bind_fact_contract(
    fact: DataFact | None,
    requests: list[DataFactRequest],
    *,
    preserve_dependencies: bool = False,
) -> DataFact | None:
    if fact is None or not requests:
        return fact
    aliases = {fact.fact_key, normalize_fact_key(fact.name)}
    request = next(
        (
            item
            for item in requests
            if item.fact_key in aliases or normalize_fact_key(item.name) in aliases
        ),
        None,
    )
    if request is None:
        return None
    return fact.model_copy(
        update={
            "fact_key": request.fact_key,
            "name": request.name,
            "fact_type": request.fact_type,
            "subject": fact.subject or request.subject,
            "time_range": fact.time_range or request.time_range,
            "dimensions": fact.dimensions or request.dimensions,
            "derived_from": fact.derived_from if preserve_dependencies else fact.derived_from or request.derived_from,
            "value_shape": fact.value_shape or request.result_shape,
            "semantic_class": fact.semantic_class or request.semantic_class,
            "derivation": fact.derivation or request.derivation,
            "selection": fact.selection or request.selection,
        }
    )
def _facts_from_database_evidence(payload: dict, requests: list[DataFactRequest]) -> list[DataFact]:
    evidence_id = str(payload.get("evidence_id") or "")
    if not evidence_id:
        return []
    rows = _rows_from_evidence(payload)
    evidence_ref = FactEvidenceRef(source_type="query", source_id=evidence_id, label=payload.get("summary"))
    facts: list[DataFact] = []
    if not requests:
        return facts
    if not rows:
        facts.extend(
            _unavailable_fact(request, evidence_ref, "Query returned no row-like records.")
            for request in requests
        )
        return facts

    time_key = _first_key(rows, ["timestamp", "time", "_time", "date"])
    value_key = _database_value_key(payload, rows, time_key)
    sorted_rows = sorted(rows, key=lambda row: str(row.get(time_key) or "")) if time_key else rows
    for request in requests:
        fact = _database_fact_for_request(request, sorted_rows, value_key, time_key, evidence_ref)
        if fact is not None:
            facts.append(fact)
    return facts


def _database_fact_for_request(
    request: DataFactRequest,
    rows: list[dict],
    value_key: str | None,
    time_key: str | None,
    evidence_ref: FactEvidenceRef,
) -> DataFact | None:
    name = request.name
    fact_type = request.fact_type
    requirements = request.requirements or {}
    rows, row_selectors = _rows_for_fact_request(request, rows)
    if row_selectors and not rows:
        selector_text = ", ".join(f"{key}={value!r}" for key, value in row_selectors.items())
        return _unavailable_fact(
            request,
            evidence_ref,
            f"Query rows did not contain a record matching the requested dimensions: {selector_text}.",
        )
    normalized_name = str(name or "").strip().lower()
    if fact_type in {"count", "record_count", "row_count"} or normalized_name in {"record_count", "row_count", "count"}:
        value = len(rows)
        return DataFact(
            fact_id=_fact_id(evidence_ref.source_id, name, value),
            name=name,
            fact_type="count",
            fact_key=request.fact_key,
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
    if fact_type in {"time_boundary", "boundary_time"}:
        if not time_key:
            return None
        position = requirements.get("time_position")
        if position not in {"start", "end"}:
            return None
        row = rows[-1] if position == "end" else rows[0]
        value = row.get(time_key)
        return DataFact(
            fact_id=_fact_id(evidence_ref.source_id, name, value),
            name=name,
            fact_type="time_boundary",
            fact_key=request.fact_key,
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
    if fact_type == "point_value":
        if not value_key:
            return None
        position = requirements.get("time_position")
        if position not in {"start", "end"}:
            return None
        row = rows[-1] if position == "end" else rows[0]
        value = row.get(value_key)
        timestamp = row.get(time_key) if time_key else None
        return DataFact(
            fact_id=_fact_id(evidence_ref.source_id, name, value),
            name=name,
            fact_type="point_value",
            fact_key=request.fact_key,
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
    if fact_type in {"extreme", "extrema", "extreme_time"}:
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
        fact_value = timestamp if fact_type == "extreme_time" or name in {"max_time", "min_time"} else value
        return DataFact(
            fact_id=_fact_id(evidence_ref.source_id, name, fact_value),
            name=name,
            fact_type="extreme_time" if fact_type == "extreme_time" or name in {"max_time", "min_time"} else "extreme",
            fact_key=request.fact_key,
            statement=(
                f"{name} is {timestamp} for {operator}_value {value}."
                if fact_type == "extreme_time" or name in {"max_time", "min_time"}
                else f"{name} is {value}" + (f" at {timestamp}." if timestamp else ".")
            ),
            value=fact_value,
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


def _rows_for_fact_request(request: DataFactRequest, rows: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Bind a Fact contract to rows using dimensions grounded in result columns."""

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


def _facts_from_analysis(payload: dict, requests: list[DataFactRequest]) -> list[DataFact]:
    analysis_id = str(payload.get("analysis_id") or "")
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    evidence_refs = [
        FactEvidenceRef(source_type="analysis", source_id=analysis_id, label=payload.get("analysis_goal")),
    ]
    input_evidence_id = payload.get("input_evidence_id")
    if input_evidence_id:
        evidence_refs.append(FactEvidenceRef(source_type="query", source_id=str(input_evidence_id)))
    facts: list[DataFact] = []
    requested_keys = {request.fact_key for request in requests}
    for raw in result.get("facts") or []:
        if not isinstance(raw, dict):
            continue
        raw_key = normalize_fact_key(raw.get("fact_key") or raw.get("name") or "")
        if requested_keys and raw_key not in requested_keys:
            continue
        raw_payload = dict(raw)
        if not isinstance(raw_payload.get("items"), list) and isinstance(raw_payload.get("value"), list):
            raw_payload["items"] = _fact_items_from_value(raw_key, raw_payload["value"])
        fact = _validate_fact(
            {
                **raw_payload,
                "method": raw_payload.get("method") or "code_interpreter",
                "evidence_refs": raw_payload.get("evidence_refs") or [ref.model_dump(mode="json") for ref in evidence_refs],
            },
            default_method="code_interpreter",
        )
        fact = _bind_fact_contract(fact, requests, preserve_dependencies="derived_from" in raw_payload)
        if fact is not None:
            fact = _validate_registered_fact(fact, next(
                (request for request in requests if request.fact_key == fact.fact_key),
                None,
            ))
            if fact.items:
                fact = fact.model_copy(
                    update={
                        "items": [
                            item.model_copy(update={"evidence_refs": item.evidence_refs or evidence_refs})
                            for item in fact.items
                        ]
                    }
                )
            facts.append(fact)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    metrics_by_key = {normalize_fact_key(key): (key, value) for key, value in metrics.items()}
    native_keys = {fact.fact_key for fact in facts}
    for request in requests:
        metric_match = metrics_by_key.get(request.fact_key) or metrics_by_key.get(normalize_fact_key(request.name))
        if request.fact_key not in native_keys and metric_match is not None:
            metric_key, value = metric_match
            fact = DataFact(
                    fact_id=_fact_id(analysis_id, request.name, value),
                    name=request.name,
                    fact_type=request.fact_type,
                    fact_key=request.fact_key,
                    statement=f"{request.name} is {value}.",
                    value=value,
                    subject=request.subject,
                    time_range=request.time_range,
                    method="code_interpreter",
                    evidence_refs=evidence_refs,
                    calculation_trace={"metric_key": metric_key, "code_hash": payload.get("code_hash")},
                    derived_from=request.derived_from,
                )
            facts.append(_bind_fact_contract(fact, [request]) or fact)
    return facts


def _validate_registered_fact(fact: DataFact, request: DataFactRequest | None) -> DataFact:
    """Apply the Fact contract again at registration, the final trust boundary."""

    if request is None:
        return fact
    flags = list(fact.quality_flags)
    expected_count = request.expected_item_count
    requirements = request.requirements if isinstance(request.requirements, dict) else {}
    if expected_count is None:
        expected_count = requirements.get("expected_item_count", requirements.get("limit"))
    try:
        expected_count = int(expected_count) if expected_count is not None else None
    except (TypeError, ValueError):
        expected_count = None
    if expected_count is not None and len(fact.items) != expected_count and "item_count_mismatch" not in flags:
        flags.append("item_count_mismatch")
    selection = {**requirements, **(request.selection if isinstance(request.selection, dict) else {})}
    if request.result_shape in {"ranked_set", "ranking"}:
        ranks = [item.rank for item in fact.items]
        if ranks != list(range(1, len(fact.items) + 1)) and "invalid_rank_sequence" not in flags:
            flags.append("invalid_rank_sequence")
    order_by = str(selection.get("order_by") or "").strip()
    direction = str(selection.get("direction") or "asc").strip().lower()
    if order_by and len(fact.items) > 1:
        values = [getattr(item, order_by, None) for item in fact.items]
        values = [item.dimensions.get(order_by) if value is None else value for item, value in zip(fact.items, values)]
        if all(value is not None for value in values):
            try:
                if values != sorted(values, reverse=direction == "desc") and "invalid_item_order" not in flags:
                    flags.append("invalid_item_order")
            except TypeError:
                if "unorderable_item_values" not in flags:
                    flags.append("unorderable_item_values")
    distinct_by = str(selection.get("distinct_by") or "").strip()
    if distinct_by and len(fact.items) > 1:
        values = [item.dimensions.get(distinct_by, getattr(item, distinct_by, None)) for item in fact.items]
        if len(values) != len(set(map(str, values))) and "duplicate_distinct_key" not in flags:
            flags.append("duplicate_distinct_key")
    item_ids = {item.item_id for item in fact.items}
    source_ids = {source_id for item in fact.items for source_id in item.source_item_ids}
    if not source_ids.issubset(item_ids) and "unresolved_source_item" not in flags:
        flags.append("unresolved_source_item")
    return fact.model_copy(update={
        "value_shape": fact.value_shape or request.result_shape or ("collection" if fact.items else "scalar"),
        "semantic_class": fact.semantic_class or request.semantic_class,
        "derivation": fact.derivation or request.derivation,
        "selection": fact.selection or request.selection,
        "quality_flags": flags,
        "status": "partial" if flags and fact.status == "verified" else fact.status,
    })


def _fact_items_from_value(fact_key: str, value: list) -> list[dict]:
    items: list[dict] = []
    for index, item in enumerate(value):
        payload = dict(item) if isinstance(item, dict) else {"value": item}
        item_id = str(payload.get("item_id") or "").strip()
        if not item_id:
            fingerprint = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            item_id = f"{fact_key}:{index}:{sha1(fingerprint.encode('utf-8')).hexdigest()[:10]}"
        payload["item_id"] = item_id
        items.append(payload)
    return items


def _unavailable_fact(request: DataFactRequest, evidence_ref: FactEvidenceRef, reason: str) -> DataFact:
    return DataFact(
        fact_id=_fact_id(evidence_ref.source_id, request.name, "unavailable"),
        name=request.name,
        fact_type=request.fact_type,
        fact_key=request.fact_key,
        statement=f"{request.name} is unavailable: {reason}",
        subject=request.subject,
        time_range=request.time_range,
        method="sql_query",
        evidence_refs=[evidence_ref],
        status="unavailable",
        unavailable_reason=reason,
        derived_from=request.derived_from,
    )


def _coverage_for(requests: list[DataFactRequest], produced: list[DataFact], rejected: list[DataFact]) -> FactCoverage:
    requested = list(dict.fromkeys(request.name for request in requests))
    requests_by_key = {request.fact_key: request for request in requests}
    by_key = {fact.fact_key: fact for fact in produced}
    rejected_keys = {fact.fact_key for fact in rejected}
    names_for_status = lambda status: [
        request.name
        for key, request in requests_by_key.items()
        if key in by_key and by_key[key].status == status
    ]
    return FactCoverage(
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


def _merge_fact_state(
    request_state,
    tool_name: str,
    requests: list[DataFactRequest],
    produced: list[DataFact],
    rejected: list[DataFact],
    event_coverage: FactCoverage,
) -> FactCoverage:
    fact_set = request_state.fact_set
    planned = {request.fact_key: request for request in fact_set.requests}
    planned.update({request.fact_key: request for request in requests})
    fact_set.requests = list(planned.values())
    existing = {fact.fact_key: fact for fact in fact_set.facts}
    for fact in [*produced, *rejected]:
        current = existing.get(fact.fact_key)
        if current is None or _fact_status_rank(fact.status) >= _fact_status_rank(current.status):
            existing[fact.fact_key] = fact
    fact_set.facts = list(existing.values())
    coverage = _coverage_for(fact_set.requests, fact_set.facts, [fact for fact in fact_set.facts if fact.status == "rejected"])
    fact_set.coverage = coverage
    request_state.fact_coverage = coverage
    request_state.fact_events.append(
        FactEvent(
            iteration=request_state.iteration,
            tool_name=tool_name,
            produced_fact_ids=[fact.fact_id for fact in produced if fact.status == "verified"],
            unavailable_fact_ids=[fact.fact_id for fact in produced if fact.status == "unavailable"],
            rejected_fact_ids=[fact.fact_id for fact in rejected],
            coverage=event_coverage,
        )
    )
    return coverage


def _fact_status_rank(status: str) -> int:
    return {"rejected": 0, "unavailable": 1, "partial": 2, "verified": 3}.get(status, 0)


def _attach_facts_to_artifact(
    request_state,
    tool_name: str,
    payload: dict,
    produced: list[DataFact],
    rejected: list[DataFact],
    coverage: FactCoverage,
) -> None:
    updates = {
        "produced_facts": produced,
        "rejected_facts": rejected,
        "fact_coverage": coverage,
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


def _dedupe_facts(facts: list[DataFact]) -> list[DataFact]:
    result: dict[str, DataFact] = {}
    for fact in facts:
        result[fact.fact_key or fact.fact_id] = fact
    return list(result.values())


def _verify_fact_dependencies(produced: list[DataFact], existing: list[DataFact]) -> list[DataFact]:
    """Verify a Fact DAG and inherit source evidence through verified parents."""

    existing_index: dict[str, DataFact] = {}
    pending: dict[str, DataFact] = {}
    aliases: dict[str, str] = {}
    for fact in existing:
        existing_index[normalize_fact_key(fact.fact_id)] = fact
        existing_index[fact.fact_key] = fact
    for fact in produced:
        pending[fact.fact_key] = fact
        aliases[normalize_fact_key(fact.fact_id)] = fact.fact_key
    resolved: dict[str, DataFact] = {}

    def verify(fact_key: str, stack: set[str]) -> DataFact:
        if fact_key in resolved:
            return resolved[fact_key]
        fact = pending[fact_key]
        if fact.method == "code_interpreter" and not fact.calculation_trace:
            quality_flags = list(fact.quality_flags)
            if "missing_calculation_trace" not in quality_flags:
                quality_flags.append("missing_calculation_trace")
            fact = fact.model_copy(update={"status": "partial", "quality_flags": quality_flags})
        if not fact.derived_from:
            resolved[fact_key] = fact
            return fact
        dependencies: list[DataFact | None] = []
        for reference in fact.derived_from:
            reference_key = normalize_fact_key(reference)
            dependency_key = reference_key if reference_key in pending else aliases.get(reference_key)
            if dependency_key in stack:
                dependencies.append(None)
            elif dependency_key in pending:
                dependencies.append(verify(dependency_key, {*stack, fact_key}))
            else:
                dependencies.append(existing_index.get(reference_key))
        quality_flags = list(fact.quality_flags)
        if any(dependency is None or dependency.status != "verified" for dependency in dependencies):
            if "unverified_dependencies" not in quality_flags:
                quality_flags.append("unverified_dependencies")
            result = fact.model_copy(update={"status": "partial", "quality_flags": quality_flags})
            resolved[fact_key] = result
            return result
        evidence_refs = list(fact.evidence_refs)
        seen_refs = {(ref.source_type, ref.source_id) for ref in evidence_refs}
        for dependency in dependencies:
            for ref in dependency.evidence_refs:
                key = (ref.source_type, ref.source_id)
                if key not in seen_refs:
                    evidence_refs.append(ref)
                    seen_refs.add(key)
        if not fact.calculation_trace:
            if "missing_calculation_trace" not in quality_flags:
                quality_flags.append("missing_calculation_trace")
            result = fact.model_copy(
                update={
                    "status": "partial",
                    "quality_flags": quality_flags,
                    "evidence_refs": evidence_refs,
                }
            )
            resolved[fact_key] = result
            return result
        result = fact.model_copy(update={"evidence_refs": evidence_refs, "quality_flags": quality_flags})
        resolved[fact_key] = result
        return result

    return [verify(fact.fact_key, set()) for fact in produced]


def _fact_has_evidence_or_unavailable(fact: DataFact) -> bool:
    return fact.status == "unavailable" or bool(fact.evidence_refs) or bool(fact.derived_from)


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


def _fact_id(*parts: Any) -> str:
    base = ":".join(str(part) for part in parts if part is not None)
    return "fact_" + sha1(base.encode("utf-8")).hexdigest()[:16]
