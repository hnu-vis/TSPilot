"""Runtime registration for DataFacts produced by tools."""
from __future__ import annotations

from hashlib import sha1
from typing import Any

from schemas.data_fact import (
    DataFact,
    DataFactRequest,
    FactCoverage,
    FactEvent,
    FactEvidenceRef,
    normalize_fact_key,
)
from core.data_fact.memory import observe_fact_usage


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

    if tool_name in {"sql_query", "query_database"}:
        produced.extend(_facts_from_database_evidence(full_payload, requests))
    elif tool_name == "code_interpreter":
        produced.extend(_facts_from_analysis(full_payload, requests))
    elif tool_name == "forecast":
        produced.extend(_facts_from_forecast(full_payload, requests))
    elif tool_name == "anomaly":
        produced.extend(_facts_from_anomaly(full_payload, requests))

    produced = _verify_fact_dependencies(
        _dedupe_facts([fact for fact in produced if _fact_has_evidence_or_unavailable(fact)]),
        request_state.fact_set.facts,
    )
    rejected = _dedupe_facts(rejected)
    event_coverage = _coverage_for(requests, produced, rejected)
    coverage = _merge_fact_state(request_state, tool_name, requests, produced, rejected, event_coverage)
    _attach_facts_to_artifact(request_state, tool_name, full_payload, produced, rejected, event_coverage)
    observe_fact_usage(
        database_id=_database_id_for_memory(request_state),
        tool_name=tool_name,
        requests=requests,
        facts=produced,
    )
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


def _database_id_for_memory(request_state) -> str | None:
    selected = getattr(request_state, "selected_database", None)
    if selected:
        return str(selected)
    context = getattr(request_state, "database_context", None)
    database_id = getattr(context, "database_id", None)
    return str(database_id) if database_id else None


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


def _bind_fact_contract(fact: DataFact | None, requests: list[DataFactRequest]) -> DataFact | None:
    if fact is None or not requests:
        return fact
    request = next((item for item in requests if item.fact_key == fact.fact_key), None)
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
            "derived_from": fact.derived_from or request.derived_from,
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
    value_key = _first_numeric_key(rows, ["value", "price", "_value", "close", "amount"])
    if not value_key:
        return facts
    sorted_rows = sorted(rows, key=lambda row: str(row.get(time_key) or "")) if time_key else rows
    for request in requests:
        fact = _database_fact_for_request(request, sorted_rows, value_key, time_key, evidence_ref)
        if fact is not None:
            facts.append(fact)
    return facts


def _database_fact_for_request(
    request: DataFactRequest,
    rows: list[dict],
    value_key: str,
    time_key: str | None,
    evidence_ref: FactEvidenceRef,
) -> DataFact | None:
    name = request.name
    fact_type = request.fact_type
    requirements = request.requirements or {}
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
            calculation_trace={"source": "normalized_database_evidence", "count_target": requirements.get("count_target") or "rows"},
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
            calculation_trace={"row": row, "time_key": time_key, "position": position},
            derived_from=request.derived_from,
        )
    if fact_type == "point_value":
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
            calculation_trace={"row": row, "value_key": value_key, "time_key": time_key, "position": position},
            derived_from=request.derived_from,
        )
    if fact_type in {"extreme", "extrema", "extreme_time"}:
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
            calculation_trace={"row": row, "value_key": value_key, "time_key": time_key, "operator": operator},
            derived_from=request.derived_from,
        )
    return None


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
        fact = _validate_fact(
            {
                **raw,
                "method": raw.get("method") or "code_interpreter",
                "evidence_refs": raw.get("evidence_refs") or [ref.model_dump(mode="json") for ref in evidence_refs],
            },
            default_method="code_interpreter",
        )
        fact = _bind_fact_contract(fact, requests)
        if fact is not None:
            facts.append(fact)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    native_keys = {fact.fact_key for fact in facts}
    for request in requests:
        if request.fact_key not in native_keys and request.name in metrics:
            value = metrics[request.name]
            facts.append(
                DataFact(
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
                    calculation_trace={"metric_key": request.name, "code_hash": payload.get("code_hash")},
                    derived_from=request.derived_from,
                )
            )
    return facts


def _facts_from_forecast(payload: dict, requests: list[DataFactRequest]) -> list[DataFact]:
    forecast_id = str(payload.get("forecast_id") or "")
    points = payload.get("forecast_points") if isinstance(payload.get("forecast_points"), list) else []
    return [
        DataFact(
            fact_id=_fact_id(forecast_id, "forecast_coverage", len(points)),
            name="forecast_coverage",
            fact_type="forecast",
            statement=f"Forecast returned {len(points)} points with status {payload.get('status')}.",
            value={"point_count": len(points), "status": payload.get("status")},
            method="forecast",
            evidence_refs=[FactEvidenceRef(source_type="forecast", source_id=forecast_id)],
            status="verified" if points else "partial",
            quality_flags=["requires_rolling"] if payload.get("status") == "requires_rolling" else [],
        )
    ]


def _facts_from_anomaly(payload: dict, requests: list[DataFactRequest]) -> list[DataFact]:
    anomaly_id = str(payload.get("anomaly_id") or "")
    points = payload.get("anomaly_points") if isinstance(payload.get("anomaly_points"), list) else []
    return [
        DataFact(
            fact_id=_fact_id(anomaly_id, "anomaly_count", len(points)),
            name="anomaly_count",
            fact_type="anomaly",
            statement=f"Anomaly detection returned {len(points)} anomaly points.",
            value={"anomaly_count": len(points)},
            method="anomaly",
            evidence_refs=[FactEvidenceRef(source_type="anomaly", source_id=anomaly_id)],
        )
    ]


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
    if tool_name in {"sql_query", "query_database"}:
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
    elif tool_name == "forecast" and getattr(request_state, "latest_forecast", None) is not None:
        forecast_id = str(payload.get("forecast_id") or "")
        if request_state.latest_forecast.forecast_id == forecast_id:
            request_state.latest_forecast = request_state.latest_forecast.model_copy(update=updates)
            request_state.forecast_artifacts[forecast_id] = request_state.latest_forecast
    elif tool_name == "anomaly" and getattr(request_state, "latest_anomaly", None) is not None:
        anomaly_id = str(payload.get("anomaly_id") or "")
        if request_state.latest_anomaly.anomaly_id == anomaly_id:
            request_state.latest_anomaly = request_state.latest_anomaly.model_copy(update=updates)
            request_state.anomaly_artifacts[anomaly_id] = request_state.latest_anomaly


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


def _first_numeric_key(rows: list[dict], candidates: list[str]) -> str | None:
    keys = list(dict.fromkeys([*candidates, *(key for row in rows[:20] for key in row.keys())]))
    for key in keys:
        if any(_number(row.get(key)) is not None for row in rows[:20]):
            return key
    return None


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
