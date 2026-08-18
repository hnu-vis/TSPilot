"""Runtime action contract validation."""
from __future__ import annotations

import json
import re

from core.completion import _requires_code_analysis, evaluate_goal_completion, latest_gap_assessment
from core.harness import build_action_space, build_observation_frame
from core.harness.action_space import VALID_ACTIONS
from core.harness.observation import state_capabilities
from runtime.output_selection import select_outputs_for_action
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation

TERMINAL_ACTIONS = {"terminate"}


def validate_action(
    request_state: RequestStateModel,
    action_name: str,
    action_input: dict | None = None,
    *,
    action_reason: str | None = None,
) -> tuple[bool, str | None]:
    if action_name not in VALID_ACTIONS:
        return False, f"Action '{action_name}' is not part of the runtime contract."
    capabilities = _effective_capabilities(request_state)
    if action_name == "rag" and request_state.database_context is not None and not ({"rag", "external_knowledge"} & capabilities):
        return (
            False,
            "RAG is not required by the structured intent profile for this database task. "
            "Use database evidence, analysis, anomaly, forecast, or terminate with grounded caveats.",
        )
    if action_name == "skill" and "skill" not in capabilities:
        return (
            False,
            "Skill is not required by the structured task contract or capability profile. "
            "Use database evidence, analysis, anomaly, forecast, or terminate with grounded caveats.",
        )
    if action_name == "sql_query":
        boundary_reason = _outer_sql_query_boundary_violation(request_state, action_input)
        if boundary_reason:
            return False, boundary_reason
    if action_name == "todowrite":
        boundary_reason = _outer_todo_boundary_violation(action_input)
        if boundary_reason:
            return False, boundary_reason
        if request_state.todo_list:
            return (
                False,
                "A todo plan already exists. Runtime advances plan status after successful actions; choose the next analysis action or terminate.",
            )
        contract_reason = _initial_todo_contract_violation(request_state, action_input)
        if contract_reason:
            return False, contract_reason
        if _is_initial_todowrite_action(request_state):
            return True, None
        if request_state.tool_history or request_state.observations:
            return (
                False,
                "todowrite is only valid before evidence or analysis work starts. "
                "Continue with the current evidence gap or terminate if the answer is covered.",
            )
    constraints = runtime_action_constraints(request_state)
    required_actions = {
        str(item.get("action") or "").strip()
        for item in constraints.get("required_actions", [])
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    }
    prohibited_actions = {
        str(item).strip()
        for item in constraints.get("prohibited_actions", [])
        if str(item).strip()
    }
    if action_name in prohibited_actions and action_name not in TERMINAL_ACTIONS:
        return False, constraints.get("reason") or f"Action '{action_name}' is not allowed in the current state."
    if required_actions and action_name not in required_actions and action_name not in TERMINAL_ACTIONS:
        return (
            False,
            (constraints.get("reason") or "The current state requires a different next action.")
            + f" Required actions: {sorted(required_actions)}.",
        )
    covered_repeat_reason = _covered_action_repeat_reason(
        request_state,
        action_name,
        required_actions,
        action_input=action_input,
    )
    if covered_repeat_reason:
        return False, covered_repeat_reason
    if action_name != "todowrite" and _requires_initial_todo_plan(request_state):
        return (
            False,
            "This request has an explicit multi-deliverable list. Create the initial todo plan with todowrite before querying or answering.",
        )
    repeat_failure_reason = _repeated_failed_action_without_strategy_change(
        request_state,
        action_name,
        action_reason=action_reason,
    )
    if repeat_failure_reason:
        return False, repeat_failure_reason
    if action_name in TERMINAL_ACTIONS:
        terminal_reason = _terminal_boundary_violation(request_state, action_input)
        if terminal_reason:
            return False, terminal_reason
        return True, None
    return True, None


def _outer_sql_query_boundary_violation(
    request_state: RequestStateModel,
    action_input: dict | None,
) -> str | None:
    if not isinstance(action_input, dict):
        return None
    constraints = request_state.constraints if isinstance(request_state.constraints, dict) else {}
    if constraints.get("allow_outer_explicit_query") is True:
        return None
    exposed_query_fields = [
        key for key in ("query", "message|query", "query_language")
        if key in action_input and action_input.get(key) not in (None, "", [], {})
    ]
    if not exposed_query_fields:
        return None
    return (
        "Outer ReAct SQL boundary violation: data_agent must not write SQL/Flux/PromQL or pass query-language fields. "
        "Call sql_query with natural-language action_input.message and optional purpose only; "
        "schema linking, query generation, dialect handling, and repair are internal to sql_query. "
        f"Remove fields: {exposed_query_fields}."
    )


def _outer_todo_boundary_violation(action_input: dict | None) -> str | None:
    if not isinstance(action_input, dict):
        return None
    if not _contains_database_query_code(action_input):
        return None
    return (
        "Outer ReAct boundary violation: todowrite must describe user-facing plan steps in natural language. "
        "Do not place database query code, dialect syntax, schema-linking details, or repair instructions inside todo content."
    )


def _initial_todo_contract_violation(
    request_state: RequestStateModel,
    action_input: dict | None,
) -> str | None:
    return None


def _contains_database_query_code(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_database_query_code(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_database_query_code(item) for item in value)
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    patterns = (
        r"\bselect\s+.+\bfrom\b",
        r"\bwith\s+\w+\s+as\s*\(",
        r"\bfrom\s*\(",
        r"\|\s*>",
        r"\brange\s*\(",
        r"\bfilter\s*\(",
        r"\baggregatewindow\s*\(",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _terminal_boundary_violation(
    request_state: RequestStateModel,
    action_input: dict | None = None,
) -> str | None:
    """Return a minimal DB-GPT-style terminal guard violation.

    Terminate should end the ReAct loop once useful tool observations exist. It
    should not prove full semantic coverage of every model-authored contract
    output; that made finalization brittle. The hard boundary here is only:
    database-backed answers need at least one evidence/artifact, and explicitly
    required specialized capabilities need their corresponding artifact.
    """

    if request_state.database_context is None:
        return None
    missing = _missing_explicit_terminal_capabilities(request_state)
    if missing:
        if _repair_retry_exhausted(request_state) and _terminal_input_covers_items(action_input, missing):
            return None
        if _latest_database_evidence_is_empty(request_state) and _terminal_input_explains_unavailable_outputs(request_state, action_input):
            return None
        return (
            "Final answer is blocked because an explicitly requested tool output is missing: "
            + ", ".join(missing)
            + "."
        )
    if not _available_artifact_refs(request_state):
        return (
            "Final answer is blocked because no database-backed observation or artifact is available yet. "
            "Call sql_query or another evidence-producing tool first."
        )
    return None


def _repair_retry_exhausted(request_state: RequestStateModel) -> bool:
    latest = request_state.observations[-1] if request_state.observations else None
    if latest is None or latest.success:
        return False
    payload = latest.payload if isinstance(latest.payload, dict) else {}
    validation_failure = payload.get("validation_failure") if isinstance(payload.get("validation_failure"), dict) else {}
    retry_policy = validation_failure.get("retry_policy") if isinstance(validation_failure.get("retry_policy"), dict) else {}
    repeated = int(payload.get("repeated_failure_count") or 1)
    max_retries = int(retry_policy.get("max_equivalent_retries") or 2)
    return retry_policy.get("terminal_after_exhausted") is True and repeated > max_retries


def _terminal_input_covers_items(action_input: dict | None, items: list[str]) -> bool:
    if not isinstance(action_input, dict) or not str(action_input.get("unavailable_reason") or "").strip():
        return False
    unavailable = {
        str(item).strip()
        for item in action_input.get("unavailable_outputs", [])
        if str(item).strip()
    }
    return bool(unavailable) and all(item in unavailable for item in items)


def _missing_explicit_terminal_capabilities(request_state: RequestStateModel) -> list[str]:
    capabilities = _effective_capabilities(request_state)
    missing: list[str] = []
    if "forecast" in capabilities and not _latest_forecast_is_usable(request_state):
        missing.append("forecast")
    if "anomaly" in capabilities and request_state.latest_anomaly is None:
        missing.append("anomaly")
    if (
        "analysis" in capabilities
        and request_state.latest_analysis_id is None
        and _requires_code_analysis(request_state)
    ):
        missing.append("analysis")
    if "visualization" in capabilities and not request_state.visualizations:
        missing.append("visualization")
    if "query" in capabilities or "database_evidence" in capabilities or "database" in capabilities:
        if request_state.latest_database_evidence is None or _latest_database_evidence_is_empty(request_state):
            missing.append("database_evidence")
    return missing


def _repeated_failed_action_without_strategy_change(
    request_state: RequestStateModel,
    action_name: str,
    *,
    action_reason: str | None,
) -> str | None:
    if action_name in TERMINAL_ACTIONS or action_name == "todowrite":
        return None
    failures: list[ToolObservation] = []
    for observation in reversed(request_state.observations):
        if observation.tool_name != action_name:
            break
        if observation.success:
            return None
        failures.append(observation)
        if len(failures) >= 2:
            break
    if len(failures) < 2:
        return None
    gap = latest_gap_assessment(request_state) or {}
    strategy_text = " ".join(
        str(value).strip()
        for value in (action_reason, gap.get("next_action_reason"))
        if str(value or "").strip()
    )
    if strategy_text:
        return None
    return (
        f"Action '{action_name}' has failed repeatedly. "
        "Assess the latest failure and provide a changed next_action_reason/action_reason before retrying the same tool."
    )


def _covered_action_repeat_reason(
    request_state: RequestStateModel,
    action_name: str,
    required_actions: set[str],
    *,
    action_input: dict | None,
) -> str | None:
    if action_name in TERMINAL_ACTIONS or action_name in required_actions:
        return None
    latest = request_state.observations[-1] if request_state.observations else None
    if latest is not None and not latest.success:
        return None
    refs = _available_artifact_refs(request_state)
    if action_name == "sql_query" and any(ref.startswith("evidence:") for ref in refs):
        current_signature = _query_action_signature(action_input)
        if current_signature and any(
            step.action == "sql_query"
            and step.observation is not None
            and step.observation.success
            and _query_action_signature(step.action_input) == current_signature
            for step in request_state.react_transcript
        ):
            return (
                "Equivalent query evidence already exists. Reuse the available evidence artifact "
                "and choose the next missing capability action, or terminate if the request is covered."
            )
    if action_name == "code_interpreter" and any(ref.startswith("analysis:") for ref in refs):
        if _has_successful_action(request_state, action_name):
            return "Analysis artifact already exists. Reuse it or choose the next missing capability action."
    if action_name == "forecast" and any(ref.startswith("forecast:") for ref in refs):
        return "Forecast artifact already exists. Reuse it or terminate if the request is covered."
    if action_name == "anomaly" and any(ref.startswith("anomaly:") for ref in refs):
        return "Anomaly artifact already exists. Reuse it or choose the next missing capability action."
    if action_name == "visualization" and request_state.visualizations:
        latest_visualization_iteration = _latest_successful_action_iteration(request_state, "visualization")
        latest_source_iteration = max(
            (_latest_successful_action_iteration(request_state, tool) for tool in ("sql_query", "code_interpreter", "forecast", "anomaly")),
            default=-1,
        )
        requested_refs = {
            _outer_artifact_ref(ref)
            for ref in (action_input or {}).get("source_refs", [])
            if _outer_artifact_ref(ref)
        }
        covered_refs = {
            _outer_artifact_ref(ref)
            for visualization in request_state.visualizations
            for ref in visualization.source_refs
            if _outer_artifact_ref(ref)
        }
        if latest_visualization_iteration >= latest_source_iteration and (
            not requested_refs or requested_refs.issubset(covered_refs)
        ):
            return "A current visualization artifact already covers these grounded sources. Reuse its visualization_id or terminate."
    return None


def _latest_successful_action_iteration(request_state: RequestStateModel, action_name: str) -> int:
    iterations = [
        int((output.meta or {}).get("iteration") or -1)
        for output in request_state.action_outputs
        if output.tool_name == action_name and output.success
    ]
    return max(iterations, default=-1)


def _outer_artifact_ref(ref) -> str | None:
    value = str(ref or "").strip()
    if value.startswith("view:evidence:"):
        parts = value.split(":", 3)
        return f"evidence:{parts[2]}" if len(parts) > 2 else None
    if value.startswith("view:analysis:"):
        parts = value.split(":", 3)
        return f"analysis:{parts[2]}" if len(parts) > 2 else None
    return value or None


def _query_action_signature(action_input: dict | None) -> str | None:
    if not isinstance(action_input, dict):
        return None
    database_context = action_input.get("database_context") if isinstance(action_input.get("database_context"), dict) else {}
    payload = {
        "database_id": database_context.get("database_id"),
        "query": str(action_input.get("query") or "").strip(),
        "message": " ".join(str(action_input.get("message") or "").lower().split()),
        "purpose": " ".join(str(action_input.get("purpose") or "").lower().split()),
        "time_range": action_input.get("time_range"),
        "insights": sorted(
            str(item.get("insight_key") or item.get("name") or "")
            for item in action_input.get("insight_requests", [])
            if isinstance(item, dict)
        ),
    }
    if not any(payload[key] for key in ("query", "message", "purpose", "insights")):
        return None
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _has_successful_action(request_state: RequestStateModel, action_name: str) -> bool:
    return any(
        observation.tool_name == action_name and observation.success
        for observation in request_state.observations
    )


def _terminal_input_explains_unavailable_outputs(
    request_state: RequestStateModel,
    action_input: dict | None,
) -> bool:
    gap = latest_gap_assessment(request_state)
    if not gap or not isinstance(action_input, dict):
        latest_goal = request_state.completion_state.get("latest_goal")
        if not isinstance(latest_goal, dict) or not isinstance(action_input, dict):
            return False
        blocking_items = [
            str(item).strip()
            for item in latest_goal.get("missing_evidence", [])
            if str(item).strip()
        ]
    else:
        blocking_items = _gap_blocking_items(gap)
        if not blocking_items:
            latest_goal = request_state.completion_state.get("latest_goal")
            if isinstance(latest_goal, dict):
                blocking_items = [
                    str(item).strip()
                    for item in latest_goal.get("missing_evidence", [])
                    if str(item).strip()
                ]
    if not blocking_items:
        return False
    unavailable_outputs = action_input.get("unavailable_outputs")
    unavailable_reason = str(action_input.get("unavailable_reason") or "").strip()
    if not isinstance(unavailable_outputs, list) or not unavailable_reason:
        return False
    if _latest_database_evidence_is_empty(request_state):
        return any(str(item).strip() for item in unavailable_outputs)
    unavailable_set = {
        str(item).strip()
        for item in unavailable_outputs
        if str(item).strip()
    }
    return all(item in unavailable_set for item in blocking_items)


def _latest_database_evidence_is_empty(request_state: RequestStateModel) -> bool:
    evidence = request_state.latest_database_evidence
    if evidence is None:
        return False
    data = evidence.data if isinstance(evidence.data, dict) else {}
    rows = data.get("rows")
    points = data.get("points")
    if isinstance(rows, list) and len(rows) > 0:
        return False
    if isinstance(points, list) and len(points) > 0:
        return False
    if isinstance(rows, list) or isinstance(points, list):
        return True
    series = data.get("series")
    if isinstance(series, list):
        return all(
            not isinstance(item, dict)
            or not isinstance(item.get("points"), list)
            or len(item.get("points")) == 0
            for item in series
        )
    return "no rows" in str(evidence.summary or "").lower()


def _gap_blocking_items(gap: dict) -> list[str]:
    items: list[str] = []
    values = gap.get("missing")
    if isinstance(values, list):
        items.extend(str(item).strip() for item in values if str(item).strip())
    return items


def _requires_initial_todo_plan(request_state: RequestStateModel) -> bool:
    if request_state.todo_list or request_state.database_context is None:
        return False
    profile = request_state.intent_profile if isinstance(request_state.intent_profile, dict) else {}
    if profile.get("needs_plan") is True:
        return True
    message = str(request_state.message or "")
    if _explicitly_requests_todo_plan(message):
        return True
    numbered_items = re.findall(
        r"(?:^|[\n；;。])\s*(?:\d+|[一二三四五六七八九十]+)(?:[\.\、\)]|\s+)\s*\S",
        message,
    )
    bullet_items = re.findall(r"(?:^|\n)\s*[-*]\s+\S", message)
    return len(numbered_items) + len(bullet_items) >= 3


def _is_initial_todowrite_action(request_state: RequestStateModel) -> bool:
    return (
        not request_state.todo_list
        and not request_state.tool_history
        and not request_state.observations
        and _requires_initial_todo_plan(request_state)
    )


def _explicitly_requests_todo_plan(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in (
            r"\btodo(?:\s+list)?\b",
            r"待办",
            r"任务列表",
            r"先\s*(?:写|制定|列|创建|生成|做)\s*(?:一个)?\s*(?:todo|计划|任务)",
            r"先.*(?:todo|计划|任务列表)",
            r"按(?:照)?(?:这个|该)?计划执行",
        )
    )


def runtime_action_constraints(request_state: RequestStateModel) -> dict:
    """Return runtime-owned next-action constraints derived from structured state."""

    downstream_analysis = _latest_query_requests_downstream_analysis(request_state)
    completion = evaluate_goal_completion(request_state)
    frame = build_observation_frame(
        request_state,
        requires_initial_todo_plan=_requires_initial_todo_plan(request_state),
        latest_database_evidence_empty=_latest_database_evidence_is_empty(request_state),
        downstream_analysis_request=downstream_analysis,
        pending_source_request=_latest_visualization_source_request(request_state),
        shape_recovery_request=_latest_query_shape_recovery(request_state),
        completion_missing_outputs=([] if completion.can_answer else completion.missing_evidence),
        completion_reason=completion.reason,
    )
    return build_action_space(frame).model_view()


def _latest_visualization_source_request(request_state: RequestStateModel) -> dict | None:
    observations = list(request_state.observations or [])
    for index in range(len(observations) - 1, -1, -1):
        observation = observations[index]
        if observation.tool_name != "visualization" or not observation.success:
            continue
        payload = observation.payload if isinstance(observation.payload, dict) else {}
        if str(payload.get("status") or "").strip() != "needs_sources":
            return None
        request = payload.get("required_data_request")
        if not isinstance(request, dict):
            return None
        required_action = str(request.get("required_action") or "").strip()
        fulfilled = any(
            later.success and later.tool_name == required_action
            for later in observations[index + 1:]
        )
        return None if fulfilled else request
    return None


def _effective_capabilities(request_state: RequestStateModel) -> set[str]:
    values = set(state_capabilities(request_state))
    if "rag" in values:
        values.add("external_knowledge")
    return values


def _latest_query_requests_downstream_analysis(request_state: RequestStateModel) -> dict | None:
    evidence = request_state.latest_database_evidence
    if evidence is None:
        return None
    diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
    task_coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    contract = task_coverage.get("query_task_contract") if isinstance(task_coverage.get("query_task_contract"), dict) else None
    if contract is None:
        llm_generation = diagnostics.get("llm_query_generation") if isinstance(diagnostics.get("llm_query_generation"), dict) else {}
        contract = llm_generation.get("query_task_contract") if isinstance(llm_generation.get("query_task_contract"), dict) else None
    if not isinstance(contract, dict):
        return None
    if str(contract.get("downstream_action") or "").strip().lower() != "code_interpreter":
        return None
    missing = task_coverage.get("missing") if isinstance(task_coverage.get("missing"), list) else []
    if not missing:
        missing = task_coverage.get("missing_or_uncertain") if isinstance(task_coverage.get("missing_or_uncertain"), list) else []
    selected = select_outputs_for_action(
        request_state,
        "code_interpreter",
        fallback_outputs=missing,
        query_task_contract=contract,
    )
    return selected if selected.get("required_outputs") else None


def _latest_forecast_is_usable(request_state: RequestStateModel) -> bool:
    forecast = request_state.latest_forecast
    if forecast is None:
        return False
    points = getattr(forecast, "forecast_points", None)
    return isinstance(points, list) and bool(points)


def _latest_query_shape_recovery(request_state: RequestStateModel) -> dict | None:
    latest = request_state.observations[-1] if request_state.observations else None
    if latest is None or latest.tool_name != "sql_query":
        return None
    if latest.success:
        evidence = request_state.latest_database_evidence
        diagnostics = evidence.diagnostics if evidence is not None and isinstance(evidence.diagnostics, dict) else {}
        coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
        if coverage.get("runtime_requires_followup") is not True:
            return None
        missing = [str(item).strip() for item in coverage.get("runtime_missing", []) if str(item).strip()]
        hint = str(coverage.get("next_action_hint") or "").strip()
        return {
            "message": hint or "Repair the latest query so its evidence satisfies the declared analysis contract.",
            "purpose": "Provide complete raw evidence required by downstream analytical tools.",
            "constraints": {"evidence_shape": "raw_timeseries"},
            "previous_query": evidence.query if evidence is not None else None,
            "runtime_missing": missing,
        }
    payload = latest.payload if isinstance(latest.payload, dict) else {}
    error_type = str(payload.get("error_type") or "").strip()
    if error_type != "query_shape_invalid":
        return None
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else payload
    issues = diagnostics.get("query_shape_issues") if isinstance(diagnostics.get("query_shape_issues"), list) else []
    recommended_shape = next(
        (
            str(issue.get("recommended_shape") or "").strip()
            for issue in issues
            if isinstance(issue, dict) and str(issue.get("recommended_shape") or "").strip()
        ),
        None,
    )
    if recommended_shape != "raw_series":
        return None
    return {
        "constraints": {"evidence_shape": "raw_timeseries"},
        "recommended_shape": recommended_shape,
        "recommended_downstream_action": diagnostics.get("recommended_downstream_action"),
        "query_shape_issues": issues,
    }


def build_policy_observation(
    request_state: RequestStateModel,
    action_name: str,
    reason: str,
) -> ToolObservation:
    next_constraints = runtime_action_constraints(request_state)
    prohibited = {
        str(item).strip()
        for item in next_constraints.get("prohibited_actions", [])
        if str(item).strip()
    }
    required = [
        str(item.get("action") or "").strip()
        for item in next_constraints.get("required_actions", [])
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    ]
    allowed_next_actions = required or sorted(VALID_ACTIONS - prohibited)
    latest_goal = request_state.completion_state.get("latest_goal")
    missing_capabilities = []
    if isinstance(latest_goal, dict):
        missing_capabilities = [
            str(item).strip()
            for item in latest_goal.get("missing_evidence", [])
            if str(item).strip()
        ]
    return ToolObservation(
        tool_name=action_name,
        success=False,
        summary=reason,
        payload={
            "rejected_action": action_name,
            "allowed_next_actions": allowed_next_actions,
            "next_action_constraints": next_constraints,
            "missing_capabilities": missing_capabilities or next_constraints.get("missing_outputs", []),
            "available_artifacts": _available_artifact_refs(request_state),
            "recommended_next_action": allowed_next_actions[0] if allowed_next_actions else None,
            "recovery_hint": (
                "Choose exactly one allowed next action and return one JSON object. "
                "The field next_action_constraints is authoritative: do not choose prohibited_actions, and choose required_actions when present. "
                "Do not call todowrite again when a plan already exists. "
                "Use available_artifacts and missing_capabilities to decide whether to query, analyze, run a specialized tool, or terminate."
            ),
        },
        error=reason,
        payload_truncated=False,
        payload_ref=None,
    )


def _available_artifact_refs(request_state: RequestStateModel) -> list[str]:
    refs: list[str] = []
    if request_state.latest_database_evidence is not None:
        refs.append(f"evidence:{request_state.latest_database_evidence.evidence_id}")
    if request_state.latest_analysis_id:
        refs.append(f"analysis:{request_state.latest_analysis_id}")
    if request_state.latest_anomaly is not None:
        refs.append(f"anomaly:{request_state.latest_anomaly.anomaly_id}")
    if request_state.latest_forecast is not None:
        refs.append(f"forecast:{request_state.latest_forecast.forecast_id}")
    if request_state.latest_rag is not None:
        refs.append("rag:latest")
    if request_state.latest_skill is not None:
        skill_name = request_state.latest_skill.get("skill_name") if isinstance(request_state.latest_skill, dict) else None
        refs.append(f"skill:{skill_name}" if skill_name else "skill:latest")
    refs.extend(f"visualization:{item.visualization_id}" for item in request_state.visualizations)
    return refs
