"""State-driven action guidance and validation."""
from __future__ import annotations

from dataclasses import dataclass, field

from schemas.state import RequestStateModel
from schemas.tool import ToolObservation

VALID_ACTIONS = {
    "todowrite",
    "query_database",
    "insight",
    "forecast",
    "anomaly",
    "rag",
    "skill",
    "format_answer",
}

@dataclass
class ActionPolicy:
    recommended_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    state_gap_summary: str = ""


def build_action_policy(request_state: RequestStateModel) -> ActionPolicy:
    requested_capabilities = _requested_capabilities(request_state.message)
    should_plan_first = _should_start_with_plan(request_state)

    if request_state.database_context is None:
        return ActionPolicy(
            recommended_actions=[],
            blocked_actions=["query_database", "insight", "forecast", "anomaly", "format_answer", "rag", "skill"],
            state_gap_summary="Database context is missing, so no data action can run.",
        )

    if request_state.latest_database_evidence is None:
        blocked = []
        if request_state.todo_list:
            blocked.append("todowrite")
        recommended_actions = ["todowrite"] if should_plan_first and not request_state.todo_list else ["query_database"]
        return ActionPolicy(
            recommended_actions=recommended_actions,
            blocked_actions=blocked,
            state_gap_summary=(
                "A compact plan is still needed before data retrieval."
                if should_plan_first and not request_state.todo_list
                else "No database evidence is loaded yet. The next useful step is to retrieve evidence."
            ),
        )

    evidence = request_state.latest_database_evidence
    result_type = evidence.result_type

    if result_type != "timeseries":
        return ActionPolicy(
            recommended_actions=["format_answer"],
            blocked_actions=["forecast", "anomaly"],
            state_gap_summary=(
                f"Current evidence is {result_type}. Answer from the evidence directly unless the user explicitly needs another query."
            ),
        )

    recommended: list[str] = []
    blocked: list[str] = []

    if request_state.latest_insight is None:
        recommended.append("insight")

    if "anomaly" in requested_capabilities and request_state.latest_anomaly is None:
        recommended.append("anomaly")

    if "forecast" in requested_capabilities and request_state.latest_forecast is None:
        recommended.append("forecast")

    if not recommended:
        recommended.append("format_answer")

    if request_state.tool_history and request_state.tool_history[-1].tool_name == "todowrite":
        blocked.append("todowrite")

    if request_state.latest_insight is None:
        blocked.append("format_answer")
    if "anomaly" in requested_capabilities and request_state.latest_anomaly is None:
        blocked.append("format_answer")
    if "forecast" in requested_capabilities and request_state.latest_forecast is None:
        blocked.append("format_answer")

    summary_parts = []
    if request_state.latest_insight is None:
        summary_parts.append("verified facts are still missing")
    if "anomaly" in requested_capabilities and request_state.latest_anomaly is None:
        summary_parts.append("anomaly analysis is still missing")
    if "forecast" in requested_capabilities and request_state.latest_forecast is None:
        summary_parts.append("forecast output is still missing")
    if not summary_parts:
        summary_parts.append("enough evidence-backed outputs are available to answer")

    return ActionPolicy(
        recommended_actions=_dedupe(recommended),
        blocked_actions=_dedupe(blocked),
        state_gap_summary="Current actionable gap: " + ", ".join(summary_parts) + ".",
    )


def validate_action(request_state: RequestStateModel, action_name: str) -> tuple[bool, str | None, ActionPolicy]:
    policy = build_action_policy(request_state)
    explicit_actions = _explicit_actions(request_state.message)

    if action_name not in VALID_ACTIONS:
        return False, f"Action '{action_name}' is not part of the runtime contract.", policy

    if _should_start_with_plan(request_state) and not request_state.todo_list and action_name != "todowrite":
        return False, _blocked_message(action_name, policy), policy

    if action_name in policy.blocked_actions:
        return False, _blocked_message(action_name, policy), policy

    if action_name == "todowrite" and request_state.todo_list:
        return False, _blocked_message(action_name, policy), policy

    current_todo = _current_todo(request_state)
    if current_todo is not None:
        if not _matches_current_step(action_name, current_todo):
            return False, _todo_mismatch_message(action_name, current_todo, policy), policy

    if action_name == "rag":
        if request_state.latest_database_evidence is not None or "rag" not in explicit_actions:
            return False, _blocked_message(action_name, policy), policy

    if action_name == "skill":
        if "skill" not in explicit_actions:
            return False, _blocked_message(action_name, policy), policy

    if action_name in {"forecast", "anomaly"}:
        evidence = request_state.latest_database_evidence
        if evidence is None or evidence.result_type != "timeseries":
            return False, _blocked_message(action_name, policy), policy

    if action_name == "format_answer":
        has_outputs = bool(
            request_state.verified_facts
            or request_state.latest_forecast
            or request_state.latest_anomaly
            or request_state.latest_rag
            or request_state.latest_skill
            or (
                request_state.latest_database_evidence is not None
                and request_state.latest_database_evidence.result_type != "timeseries"
            )
        )
        if not has_outputs:
            return False, _blocked_message(action_name, policy), policy

    return True, None, policy


def build_policy_observation(
    request_state: RequestStateModel,
    action_name: str,
    policy: ActionPolicy,
    reason: str,
) -> ToolObservation:
    return ToolObservation(
        tool_name=action_name,
        success=False,
        summary=reason,
        payload={
            "recommended_next_actions": policy.recommended_actions,
            "blocked_actions": policy.blocked_actions,
            "state_gap_summary": policy.state_gap_summary,
        },
        error=reason,
        payload_truncated=False,
        payload_ref=None,
    )


def _requested_capabilities(message: str) -> set[str]:
    normalized = message.lower()
    capabilities: set[str] = set()
    if any(token in normalized for token in ["forecast", "predict", "prediction", "预测", "预估"]):
        capabilities.add("forecast")
    if any(token in normalized for token in ["anomaly", "abnormal", "异常", "尖峰", "离群", "异常点", "outlier", "spike"]):
        capabilities.add("anomaly")
    return capabilities


def _should_start_with_plan(request_state: RequestStateModel) -> bool:
    if request_state.todo_list:
        return False
    normalized = request_state.message.lower()
    if any(token in normalized for token in ["规划", "计划", "步骤", "todo", "plan"]):
        return True
    if any(token in normalized for token in ["执行过程", "分析过程", "展示过程", "step by step", "show process"]):
        return True
    fact_types = list(getattr(request_state, "requested_fact_types", []) or [])
    if len(fact_types) >= 2:
        return True
    if any(token in normalized for token in ["周期", "季节性", "seasonality", "seasonal", "periodic", "cycle"]):
        return True
    return False


def _explicit_actions(message: str) -> set[str]:
    normalized = message.lower()
    actions: set[str] = set()
    if any(token in normalized for token in ["rag", "retrieve", "retrieve knowledge", "external knowledge", "知识检索"]):
        actions.add("rag")
    if any(token in normalized for token in ["skill", "workflow", "预定义技能", "预定义流程"]):
        actions.add("skill")
    return actions


def _blocked_message(action_name: str, policy: ActionPolicy) -> str:
    recommendations = ", ".join(policy.recommended_actions) or "no alternative action"
    return (
        f"Action '{action_name}' does not match the current state. "
        f"{policy.state_gap_summary} Recommended next action(s): {recommendations}."
    )


def _current_todo(request_state: RequestStateModel) -> dict | None:
    for todo in request_state.todo_list:
        if todo.get("status") == "in_progress":
            return todo
    for todo in request_state.todo_list:
        if todo.get("status") == "pending":
            return todo
    return None


def _matches_current_step(action_name: str, todo: dict) -> bool:
    task_type = str(todo.get("task_type") or "").strip().lower()
    if task_type == "generic":
        return True
    if task_type == "plan":
        return action_name == "todowrite"
    if task_type == "query":
        return action_name == "query_database"
    if task_type == "insight":
        return action_name == "insight"
    if task_type == "anomaly":
        return action_name == "anomaly"
    if task_type == "forecast":
        return action_name == "forecast"
    if task_type == "answer":
        return action_name == "format_answer"
    if task_type == "rag":
        return action_name == "rag"
    if task_type == "skill":
        return action_name == "skill"
    content = str(todo.get("content") or "").lower()
    if any(token in content for token in ["查询", "查库", "取数", "query"]):
        return action_name == "query_database"
    if any(token in content for token in ["异常", "anomaly"]):
        return action_name == "anomaly"
    if any(token in content for token in ["预测", "forecast", "predict"]):
        return action_name == "forecast"
    if any(token in content for token in ["总结", "回答", "format", "answer"]):
        return action_name == "format_answer"
    if any(token in content for token in ["洞察", "事实", "趋势", "周期", "insight", "trend", "seasonality"]):
        return action_name == "insight"
    if any(token in content for token in ["规划", "计划", "todo", "plan"]):
        return action_name == "todowrite"
    return True


def _todo_mismatch_message(
    action_name: str,
    current_todo: dict,
    policy: ActionPolicy,
) -> str:
    content = str(current_todo.get("content") or "")
    task_type = str(current_todo.get("task_type") or "generic")
    recommendations = ", ".join(policy.recommended_actions) or "follow the current plan step"
    return (
        f"Action '{action_name}' deviates from the current todo step "
        f"({task_type}: {content}). Recommended next action(s): {recommendations}."
    )


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped
