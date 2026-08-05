"""Capability registry for the outer ReAct harness.

This module centralizes action/tool capability metadata. Runtime code should
read this registry instead of scattering tool-name branches through prompts,
policy, and state transition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class ActionParameterCard:
    action: str
    use_when: str
    parameters: tuple[str, ...] = ()
    input_defaults: dict = field(default_factory=dict)

    def model_view(self) -> dict:
        return {
            "action": self.action,
            "use_when": self.use_when,
            "parameters": list(self.parameters),
        }


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    required_actions: tuple[str, ...] = ()
    produced_artifact_kinds: tuple[str, ...] = ()
    evidence_kinds: tuple[str, ...] = ()
    terminal_output_names: tuple[str, ...] = ()
    card: ActionParameterCard | None = None
    aliases: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    prerequisite_actions: tuple[str, ...] = ()


class CapabilityRegistry:
    def __init__(self, specs: Iterable[CapabilitySpec]):
        self._specs = {spec.capability_id: spec for spec in specs}
        self._aliases: dict[str, str] = {}
        for spec in specs:
            self._aliases[spec.capability_id] = spec.capability_id
            for alias in spec.aliases:
                self._aliases[alias] = spec.capability_id
            for task_type in spec.task_types:
                self._aliases[task_type] = spec.capability_id

    def get(self, capability_id: str) -> CapabilitySpec | None:
        normalized = self.normalize_id(capability_id)
        return self._specs.get(normalized)

    def normalize_id(self, value: str) -> str:
        text = str(value or "").strip().lower()
        return self._aliases.get(text, text)

    def normalize_many(self, values: Iterable[str], *, include_query: bool = False) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = self.normalize_id(value)
            if item in self._specs and item not in normalized:
                normalized.append(item)
        if include_query and "query" not in normalized:
            normalized.insert(0, "query")
        return normalized

    def action_cards(self) -> list[dict]:
        cards = [
            spec.card.model_view()
            for spec in self._specs.values()
            if spec.card is not None
        ]
        return sorted(cards, key=lambda item: _ACTION_ORDER.get(item["action"], 99))

    def actions_for_capability(self, capability_id: str) -> tuple[str, ...]:
        spec = self.get(capability_id)
        return spec.required_actions if spec is not None else ()

    def default_input_for_action(self, action_name: str) -> dict:
        for spec in self._specs.values():
            if spec.card and spec.card.action == action_name:
                return dict(spec.card.input_defaults)
        return {}

    def task_type_for_capability(self, capability_id: str) -> str:
        normalized = self.normalize_id(capability_id)
        spec = self._specs.get(normalized)
        if spec is None:
            return str(capability_id or "").strip().lower()
        if normalized == "external_knowledge":
            return "rag"
        action = spec.required_actions[0] if spec.required_actions else normalized
        if normalized == "analysis":
            return "code_interpreter"
        if normalized == "answer":
            return "answer"
        return action

    def capability_for_action(self, action_name: str) -> str | None:
        action = str(action_name or "").strip().lower()
        for spec in self._specs.values():
            if action in spec.required_actions:
                return spec.capability_id
        return None

    def action_matches_task_type(self, action_name: str, task_type: str) -> bool:
        task_capability = self.normalize_id(task_type)
        if not task_capability or task_capability == "generic":
            return True
        action_capability = self.capability_for_action(action_name)
        return action_capability == task_capability

    def action_is_prerequisite(self, action_name: str, task_type: str) -> bool:
        task_capability = self.normalize_id(task_type)
        action = str(action_name or "").strip().lower()
        spec = self.get(task_capability)
        if spec is None:
            return False
        return action in spec.prerequisite_actions

    def artifact_ref_for_payload(self, action_name: str, payload: dict) -> str | None:
        action = str(action_name or "").strip().lower()
        if action in {"sql_query", "query_database"} and payload.get("evidence_id"):
            return f"evidence:{payload['evidence_id']}"
        if action == "code_interpreter" and payload.get("analysis_id"):
            return f"analysis:{payload['analysis_id']}"
        if action == "forecast" and payload.get("forecast_id"):
            return f"forecast:{payload['forecast_id']}"
        if action == "anomaly" and payload.get("anomaly_id"):
            return f"anomaly:{payload['anomaly_id']}"
        if action == "rag":
            return "rag:latest"
        if action == "skill" and payload.get("skill_name"):
            return f"skill:{payload['skill_name']}"
        return None

    def hint_for_task_type(self, task_type: str) -> str | None:
        capability_id = self.normalize_id(task_type)
        return {
            "query": "Call sql_query with the missing filters, fields, aggregation, or time range.",
            "analysis": "Run code_interpreter over the full evidence artifact.",
            "anomaly": "Run anomaly after time-series evidence is available.",
            "forecast": "Run forecast after time-series evidence is available.",
            "external_knowledge": "Call rag only if external knowledge is required.",
            "skill": "Invoke the requested packaged skill.",
            "answer": "Terminate only after final answer verification can pass.",
        }.get(capability_id or "")


_ACTION_ORDER = {
    "todowrite": 0,
    "sql_query": 1,
    "code_interpreter": 2,
    "anomaly": 3,
    "forecast": 4,
    "rag": 5,
    "skill": 6,
    "terminate": 7,
}


def default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(
        [
            CapabilitySpec(
                capability_id="todo_plan",
                required_actions=("todowrite",),
                produced_artifact_kinds=("todo",),
                card=ActionParameterCard(
                    action="todowrite",
                    use_when="Complex task needs 3 or more independently verifiable user-visible steps and no plan exists.",
                    parameters=("message", "current_intent?", "focus?", "task_contract?", "todos", "evidence_summary?"),
                ),
                aliases=("plan", "todo"),
                task_types=("plan",),
            ),
            CapabilitySpec(
                capability_id="query",
                required_actions=("sql_query",),
                produced_artifact_kinds=("database_evidence",),
                evidence_kinds=("database",),
                card=ActionParameterCard(
                    action="sql_query",
                    use_when=(
                        "Need grounded database evidence. Describe the evidence needed in natural language; "
                        "schema linking, query generation, dialect handling, and validation happen inside the tool."
                    ),
                    parameters=("message", "purpose?", "time_range?", "constraints?", "fact_requests?"),
                ),
                aliases=("database_query", "database", "sql"),
                task_types=("query", "database", "database_evidence", "sql"),
            ),
            CapabilitySpec(
                capability_id="analysis",
                required_actions=("code_interpreter",),
                produced_artifact_kinds=("analysis",),
                evidence_kinds=("analysis",),
                terminal_output_names=("analysis",),
                card=ActionParameterCard(
                    action="code_interpreter",
                    use_when="Existing evidence needs derived metrics, statistics, ratios, windows, or user-requested custom computation.",
                    parameters=("database_evidence", "analysis_goal", "analysis_request?", "required_outputs?", "code?", "expected_result_schema?", "constraints?", "fact_requests?"),
                    input_defaults={"database_evidence": "latest", "analysis_request": {"mode": "canonical_timeseries_metrics"}},
                ),
                aliases=("derived_metric", "statistics", "statistical_summary", "calculation"),
                task_types=("analysis", "code_interpreter", "derived", "statistical", "statistics"),
                prerequisite_actions=("sql_query",),
            ),
            CapabilitySpec(
                capability_id="anomaly",
                required_actions=("anomaly",),
                produced_artifact_kinds=("anomaly",),
                evidence_kinds=("anomaly",),
                terminal_output_names=("anomaly",),
                card=ActionParameterCard(
                    action="anomaly",
                    use_when="Need anomaly/spike/outlier detection on time-series evidence.",
                    parameters=("database_evidence", "detector_name?", "series_name?", "constraints?", "fact_requests?"),
                    input_defaults={"database_evidence": "latest"},
                ),
                aliases=("outlier", "spike"),
                task_types=("anomaly",),
                prerequisite_actions=("sql_query", "code_interpreter"),
            ),
            CapabilitySpec(
                capability_id="forecast",
                required_actions=("forecast",),
                produced_artifact_kinds=("forecast",),
                evidence_kinds=("forecast",),
                terminal_output_names=("forecast",),
                card=ActionParameterCard(
                    action="forecast",
                    use_when="Need prediction/forecast on time-series evidence.",
                    parameters=("database_evidence", "horizon", "model_name?", "series_name?", "constraints?", "fact_requests?"),
                    input_defaults={"database_evidence": "latest"},
                ),
                aliases=("prediction", "predict"),
                task_types=("forecast",),
                prerequisite_actions=("sql_query", "code_interpreter", "anomaly"),
            ),
            CapabilitySpec(
                capability_id="external_knowledge",
                required_actions=("rag",),
                produced_artifact_kinds=("rag",),
                evidence_kinds=("rag",),
                card=ActionParameterCard(
                    action="rag",
                    use_when="External/local knowledge is explicitly needed beyond database evidence.",
                    parameters=("query", "filters?"),
                ),
                aliases=("rag", "knowledge", "retrieval"),
                task_types=("rag", "knowledge"),
            ),
            CapabilitySpec(
                capability_id="skill",
                required_actions=("skill",),
                produced_artifact_kinds=("skill",),
                evidence_kinds=("skill",),
                card=ActionParameterCard(
                    action="skill",
                    use_when="User explicitly asks for a named packaged workflow or skill.",
                    parameters=("skill_name", "parameters"),
                ),
                task_types=("skill",),
            ),
            CapabilitySpec(
                capability_id="answer",
                required_actions=("terminate",),
                produced_artifact_kinds=("final_answer",),
                card=ActionParameterCard(
                    action="terminate",
                    use_when="Evidence covers the request, or task cannot proceed with available context.",
                    parameters=("result? natural-language prose", "summary_goal?", "direct_answer? natural-language prose", "include_analysis_ids", "include_fact_ids", "include_visualization_ids", "section_plan", "unavailable_outputs", "unavailable_reason?"),
                ),
                aliases=("conclusion", "final"),
                task_types=("answer",),
                prerequisite_actions=("sql_query", "code_interpreter", "anomaly", "forecast"),
            ),
        ]
    )
