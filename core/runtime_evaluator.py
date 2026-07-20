"""LLM evaluators for runtime planning and completion contracts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.completion import CompletionEvaluation, GoalCompletionEvaluation
from schemas.state import RequestStateModel


class PlanRequirementVerdict(BaseModel):
    requires_plan: bool = False
    reason: str = ""
    deliverables: list[str] = Field(default_factory=list)
    confidence: float | None = None
    next_action_hint: str | None = None


class StepCompletionVerdict(BaseModel):
    completed: bool = False
    reason: str = ""
    missing_items: list[str] = Field(default_factory=list)
    satisfied_items: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    next_action_hint: str | None = None
    confidence: float | None = None


class AnswerabilityVerdict(BaseModel):
    can_answer: bool = False
    reason: str = ""
    missing_items: list[str] = Field(default_factory=list)
    answerable_from: list[str] = Field(default_factory=list)
    next_action_hint: str | None = None
    confidence: float | None = None


@dataclass
class RuntimeLLMEvaluator:
    """Use an LLM for semantic plan and evidence completion decisions."""

    llm: Any
    max_rows_preview: int = 6
    max_query_history: int = 6
    last_plan_verdict: dict[str, Any] | None = None
    last_step_verdict: dict[str, Any] | None = None
    last_answerability_verdict: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    async def evaluate_plan_requirement(
        self,
        *,
        request_state: RequestStateModel,
        proposed_action: str,
        action_input: dict,
    ) -> PlanRequirementVerdict:
        if self.llm is None or not self._should_check_plan_requirement(request_state.message):
            verdict = PlanRequirementVerdict(requires_plan=False, reason="Plan requirement LLM check was not needed.")
            self.last_plan_verdict = verdict.model_dump(mode="json")
            return verdict
        payload = {
            "message": request_state.message,
            "database_context": request_state.database_context.model_dump(mode="json") if request_state.database_context else None,
            "proposed_action": proposed_action,
            "action_input": action_input,
        }
        raw = await self._invoke(
            "TSPilot Plan Requirement JSON",
            (
                "Decide whether this user request needs an explicit todo plan before executing the proposed action. "
                "Require a plan only when the user asks for multiple independently verifiable deliverables, ordered tasks, "
                "or per-item query/result reporting. Return JSON with requires_plan, reason, deliverables, confidence, next_action_hint."
            ),
            payload,
        )
        verdict = self._parse(raw, PlanRequirementVerdict)
        self.last_plan_verdict = verdict.model_dump(mode="json")
        return verdict

    async def evaluate_step_completion(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        full_payload: dict,
    ) -> CompletionEvaluation:
        current = next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)
        if self.llm is None or current is None:
            return CompletionEvaluation(False, "LLM completion evaluation was not available.")
        payload = {
            "message": request_state.message,
            "current_todo": current,
            "tool_name": tool_name,
            "tool_payload": self._summarize_payload(full_payload),
            "query_history": self._query_history(request_state),
        }
        raw = await self._invoke(
            "TSPilot Step Completion JSON",
            (
                "Decide whether the latest tool evidence satisfies the current todo acceptance criteria. "
                "Judge semantic completion, not just result shape. Return JSON with completed, reason, missing_items, "
                "satisfied_items, evidence_refs, next_action_hint, confidence."
            ),
            payload,
        )
        verdict = self._parse(raw, StepCompletionVerdict)
        verdict_payload = verdict.model_dump(mode="json")
        self.last_step_verdict = verdict_payload
        return CompletionEvaluation(
            completed=verdict.completed,
            reason=verdict.reason or ("Completed by LLM verdict." if verdict.completed else "Not completed by LLM verdict."),
            missing_evidence=verdict.missing_items,
            evidence_refs=verdict.evidence_refs,
            next_action_hint=verdict.next_action_hint,
        )

    async def evaluate_answerability(self, *, request_state: RequestStateModel) -> GoalCompletionEvaluation:
        if self.llm is None:
            return GoalCompletionEvaluation(False, "LLM answerability evaluation was not available.")
        payload = {
            "message": request_state.message,
            "todo_list": request_state.todo_list,
            "latest_evidence": self._summarize_payload(
                request_state.latest_database_evidence.model_dump(mode="json")
                if request_state.latest_database_evidence
                else {}
            ),
            "query_history": self._query_history(request_state),
            "completion_state": request_state.completion_state,
        }
        raw = await self._invoke(
            "TSPilot Answerability JSON",
            (
                "Decide whether available evidence is sufficient to answer the user's request fully. "
                "Return JSON with can_answer, reason, missing_items, answerable_from, next_action_hint, confidence."
            ),
            payload,
        )
        verdict = self._parse(raw, AnswerabilityVerdict)
        self.last_answerability_verdict = verdict.model_dump(mode="json")
        return GoalCompletionEvaluation(
            can_answer=verdict.can_answer,
            reason=verdict.reason,
            missing_evidence=verdict.missing_items,
            answerable_from=verdict.answerable_from,
            next_action_hint=verdict.next_action_hint,
        )

    def _should_check_plan_requirement(self, message: str) -> bool:
        text = str(message or "")
        numbered = len(re.findall(r"(?:^|[;；。:\n])\s*\d+[.)、．]", text)) >= 2
        return bool(
            numbered
            or "完成以下任务" in text
            or "每项结果" in text
            or "每项对应" in text
            or "分别返回" in text
            or "展示执行过程" in text
            or "按步骤" in text
        )

    async def _invoke(self, marker: str, instruction: str, payload: dict) -> str:
        messages = [
            (
                "system",
                "You are TSPilot's runtime contract evaluator. Return exactly one JSON object and no markdown.",
            ),
            (
                "user",
                marker + ":\n" + json.dumps({"instruction": instruction, "context": payload}, ensure_ascii=False, default=str),
            ),
        ]
        response = await self.llm.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        return str(content)

    def _parse(self, raw: str, model):
        stripped = raw.strip()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise ValueError(f"Runtime LLM evaluator did not return JSON: {raw}")
            decoded = json.loads(match.group(0))
        try:
            return model.model_validate(decoded)
        except ValidationError as exc:
            raise ValueError(f"Runtime LLM evaluator returned invalid JSON: {exc}") from exc

    def _query_history(self, request_state: RequestStateModel) -> list[dict]:
        history = []
        for evidence in request_state.database_evidence_artifacts.values():
            history.append(self._summarize_payload(evidence.model_dump(mode="json")))
        return history[-self.max_query_history :]

    def _summarize_payload(self, payload: dict) -> dict:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        points = data.get("points") if isinstance(data.get("points"), list) else []
        summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
        return {
            "evidence_id": payload.get("evidence_id"),
            "result_type": payload.get("result_type"),
            "summary": payload.get("summary"),
            "query_language": payload.get("query_language"),
            "query": payload.get("query"),
            "columns": payload.get("columns") or [],
            "row_count": summary_stats.get("rows_count") if summary_stats else payload.get("row_count"),
            "point_count": summary_stats.get("points_count") if summary_stats else payload.get("point_count"),
            "sample_rows": rows[: self.max_rows_preview],
            "sample_points": points[: self.max_rows_preview],
            "metadata": payload.get("metadata") or {},
            "diagnostics": {
                "summary_stats": summary_stats,
                "llm_query_generation": diagnostics.get("llm_query_generation"),
            },
        }
