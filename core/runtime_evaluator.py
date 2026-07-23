"""LLM evaluators for runtime completion diagnostics."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.completion import CompletionEvaluation, GoalCompletionEvaluation
from schemas.output import FinalAnswer
from schemas.state import RequestStateModel


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


class GoalVerificationResult(BaseModel):
    can_answer: bool = False
    reason: str = ""
    missing_items: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    answerable_from: list[str] = Field(default_factory=list)
    next_action_hint: str | None = None
    confidence: float | None = None


@dataclass
class RuntimeLLMEvaluator:
    """Use an LLM for semantic evidence completion diagnostics."""

    llm: Any
    max_rows_preview: int = 6
    max_query_history: int = 6
    last_step_verdict: dict[str, Any] | None = None
    last_answerability_verdict: dict[str, Any] | None = None
    last_goal_verification: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

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
                "Decide whether available observations and artifacts satisfy the user's requested answer. "
                "Focus on task coverage: requested fields, filters, time range, statistics, grouping, comparisons, extrema, and required query text. "
                "Use data_completeness to distinguish complete query results from prompt previews. "
                "When data_completeness says the visible rows/points are complete for the executed query, do not call them sample-only evidence just because they appear in a preview field. "
                "Do not block just because evidence has caveats, outliers, or prompt sampling if the answer can state those caveats. "
                "Do block when the final answer would require deriving facts that no query or analysis artifact actually computed. "
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

    async def verify_final_answer(
        self,
        *,
        request_state: RequestStateModel,
        candidate_answer: FinalAnswer,
    ) -> GoalVerificationResult:
        """Semantically verify a terminal answer candidate against the user goal."""
        if self.llm is None:
            return GoalVerificationResult(
                can_answer=False,
                reason="LLM goal verifier was not available.",
                missing_items=["goal_verifier"],
                next_action_hint="Continue with more evidence or provide a datasource-specific answer only when verification is available.",
            )
        payload = {
            "message": request_state.message,
            "time_range": request_state.time_range,
            "constraints": request_state.constraints,
            "todo_list": request_state.todo_list,
            "candidate_answer": self._summarize_final_answer(candidate_answer),
            "latest_evidence": self._summarize_payload(
                request_state.latest_database_evidence.model_dump(mode="json")
                if request_state.latest_database_evidence
                else {}
            ),
            "query_history": self._query_history(request_state),
            "analysis_workspace": self._analysis_history(request_state),
            "verified_facts": [
                fact.model_dump(mode="json")
                for fact in request_state.verified_facts[:12]
            ],
            "recent_observations": self._recent_observations(request_state),
            "available_refs": self._available_refs(request_state),
            "completion_state": request_state.completion_state,
        }
        raw = await self._invoke(
            "TSPilot Goal Verification JSON",
            (
                "Judge whether the candidate final answer fully satisfies the user's task. "
                "This is semantic verification, not action-success checking. Check every explicit user deliverable, "
                "requested time range, entity/filter constraints, fields, aggregation/granularity, ordering/limits, "
                "and whether every substantive answer claim is supported by observations, evidence, facts, or analysis refs. "
                "Reject answers that infer facts from prompt previews when full-data computation was required. "
                "Accept caveated answers only when the caveat itself is supported and no requested deliverable is missing. "
                "Return JSON with can_answer, reason, missing_items, unsupported_claims, answerable_from, next_action_hint, confidence."
            ),
            payload,
        )
        verdict = self._parse(raw, GoalVerificationResult)
        self.last_goal_verification = verdict.model_dump(mode="json")
        return verdict

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

    def _analysis_history(self, request_state: RequestStateModel) -> list[dict]:
        analyses = []
        for analysis in request_state.analysis_artifacts.values():
            payload = analysis.model_dump(mode="json")
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            analyses.append(
                {
                    "analysis_id": payload.get("analysis_id"),
                    "analysis_goal": payload.get("analysis_goal"),
                    "summary": payload.get("summary"),
                    "status": payload.get("status"),
                    "input_evidence_id": payload.get("input_evidence_id"),
                    "input_row_count": payload.get("input_row_count"),
                    "result_preview": self._bounded_value(result, max_string_chars=1000, max_list_items=8, max_dict_items=12),
                }
            )
        return analyses[-8:]

    def _recent_observations(self, request_state: RequestStateModel) -> list[dict]:
        observations = []
        for observation in request_state.observations[-6:]:
            observations.append(
                {
                    "tool_name": observation.tool_name,
                    "success": observation.success,
                    "summary": observation.summary,
                    "error": observation.error,
                    "payload": self._bounded_value(observation.payload, max_string_chars=800, max_list_items=6, max_dict_items=12),
                }
            )
        return observations

    def _available_refs(self, request_state: RequestStateModel) -> list[str]:
        refs = []
        refs.extend(f"evidence:{evidence_id}" for evidence_id in request_state.database_evidence_artifacts)
        refs.extend(f"analysis:{analysis_id}" for analysis_id in request_state.analysis_artifacts)
        refs.extend(f"insight:{insight_id}" for insight_id in request_state.insight_artifacts)
        refs.extend(f"forecast:{forecast_id}" for forecast_id in request_state.forecast_artifacts)
        refs.extend(f"anomaly:{anomaly_id}" for anomaly_id in request_state.anomaly_artifacts)
        if request_state.latest_rag:
            refs.append("rag:latest")
        if request_state.latest_skill:
            refs.append(f"skill:{request_state.latest_skill.get('skill_name', 'latest')}")
        return refs

    def _summarize_final_answer(self, answer: FinalAnswer) -> dict:
        payload = answer.model_dump(mode="json")
        return {
            "title": payload.get("title"),
            "summary": payload.get("summary"),
            "sections": [
                {
                    "section_type": section.get("section_type"),
                    "heading": section.get("heading"),
                    "content": self._bounded_value(section.get("content"), max_string_chars=1000),
                    "structured_payload": self._bounded_value(section.get("structured_payload") or {}, max_string_chars=600, max_list_items=8, max_dict_items=12),
                }
                for section in payload.get("sections", [])[:8]
                if isinstance(section, dict)
            ],
            "references": [
                {
                    "source_type": ref.get("source_type"),
                    "source_id": ref.get("source_id"),
                    "label": ref.get("label"),
                }
                for ref in payload.get("references", [])[:16]
                if isinstance(ref, dict)
            ],
            "visualization_count": len(payload.get("visualizations") or []),
        }

    def _summarize_payload(self, payload: dict) -> dict:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        points = data.get("points") if isinstance(data.get("points"), list) else []
        summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
        prompt_sampling = diagnostics.get("prompt_sampling") if isinstance(diagnostics.get("prompt_sampling"), dict) else {}
        row_count = summary_stats.get("rows_count") if summary_stats else payload.get("row_count")
        point_count = summary_stats.get("points_count") if summary_stats else payload.get("point_count")
        series_count = summary_stats.get("series_count") if summary_stats else payload.get("series_count")
        sampled_for_prompt = bool(prompt_sampling.get("sampled_for_prompt"))
        is_full_fidelity = diagnostics.get("is_full_fidelity")
        if is_full_fidelity is None:
            is_full_fidelity = not sampled_for_prompt and (
                (not isinstance(row_count, int) or row_count == len(rows))
                and (not isinstance(point_count, int) or point_count == len(points))
            )
        visible_row_count = len(rows)
        visible_point_count = len(points)
        return {
            "evidence_id": payload.get("evidence_id"),
            "result_type": payload.get("result_type"),
            "summary": payload.get("summary"),
            "query_language": payload.get("query_language"),
            "query": payload.get("query"),
            "columns": payload.get("columns") or [],
            "row_count": row_count,
            "point_count": point_count,
            "series_count": series_count,
            "result_rows_preview": rows[: self.max_rows_preview],
            "result_points_preview": points[: self.max_rows_preview],
            "data_completeness": {
                "is_full_fidelity": bool(is_full_fidelity),
                "sampled_for_prompt": sampled_for_prompt,
                "visible_row_count": visible_row_count,
                "visible_point_count": visible_point_count,
                "full_row_count": row_count,
                "full_point_count": point_count,
                "full_series_count": series_count,
                "preview_contains_all_visible_rows": visible_row_count <= self.max_rows_preview,
                "preview_contains_all_visible_points": visible_point_count <= self.max_rows_preview,
                "full_artifact_ref": prompt_sampling.get("full_artifact_ref") or diagnostics.get("artifact_ref"),
            },
            "metadata": payload.get("metadata") or {},
            "diagnostics": {
                "summary_stats": summary_stats,
                "prompt_sampling": prompt_sampling or None,
                "llm_query_generation": diagnostics.get("llm_query_generation"),
                "schema_linking_generation": diagnostics.get("schema_linking_generation"),
            },
        }

    def _bounded_value(
        self,
        value,
        *,
        max_string_chars: int = 800,
        max_list_items: int = 8,
        max_dict_items: int = 12,
    ):
        if isinstance(value, str):
            if len(value) <= max_string_chars:
                return value
            return value[:max_string_chars] + f"... [truncated {len(value) - max_string_chars} chars]"
        if isinstance(value, list):
            items = [
                self._bounded_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                )
                for item in value[:max_list_items]
            ]
            if len(value) > max_list_items:
                items.append({"truncated_items": len(value) - max_list_items})
            return items
        if isinstance(value, dict):
            bounded = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_dict_items:
                    bounded["truncated_keys"] = len(value) - max_dict_items
                    break
                bounded[key] = self._bounded_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                )
            return bounded
        return value
