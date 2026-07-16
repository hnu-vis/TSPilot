"""Final answer assembly tool."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from core.report.composer import (
    build_anomaly_section,
    build_forecast_section,
    build_summary,
    missing_requirements,
    ordered_sections,
)
from schemas.output import AnswerReference, AnswerSection, FinalAnswer
from schemas.state import RequestStateModel
from tools.base import BaseTool


class FormatAnswerInput(BaseModel):
    summary_goal: str | None = None
    direct_answer: str | None = None
    include_fact_ids: list[str] = Field(default_factory=list)
    include_visualization_ids: list[str] = Field(default_factory=list)
    section_plan: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("summary_goal"):
            normalized["summary_goal"] = normalized.get("message") or "Assemble the final answer."
        return normalized


class FormatAnswerTool(BaseTool):
    async def execute(
        self,
        validated_input: FormatAnswerInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        missing = missing_requirements(request_state)
        if missing:
            raise ValueError(
                "Final answer cannot be assembled yet. Missing required outputs: "
                + ", ".join(missing)
            )
        facts = [
            fact
            for fact in request_state.verified_facts
            if not validated_input.include_fact_ids or fact.fact_id in validated_input.include_fact_ids
        ]
        summary = build_summary(
            request_state,
            facts,
            self._fallback_summary(
                request_state,
                validated_input.summary_goal,
                validated_input.direct_answer,
            ),
        )
        if request_state.database_context is None and validated_input.direct_answer:
            answer = FinalAnswer(
                title=None,
                summary=summary,
                sections=[],
                references=[],
                visualizations=[],
            )
            return answer.model_dump(mode="json")

        sections_by_type: dict[str, AnswerSection] = {}
        sections_by_type["summary"] = AnswerSection(
            section_type="summary",
            heading="Summary",
            content=summary,
            structured_payload=None,
        )
        if facts:
            sections_by_type["facts"] = AnswerSection(
                section_type="facts",
                heading="Verified Facts",
                content="\n".join(f"- {fact.statement}" for fact in facts),
                structured_payload={"fact_ids": [fact.fact_id for fact in facts]},
            )
        if request_state.latest_database_evidence is not None and not facts:
            evidence = request_state.latest_database_evidence
            for section in self._evidence_sections(evidence):
                sections_by_type[section.section_type] = section
        if request_state.todo_list and "plan" in request_state.answer_requirements:
            sections_by_type["plan"] = AnswerSection(
                section_type="plan",
                heading="Plan",
                content="\n".join(
                    f"- [{todo.get('status', 'pending')}] {todo.get('content', '')}"
                    for todo in request_state.todo_list
                ),
                structured_payload={"todo_count": len(request_state.todo_list)},
            )
        if request_state.latest_forecast is not None:
            sections_by_type["forecast"] = build_forecast_section(request_state.latest_forecast)
        if request_state.latest_anomaly is not None:
            sections_by_type["anomaly"] = build_anomaly_section(request_state.latest_anomaly)
        if request_state.latest_rag:
            sections_by_type["rag"] = AnswerSection(
                section_type="rag",
                heading="Knowledge Context",
                content=request_state.latest_rag.get("summary", ""),
                structured_payload={"result_count": len(request_state.latest_rag.get("results", []))},
            )
        if request_state.latest_skill:
            sections_by_type["skill"] = AnswerSection(
                section_type="skill",
                heading="Skill Output",
                content=request_state.latest_skill.get("summary", ""),
                structured_payload={"skill_name": request_state.latest_skill.get("skill_name")},
            )
        sections_by_type["conclusion"] = AnswerSection(
            section_type="conclusion",
            heading="Conclusion",
            content=summary,
            structured_payload={
                "has_facts": bool(facts),
                "has_anomaly": request_state.latest_anomaly is not None,
                "has_forecast": request_state.latest_forecast is not None,
            },
        )
        sections = ordered_sections(sections_by_type, validated_input.section_plan)
        references = []
        if request_state.latest_database_evidence is not None:
            references.append(
                AnswerReference(
                    source_type="query",
                    source_id=request_state.latest_database_evidence.evidence_id,
                    label="Database evidence",
                    evidence={"summary": request_state.latest_database_evidence.summary},
                )
            )
        references.extend(
            AnswerReference(
                source_type="fact",
                source_id=fact.fact_id,
                label=fact.fact_type,
                evidence=fact.evidence,
            )
            for fact in facts
        )
        if request_state.latest_forecast is not None:
            references.append(
                AnswerReference(
                    source_type="forecast",
                    source_id=request_state.latest_forecast.forecast_id,
                    label=request_state.latest_forecast.model_name,
                    evidence=request_state.latest_forecast.diagnostics,
                )
            )
        if request_state.latest_anomaly is not None:
            references.append(
                AnswerReference(
                    source_type="anomaly",
                    source_id=request_state.latest_anomaly.anomaly_id,
                    label=request_state.latest_anomaly.detector_name,
                    evidence=request_state.latest_anomaly.diagnostics,
                )
            )
        if request_state.latest_rag:
            references.append(
                AnswerReference(
                    source_type="rag",
                    source_id=request_state.latest_rag.get("results", [{}])[0].get("source_id")
                    if request_state.latest_rag.get("results")
                    else None,
                    label="Knowledge retrieval",
                    evidence={"summary": request_state.latest_rag.get("summary")},
                )
            )
        if request_state.latest_skill:
            references.append(
                AnswerReference(
                    source_type="skill",
                    source_id=request_state.latest_skill.get("skill_name"),
                    label=request_state.latest_skill.get("skill_name") or "skill",
                    evidence={"summary": request_state.latest_skill.get("summary")},
                )
            )
        visualizations = [
            visualization
            for visualization in request_state.visualizations
            if (
                not validated_input.include_visualization_ids
                or visualization.visualization_id in validated_input.include_visualization_ids
            )
        ]
        answer = FinalAnswer(
            title="TSPilot v0.2 Analysis",
            summary=summary,
            sections=sections,
            references=references,
            visualizations=visualizations,
        )
        return answer.model_dump(mode="json")

    def _build_summary(self, request_state: RequestStateModel, facts: list, summary_goal: str) -> str:
        subject = self._subject_label(request_state)
        parts: list[str] = []

        trend_fact = next((fact for fact in facts if fact.fact_type == "trend"), None)
        extrema_fact = next((fact for fact in facts if fact.fact_type in {"extreme", "extrema"}), None)

        if trend_fact is not None:
            parts.append(trend_fact.statement)
        elif facts:
            parts.append(facts[0].statement)
        elif request_state.latest_database_evidence is not None:
            parts.append(request_state.latest_database_evidence.summary)

        if extrema_fact is not None:
            parts.append(extrema_fact.statement)

        if request_state.latest_anomaly is not None:
            anomaly_points = request_state.latest_anomaly.anomaly_points
            if anomaly_points:
                preview = ", ".join(
                    f"{point.get('timestamp')}={point.get('value')}"
                    for point in anomaly_points[:3]
                )
                parts.append(
                    f"异常检测发现 {len(anomaly_points)} 个异常点，典型异常包括 {preview}。"
                )
            else:
                parts.append("异常检测未发现显著异常点。")

        if request_state.latest_forecast is not None:
            forecast_points = request_state.latest_forecast.forecast_points
            if forecast_points:
                first_point = forecast_points[0]
                last_point = forecast_points[-1]
                direction = "上升" if last_point.value > first_point.value else "下降" if last_point.value < first_point.value else "基本持平"
                parts.append(
                    f"{subject} 的短期预测共 {len(forecast_points)} 个点，预测区间内整体{direction}，"
                    f"从 {first_point.value:.2f} 变化到 {last_point.value:.2f}。"
                )

        compact = " ".join(part.strip() for part in parts if part and part.strip())
        return compact or self._fallback_summary(request_state, summary_goal)

    def _fallback_summary(
        self,
        request_state: RequestStateModel,
        summary_goal: str,
        direct_answer: str | None = None,
    ) -> str:
        if request_state.latest_database_evidence is not None:
            return request_state.latest_database_evidence.summary
        if request_state.latest_rag:
            return request_state.latest_rag.get("summary", summary_goal)
        if request_state.latest_skill:
            return request_state.latest_skill.get("summary", summary_goal)
        if direct_answer and direct_answer.strip():
            return direct_answer.strip()
        return summary_goal

    def _evidence_sections(self, evidence) -> list[AnswerSection]:
        if evidence.result_type == "statistics":
            stats = evidence.data.get("statistics", {})
            lines = [f"- {key}: {value}" for key, value in stats.items()]
            return [
                AnswerSection(
                    section_type="statistics",
                    heading="Statistics",
                    content="\n".join(lines) if lines else evidence.summary,
                    structured_payload=stats or None,
                )
            ]
        if evidence.result_type == "metric_list":
            metrics = evidence.data.get("metrics", [])
            preview = metrics[:20]
            return [
                AnswerSection(
                    section_type="metric_list",
                    heading="Available Metrics",
                    content="\n".join(f"- {metric}" for metric in preview) if preview else evidence.summary,
                    structured_payload={"metric_count": len(metrics)},
                )
            ]
        if evidence.result_type == "schema":
            tables = evidence.data.get("tables_or_measurements", [])
            preview = [table.get("name") for table in tables[:20]]
            return [
                AnswerSection(
                    section_type="schema",
                    heading="Schema Preview",
                    content="\n".join(f"- {name}" for name in preview) if preview else evidence.summary,
                    structured_payload={"table_count": len(tables)},
                )
            ]
        if evidence.result_type == "table":
            rows = evidence.data.get("rows", [])
            return [
                AnswerSection(
                    section_type="table",
                    heading="Table Result",
                    content=evidence.summary,
                    structured_payload={"row_count": len(rows), "columns": evidence.columns},
                )
            ]
        return []
