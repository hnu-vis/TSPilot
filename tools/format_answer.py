"""Final answer assembly tool."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.database.dialects import dialect_for_database
from core.report.composer import (
    build_anomaly_section,
    build_forecast_section,
    build_summary,
    forecast_point_bounds,
    ordered_sections,
)
from schemas.data_fact import DataFact
from schemas.output import AnswerReference, AnswerSection, FinalAnswer
from schemas.state import RequestStateModel
from tools.base import BaseTool


class FormatAnswerInput(BaseModel):
    summary_goal: str | None = None
    direct_answer: str | None = None
    include_analysis_ids: list[str] = Field(default_factory=list)
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
    _LABELS = {
        "zh": {
            "summary": "摘要",
            "facts": "计算依据",
            "analysis": "分析",
            "plan": "计划",
            "knowledge_context": "知识上下文",
            "skill_output": "技能输出",
            "conclusion": "结论",
            "database_evidence": "数据库证据",
            "knowledge_retrieval": "知识检索",
            "statistics": "统计结果",
            "available_metrics": "可用指标",
            "schema_preview": "Schema 预览",
            "table_result": "表格结果",
            "query_results": "查询结果",
            "query": "查询",
            "metrics": "指标",
        },
        "en": {
            "summary": "Summary",
            "facts": "Evidence",
            "analysis": "Analysis",
            "plan": "Plan",
            "knowledge_context": "Knowledge Context",
            "skill_output": "Skill Output",
            "conclusion": "Conclusion",
            "database_evidence": "Database evidence",
            "knowledge_retrieval": "Knowledge retrieval",
            "statistics": "Statistics",
            "available_metrics": "Available Metrics",
            "schema_preview": "Schema Preview",
            "table_result": "Table Result",
            "query_results": "Query Results",
            "query": "Query",
            "metrics": "Metrics",
        },
    }

    async def execute(
        self,
        validated_input: FormatAnswerInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        selected_fact_ids = set(self._resource_ids(validated_input.include_fact_ids, "fact"))
        data_facts = [
            fact
            for fact in request_state.fact_set.facts
            if (
                fact.status == "verified"
                and fact.fact_id in selected_fact_ids
            )
        ]
        unavailable_facts = [
            fact
            for fact in request_state.fact_set.facts
            if (
                fact.status == "unavailable"
                and fact.fact_id in selected_fact_ids
            )
        ]
        analyses = self._selected_analyses(request_state, validated_input.include_analysis_ids)
        fallback_summary = self._fallback_summary(
            request_state,
            validated_input.summary_goal,
            validated_input.direct_answer,
        )
        direct_answer = self._usable_direct_answer(validated_input.direct_answer)
        if direct_answer:
            summary = direct_answer
        elif analyses:
            analysis_summary = " ".join(analysis.summary.strip() for analysis in analyses if analysis.summary.strip())
            summary = build_summary(
                request_state,
                [],
                analysis_summary or fallback_summary,
                prefer_fallback=bool(analysis_summary),
            )
        elif data_facts:
            summary = self._data_fact_summary(data_facts, fallback_summary)
        else:
            summary = build_summary(
                request_state,
                [],
                fallback_summary,
            )
        if request_state.database_context is None and direct_answer:
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
            heading=self._label(request_state, "summary"),
            content=summary,
            structured_payload=None,
        )
        if data_facts or unavailable_facts:
            sections_by_type["facts"] = AnswerSection(
                section_type="facts",
                heading=self._label(request_state, "facts"),
                content=self._render_data_fact_section(data_facts, unavailable_facts),
                structured_payload={
                    "fact_ids": [fact.fact_id for fact in data_facts],
                    "unavailable_fact_ids": [fact.fact_id for fact in unavailable_facts],
                },
            )
        if analyses:
            sections_by_type["analysis"] = AnswerSection(
                section_type="analysis",
                heading=self._label(request_state, "analysis"),
                content=self._render_analysis_section(analyses, request_state),
                structured_payload={
                    "analysis_ids": [analysis.analysis_id for analysis in analyses],
                    "metrics": [
                        metrics
                        for analysis in analyses
                        if isinstance(analysis.result, dict)
                        and isinstance((metrics := analysis.result.get("metrics")), dict)
                        and metrics
                    ],
                    "results": [analysis.result for analysis in analyses],
                },
            )
        database_evidence = self._database_evidence_inventory(request_state)
        if len(database_evidence) > 1:
            sections_by_type["query_results"] = self._database_query_results_section(database_evidence, request_state)
        if (
            len(database_evidence) <= 1
            and request_state.latest_database_evidence is not None
            and not data_facts
        ):
            evidence = request_state.latest_database_evidence
            for section in self._evidence_sections(evidence, request_state):
                sections_by_type[section.section_type] = section
        if request_state.todo_list and "plan" in validated_input.section_plan:
            sections_by_type["plan"] = AnswerSection(
                section_type="plan",
                heading=self._label(request_state, "plan"),
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
                heading=self._label(request_state, "knowledge_context"),
                content=request_state.latest_rag.get("summary", ""),
                structured_payload={"result_count": len(request_state.latest_rag.get("results", []))},
            )
        if request_state.latest_skill:
            sections_by_type["skill"] = AnswerSection(
                section_type="skill",
                heading=self._label(request_state, "skill_output"),
                content=request_state.latest_skill.get("summary", ""),
                structured_payload={"skill_name": request_state.latest_skill.get("skill_name")},
            )
        sections_by_type["conclusion"] = AnswerSection(
            section_type="conclusion",
            heading=self._label(request_state, "conclusion"),
            content=summary,
            structured_payload={
                "has_facts": bool(data_facts),
                "has_analysis": bool(analyses),
                "has_anomaly": request_state.latest_anomaly is not None,
                "has_forecast": request_state.latest_forecast is not None,
            },
        )
        sections = ordered_sections(sections_by_type, validated_input.section_plan)
        references = []
        for evidence in database_evidence:
            references.append(
                AnswerReference(
                    source_type="query",
                    source_id=evidence.evidence_id,
                    label=self._evidence_purpose(evidence) or self._label(request_state, "database_evidence"),
                    evidence=self._database_reference_payload(evidence),
                )
            )
        if selected_fact_ids:
            references.extend(
                AnswerReference(
                    source_type="fact",
                    source_id=fact.fact_id,
                    label=fact.fact_type,
                    evidence=self._data_fact_reference_payload(fact),
                )
                for fact in [*data_facts, *unavailable_facts]
            )
        references.extend(
            AnswerReference(
                source_type="analysis",
                source_id=analysis.analysis_id,
                label=analysis.analysis_goal,
                evidence={
                    "summary": analysis.summary,
                    "result": analysis.result,
                    "input_evidence_id": analysis.input_evidence_id,
                    "input_row_count": analysis.input_row_count,
                    "code_hash": analysis.code_hash,
                    "code_type": analysis.code_type,
                },
            )
            for analysis in analyses
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
                    label=self._label(request_state, "knowledge_retrieval"),
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
            title=None,
            summary=summary,
            sections=sections,
            references=references,
            visualizations=visualizations,
        )
        return answer.model_dump(mode="json")

    def _data_fact_summary(self, facts: list[DataFact], fallback_summary: str) -> str:
        statements = [fact.statement.strip() for fact in facts if fact.statement.strip()]
        return " ".join(statements[:3]) or fallback_summary

    def _render_data_fact_section(self, facts: list[DataFact], unavailable_facts: list[DataFact]) -> str:
        lines = [f"- {fact.statement}" for fact in facts if fact.statement]
        lines.extend(
            f"- {fact.statement}"
            for fact in unavailable_facts
            if fact.statement
        )
        return "\n".join(lines)

    def _data_fact_reference_payload(self, fact: DataFact) -> dict:
        return {
            "name": fact.name,
            "status": fact.status,
            "value": fact.value,
            "method": fact.method,
            "evidence_refs": [ref.model_dump(mode="json") for ref in fact.evidence_refs],
            "calculation_trace": fact.calculation_trace,
            "quality_flags": fact.quality_flags,
            "unavailable_reason": fact.unavailable_reason,
            "derived_from": fact.derived_from,
        }

    def _build_summary(self, request_state: RequestStateModel, facts: list, summary_goal: str) -> str:
        subject = self._subject_label(request_state)
        parts: list[str] = []
        language = self._language(request_state)

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
                if language == "zh":
                    parts.append(f"异常检测发现 {len(anomaly_points)} 个异常点，典型异常包括 {preview}。")
                else:
                    parts.append(f"Anomaly detection found {len(anomaly_points)} anomalies, including {preview}.")
            else:
                parts.append("异常检测未发现显著异常点。" if language == "zh" else "Anomaly detection found no significant anomalies.")

        if request_state.latest_forecast is not None:
            forecast_points = request_state.latest_forecast.forecast_points
            if forecast_points:
                first_point, last_point, point_count = forecast_point_bounds(request_state.latest_forecast)
                if language == "zh":
                    direction = "上升" if last_point.value > first_point.value else "下降" if last_point.value < first_point.value else "基本持平"
                    parts.append(
                        f"{subject} 的短期预测共 {point_count} 个点，预测区间内整体{direction}，"
                        f"从 {first_point.value:.2f} 变化到 {last_point.value:.2f}。"
                    )
                else:
                    direction = "rises" if last_point.value > first_point.value else "falls" if last_point.value < first_point.value else "stays roughly flat"
                    parts.append(
                        f"The short-term forecast for {subject} contains {point_count} points and overall {direction}, "
                        f"moving from {first_point.value:.2f} to {last_point.value:.2f}."
                    )
            elif request_state.latest_forecast.status == "requires_rolling":
                plan = request_state.latest_forecast.forecast_plan
                requested = plan.requested_steps if plan else request_state.latest_forecast.horizon
                chunk = plan.recommended_chunk_steps if plan else None
                if language == "zh":
                    suffix = f"建议每次最多 {chunk} 步" if chunk else "需要分段执行"
                    parts.append(f"{subject} 的预测跨度为 {requested} 个时间步，超过单次直接预测窗口，{suffix}。")
                else:
                    suffix = f"recommended chunk size is {chunk} steps" if chunk else "chunked execution is required"
                    parts.append(
                        f"The forecast horizon for {subject} is {requested} steps, which exceeds the direct forecast window; {suffix}."
                    )

        compact = " ".join(part.strip() for part in parts if part and part.strip())
        return compact or self._fallback_summary(request_state, summary_goal)

    def _fallback_summary(
        self,
        request_state: RequestStateModel,
        summary_goal: str,
        direct_answer: str | None = None,
    ) -> str:
        usable_direct_answer = self._usable_direct_answer(direct_answer)
        if usable_direct_answer:
            return usable_direct_answer
        if request_state.latest_database_evidence is not None:
            return request_state.latest_database_evidence.summary
        if request_state.latest_rag:
            return request_state.latest_rag.get("summary", summary_goal)
        if request_state.latest_skill:
            return request_state.latest_skill.get("summary", summary_goal)
        return summary_goal

    def _subject_label(self, request_state: RequestStateModel) -> str:
        evidence = request_state.latest_database_evidence
        if evidence is not None:
            metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
            for value in (
                metadata.get("series_name"),
                metadata.get("metric"),
                metadata.get("measurement"),
                evidence.database,
            ):
                text = str(value or "").strip()
                if text:
                    return text
        return "该序列" if self._language(request_state) == "zh" else "the series"

    def _has_explicit_sql_query_evidence(self, request_state: RequestStateModel) -> bool:
        evidence = request_state.latest_database_evidence
        return bool(
            evidence is not None
            and isinstance(evidence.metadata, dict)
            and evidence.metadata.get("sql_query_mode") == "explicit"
        )

    def _has_database_answer_evidence(self, request_state: RequestStateModel) -> bool:
        evidence = request_state.latest_database_evidence
        return evidence is not None

    def _has_requested_facts(self, request_state: RequestStateModel, include_fact_ids: list[str]) -> bool:
        if not include_fact_ids:
            return False
        available = {fact.fact_id for fact in request_state.fact_set.facts if fact.status == "verified"}
        return all(fact_id in available for fact_id in include_fact_ids)

    def _has_requested_analyses(self, request_state: RequestStateModel, include_analysis_ids: list[str]) -> bool:
        if not include_analysis_ids:
            return False
        normalized_ids = self._resource_ids(include_analysis_ids, "analysis")
        return all(analysis_id in request_state.analysis_artifacts for analysis_id in normalized_ids)

    def _selected_analyses(self, request_state: RequestStateModel, include_analysis_ids: list[str]):
        if include_analysis_ids:
            return [
                request_state.analysis_artifacts[analysis_id]
                for analysis_id in self._resource_ids(include_analysis_ids, "analysis")
                if analysis_id in request_state.analysis_artifacts
            ]
        return list(request_state.analysis_artifacts.values())

    def _resource_ids(self, values: list[str], resource_type: str) -> list[str]:
        normalized = []
        prefix = f"{resource_type}:"
        for value in values:
            resource_id = str(value or "").strip()
            if resource_id.startswith(prefix):
                resource_id = resource_id.split(":", 1)[1].strip()
            if resource_id and resource_id not in normalized:
                normalized.append(resource_id)
        return normalized

    def _render_analysis_section(self, analyses: list, request_state: RequestStateModel) -> str:
        blocks = []
        for analysis in analyses:
            lines = [f"- {analysis.summary}"]
            transparency_lines = self._render_analysis_transparency_details(analysis)
            if transparency_lines:
                lines.extend(f"  {line}" for line in transparency_lines)
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def _render_analysis_transparency_details(self, analysis) -> list[str]:
        result = analysis.result if isinstance(analysis.result, dict) else {}
        details = result.get("details")
        if not isinstance(details, dict):
            return []
        transparency_keys = {
            "outlier_rule",
            "threshold_or_formula",
            "rationale",
            "excluded_rows",
            "raw_metrics",
            "adjusted_metrics",
        }
        if not transparency_keys & set(details):
            return []

        lines: list[str] = []
        for key, label in (
            ("outlier_rule", "outlier_rule"),
            ("threshold_or_formula", "threshold_or_formula"),
            ("rationale", "rationale"),
        ):
            value = details.get(key)
            if value is not None and str(value).strip():
                lines.append(f"{label}: {value}")

        excluded_rows = details.get("excluded_rows")
        if isinstance(excluded_rows, list):
            lines.append(f"excluded_rows: {len(excluded_rows)}")

        for key in ("raw_metrics", "adjusted_metrics"):
            value = details.get(key)
            if isinstance(value, dict) and value:
                metric_text = ", ".join(
                    f"{metric_key}: {metric_value}"
                    for metric_key, metric_value in value.items()
                    if metric_value is not None
                )
                if metric_text:
                    lines.append(f"{key}: {metric_text}")
        return lines

    def _usable_direct_answer(self, direct_answer: str | None) -> str | None:
        if not direct_answer or not direct_answer.strip():
            return None
        normalized = direct_answer.strip()
        lowered = normalized.lower()
        blocked_phrases = (
            "格式不符合要求",
            "单一json对象",
            "json object",
            "contract violation",
            "parser error",
            "repair instruction",
        )
        if any(phrase in lowered for phrase in blocked_phrases):
            return None
        return normalized

    def _evidence_sections(self, evidence, request_state: RequestStateModel) -> list[AnswerSection]:
        sections = self._query_sections(evidence, request_state)
        if evidence.result_type == "statistics":
            stats = evidence.data.get("statistics", {})
            lines = [f"- {key}: {value}" for key, value in stats.items()]
            return sections + [
                AnswerSection(
                    section_type="statistics",
                    heading=self._label(request_state, "statistics"),
                    content="\n".join(lines) if lines else evidence.summary,
                    structured_payload=stats or None,
                )
            ]
        if evidence.result_type == "metric_list":
            metrics = evidence.data.get("metrics", [])
            preview = metrics[:20]
            return sections + [
                AnswerSection(
                    section_type="metric_list",
                    heading=self._label(request_state, "available_metrics"),
                    content="\n".join(f"- {metric}" for metric in preview) if preview else evidence.summary,
                    structured_payload={"metric_count": len(metrics)},
                )
            ]
        if evidence.result_type == "schema":
            tables = evidence.data.get("tables_or_measurements", [])
            preview = [table.get("name") for table in tables[:20]]
            return sections + [
                AnswerSection(
                    section_type="schema",
                    heading=self._label(request_state, "schema_preview"),
                    content="\n".join(f"- {name}" for name in preview) if preview else evidence.summary,
                    structured_payload={"table_count": len(tables)},
                )
            ]
        if evidence.result_type == "table":
            rows = evidence.data.get("rows", [])
            return sections + [
                AnswerSection(
                    section_type="table",
                    heading=self._label(request_state, "table_result"),
                    content=evidence.summary,
                    structured_payload={"row_count": len(rows), "columns": evidence.columns},
                )
            ]
        return sections

    def _database_evidence_inventory(self, request_state: RequestStateModel) -> list:
        """Return database evidence in insertion order without duplicating latest."""

        items = list(request_state.database_evidence_artifacts.values())
        latest = request_state.latest_database_evidence
        if latest is not None and latest.evidence_id not in {item.evidence_id for item in items}:
            items.append(latest)
        return items

    def _database_query_results_section(self, evidence_items: list, request_state: RequestStateModel) -> AnswerSection:
        blocks = []
        structured_items = []
        for index, evidence in enumerate(evidence_items, start=1):
            summary = self._database_evidence_summary(evidence)
            structured_items.append(summary)
            blocks.append(self._render_database_evidence_block(index, evidence, summary, request_state))
        return AnswerSection(
            section_type="query_results",
            heading=self._label(request_state, "query_results"),
            content="\n\n".join(blocks),
            structured_payload={"items": structured_items},
        )

    def _database_evidence_summary(self, evidence) -> dict[str, Any]:
        data = evidence.data if isinstance(evidence.data, dict) else {}
        diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
        prompt_sampling = diagnostics.get("prompt_sampling") if isinstance(diagnostics.get("prompt_sampling"), dict) else {}
        summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
        row_count = self._evidence_row_count(evidence, data, diagnostics, summary_stats, prompt_sampling)
        rows_preview = self._preview_rows(data)
        artifact_ref = (
            diagnostics.get("artifact_ref")
            or diagnostics.get("snapshot_ref")
            or prompt_sampling.get("full_artifact_ref")
        )
        sampled = bool(prompt_sampling.get("sampled_for_prompt"))
        return {
            "evidence_id": evidence.evidence_id,
            "purpose": self._evidence_purpose(evidence),
            "summary": evidence.summary,
            "query_language": evidence.query_language,
            "query": evidence.query,
            "row_count": row_count,
            "columns": evidence.columns,
            "rows_preview": rows_preview,
            "sampled_for_prompt": sampled,
            "artifact_ref": artifact_ref,
        }

    def _evidence_row_count(
        self,
        evidence,
        data: dict,
        diagnostics: dict,
        summary_stats: dict,
        prompt_sampling: dict,
    ) -> int | None:
        candidates = [
            diagnostics.get("row_count_total"),
            diagnostics.get("row_count_materialized"),
            (diagnostics.get("sql_query") or {}).get("row_count") if isinstance(diagnostics.get("sql_query"), dict) else None,
            summary_stats.get("rows_count"),
            summary_stats.get("points_count"),
            (prompt_sampling.get("full_counts") or {}).get("rows_count") if isinstance(prompt_sampling.get("full_counts"), dict) else None,
            (prompt_sampling.get("full_counts") or {}).get("points_count") if isinstance(prompt_sampling.get("full_counts"), dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
        rows = data.get("rows")
        if isinstance(rows, list):
            return len(rows)
        points = data.get("points")
        if isinstance(points, list):
            return len(points)
        return None

    def _preview_rows(self, data: dict) -> list[dict]:
        rows = data.get("rows")
        if isinstance(rows, list):
            return [row for row in rows[:5] if isinstance(row, dict)]
        points = data.get("points")
        if isinstance(points, list):
            return [point for point in points[:5] if isinstance(point, dict)]
        statistics = data.get("statistics")
        if isinstance(statistics, dict):
            return [statistics]
        return []

    def _evidence_purpose(self, evidence) -> str | None:
        metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
        diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
        sql_query = diagnostics.get("sql_query") if isinstance(diagnostics.get("sql_query"), dict) else {}
        for value in (
            metadata.get("purpose"),
            sql_query.get("purpose"),
            metadata.get("expected_result_type"),
        ):
            text = str(value or "").strip()
            if text:
                return text
        return None

    def _render_database_evidence_block(
        self,
        index: int,
        evidence,
        summary: dict[str, Any],
        request_state: RequestStateModel,
    ) -> str:
        language = self._language(request_state)
        if language == "zh":
            lines = [f"{index}. 查询目的：{summary.get('purpose') or evidence.summary}"]
        else:
            lines = [f"{index}. Query purpose: {summary.get('purpose') or evidence.summary}"]
        row_count = summary.get("row_count")
        if row_count is not None:
            lines.append(f"实际返回行数：{row_count}" if language == "zh" else f"Rows returned: {row_count}")
        columns = summary.get("columns") or []
        if columns:
            prefix = "返回列：" if language == "zh" else "Columns: "
            lines.append(prefix + ", ".join(str(column) for column in columns))
        rows_preview = summary.get("rows_preview") or []
        result_digest = self._result_digest(rows_preview)
        if result_digest:
            lines.append(f"结果值：{result_digest}" if language == "zh" else f"Result values: {result_digest}")
        concise_summary = self._concise_evidence_summary(evidence.summary)
        if concise_summary:
            lines.append(f"结果摘要：{concise_summary}" if language == "zh" else f"Result summary: {concise_summary}")
        if rows_preview:
            if summary.get("sampled_for_prompt"):
                lines.append("当前展示的是采样预览；完整结果见 artifact。" if language == "zh" else "The current display is a sampled preview; see the artifact for the full result.")
            else:
                lines.append("当前展示的是实际返回预览。" if language == "zh" else "The current display is the returned preview.")
            lines.append(self._markdown_table(rows_preview, columns))
        elif summary.get("sampled_for_prompt"):
            lines.append("当前仅有采样摘要可用于最终展示；完整结果见 artifact。" if language == "zh" else "Only a sampled summary is available for display; see the artifact for the full result.")
        artifact_ref = summary.get("artifact_ref")
        if artifact_ref:
            lines.append(f"结果引用：{artifact_ref}" if language == "zh" else f"Result reference: {artifact_ref}")
        query = str(summary.get("query") or "").strip()
        if query and not self._is_internal_query(evidence):
            language = self._markdown_language(evidence.query_language)
            fence = f"```{language}\n{query}\n```" if language else f"```\n{query}\n```"
            lines.append("查询语句：" if language == "zh" else "Query statement:")
            lines.append(fence)
        return "\n".join(lines)

    def _markdown_table(self, rows: list[dict], preferred_columns: list | None = None) -> str:
        if not rows:
            return ""
        columns = [
            str(column)
            for column in (preferred_columns or [])
            if str(column) and any(str(column) in row for row in rows)
        ]
        for row in rows:
            for column in row.keys():
                if column not in columns:
                    columns.append(column)
        if not columns:
            return ""
        lines = [
            "| " + " | ".join(str(column) for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(self._table_cell(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _result_digest(self, rows: list[dict]) -> str | None:
        if not rows:
            return None
        if len(rows) == 1:
            row = rows[0]
            if len(row) == 1:
                key, value = next(iter(row.items()))
                return f"{key} = {self._table_cell(value)}"
            return ", ".join(
                f"{key} = {self._table_cell(value)}"
                for key, value in row.items()
            )
        if len(rows) <= 3:
            return "; ".join(self._compact_row_digest(row) for row in rows)
        return None

    def _compact_row_digest(self, row: dict) -> str:
        label_key = next(
            (key for key in ("bound", "label", "metric", "type") if key in row),
            None,
        )
        if label_key:
            rest = ", ".join(
                f"{key} = {self._table_cell(value)}"
                for key, value in row.items()
                if key != label_key
            )
            return f"{self._table_cell(row.get(label_key))}: {rest}"
        return ", ".join(
            f"{key} = {self._table_cell(value)}"
            for key, value in row.items()
        )

    def _concise_evidence_summary(self, summary: str | None) -> str | None:
        text = str(summary or "").strip()
        if not text:
            return None
        marker = " for query '"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip(".")
        return text

    def _table_cell(self, value) -> str:
        text = str(value if value is not None else "")
        return text.replace("|", "\\|").replace("\n", " ")

    def _database_reference_payload(self, evidence) -> dict:
        summary = self._database_evidence_summary(evidence)
        return {
            "summary": evidence.summary,
            "query_language": evidence.query_language,
            "query": evidence.query,
            "row_count": summary.get("row_count"),
            "columns": summary.get("columns"),
            "rows_preview": summary.get("rows_preview"),
            "sampled_for_prompt": summary.get("sampled_for_prompt"),
            "artifact_ref": summary.get("artifact_ref"),
        }

    def _query_sections(self, evidence, request_state: RequestStateModel | None = None) -> list[AnswerSection]:
        query = str(evidence.query or "").strip()
        if not query or self._is_internal_query(evidence):
            return []
        language = self._markdown_language(evidence.query_language)
        fence = f"```{language}\n{query}\n```" if language else f"```\n{query}\n```"
        return [
            AnswerSection(
                section_type="query",
                heading=self._label(request_state, "query"),
                content=fence,
                structured_payload={
                    "query_language": evidence.query_language,
                    "database": evidence.database,
                },
            )
        ]

    def _is_internal_query(self, evidence) -> bool:
        language = str(evidence.query_language or "").strip().lower()
        query = str(evidence.query or "").strip().lower()
        return language == "reference_dataset" or query.startswith("reference_dataset:")

    def _markdown_language(self, query_language: str | None) -> str | None:
        if not str(query_language or "").strip():
            return None
        return dialect_for_database(query_language).markdown_language(query_language)

    def _language(self, request_state: RequestStateModel | None) -> str:
        return "zh" if getattr(request_state, "response_language", "en") == "zh" else "en"

    def _label(self, request_state: RequestStateModel | None, key: str) -> str:
        language = self._language(request_state)
        return self._LABELS[language].get(key, self._LABELS["en"].get(key, key.replace("_", " ").title()))
