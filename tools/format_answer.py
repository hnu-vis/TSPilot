"""Grounded final-answer assembly without an internal LLM call."""
from __future__ import annotations

from pydantic import BaseModel

from core.visualization import PresentationCatalog, VisualizationMaterializer
from schemas.output import (
    AnswerClaim,
    AnswerReference,
    AnswerSection,
    FinalAnswer,
    FinalResponsePlan,
)
from schemas.state import RequestStateModel
from tools.base import BaseTool


class FormatAnswerInput(BaseModel):
    response_plan: FinalResponsePlan


class FormatAnswerTool(BaseTool):
    """Validate a final response plan and materialize its grounded views.

    The outer ReAct model already authored both prose and visual semantics in its
    terminal action.  Keeping this layer model-free makes the presentation
    boundary fast, observable, and consistent.
    """

    def __init__(self, llm=None):
        # Kept in the constructor signature so registry wiring remains stable.
        # Deliberately do not store or invoke the model here.
        _ = llm

    async def execute(
        self,
        validated_input: FormatAnswerInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        plan = validated_input.response_plan
        catalog = PresentationCatalog(request_state)
        self._validate_plan_sources(plan, catalog)
        visualizations = VisualizationMaterializer(request_state).materialize_all(plan.visual_intents)
        visualization_ids_by_ref: dict[str, list[str]] = {}
        for visualization in visualizations:
            for ref in [*visualization.source_refs, *visualization.fact_refs]:
                visualization_ids_by_ref.setdefault(ref, []).append(visualization.visualization_id)

        sections = [
            AnswerSection(
                section_type=section.section_type,
                heading=section.heading,
                content=section.content,
                structured_payload={"source_refs": self._canonical_refs(section.source_refs, catalog)},
            )
            for section in plan.sections
        ]
        referenced = self._referenced_sources(plan, catalog)
        references = [self._reference(source) for source in referenced]
        claims = self._claims(plan, catalog, visualization_ids_by_ref)
        answer = FinalAnswer(
            title=plan.title,
            summary=plan.summary,
            sections=sections,
            references=references,
            claims=claims,
            visualizations=visualizations,
        )
        return answer.model_dump(mode="json")

    def _validate_plan_sources(self, plan: FinalResponsePlan, catalog: PresentationCatalog) -> None:
        for section in plan.sections:
            self._canonical_refs(section.source_refs, catalog)
        primary_by_purpose: set[str] = set()
        for intent in plan.visual_intents:
            self._canonical_refs(intent.source_refs, catalog)
            fact_sources = [catalog.resolve(ref) for ref in intent.fact_refs]
            invalid = [source.ref for source in fact_sources if source.kind != "fact"]
            if invalid:
                raise ValueError(f"visual intent fact_refs contain non-Fact sources: {invalid}")
            purpose_key = intent.purpose.strip().casefold()
            if intent.priority == "primary" and purpose_key in primary_by_purpose:
                raise ValueError(f"multiple primary visualizations cover the same purpose: {intent.purpose}")
            if intent.priority == "primary":
                primary_by_purpose.add(purpose_key)

    def _canonical_refs(self, refs: list[str], catalog: PresentationCatalog) -> list[str]:
        return list(dict.fromkeys(catalog.resolve(ref).ref for ref in refs))

    def _referenced_sources(self, plan: FinalResponsePlan, catalog: PresentationCatalog):
        refs: list[str] = []
        for section in plan.sections:
            refs.extend(section.source_refs)
        for intent in plan.visual_intents:
            refs.extend(intent.source_refs)
            refs.extend(intent.fact_refs)
        return [catalog.resolve(ref) for ref in dict.fromkeys(refs)]

    def _reference(self, source) -> AnswerReference:
        value = source.value
        if source.kind == "evidence":
            return AnswerReference(
                source_type="query",
                source_id=value.evidence_id,
                label=value.summary or value.result_type,
                evidence={
                    "result_type": value.result_type,
                    "database": value.database,
                    "query_language": value.query_language,
                    "query": value.query,
                    "columns": value.columns,
                    "metadata": value.metadata,
                },
            )
        if source.kind == "fact":
            return AnswerReference(
                source_type="fact",
                source_id=value.fact_id,
                label=value.name,
                evidence={
                    "fact_key": value.fact_key,
                    "fact_type": value.fact_type,
                    "statement": value.statement,
                    "value": value.value,
                    "unit": value.unit,
                    "status": value.status,
                    "evidence_refs": [item.model_dump(mode="json") for item in value.evidence_refs],
                },
            )
        if source.kind == "analysis":
            return AnswerReference(
                source_type="analysis",
                source_id=value.analysis_id,
                label=value.analysis_goal,
                evidence={
                    "summary": value.summary,
                    "result": value.result,
                    "input_evidence_id": value.input_evidence_id,
                    "input_row_count": value.input_row_count,
                    "code_hash": value.code_hash,
                    "code_type": value.code_type,
                },
            )
        if source.kind == "forecast":
            return AnswerReference(
                source_type="forecast",
                source_id=value.forecast_id,
                label=value.model_name,
                evidence={"horizon": value.horizon, "status": value.status, "diagnostics": value.diagnostics},
            )
        return AnswerReference(
            source_type="anomaly",
            source_id=value.anomaly_id,
            label=value.detector_name,
            evidence={"anomaly_count": len(value.anomaly_points), "diagnostics": value.diagnostics},
        )

    def _claims(self, plan, catalog, visualization_ids_by_ref):
        claims: list[AnswerClaim] = []
        for index, section in enumerate(plan.sections):
            sources = [catalog.resolve(ref) for ref in section.source_refs]
            canonical = [source.ref for source in sources]
            claims.append(AnswerClaim(
                claim_id=f"claim_section_{index + 1}",
                text=section.content,
                fact_ids=[source.value.fact_id for source in sources if source.kind == "fact"],
                item_ids=[item.item_id for source in sources if source.kind == "fact" for item in source.value.items],
                analysis_ids=[source.value.analysis_id for source in sources if source.kind == "analysis"],
                artifact_ids=[source.ref.split(":", 1)[1] for source in sources if source.kind in {"forecast", "anomaly"}],
                evidence_ids=[source.value.evidence_id for source in sources if source.kind == "evidence"],
                visualization_ids=list(dict.fromkeys(
                    visualization_id
                    for ref in canonical
                    for visualization_id in visualization_ids_by_ref.get(ref, [])
                )),
            ))
        return claims
