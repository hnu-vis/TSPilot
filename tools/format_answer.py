"""Grounded final-answer assembly without an internal LLM call."""
from __future__ import annotations

from pydantic import BaseModel

from core.visualization import InvalidPresentationLineageError, PresentationCatalog
from schemas.output import (
    AnswerClaim,
    AnswerReference,
    AnswerSection,
    FinalAnswer,
    FinalResponsePlan,
)
from schemas.state import RequestStateModel
from tools.base import BaseTool, StructuredToolError


class FormatAnswerInput(BaseModel):
    response_plan: FinalResponsePlan


class FormatAnswerTool(BaseTool):
    """Assemble prose with already validated visualization artifacts."""

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
        try:
            visualizations = self._selected_visualizations(plan, request_state, catalog)
        except InvalidPresentationLineageError as exc:
            raise _analysis_lineage_error(exc, request_state, catalog) from exc
        except ValueError as exc:
            raise _visualization_reference_error(exc, request_state) from exc
        try:
            selected_visualization_ids = {item.visualization_id for item in visualizations}
            self._validate_plan_sources(plan, catalog, selected_visualization_ids)
            referenced = self._referenced_sources(plan, visualizations, catalog)
        except InvalidPresentationLineageError as exc:
            raise _analysis_lineage_error(exc, request_state, catalog) from exc
        except ValueError as exc:
            raise _response_plan_reference_error(exc, catalog) from exc
        visualization_ids_by_ref: dict[str, list[str]] = {}
        for visualization in visualizations:
            for ref in visualization.source_refs:
                source = catalog.resolve(ref)
                related_refs = [source.ref]
                if source.kind == "view":
                    related_refs.extend(source.value.lineage)
                for related_ref in related_refs:
                    visualization_ids_by_ref.setdefault(related_ref, []).append(visualization.visualization_id)

        sections = [
            AnswerSection(
                section_type=section.section_type,
                heading=section.heading,
                content=section.content,
                structured_payload={
                    "source_refs": self._canonical_refs(
                        section.source_refs,
                        catalog,
                        selected_visualization_ids,
                    )
                },
            )
            for section in plan.sections
        ]
        references = [self._reference(source, catalog) for source in referenced]
        claims = self._claims(
            plan,
            catalog,
            visualization_ids_by_ref,
            selected_visualization_ids,
        )
        answer = FinalAnswer(
            title=plan.title,
            summary=plan.summary,
            sections=sections,
            references=references,
            claims=claims,
            visualizations=visualizations,
        )
        return answer.model_dump(mode="json")

    def _validate_plan_sources(
        self,
        plan: FinalResponsePlan,
        catalog: PresentationCatalog,
        visualization_ids: set[str],
    ) -> None:
        for section in plan.sections:
            self._canonical_refs(section.source_refs, catalog, visualization_ids)

    def _selected_visualizations(self, plan, request_state, catalog):
        available = {item.visualization_id: item for item in request_state.visualizations}
        missing = [item for item in plan.visualization_ids if item not in available]
        if missing:
            raise ValueError(f"unknown visualization artifact ids: {missing}")
        selected = [available[item] for item in dict.fromkeys(plan.visualization_ids)]
        for visualization in selected:
            if not visualization.data_ref:
                raise ValueError(f"visualization artifact '{visualization.visualization_id}' has no full-data reference")
            for ref in visualization.source_refs:
                source = catalog.resolve(ref)
                if source.kind == "view":
                    catalog.resolve_lineage(source)
        return selected

    def _canonical_refs(
        self,
        refs: list[str],
        catalog: PresentationCatalog,
        visualization_ids: set[str] | None = None,
    ) -> list[str]:
        canonical: list[str] = []
        for ref in refs:
            visualization_id = _selected_visualization_id(ref, visualization_ids or set())
            if visualization_id:
                canonical.append(f"visualization:{visualization_id}")
            else:
                canonical.append(catalog.resolve(ref).ref)
        return list(dict.fromkeys(canonical))

    def _referenced_sources(self, plan: FinalResponsePlan, visualizations, catalog: PresentationCatalog):
        refs: list[str] = []
        for section in plan.sections:
            refs.extend(
                ref
                for ref in section.source_refs
                if not _selected_visualization_id(ref, {item.visualization_id for item in visualizations})
            )
        for visualization in visualizations:
            refs.extend(visualization.source_refs)
        referenced = []
        for ref in refs:
            source = catalog.resolve(ref)
            if source.kind == "view":
                referenced.extend(catalog.resolve_lineage(source))
            else:
                referenced.append(source)
        return list({source.ref: source for source in referenced}.values())

    def _reference(self, source, catalog: PresentationCatalog) -> AnswerReference:
        presentation = catalog.reference_presentation(source)
        return AnswerReference(
            source_type=presentation.source_type,
            source_id=presentation.source_id,
            label=presentation.label,
            evidence=presentation.evidence,
        )

    def _claims(self, plan, catalog, visualization_ids_by_ref, selected_visualization_ids):
        claims: list[AnswerClaim] = []
        for index, section in enumerate(plan.sections):
            explicit_visualization_ids = [
                visualization_id
                for ref in section.source_refs
                if (visualization_id := _selected_visualization_id(ref, selected_visualization_ids))
            ]
            sources = [
                catalog.resolve(ref)
                for ref in section.source_refs
                if not _selected_visualization_id(ref, selected_visualization_ids)
            ]
            canonical = [source.ref for source in sources]
            presentations = [
                (source, catalog.reference_presentation(source))
                for source in sources
                if source.reference is not None
            ]
            claims.append(AnswerClaim(
                claim_id=f"claim_section_{index + 1}",
                text=section.content,
                insight_ids=list(dict.fromkeys(
                    presentation.source_id
                    for _source, presentation in presentations
                    if presentation.source_type == "insight"
                )),
                item_ids=list(dict.fromkeys(
                    source.ref.rsplit("#", 1)[1]
                    for source, _presentation in presentations
                    if "#" in source.ref
                )),
                analysis_ids=[
                    presentation.source_id
                    for _source, presentation in presentations
                    if presentation.source_type == "analysis"
                ],
                artifact_ids=[
                    presentation.source_id
                    for _source, presentation in presentations
                    if presentation.source_type in {"forecast", "anomaly", "derived_evidence"}
                ],
                evidence_ids=[
                    presentation.source_id
                    for _source, presentation in presentations
                    if presentation.source_type in {"query", "derived_evidence"}
                ],
                visualization_ids=list(dict.fromkeys(
                    explicit_visualization_ids
                    + [
                        visualization_id
                        for ref in canonical
                        for visualization_id in visualization_ids_by_ref.get(ref, [])
                    ]
                )),
            ))
        return claims


def _selected_visualization_id(ref: str, selected_ids: set[str]) -> str | None:
    """Accept both the tool-returned bare id and its canonical artifact ref."""
    candidate = str(ref or "").strip()
    if candidate.startswith("visualization:"):
        candidate = candidate.split(":", 1)[1]
    return candidate if candidate in selected_ids else None


def _visualization_reference_error(exc: ValueError, request_state: RequestStateModel) -> StructuredToolError:
    message = f"Visualization artifact validation failed: {exc}"
    repair_contract = {
        "mode": "visualization_artifact_required",
        "instruction": (
            "Call visualization to create or repair grounded artifacts, then reference only returned visualization_ids."
        ),
        "available_visualization_ids": [item.visualization_id for item in request_state.visualizations],
    }
    return StructuredToolError(
        message,
        error_type="visualization_artifact_validation",
        retryable=True,
        recommended_next_action="visualization",
        diagnostics=repair_contract,
        validation_failure={
            "scope": "final_response_plan",
            "capability": "visualization",
            "tool": "visualization",
            "error_code": "visualization_artifact_validation",
            "message": message,
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "visualization",
                "max_equivalent_retries": 1,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )


def _response_plan_reference_error(exc: ValueError, catalog: PresentationCatalog) -> StructuredToolError:
    message = f"Final response source validation failed: {exc}"
    repair_contract = {
        "mode": "final_response_reference_repair",
        "instruction": (
            "Retry terminate using exact canonical artifact refs from state.artifacts.refs and insight_state.recent_insights. "
            "Do not regenerate an already successful visualization."
        ),
        "available_source_refs": catalog.canonical_refs(),
    }
    return StructuredToolError(
        message,
        error_type="final_response_reference_invalid",
        retryable=True,
        recommended_next_action="terminate",
        diagnostics=repair_contract,
        validation_failure={
            "scope": "final_response_plan",
            "capability": "answer",
            "tool": "terminate",
            "error_code": "final_response_reference_invalid",
            "message": message,
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "terminate",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )


def _analysis_lineage_error(
    exc: InvalidPresentationLineageError,
    request_state: RequestStateModel,
    catalog: PresentationCatalog,
) -> StructuredToolError:
    derived_id = exc.view_ref.split(":", 2)[2] if exc.view_ref.startswith("view:derived_evidence:") else None
    derived = request_state.derived_evidence_artifacts.get(derived_id) if derived_id else None
    analysis = next(
        (
            item for item in request_state.analysis_artifacts.values()
            if derived_id and any(derived_id in candidate.derived_evidence_ids for candidate in item.computed_insights)
        ),
        None,
    )
    analysis_id = getattr(analysis, "analysis_id", None)
    evidence_id = getattr(analysis, "input_evidence_id", None) or next(
        (ref.removeprefix("evidence:") for ref in getattr(derived, "lineage", []) if ref.startswith("evidence:")),
        None,
    )
    allowed_refs = [f"evidence:{evidence_id}"] if evidence_id else []
    repair_contract = {
        "mode": "analysis_artifact_repair",
        "input_evidence": evidence_id or "latest",
        "analysis_goal": getattr(analysis, "analysis_goal", None) or request_state.message,
        "failed_analysis_id": analysis_id,
        "failed_derived_evidence_id": derived_id,
        "failed_view_ref": exc.view_ref,
        "unknown_lineage_ref": exc.lineage_ref,
        "allowed_lineage_refs": allowed_refs,
        "failed_code": str((getattr(analysis, "diagnostics", {}) or {}).get("executed_code") or ""),
        "instruction": (
            "Recompute the requested Insight and its derived Evidence using only grounded input refs."
        ),
    }
    message = f"Derived Evidence lineage validation failed: {exc}"
    return StructuredToolError(
        message,
        error_type="analysis_lineage_invalid",
        retryable=True,
        recommended_next_action="code_interpreter",
        diagnostics={"repair_contract": repair_contract, "available_source_refs": catalog.canonical_refs()},
        validation_failure={
            "scope": "artifact_output",
            "capability": "analysis",
            "tool": "code_interpreter",
            "error_code": "analysis_lineage_invalid",
            "message": message,
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "code_interpreter",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )
