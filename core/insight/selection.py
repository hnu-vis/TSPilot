"""Select verified insight outputs and assemble the tool-facing result."""
from __future__ import annotations

from core.insight.completion import complete_fact_candidates
from core.insight.fact_engine import normalize_requested_fact_types, series_count, supports_multi_series_runtime
from core.insight.proposal import propose_fact_candidates
from core.insight.verification import verify_completed_facts
from schemas.database import DatabaseEvidence
from schemas.insight import InsightResult
from schemas.visualization import VisualizationPayload


def build_insight_result(
    evidence: DatabaseEvidence,
    requested_fact_types: list[str],
    focus: str | None,
) -> InsightResult:
    normalized_fact_types = normalize_requested_fact_types(requested_fact_types)
    candidates = propose_fact_candidates(evidence, normalized_fact_types)
    completed = complete_fact_candidates(candidates, evidence, focus)
    verified, rejected = verify_completed_facts(completed, evidence)
    visualizations = _build_visualizations(evidence, verified, normalized_fact_types)
    supported = sorted({fact.fact_type for fact in verified})
    summary_blocks = [{"kind": "summary", "content": fact.statement} for fact in verified[:1]]
    diagnostics = {
        "point_count": len(evidence.data.get("points", [])),
        "series_count": series_count(evidence),
        "multi_series_evidence_detected": supports_multi_series_runtime(evidence),
    }
    return InsightResult(
        insight_id=f"ins_{evidence.evidence_id}",
        requested_fact_types=normalized_fact_types,
        supported_fact_types=supported,
        fact_candidates=candidates,
        completed_facts=completed,
        verified_facts=verified,
        rejected_facts=rejected,
        summary_blocks=summary_blocks,
        visualizations=visualizations,
        diagnostics=diagnostics,
    )


def _build_visualizations(
    evidence: DatabaseEvidence,
    verified_facts: list,
    requested_fact_types: list[str],
) -> list[VisualizationPayload]:
    points = evidence.data.get("points", [])
    if not points:
        return []
    value_field = evidence.data.get("value_field", "value")
    return [
        VisualizationPayload(
            visualization_id=f"viz_{evidence.evidence_id}",
            visualization_type="chart",
            visualization_kind="line",
            renderer="linechart",
            title=f"{value_field} trend",
            summary=f"{value_field} 的时序折线图。",
            chart={
                "x_axis_data": [point["timestamp"] for point in points],
                "series_data": [{"name": value_field, "data": [point["value"] for point in points]}],
            },
            binding_fact_ids=[fact.fact_id for fact in verified_facts],
            binding_evidence_ids=[evidence.evidence_id],
            requested_fact_types=requested_fact_types,
            time_column=evidence.data.get("time_field"),
            primary_measure=value_field,
            display_priority=1,
        )
    ]
