"""Complete fact candidates with concrete evidence context."""
from __future__ import annotations

from core.insight.fact_engine import evidence_rows, series_count, supports_multi_series_runtime
from schemas.database import DatabaseEvidence
from schemas.insight import CompletedFact, FactCandidate


def complete_fact_candidates(
    candidates: list[FactCandidate],
    evidence: DatabaseEvidence,
    focus: str | None,
) -> list[CompletedFact]:
    rows, columns, time_field, value_field = evidence_rows(evidence)
    if not rows:
        return []
    return [
        CompletedFact(
            fact_id=candidate.fact_id,
            fact_type=candidate.fact_type,
            statement=candidate.statement or "",
            focus=focus,
            required_evidence=[evidence.evidence_id],
            evidence={
                "evidence_id": evidence.evidence_id,
                "row_count": len(rows),
                "columns": columns,
                "time_field": time_field,
                "value_field": value_field,
                "series_count": series_count(evidence),
                "supports_multi_series_runtime": supports_multi_series_runtime(evidence),
            },
            confidence=candidate.confidence,
        )
        for candidate in candidates
    ]
