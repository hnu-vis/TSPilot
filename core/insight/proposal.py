"""Generate fact candidates from normalized evidence."""
from __future__ import annotations

from core.insight.fact_engine import normalize_requested_fact_types
from schemas.database import DatabaseEvidence
from schemas.insight import FactCandidate


def propose_fact_candidates(evidence: DatabaseEvidence, requested_fact_types: list[str]) -> list[FactCandidate]:
    points = evidence.data.get("points", [])
    if evidence.result_type != "timeseries" or len(points) < 2:
        return []
    value_field = evidence.data.get("value_field", "value")
    candidates: list[FactCandidate] = []
    for fact_type in normalize_requested_fact_types(requested_fact_types):
        candidates.append(
            FactCandidate(
                fact_id=f"fact_{evidence.evidence_id}_{fact_type}",
                fact_type=fact_type,
                statement=f"{value_field} {fact_type} fact candidate.",
                confidence=0.8,
                evidence_refs=[evidence.evidence_id],
            )
        )
    return candidates
