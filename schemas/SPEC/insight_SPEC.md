# schemas/insight.py SPEC

## Purpose

Define the structured result of converting evidence into grounded facts.

## Models

- `FactCandidate`
- `CompletedFact`
- `VerifiedFact`
- `RejectedFact`
- `InsightResult`

## `FactCandidate`

Fields:

- `fact_id: str`
- `fact_type: str`
- `statement: str | null`
- `confidence: float | null`
- `evidence_refs: list[str]`

## `CompletedFact`

Fields:

- `fact_id: str`
- `fact_type: str`
- `statement: str`
- `focus: str | null`
- `required_evidence: list[str]`
- `evidence: dict`
- `confidence: float | null`

## `VerifiedFact`

Fields:

- `fact_id: str`
- `fact_type: str`
- `statement: str`
- `confidence: float`
- `evidence: dict`
- `verification_rule: str`
- `verification_status: Literal["verified"]`

## `RejectedFact`

Fields:

- `fact_id: str`
- `fact_type: str`
- `statement: str | null`
- `reason: str`
- `evidence: dict | null`
- `verification_rule: str | null`

## `InsightResult`

Fields:

- `insight_id: str`
- `requested_fact_types: list[str]`
- `supported_fact_types: list[str]`
- `fact_candidates: list[FactCandidate]`
- `completed_facts: list[CompletedFact]`
- `verified_facts: list[VerifiedFact]`
- `rejected_facts: list[RejectedFact]`
- `summary_blocks: list[dict]`
- `visualizations: list[VisualizationPayload]`
- `diagnostics: dict`

## Contract notes

- `fact_type` is an open string label, not a closed enum
- `insight_id` must be stable within one request so downstream references can point to the selected analysis output
- `requested_fact_types` may be empty when the request does not need insight
- only `verified_facts` may feed final answer narration
- `completed_facts` exist so incomplete LLM ideas can be normalized before verification
- `visualizations` must be grounded in verified facts or model outputs with evidence

## Responsibilities

- represent fact extraction as a verifiable structured result
- keep analysis output separate from final answer text

## Must not do

- query databases
- perform hidden presentation assembly
