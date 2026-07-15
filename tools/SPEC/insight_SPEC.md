# tools/insight.py SPEC

## Purpose

Convert database evidence into grounded facts and chartable insight.

## Input

- `database_evidence: DatabaseEvidence`
- `requested_fact_types: list[str]`
- `focus: str | null`
- `constraints: dict | null`

## Output

- `InsightResult`

## Reads

- latest database evidence
- request intent and focus
- requested fact family list

## Writes

- fact candidates
- completed facts
- verified facts
- rejected facts
- visualization payloads

## Internal pipeline

1. inspect evidence family and shape
2. determine which fact families are supported
3. propose candidate facts
4. complete candidate facts with required evidence
5. verify each candidate against deterministic rules
6. select the verified facts that best match the request
7. build chartable visualization payloads from verified facts

## Contract notes

- fact families are open string labels
- only verified facts may flow into final answer narration
- visualizations must be grounded in verified facts or evidence-backed model output
- `insight` must not query databases directly

## Fact-family responsibility

`insight` is the preferred execution layer for analysis-native facts that require
deterministic computation over already materialized rows or time-series points.

Typical analysis-native fact families:

- `trend`
- `seasonality`
- `outlier`
- `association`
- `distribution`
- `categorization`

Hybrid fact families:

- `difference`
- `rank`
- `proportion`
- `extreme`
- `aggregation`

For hybrid families, the long-term design intent is:

- prefer `query_database` when the datasource can produce the exact fact or the
  exact grouped evidence directly
- use `insight` when the fact must be derived from raw materialized evidence,
  especially time-series rows/points

Current implementation note:

- `insight` can currently verify all 11 canonical fact families against
  materialized evidence
- this does not mean every fact family should always be executed here
- when upstream evidence has already been sampled, any downstream fact computed
  by `insight` reflects that sampled evidence rather than an implicit full-table
  recomputation

## Canonical fact families

- `aggregation`
- `extreme`
- `trend`
- `difference`
- `rank`
- `distribution`
- `association`
- `outlier`
- `seasonality`
- `proportion`
- `categorization`

Supported alias examples:

- `extrema -> extreme`
- `change_percent -> difference`
- `comparison -> difference`
- `correlation -> association`
- `anomaly -> outlier`

## Must not do

- write final answer text
- call forecast or anomaly directly
