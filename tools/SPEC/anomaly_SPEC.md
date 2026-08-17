# tools/anomaly.py SPEC

## Purpose

Run anomaly detection on normalized time-series evidence.

## Input

- `database_evidence: DatabaseEvidence`
- `constraints: dict | null`

## Output

- `AnomalyResult`

## Reads

- time-series evidence
- anomaly options

## Writes

- anomaly result
- visualization payloads
- diagnostics

## Internal pipeline

1. validate that evidence is time-series shaped
2. normalize timestamps and values
3. choose a registered anomaly detector, which may be local or API-backed
4. run anomaly detection
5. package anomaly points, spans, and scores

## Contract notes

- fail fast when input evidence is not `timeseries`
- anomaly detector implementations must be selected through the registry
- do not infer unrelated insights

## Must not do

- query databases directly
- write final answer text
