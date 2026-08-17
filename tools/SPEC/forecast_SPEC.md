# tools/forecast.py SPEC

## Purpose

Run forecast analysis on normalized time-series evidence.

## Input

- `database_evidence: DatabaseEvidence`
- `horizon: int | str | dict | null`
- `constraints: dict | null`

## Output

- `ForecastResult`

## Reads

- time-series evidence
- forecast options

## Writes

- forecast result
- visualization payloads
- diagnostics

## Internal pipeline

1. validate that evidence is time-series shaped
2. normalize timestamps and values
3. validate forecast evidence coverage for the requested range
4. resolve a forecast plan from explicit steps, user duration, or a short-term default
5. choose a registered forecast model, which may be local or API-backed
6. run direct forecast only when the resolved plan is within the direct forecast window
7. package forecast points, confidence intervals, forecast plan, and diagnostics

## Contract notes

- fail fast when input evidence is not `timeseries`
- reject raw limited evidence when it does not cover the requested time range
- return `requires_rolling` plus a plan when the requested horizon exceeds the direct forecast window
- forecast model implementations must be selected through the registry
- do not infer unrelated insights

## Must not do

- query databases directly
- write final answer text
