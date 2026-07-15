# tools/forecast.py SPEC

## Purpose

Run forecast analysis on normalized time-series evidence.

## Input

- `database_evidence: DatabaseEvidence`
- `horizon: int | null`
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
3. choose forecast adapter / model
4. run forecast
5. package forecast points and confidence intervals

## Contract notes

- fail fast when input evidence is not `timeseries`
- do not infer unrelated facts

## Must not do

- query databases directly
- write final answer text
