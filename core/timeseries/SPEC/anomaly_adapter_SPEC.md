# core/timeseries/anomaly_adapter.py SPEC

## Purpose

Bridge normalized series to an anomaly detector backend.

## Responsibilities

- prepare input series
- call the anomaly backend
- return detector output in a stable shape

## Must not do

- infer facts
- write final answer text
