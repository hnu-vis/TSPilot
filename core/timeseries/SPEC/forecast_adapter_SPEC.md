# core/timeseries/forecast_adapter.py SPEC

## Purpose

Bridge normalized series to a forecast backend.

## Responsibilities

- prepare input series
- call the forecast backend
- return model output in a stable shape

## Must not do

- infer insights
- write final answer text
