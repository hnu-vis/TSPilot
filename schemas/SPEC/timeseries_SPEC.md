# schemas/timeseries.py SPEC

## Purpose

Define normalized time-series structures and the outputs of forecast/anomaly tools.

## Models

- `TimeSeriesPoint`
- `TimeSeriesSeries`
- `ForecastResult`
- `AnomalyResult`

## `TimeSeriesPoint`

Fields:

- `timestamp: str`
- `value: float`

## `TimeSeriesSeries`

Fields:

- `series_name: str | null`
- `time_field: str`
- `value_field: str`
- `points: list[TimeSeriesPoint]`
- `labels: dict`

## `ForecastResult`

Fields:

- `forecast_id: str`
- `model_name: str`
- `horizon: int`
- `forecast_points: list[TimeSeriesPoint]`
- `confidence_interval: list[dict]`
- `diagnostics: dict`
- `visualizations: list[VisualizationPayload]`

## `AnomalyResult`

Fields:

- `anomaly_id: str`
- `detector_name: str`
- `anomaly_points: list[dict]`
- `anomaly_spans: list[dict]`
- `scores: list[dict]`
- `diagnostics: dict`
- `visualizations: list[VisualizationPayload]`

## Contract notes

- time-series tools must fail fast when the input evidence is not time-series shaped
- normalized timestamps must be comparable and stable
- forecast and anomaly results may carry chart payloads, but not final answer text
- `forecast_id` and `anomaly_id` must be stable within one request for references and traceability

## Responsibilities

- define the canonical series shape used by forecast/anomaly
- keep time-based analysis payloads consistent

## Must not do

- infer final user narrative
- query databases directly
