# Real HTTP E2E Broader Regression Report

- Date: 2026-07-30
- Backend: real FastAPI on `127.0.0.1:18081`
- LLM: real configured model from project `.env`
- Database: real InfluxDB Bitcoin sample on `localhost:18086`
- Frontend build: not executed

## Summary

Ran four additional real end-to-end cases after the schema-linking/dialect-preview fix.

Result: 2 passed, 2 failed by timeout.

| Case | Mode | Result | Time | Notes |
|---|---|---:|---:|---|
| `btc_multi_fact_sse` | SSE | Failed | `150.846s` | 21 ReAct rounds; no `final_answer`/`error`; repeated `required_filter_missing` for `code='USD'`. |
| `btc_anomaly_sse` | SSE | Failed | `150.128s` | 21 ReAct rounds; no `final_answer`/`error`; repeated `code_interpreter` failure about missing transparent outlier details. |
| `btc_forecast_sse` | SSE | Passed | `26.985s` | final answer reached; 3 ReAct rounds: `sql_query -> code_interpreter -> terminate`. |
| `btc_schema_explain_json` | JSON | Passed | `20.508s` | HTTP 200; status `completed`; used tools `["sql_query"]`; answer included measurement/field/tag and Flux example. |

Artifacts:

- `cache_data/e2e_runs/real_http_broader_regression_2026-07-30/summary.json`
- `cache_data/e2e_runs/real_http_broader_regression_2026-07-30/btc_multi_fact_sse.json`
- `cache_data/e2e_runs/real_http_broader_regression_2026-07-30/btc_anomaly_sse.json`
- `cache_data/e2e_runs/real_http_broader_regression_2026-07-30/btc_forecast_sse.json`
- `cache_data/e2e_runs/real_http_broader_regression_2026-07-30/btc_schema_explain_json.json`

## Failure 1: Multi-Fact Query

Request asked for:

- USD count
- earliest 5 rows
- latest 5 rows
- earliest/latest timestamp
- Flux query and row count for each result

Observed:

```text
elapsed_seconds=150.846
step.start=21
tool_call=21
tool_result=21
final_answer=0
error=0
```

Repeated terminal failure:

```text
Tool 'sql_query' failed:
Explicit query is missing filters required by the user request.
Rendered query is missing the required filter code='USD'.
```

Diagnosis:

- The model switched into explicit query mode for follow-up queries.
- Required-filter validation correctly detected that generated Flux did not preserve `code='USD'`.
- The recovery observation still led the model to repeat similar explicit queries instead of returning to automatic planning or using the structured schema contract.

This is a real robustness gap in explicit-query repair/action transition.

## Failure 2: Anomaly Detection

Request asked for anomaly points in Bitcoin USD price.

Observed:

```text
elapsed_seconds=150.128
step.start=21
tool_call=21
tool_result=21
final_answer=0
error=0
```

Repeated failure:

```text
Tool 'code_interpreter' failed:
analysis result using outlier treatment must include transparent details fields:
adjusted_metrics, excluded_rows, outlier_rule, rationale, raw_metrics, threshold_or_formula
```

Diagnosis:

- The system correctly requires transparent outlier-treatment details.
- The model/code-interpreter path repeatedly failed to satisfy that artifact contract.
- Repeated failure strategy did not escalate to a different action or produce a terminal error with actionable unavailable output.

This is a real robustness gap in analysis artifact contract repair.

## Passed Case: Forecast

Request asked for future 5 Bitcoin USD price points and evidence.

Observed:

```text
elapsed_seconds=26.985
step.start=3
tool_call=3
tool_result=3
final_answer=1
```

Tool chain:

```text
sql_query -> code_interpreter -> terminate
```

## Passed Case: Schema Explanation

Request asked to explain the Bitcoin datasource measurement/field/tag and how to query USD price.

Observed:

```text
elapsed_seconds=20.508
HTTP 200
status=completed
used_tools=["sql_query"]
```

Answer included:

- measurement: `coindesk`
- field: `price`
- tags: `code`, `crypto`, `description`, `symbol`
- Flux query using `_value` long-form semantics

## Follow-Up Fix Direction

Do not patch individual requests.

Recommended next fixes:

1. Explicit query repair should preserve required filters through structured repair context and avoid repeated explicit query attempts with the same missing-filter signature.
2. Multi-fact tasks should use a query-batch/task-plan contract so each required output has its own evidence slot and required filters.
3. Analysis/code-interpreter artifacts should expose a repairable schema contract for required transparency fields instead of relying on repeated free-form retries.
4. Repeated failure transition should be terminal-aware: after repeated equivalent failures, either force a materially different valid action or return a final answer with explicitly unavailable outputs.

