# Real HTTP E2E Test Report

- Date: 2026-07-30
- Time window: 18:48-19:01 CST
- Workspace: `/home/feilvvl/TSPilot-v0.1`
- Environment: `/home/feilvvl/TSPilot/tspilot_env`
- Backend under test: real FastAPI service on `127.0.0.1:18081`
- LLM path: real configured `ChatOpenAI` from project `.env`
- Database path: real InfluxDB on `localhost:18086`
- Frontend build: not executed

## Conclusion

This run was a real backend HTTP E2E test, not a pytest fake-agent test.

The system is not passing real E2E for `bitcoin 的 usd 最大值是多少`.

Observed behavior:

- Non-stream JSON request timed out after 240.2 seconds with no HTTP response.
- SSE request produced first UI-visible events quickly, but did not reach `final_answer` or `error` after 306.6 seconds.
- The ReAct loop repeatedly called `sql_query` with an invalid Flux query using `max(column: "price")`.
- The real InfluxDB data stores the field name in `_field == "price"` and the numeric value in `_value`; therefore the correct aggregation column is `_value`, not `price`.
- The repeated tool failure was not converted into a robust schema-repair/action-space transition, so the loop kept retrying instead of terminating or escalating to a schema inspection / corrected query.

## Real Service Setup

Health check:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 18081
```

```text
GET http://127.0.0.1:18081/health
200 {"status":"ok"}
```

Configuration checks:

```text
settings_OPENAI_API_KEY=True
settings_OPENAI_API_BASE=True
settings_OPENAI_MODEL=gpt-4o-mini
database config dir=/home/feilvvl/TSPilot-v0.1/configs/databases
InfluxDB localhost:18086 reachable=True
```

## Test Cases

| # | Case | Request mode | Test content | Result | Key output |
|---:|---|---|---|---|---|
| 1 | `btc_usd_max` | JSON HTTP | `bitcoin 的 usd 最大值是多少` against `influxdb2-bitcoin-sample` | Failed | Client timed out after `240.226s`; no HTTP response body returned. |
| 2 | `btc_usd_max_sse_first_events` | SSE HTTP | Same Bitcoin max question | Partial pass | HTTP `200`; first chunk at `0.130s`; received `conversation_id` and heartbeat `agent_step` events. |
| 3 | `btc_complex_multi_fact_sse_first_events` | SSE HTTP | Multi-deliverable Bitcoin query: count, earliest 5, latest 5, min/max time, Flux and row counts | Partial pass | HTTP `200`; first chunk at `0.135s`; received initial SSE events, but this short probe did not wait for final answer. |
| 4 | `btc_usd_max_sse_90s_probe` | SSE HTTP | Same Bitcoin max question; probe until first non-heartbeat action | Partial pass | HTTP `200`; first chunk at `0.127s`; first `thought` at `3.513s`. |
| 5 | `btc_usd_max_sse_full` | SSE HTTP | Same Bitcoin max question; full completion probe | Failed | After `306.633s`, received 23 ReAct rounds, 23 `tool_call`, 22 `tool_result`, but no `final_answer` or `error`. |

Artifacts:

- `cache_data/e2e_runs/real_http_harness_refactor_2026-07-30/btc_usd_max.json`
- `cache_data/e2e_runs/real_http_harness_refactor_2026-07-30/btc_usd_max_sse_first_events.json`
- `cache_data/e2e_runs/real_http_harness_refactor_2026-07-30/btc_complex_multi_fact_sse_first_events.json`
- `cache_data/e2e_runs/real_http_harness_refactor_2026-07-30/btc_usd_max_sse_90s_probe.json`
- `cache_data/e2e_runs/real_http_harness_refactor_2026-07-30/btc_usd_max_sse_full.json`

## Timing Breakdown

From `btc_usd_max_sse_full`:

```text
first_chunk_seconds=0.132
first thought/action generation completed around 3.731s
first sql_query tool_result returned at 11.393s
probe interrupted at 306.633s
terminal final_answer/error was not reached
```

Event counts:

```text
conversation_id: 1
agent_step: 307
step.start: 23
thought: 23
step.chunk: 23
step.meta: 23
tool_call: 23
tool_result: 22
step.done: 22
final_answer: 0
error: 0
```

This means the main delay is not simply “the first action takes 30s”. In the full real run, the first action appeared in about 3.7s. The actual failure mode is repeated ReAct retries caused by failed SQL generation/repair.

## Failure Evidence

First failed tool result from the real SSE full run:

```text
Tool 'sql_query' failed: Generated query execution failed:
runtime error @6:6-6:26: max: no column "price" exists
Repair hint: rewrite the read-only query based on the error.
```

The first tool call asked for a Flux query equivalent to:

```text
from coindesk, compute Bitcoin USD max using price
```

The generated query attempted to aggregate `max(column: "price")`.

Real InfluxDB schema/value probe showed:

```text
_measurement = coindesk
_field = price
code = USD
crypto = bitcoin
numeric value column = _value
```

Correct direct Flux verification:

```flux
from(bucket: "bitcoin")
  |> range(start: 0)
  |> filter(fn: (r) => r._measurement == "coindesk" and r._field == "price" and r.code == "USD" and r.crypto == "bitcoin")
  |> max(column: "_value")
```

Direct database result:

```text
_time=2023-01-04T23:04:00Z
_value=168249475888010.0
_field=price
code=USD
crypto=bitcoin
rows=1
```

## Diagnosis

This is a system-level robustness issue, not a single local bug.

The failing chain is:

```text
schema context too shallow
  -> LLM/sql_query treats logical field name "price" as a physical value column
  -> Flux execution fails because InfluxDB stores values in _value
  -> tool observation recommends another sql_query
  -> model repeats same repair direction
  -> ReAct loop burns iterations/time and never reaches terminate
```

The previous fake E2E passed because fake tools/LLM provided successful observations. The real HTTP path exposes that the schema-linking and query-repair contract is not strong enough for real Flux execution.

## Frontend-Relevant Finding

If the frontend sends non-stream JSON, it will receive no intermediate response while the backend loop runs. That matches the observed “前端没任何返回”.

If the frontend sends `stream: true`, the backend can emit:

```text
conversation_id at ~0.13s
agent_step heartbeat every ~1s
thought/tool events when available
```

However, streaming only fixes visibility. It does not fix completion. The backend still fails to finish because the ReAct loop repeats invalid SQL repair.

## Fix Direction

Do not fix this with a Bitcoin-specific rule or a deterministic fallback.

The robust fix should target the generic SQL/query capability:

1. Make schema linking produce physical-column semantics, not just measurement/table names.
   - For InfluxDB, distinguish `_field` values from physical value columns like `_value`.
   - For SQL-like DBs, distinguish table columns, measures, dimensions, and aggregate targets.
2. Make `sql_query` expose a structured query contract to the LLM/tool layer:
   - logical measure requested
   - physical value column
   - required filters/tags
   - query language/dialect rules
3. Make repeated tool failures part of state transition/action-space logic.
   - Repeated same failure signature should not blindly recommend the same action without new evidence.
   - The next valid action should be schema inspection / query repair with explicit failed mapping context, not another unconstrained query attempt.
4. Keep ReAct as the core loop.
   - The loop should still be: observe state, choose one action, execute tool, update state.
   - The change should improve the observation/action-space contract so the LLM has enough structured evidence to choose the next correct action.

