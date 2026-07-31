# Real HTTP E2E Report: Schema Linking Fix

- Date: 2026-07-30
- Workspace: `/home/feilvvl/TSPilot-v0.2`
- Environment: `/home/feilvvl/TSPilot/tspilot_env`
- Backend: real FastAPI on `127.0.0.1:18081`
- LLM: real configured `ChatOpenAI` from project `.env`
- Database: real InfluxDB on `localhost:18086`
- Frontend build: not executed

## Result

The real E2E regression for `bitcoin 的 usd 最大值是多少` now passes.

Before this fix, the same request timed out or repeated invalid `sql_query` for more than 20 ReAct rounds because Flux generated `max(column: "price")`.

After this fix:

- Influx schema preview exposes logical field values separately from physical value columns.
- schema linking exposes measure mappings such as `price -> _field == "price" -> _value`.
- Flux shape validation catches logical field values used as physical aggregate columns before execution.
- schema linking contract tolerates LLM string-list outputs such as `["price"]`.
- terminate input tolerates boolean include flags, avoiding extra repair rounds.
- The Influx physical model is now provided by the selected database dialect, not hard-coded in generic `schema.py`.

## Test Cases

| Case | Mode | Result | Timing | Event / tool summary |
|---|---|---:|---:|---|
| `btc_usd_max_sse_final_after_terminate_fix` | SSE HTTP | Passed | `19.157s` | first chunk `0.483s`; final_answer `19.157s`; 2 ReAct rounds; `sql_query -> terminate` |
| `btc_usd_max_json_final_after_fix` | JSON HTTP | Passed | `14.618s` | HTTP 200; status `completed`; response kind `final_answer`; used tools `["sql_query"]` |
| `btc_usd_max_json_after_dialect_preview_refactor` | JSON HTTP | Passed | `31.109s` | HTTP 200; status `completed`; response kind `final_answer`; used tools `["sql_query"]`; verifies dialect-owned schema preview extension |

Artifacts:

- `cache_data/e2e_runs/real_http_schema_linking_fix_2026-07-30/btc_usd_max_sse_final_after_terminate_fix.json`
- `cache_data/e2e_runs/real_http_schema_linking_fix_2026-07-30/btc_usd_max_json_final_after_fix.json`
- `cache_data/e2e_runs/real_http_schema_linking_fix_2026-07-30/btc_usd_max_json_after_dialect_preview_refactor.json`

## Decoupling Check

The generic schema preview no longer inspects `query_language == "flux"`.

Validation output:

```text
generic_has_physical_model=False
extended_has_physical_model=True
```

The caller resolves the dialect from the selected database type and passes it into schema preview assembly. The Influx dialect contributes the `_field/_value` physical model through its schema preview extension.

## Verified Query Shape

The successful real query used long-form Flux semantics:

```flux
from(bucket:"bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r["_field"] == "price")
  |> filter(fn: (r) => r["code"] == "USD")
  |> filter(fn: (r) => r["crypto"] == "bitcoin")
  |> max()
```

Returned value:

```text
168249475888010.0
```

This value is grounded in the real query result and registered facts, not computed in `terminate`.
