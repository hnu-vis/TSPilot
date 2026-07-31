# Action Output / ReAct Observation Refactor E2E Report - 2026-07-30

## Scope

This report covers the real HTTP/SSE end-to-end checks run after introducing DB-GPT-style `ActionOutput` as the canonical tool result boundary.

Primary goals:

- Keep outer ReAct flow as the core control loop.
- Ensure model-visible observations are compact and do not expose tool-internal query/schema-linking fields.
- Treat each tool result as a structured action output with model observation, public view, resource reference, and memory fragment.
- Verify simple database QA still works.
- Stress a multi-step task involving todo planning, database query, analysis, anomaly, forecast, and final explanation.

## Code changes exercised

- Added canonical action-output schema and builder:
  - `schemas/action_output.py`
  - `core/harness/action_output.py`
- Wired action outputs into execution/state/streaming:
  - `runtime/tool_executor.py`
  - `runtime/react_loop.py`
  - `runtime/request_state.py`
  - `schemas/state.py`
- Reduced backend model-visible observation/context:
  - `prompts/data_agent.py`
  - `core/harness/observation_view.py`
- Added robust task-contract handling:
  - initial multi-step `todowrite` must provide a parseable task contract;
  - later task-contract updates are monotonic and cannot silently delete existing required outputs;
  - terminal answer/conclusion outputs are treated as terminate-produced outputs, not pre-terminate artifacts.
- Added structured time-series input repair for specialized tools:
  - `tools/anomaly.py`
  - `tools/forecast.py`
  - `core/harness/action_space.py`

## Environment

- Backend: `uvicorn app.server:app --host 127.0.0.1 --port 18081`
- Python env: `/home/feilvvl/TSPilot/tspilot_env`
- API mode: real HTTP SSE, not unit mock.
- Database context reused from real `influxdb2-bitcoin-sample` request.

## Static validation

Command:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m py_compile runtime/action_policy.py core/harness/action_space.py prompts/data_agent.py schemas/action_output.py schemas/state.py core/harness/action_output.py runtime/tool_executor.py runtime/react_loop.py runtime/request_state.py core/completion.py tools/anomaly.py tools/forecast.py
```

Result: passed.

## E2E case 1: simple max query

Request:

```text
bitcoin 的 usd 最大值是多少
```

Output file:

```text
cache_data/e2e_runs/action_output_refactor_2026-07-30/btc_usd_max_sse_after_all_changes.json
```

Result:

- Status: passed.
- Elapsed: 15.9s.
- Final event: `final_answer`.
- Tool sequence: `sql_query -> terminate`.
- `sql_query` result: success, 1067 chars, no leaked internal query/schema-linking fields.
- `terminate` result: success, 210 chars.
- Final answer preview: `比特币的最大值是168249475888010.0美元。`

Backend prompt snapshot after the run:

- System prompt: 3049 chars.
- Context: 5631 chars.
- Latest model observation: 130 chars.
- Action outputs: 2.
- Memory fragments: 2.
- Resource refs:
  - `evidence:evi_influxdb2-bitcoin-sample_timeseries_e013577c0f8e`
  - `final_answer:req_681420bde069`
- Leak check: passed for `from(bucket`, `|>`, `SELECT`, `schema_linking`, `query_trace`, `query_task_contract`, `llm_query_generation`, `_field`, `_value`, `query_language`.

## E2E case 2: complex todo / analysis / anomaly / forecast

Request:

```text
请做一个todo list：1. 查询 Bitcoin USD 的最大值和发生时间；2. 计算最近一段时间的平均值和波动；3. 判断是否存在异常点；4. 预测未来24小时趋势；5. 最后解释这些结果之间的关系。
```

Runs:

1. `btc_complex_todo_sse.json`
2. `btc_complex_todo_sse_after_contract_gate.json`
3. `btc_complex_todo_sse_after_contract_merge.json`
4. `btc_complex_todo_sse_after_timeseries_repair.json` was interrupted after no final SSE event was received for more than 300s.

Observed progression:

- Before contract gate:
  - Returned `final_answer` in 48.1s.
  - Tool sequence included `todowrite`, `sql_query`, `code_interpreter`, `forecast`, `terminate`.
  - Problem: no `anomaly` call despite user asking for anomaly detection.
  - Root cause: `intent_profile` stayed at fallback `["query"]`; no task contract existed, so completion did not know anomaly was required.

- After requiring task contract on initial `todowrite`:
  - The model produced a 5-output task contract in `action_input.task_contract`.
  - `anomaly` and `forecast` became part of the enforced flow.
  - Problem: a later top-level `task_contract` update shrank the contract to one output, causing contract/todo mismatch.

- After monotonic task-contract merge:
  - Contract shrinkage was prevented.
  - Problem: `anomaly` was repeatedly forced on a single-row max-value evidence artifact.
  - Root cause: specialized tool input shape failure was not represented as a structured repair contract; action_space kept requiring the same specialized tool instead of requiring a raw time-series evidence query first.

- After adding structured time-series repair:
  - `anomaly`/`forecast` now emit `StructuredToolError(error_type="insufficient_timeseries_evidence")` with a `validation_failure.repair_contract` requiring `sql_query` with `raw_timeseries` evidence shape.
  - The final long run did not deliver a first/next terminal SSE event before manual interruption after 300s, so this specific complex E2E is still not considered passed.

## Current assessment

Passed:

- Simple database QA works end-to-end.
- ActionOutput is now the canonical internal tool-output record.
- Frontend-facing `tool_result` payloads are compact and sanitized.
- Backend model-visible observations are compact and sanitized.
- SQL/query/schema-linking internals are no longer exposed to the outer ReAct model in observation context.
- Task contracts are now protected from later LLM contract shrinkage.
- Specialized time-series tools can produce structured repair signals instead of opaque string errors.

Not yet passed:

- The complex todo/anomaly/forecast E2E is still too slow and did not complete in the last run.
- There is still a first-token/first-event latency problem in complex cases. In the last interrupted run, the HTTP request was accepted (`200 OK`) but the client did not receive a terminal SSE event within 300s.

Most likely remaining issue:

- The ReAct loop still depends on repeated full LLM turns for policy recovery and todo progression. Even with compact observations, the model can spend many turns recovering from evidence-shape issues.
- The next robust fix should make state transition/action_space consume `validation_failure.repair_contract` earlier and deterministically at the runtime policy level, so the next action is constrained to the prerequisite evidence query immediately after the specialized-tool shape failure.

