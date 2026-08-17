# Real HTTP/SSE E2E Repair Contract Report — 2026-07-30

## Scope

This report covers real end-to-end tests through the backend HTTP/SSE chat route:

- Server: `uvicorn app.server:app --host 127.0.0.1 --port 18081`
- Endpoint: `POST /api/v1/chat`
- Database: `influxdb2-bitcoin-sample`
- Database type: `influxdb`
- Environment: `/home/feilvvl/TSPilot/tspilot_env`

Frontend was not rebuilt.

## Code changes validated

1. Structured validation failures now carry a machine-readable `validation_failure.repair_contract`.
2. `ActionSpaceBuilder` converts the latest validation failure into a required repair action.
3. `ToolExecutor` merges required action `input_guidance` into tool input and injects repair contracts into SQL/code-interpreter calls.
4. `sql_query` repair mode no longer repeats an invalid explicit query; it routes through LLM query planning with the prior failure contract.
5. `todowrite` now expands numbered multi-deliverable requests from `message`/`focus`, including inline Chinese numbered lists.
6. SSE input preview is defensive against `null` tool fields so a bad preview cannot terminate the stream.

## Test commands

Compilation:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m py_compile \
  tools/todowrite.py runtime/tool_executor.py core/harness/action_space.py runtime/react_loop.py
```

Result: pass.

Local tool behavior:

- Input: a Chinese inline numbered request with five deliverables.
- Result: `todowrite` produced 5 todos; first todo `in_progress`; last todo `answer`.

Result: pass.

## Real E2E cases

### Case A — anomaly repair path

User request:

> 检测 bitcoin 的 usd 数据有没有异常点，并解释异常处理是否影响统计结论

Artifact:

- `cache_data/e2e_runs/real_http_repair_contract_2026-07-30/btc_anomaly_repair_sse.json`

Result:

- Status: pass
- Final answer received: yes
- Elapsed: 23.874s
- ReAct rounds: 4
- Coverage:
  - SQL evidence was acquired.
  - `code_interpreter` validation required transparent outlier-treatment fields.
  - Repair contract guided the next analysis action.
  - Final answer passed grounded-output checks.

### Case B — complex multi-insight/todo query before latest fix

User request:

> 请查询 bitcoin 的 usd 数据，并完成以下任务：1. 返回USD价格数据的总记录数；2. 返回按时间升序排列的最早5条原始记录；3. 返回按时间降序排列的最晚5条原始记录；4. 返回整个数据集的最早时间和最晚时间，精确到秒；5. 展示每项结果对应的完整Flux查询语句和实际返回行数。

Artifact:

- `cache_data/e2e_runs/real_http_repair_contract_2026-07-30/btc_multi_insight_todowrite_fresh_sse.json`

Result:

- Status: fail / timeout
- Timeout: 180.976s
- Final answer received: no
- Root cause:
  - The initial todo plan collapsed the five user-visible deliverables into one giant todo.
  - The ReAct loop repeatedly queried broad evidence because state progress had no per-deliverable structure.

### Case C — complex multi-insight/todo query after todo/action-guidance fix, before SSE-preview fix

Artifact:

- `cache_data/e2e_runs/real_http_repair_contract_2026-07-30/btc_multi_insight_contract_sse.json`

Result:

- Status: fail / stream exception
- Elapsed before disconnect: 149.088s
- Final answer received: no
- Improvement:
  - Initial `todowrite` produced 5 todos.
  - SQL repair recovered missing `code='USD'` and `crypto='bitcoin'` filters.
  - Todo progress advanced through the query steps.
- Root cause:
  - SSE mapping crashed on `len(action_input.get("todos", []))` when the model emitted `todos: null` for a `todowrite` action.
  - This caused backend stream termination without a user-visible final/error event.

### Case D — complex multi-insight/todo query after all fixes

Artifact:

- `cache_data/e2e_runs/real_http_repair_contract_2026-07-30/btc_multi_insight_sse_preview_fix.json`

Result:

- Status: pass
- HTTP status: 200
- Final answer received: yes
- Elapsed: 168.561s
- Event count: 267
- ReAct rounds started: 15
- Tool calls: 15
- Tool results: 15
- Final answer events: 1

Observed behavior:

- Initial `todowrite` produced 5 todos.
- First SQL query failed due to missing required filters; the failure included structured `validation_failure.repair_contract`.
- Subsequent SQL actions preserved required filters and produced database evidence.
- Todo progress advanced to the final answer step.
- SSE remained open and delivered `final_answer`.

Remaining quality issue:

- The successful run is still slow: 168.6s for a complex five-deliverable request.
- The loop still performs more SQL calls than ideal because the outer ReAct agent decomposes evidence acquisition step-by-step rather than batching compatible query outputs.
- This is a performance/planning-efficiency issue, not the previous functional deadlock or stream-disconnect issue.

