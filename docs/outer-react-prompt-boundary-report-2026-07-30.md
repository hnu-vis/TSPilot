# Outer ReAct Prompt Boundary E2E Report - 2026-07-30

## Scope

本轮修复目标是把外层 ReAct agent 收敛为 DB-GPT 类似的 tool-calling 控制层：

- 外层只选择下一步工具和传自然语言参数。
- schema linking、query generation、dialect、repair、SQL/Flux/PromQL 细节留在工具内部。
- observation 给外层模型的是紧凑状态和 artifact refs，不是完整工具内部执行链路。
- 保持核心 ReAct 流程，不改成批量多输出一次性执行。

## Code checks

Command:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m py_compile prompts/data_agent.py runtime/action_policy.py core/harness/capabilities.py tools/registry.py
```

Result: passed.

## Prompt boundary snapshot

Constructed a real `RequestStateModel` for:

```text
bitcoin 的 usd 最大值是多少
```

Selected database:

```json
{
  "database_id": "influxdb2-bitcoin-sample",
  "database_type": "influxdb"
}
```

Result:

```json
{
  "prompt_chars": 4626,
  "leaks": []
}
```

Checked absent from model-visible user prompt:

- `schema_linking`
- `query_trace`
- `query_task_contract`
- `from(bucket`
- `|>`
- `SELECT `
- `_field`
- `_value`

## Action policy checks

Result:

```json
{
  "sql_query_explicit_query": [false, "rejected"],
  "sql_query_natural": [true, null],
  "todowrite_query_code": [false, "rejected"],
  "todowrite_natural": [true, null]
}
```

Meaning:

- `sql_query` only accepts natural-language outer action input unless runtime explicitly authorizes an exception.
- `todowrite` rejects database query code/dialect syntax in todo content.

## Real HTTP E2E: simple query

Artifact:

```text
cache_data/e2e_runs/outer_react_boundary_2026-07-30/btc_usd_max_sse.json
```

User request:

```text
bitcoin 的 usd 最大值是多少
```

Result:

```json
{
  "elapsed_seconds": 18.382,
  "final_event": "final_answer",
  "bad_sql_boundary_actions": [],
  "bad_todo_query_actions": []
}
```

Observed chain:

```text
sql_query -> terminate
```

First tool call occurred at about `6.8s`. The `sql_query` action input was natural-language only:

```json
{
  "message": "查询比特币的美元最大值",
  "purpose": "获取比特币的历史价格数据以确定其最大值",
  "time_range": null,
  "constraints": {},
  "insight_requests": null
}
```

## Real HTTP E2E: complex multi-step task

Artifact:

```text
cache_data/e2e_runs/outer_react_boundary_2026-07-30/btc_complex_todo_sse.json
```

User request:

```text
请做一个todo list：1. 查询 Bitcoin USD 的最大值和发生时间；2. 计算最近一段时间的平均值和波动；3. 判断是否存在异常点；4. 预测未来24小时趋势；5. 最后解释这些结果之间的关系。
```

Result:

```json
{
  "elapsed_seconds": 64.147,
  "final_event": "final_answer",
  "bad_sql_boundary_actions": [],
  "bad_todo_query_actions": []
}
```

Observed chain:

```text
todowrite -> sql_query -> sql_query -> anomaly(policy blocked) -> code_interpreter -> forecast(failed invalid model) -> forecast -> todowrite(policy blocked) -> terminate
```

Functional result: completed with final answer.

Boundary result:

- No public `sql_query` action input contained explicit query fields or query syntax.
- No `todowrite` action input contained query syntax.

Non-blocking issues found:

1. The model still attempted `anomaly` once when runtime required `code_interpreter` first. Policy blocked it and the loop recovered.
2. The model supplied an invalid forecast model name once. The next forecast call recovered with the available model.
3. The model attempted `todowrite` again after a plan already existed. Policy blocked it and the loop recovered.
4. SSE `tool_result.summary` and final response artifact sections still expose the actual executed query text to the frontend/logs. This is separate from model-visible prompt boundary. If the product requirement is also "frontend must not show query code", stream/final response sanitization should be handled separately.

## Conclusion

The outer ReAct prompt boundary is now materially tighter:

- Prompt state is compact and no longer duplicates full runtime state plus previous transcript.
- `sql_query` action card is natural-language oriented.
- Tool internals are moved behind artifact refs and compact summaries.
- Runtime policy rejects direct outer query fields and todo query-code contamination.
- Real HTTP E2E still completes both simple and complex tasks.

