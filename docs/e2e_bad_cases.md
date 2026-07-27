# E2E Bad Case List

This file tracks real backend E2E cases that exposed regressions or unstable behavior.
Each entry should keep the original user request, datasource, observed tool chain, failure mode,
root cause, fix status, and retest result.

## Status Legend

- `open`: not fixed or not diagnosed
- `fixed`: code fix landed
- `retest_passed`: fixed and verified by a later E2E run
- `flaky`: behavior is inconsistent across repeated runs

## Cases

| ID | Status | Area | User Request | Expected Chain | Observed Problem | Root Cause | Fix / Next Step | Last Retest |
|---|---|---|---|---|---|---|---|---|
| E2E-20260727-001 | `retest_passed` | forecast, terminate gate | 查询 `appliances_energy_wh` 1 天历史，并预测接下来 1 天趋势 | `sql_query -> forecast/code_interpreter -> terminate` | First run failed after repeated `terminate` attempts; final status was `failed` after max iterations. | `available_actions` described `terminate` without `unavailable_outputs/unavailable_reason`, while policy required those fields for explicit unavailable outputs. | Updated `prompts/data_agent.py` terminate action contract to include `unavailable_outputs` and `unavailable_reason`. | Retest passed. Chain: `sql_query -> forecast... -> code_interpreter -> terminate`; no `insight`/`format_answer`. |
| E2E-20260727-002 | `fixed` | gap assessment, no active todo | 查询 `appliances_energy_wh` 1 天数据并检测异常点 | `sql_query -> anomaly -> terminate` | Run loop produced repeated `todo_assessment` failures: `No active todo step`; `anomaly` action was never executed after the first query. | `previous_observation_assessment.completed_active_todo=true` was treated as a blocking todo-completion failure even when there was no active todo. Gap assessment and todo completion were conflated. | Updated `runtime/react_loop.py` to block on failed assessment only when an active todo existed before assessment. Added regression test `test_gap_assessment_without_active_todo_does_not_block_action_execution`. | Unit retest passed. HTTP retest was interrupted and should be rerun. |
| E2E-20260727-003 | `flaky` | query/analysis latency | 查询 `appliances_energy_wh` 1 天的起始值、结束值、涨跌幅、最高值和最低值 | `sql_query -> terminate` or `sql_query -> sql_query -> terminate` | One batch run timed out at 180s. A later retest passed in 28.64s. | Likely same no-active-todo assessment loop or LLM/tool latency during the previous run; later trace completed cleanly after loop fix. | Keep as flaky and rerun in future batches. If it recurs, inspect trace for policy blocks and repeated terminal/action loops. | Retest passed. Chain: `sql_query -> terminate`; no old actions. |
| E2E-20260727-004 | `open` | multi-deliverable todo | 查询 `appliances_energy_wh` 1 天数据，并完成：总记录数、最早5条、最晚5条、最早/最晚时间、每项查询语句和返回行数 | `todowrite -> sql_query... -> terminate` | Batch run timed out at 240s. | Not yet diagnosed. Suspect todo/gap-assessment interaction or excessive step expansion. | Rerun after E2E-20260727-002 fix. Capture full trace and policy block summaries. | Not retested after fix. |

## Full Request Payloads

### E2E-20260727-001

```json
{
  "message": "查询 appliances_energy_wh 在 2016-01-11 17:00:00 到 2016-01-12 17:00:00 的数据，并预测接下来 1 天趋势。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T17:00:00"
  },
  "stream": false
}
```

### E2E-20260727-002

```json
{
  "message": "查询 appliances_energy_wh 在 2016-01-11 17:00:00 到 2016-01-12 17:00:00 的数据，并检测异常点。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T17:00:00"
  },
  "stream": false
}
```

### E2E-20260727-003

```json
{
  "message": "查询 appliances_energy_wh 在 2016-01-11 17:00:00 到 2016-01-12 17:00:00 的起始值、结束值、涨跌幅、最高值和最低值。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T17:00:00"
  },
  "stream": false
}
```

### E2E-20260727-004

```json
{
  "message": "请查询 appliances_energy_wh 在 2016-01-11 17:00:00 到 2016-01-12 17:00:00，并完成：1.返回总记录数；2.返回最早5条；3.返回最晚5条；4.返回最早和最晚时间；5.展示每项查询语句和实际返回行数。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T17:00:00"
  },
  "stream": false
}
```

## Maintenance Rules

When adding a new bad case:

1. Keep the original user request exactly as submitted.
2. Record the expected tool chain at capability level, not a hardcoded sequence unless the request requires a specific tool.
3. Record observed actions and observations, especially repeated policy blocks or old action names.
4. Prefer marking a case `fixed` only after a code change and `retest_passed` only after a real HTTP E2E rerun.
5. If behavior differs between runs, mark `flaky` and keep both timings.
