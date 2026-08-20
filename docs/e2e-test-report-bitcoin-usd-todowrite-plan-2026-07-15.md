# Bitcoin USD 端到端测试报告（Todowrite Plan State）

日期: 2026-07-15

## 测试目标

验证两件事：

- 复杂时序分析请求是否会先进入 `todowrite`，形成可执行 plan state
- 后续执行是否会依照 todo plan 收口到 `query_database -> insight -> format_answer`，而不是无关扩展到 `forecast`

测试请求：

```json
{
  "message": "请判断 Bitcoin USD 在 2023 年 1 月 4 日 23:04:00 UTC 到 2023 年 2 月 3 日 22:47:00 UTC这个历史数据集内有没有明显每天或每周重复的周期性波动。请严格基于数据库数据分析，并展示执行过程。",
  "database_context": {
    "database_id": "influxdb2-bitcoin-sample",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2023-01-04T23:04:00Z",
    "end": "2023-02-03T22:47:00Z"
  },
  "constraints": {
    "max_points": 240
  }
}
```

## 服务健康检查

真实 `GET /health` 返回：

```json
{
  "status": "ok"
}
```

## 测试结果概览

本次真实 HTTP 请求成功完成。

关键结果：

```json
{
  "conversation_id": "conv_26c14ccdc856",
  "request_id": "req_078f2e919362",
  "status": "completed",
  "response_kind": "final_answer",
  "used_tools": [
    "todowrite",
    "query_database",
    "insight",
    "format_answer"
  ]
}
```

本次结果说明：

- 已经先执行 `todowrite`
- todo plan 收口为 3 步：`query -> insight -> answer`
- 没有越权调用 `forecast`
- 最终结论基于 `seasonality` insight，而不是趋势/预测模板

## 执行过程

### 1. `todowrite`

真实输入：

```json
{
  "message": "建立针对 Bitcoin USD 在指定时间范围内的季节性分析计划，严格基于数据库数据判断是否存在明显每日或每周重复周期。",
  "current_intent": "判断是否存在明显每天或每周重复的周期性波动",
  "requested_insight_types": [
    "seasonality"
  ],
  "focus": "Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 区间内的日/周周期性",
  "todos": [
    {
      "content": "查询该时间范围内的 Bitcoin USD 历史数据，控制点数不超过 240",
      "task_type": "query",
      "status": "in_progress",
      "priority": 1
    },
    {
      "content": "基于查询结果做季节性/周期性分析，检查日内与周内重复模式",
      "task_type": "insight",
      "status": "pending",
      "priority": 2
    },
    {
      "content": "整理并输出结论，明确说明是否存在明显每日或每周周期性，并展示执行过程",
      "task_type": "answer",
      "status": "pending",
      "priority": 3
    }
  ],
  "evidence_summary": "当前尚无数据库证据；需要先获取时间序列样本后再分析季节性。"
}
```

真实输出摘要：

```text
Todo plan updated with 3 steps. Current step: 查询该时间范围内的 Bitcoin USD 历史数据，控制点数不超过 240. Completed 0, remaining 3.
```

关键点：

- 计划状态被显式写入
- `task_type` 已收口为 `query / insight / answer`
- 没有把 `forecast` 写进 plan

### 2. `query_database`

真实输入：

```json
{
  "message": "查询 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 的历史价格数据，用于检查是否存在明显日周期或周周期重复波动。请尽量返回均匀采样结果，点数不超过 240。",
  "database_context": {
    "database_id": "influxdb2-bitcoin-sample",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2023-01-04T23:04:00Z",
    "end": "2023-02-03T22:47:00Z"
  },
  "constraints": {
    "max_points": 240
  }
}
```

真实生成的 Flux：

```flux
from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
  |> filter(fn: (r) => r.code == "USD")
  |> filter(fn: (r) => r.crypto == "bitcoin")
```

真实输出摘要：

```text
Loaded 2679 rows across 1 series for query 'from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
  |> filter(fn: (r) => r.code == "USD")
  |> filter(fn: (r) => r.crypto == "bitcoin")'.
```

关键输出：

- `result_type = "timeseries"`
- `row_count = 2679`
- `series_count = 1`
- `requested_insight_families = ["seasonality"]`

查询快照：

- [qry_influxdb2-bitcoin-sample_b7690376e354.json](../cache_data/query_snapshots/qry_influxdb2-bitcoin-sample_b7690376e354.json)

### 3. `insight`

真实输入：

```json
{
  "message": "基于已加载的 Bitcoin USD 时序数据，分析 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 区间内是否存在明显每日或每周周期性波动；请严格只依据数据库证据，输出可验证事实与判断依据。",
  "evidence_refs": [
    "latest_database_evidence"
  ],
  "requested_insight_types": [
    "seasonality"
  ],
  "focus": "检查日内与周内重复模式，确认是否存在明显周期性波动",
  "analysis_requirements": [
    "先检查数据点密度、时间覆盖和是否存在异常值或采样不均匀",
    "再基于现有证据判断日周期/周周期是否明显",
    "结论需区分“明显存在”“不明显/弱”与“证据不足”"
  ]
}
```

关键输出：

```json
{
  "requested_insight_types": [
    "seasonality"
  ],
  "supported_insight_types": [
    "seasonality"
  ],
  "verified_insight": {
    "insight_type": "seasonality",
    "statement": "value 在该时间范围内没有明显周期性。",
    "evidence": {
      "period": null,
      "autocorrelation": 0.0,
      "strength": 0.0,
      "has_seasonality": false
    }
  }
}
```

分析快照：

- [ins_evi_influxdb2-bitcoin-sample_from_bucket___bitcoin_____range_start__2023-01-_d336826bdaa2.json](../cache_data/analysis_snapshots/ins_evi_influxdb2-bitcoin-sample_from_bucket___bitcoin_____range_start__2023-01-_d336826bdaa2.json)

### 4. `format_answer`

真实输入：

```json
{
  "answer_style": "concise_with_process",
  "include_process": true,
  "include_caveats": true
}
```

真实输出摘要：

```text
value 在该时间范围内没有明显周期性。
```

最终答复状态：

- `has_insights = true`
- `has_anomaly = false`
- `has_forecast = false`

## 最终结论

基于这次真实数据库查询和后续确定性 insight 验证：

- 在 `2023-01-04T23:04:00Z` 到 `2023-02-03T22:47:00Z` 这段 `Bitcoin USD` 历史数据中，**没有发现明显的每日或每周重复周期性波动**
- 结论直接来自 `seasonality` insight：
  - `has_seasonality = false`
  - `period = null`
  - `autocorrelation = 0.0`
  - `strength = 0.0`

## 本次实现验证结果

这次测试验证了本轮改动已经达成：

- 复杂请求会先进入 `todowrite`
- todo plan 会收口到当前目标，而不是扩展成预测任务
- 后续动作会按当前 plan step 推进
- 最终工具链为：
  - `todowrite -> query_database -> insight -> format_answer`

## 仍然存在的边界

- `query_database` 实际返回了 `2679` 行，而不是严格下采样到 `240` 点；`max_points` 当前更像 prompt/plan 约束，不是数据库层硬裁剪
- 最终摘要里的主体仍然显示为 `value`，而不是更友好的 `Bitcoin USD`
- 返回数据起始处仍有两个极端大值，虽然这次没有把流程带偏，但后续如果做更严谨的季节性分析，最好增加异常值质量门控
