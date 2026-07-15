# Bitcoin USD 周期性 E2E 复测报告

日期: 2026-07-15

## 请求

```http
POST /api/v1/chat
```

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

## 响应概览

```json
{
  "http_status": 200,
  "conversation_id": "conv_93b73a7a61a0",
  "request_id": "req_8b00053c8ef5",
  "status": "completed",
  "response_kind": "final_answer",
  "used_tools": [
    "query_database",
    "insight",
    "anomaly",
    "format_answer"
  ]
}
```

完整响应已保存到：

- [cache_data/bitcoin_e2e_rerun_2026-07-15.json](/home/feilvvl/TSPilot-v0.2/cache_data/bitcoin_e2e_rerun_2026-07-15.json)

## 真实执行过程

### 1. `query_database`

生成的真实 Flux：

```flux
from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
```

结果：

- `result_type = "timeseries"`
- `row_count = 8037`
- `series_count = 1`

### 2. `insight`

请求的 fact type：

```json
["seasonality"]
```

已验证事实：

```json
{
  "fact_type": "seasonality",
  "statement": "value 在该时间范围内没有明显周期性。",
  "evidence": {
    "has_seasonality": false,
    "autocorrelation": 0.0,
    "strength": 0.0,
    "period": null
  }
}
```

### 3. 第一次 `format_answer` 被拒绝

运行时 observation：

```json
{
  "tool_name": "format_answer",
  "success": false,
  "summary": "Action 'format_answer' does not match the current state. Current actionable gap: anomaly analysis is still missing. Recommended next action(s): anomaly."
}
```

这说明经典 ReAct 控制流已经生效：

- 模型可以提出过早的 `action`
- 运行时不会终止请求
- 而是返回 `observation`，继续下一轮

### 4. `anomaly`

结果：

- `detector = "zscore"`
- `anomaly_points = 6`

典型异常点：

- `2023-01-04T23:04:00+00:00 = 163899553938400.0`
- `2023-01-04T23:21:00+00:00 = 162569929683130.0`
- `2023-01-04T23:04:00+00:00 = 140587916056220.0`

### 5. 第二次 `format_answer`

成功完成并返回最终答案。

最终摘要：

```text
value 在该时间范围内没有明显周期性。 异常检测发现 6 个异常点，典型异常包括 2023-01-04T23:04:00+00:00=163899553938400.0, 2023-01-04T23:21:00+00:00=162569929683130.0, 2023-01-04T23:04:00+00:00=140587916056220.0。
```

## 结论

这次复测表明：

- 真实 HTTP E2E 已跑通
- 去掉 `clarification` 后，链路仍能正常完成
- 控制流已经回到经典 ReAct：`thought -> action -> observation -> ... -> final_answer`
- 过早 `format_answer` 会被 runtime 拒绝并继续循环，不会走额外终态

## 当前剩余问题

这次业务结论仍然不能完全视为“严格的 Bitcoin USD 结论”，原因是：

- 查询没有加入 `code = "USD"` 过滤
- 也没有加入 `crypto = "bitcoin"` 过滤
- 最终 answer 仍然把目标称为 `value`，不是明确的 `Bitcoin USD`

从响应中的异常值也能看出，当前序列很可能混入了不应直接参与分析的记录：

- 正常价格量级约 `1.6e4`
- 异常值量级却到 `1.6e14`

所以这次更准确的判断是：

- **系统控制流已正确**
- **数据库查询已成功**
- **但字段/标签约束仍不够严格，业务语义还没有完全收敛到纯 Bitcoin USD 序列**
