# Bitcoin USD 端到端测试报告

日期: 2026-07-15

## 测试目标

验证系统是否能够对以下真实请求完成端到端分析，并确保数据库查询满足两个关键要求：

- 查询严格过滤到 `Bitcoin USD`
- 周期性分析请求不再误路由成 `max()` 这类单值聚合

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

## 测试结果概览

本次测试基于真实 HTTP 请求，成功完成。

关键结果：

```json
{
  "conversation_id": "conv_5e89ed5687f2",
  "request_id": "req_c83ec25dd3e3",
  "status": "completed",
  "used_tools": [
    "query_database",
    "insight",
    "anomaly",
    "forecast",
    "format_answer"
  ]
}
```

完整响应文件：

- [cache_data/bitcoin_e2e_after_agg_fix_2026-07-15.json](/home/feilvvl/TSPilot-v0.2/cache_data/bitcoin_e2e_after_agg_fix_2026-07-15.json)

## 执行过程

### 1. `query_database`

生成的真实 Flux：

```flux
from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
  |> filter(fn: (r) => r.code == "USD")
  |> filter(fn: (r) => r.crypto == "bitcoin")
```

关键输出：

- `result_type = "timeseries"`
- `query_shape = "raw_timeseries"`
- `evidence_family = "timeseries"`
- `row_count = 2679`

Observation 摘要：

```text
Loaded 2679 rows across 1 series for query 'from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
  |> filter(fn: (r) => r.code == "USD")
  |> filter(fn: (r) => r.crypto == "bitcoin")'.
```

结论：

- 已经严格过滤到 `code = "USD"` 和 `crypto = "bitcoin"`
- 没有再误生成 `group() |> max()`

### 2. `insight`

本次 `insight` 产出了趋势、差值、极值三类事实。

关键结论摘要：

- `value` 在所选时间范围内整体下降
- 首尾差值约为 `-100.00%`
- 最高值出现在 `2023-01-04T23:04:00+00:00`
- 最低值出现在 `2023-01-06T13:10:00+00:00`

### 3. `anomaly`

异常检测工具成功执行。

关键输出：

- `detector = "zscore"`
- `anomaly_points = 2`

典型异常点：

- `2023-01-04T23:04:00+00:00 = 168249475888010.0`
- `2023-01-04T23:21:00+00:00 = 166884563179570.0`

### 4. `forecast`

预测工具成功执行。

关键输出：

- `model_name = "linear_regression"`
- `horizon = 12`

预测结果显示后续 12 个点继续下降，但数值被前述异常点明显拉偏。

### 5. `format_answer`

最终回答成功组装。

最终摘要：

```text
value 在所选时间范围内整体下降，变化幅度约为 -100.00%。 value 的最高值为 168249475888010.00（2023-01-04T23:04:00+00:00），最低值为 16702.30（2023-01-06T13:10:00+00:00）。 异常检测发现 2 个异常点，典型异常包括 2023-01-04T23:04:00+00:00=168249475888010.0, 2023-01-04T23:21:00+00:00=166884563179570.0。 value 的短期预测共 12 个点，预测区间内整体下降，从 -250053792504.43 变化到 -253133386209.12。
```

## 结论

本次修复后的端到端测试表明：

- `Bitcoin USD` 的值过滤已经真实生效
- 周期性分析请求已不再错误路由为 `max()` 单点聚合
- 主链路能够完成 `query_database -> insight -> anomaly -> forecast -> format_answer`

当前剩余问题：

- 数据中前两条点是极端异常值，明显影响趋势、预测和最终摘要
- 因此这次结果说明“查询规划问题已修复”，但“数据质量治理”仍需继续加强
