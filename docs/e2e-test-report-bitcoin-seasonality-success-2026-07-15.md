# Bitcoin USD 周期性端到端测试报告

日期: 2026-07-15  
项目: `TSPilot-v0.2`  
测试类型: 真实 HTTP 端到端测试  
服务入口: `http://127.0.0.1:5680`

## 1. 测试目标

验证系统是否能够针对以下真实请求完成数据库取数、确定性分析和最终回答：

> 请判断 Bitcoin USD 在 2023 年 1 月 4 日 23:04:00 UTC 到 2023 年 2 月 3 日 22:47:00 UTC 这个历史数据集内有没有明显每天或每周重复的周期性波动。请严格基于数据库数据分析，并展示执行过程。

## 2. 真实请求

### 2.1 Health Check

请求:

```http
GET /health
```

响应:

```json
{
  "status": "ok"
}
```

### 2.2 Chat Request

请求:

```http
POST /api/v1/chat
Content-Type: application/json
```

请求体:

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

## 3. 真实响应概览

本次成功完成的真实响应文件:

- [cache_data/bitcoin_e2e_attempt_1.json](/home/feilvvl/TSPilot-v0.2/cache_data/bitcoin_e2e_attempt_1.json)

关键字段:

```json
{
  "conversation_id": "conv_41f194bfe31f",
  "request_id": "req_3cb912d771a4",
  "status": "completed",
  "response_kind": "final_answer",
  "used_tools": [
    "query_database",
    "insight",
    "anomaly",
    "insight",
    "insight",
    "format_answer"
  ]
}
```

最终摘要:

```text
value 在该时间范围内没有明显周期性。异常检测发现 6 个异常点，典型异常包括
2023-01-04T23:04:00+00:00=163899553938400.0,
2023-01-04T23:21:00+00:00=162569929683130.0,
2023-01-04T23:04:00+00:00=140587916056220.0。
```

## 4. 执行过程

### Step 1: `query_database`

Action Input:

```json
{
  "message": "请从数据库中提取 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 的历史时间序列数据，用于分析是否存在明显的日周期或周周期重复波动。请尽量控制返回点数不超过 240，并保留足够的时间戳精度以便后续做周期性分析。",
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

生成的真实 Flux:

```flux
from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
```

Observation Summary:

```text
Loaded 8037 rows across 1 series for query 'from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")'.
```

关键输出:

- `result_type = "timeseries"`
- `query_language = "flux"`
- `row_count = 8037`
- `series_count = 1`

### Step 2: `insight`

Action Input:

```json
{
  "message": "基于已加载的 Bitcoin USD 时序数据（2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z，8037 点），请验证并输出与季节性相关的事实：1) 是否存在明显的日周期（24小时附近）重复波动；2) 是否存在明显的周周期（7天附近）重复波动；3) 如有，请给出支持或反对的数据库证据特征（例如分时/分日模式、峰谷是否稳定重复、幅度是否显著）；4) 如无，请明确说明在该数据范围内未观察到稳定重复模式。请严格基于已提供的数据库时序证据，不要引入外部知识."
}
```

Observation Summary:

```text
InsightTool
```

关键验证事实:

```json
{
  "fact_type": "seasonality",
  "statement": "value 在该时间范围内没有明显周期性。",
  "evidence": {
    "period": null,
    "autocorrelation": 0.0,
    "strength": 0.0,
    "has_seasonality": false
  }
}
```

### Step 3: `anomaly`

Action Input:

```json
{
  "message": "基于已查询到的 Bitcoin USD 原始时序数据，检测该时间范围内是否存在明显的日周期或周周期重复波动，并输出可用于判断 seasonality 的证据。请严格基于数据库证据，不要使用外部知识。",
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
  },
  "focus": "判断 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 之间是否有明显每天或每周重复的周期性波动。",
  "requested_fact_types": [
    "seasonality"
  ]
}
```

Observation Summary:

```text
AnomalyTool
```

关键输出:

- `detector = "zscore"`
- `anomaly_points = 6`
- 典型异常值:
  - `2023-01-04T23:04:00+00:00 = 163899553938400.0`
  - `2023-01-04T23:21:00+00:00 = 162569929683130.0`
  - `2023-01-04T23:04:00+00:00 = 140587916056220.0`

### Step 4: `insight`

Action Input:

```json
{
  "fact_types": [
    "seasonality"
  ],
  "evidence_refs": [
    "evi_influxdb2-bitcoin-sample_from(bucket:_\"bitcoin\")..."
  ],
  "focus": "判断 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 内是否存在明显每天或每周重复的周期性波动，并给出可验证结论。"
}
```

Observation Summary:

```text
InsightTool
```

### Step 5: `insight`

Action Input:

```json
{
  "message": "基于已加载的 Bitcoin USD 时序数据库证据，判断是否存在明显每天或每周重复的周期性波动；请严格只使用数据库证据并给出可验证结论与分析过程。",
  "evidence_ids": [
    "evi_influxdb2-bitcoin-sample_from(bucket:_\"bitcoin\")..."
  ],
  "requested_fact_types": [
    "seasonality"
  ],
  "focus": "请判断 Bitcoin USD 在 2023 年 1 月 4 日 23:04:00 UTC 到 2023 年 2 月 3 日 22:47:00 UTC这个历史数据集内有没有明显每天或每周重复的周期性波动。请严格基于数据库数据分析，并展示执行过程。"
}
```

Observation Summary:

```text
InsightTool
```

### Step 6: `format_answer`

Action Input:

```json
{
  "answer": "基于数据库中的这段时序数据（2023-01-04 23:04:00 UTC 到 2023-02-03 22:47:00 UTC，8037 个点），没有发现 Bitcoin USD 存在明显的每天或每周重复周期性波动。\n\n执行过程（严格基于数据库数据）：\n1. 从 influxdb2-bitcoin-sample 的 bitcoin bucket 查询 coindesk.measurement 下的 price 字段。\n2. 时间范围限定为 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z。\n3. 数据共加载 8037 行、单一时间序列。\n4. 对该序列做周期性/季节性验证，结果为：\n   - has_seasonality = false\n   - autocorrelation = 0.0\n   - strength = 0.0\n   - period = null\n5. 同时查看时序趋势与异常点结果，未见稳定重复的日内或周内起伏模式；前端异常主要集中在样本起始处的少量极端值，不构成周期性证据。\n\n结论：在该数据集内，没有明显的日周期或周周期重复波动。",
  "verdict": "no_clear_daily_or_weekly_seasonality",
  "confidence": 0.45
}
```

Observation Summary:

```text
value 在该时间范围内没有明显周期性。异常检测发现 6 个异常点，典型异常包括
2023-01-04T23:04:00+00:00=163899553938400.0,
2023-01-04T23:21:00+00:00=162569929683130.0,
2023-01-04T23:04:00+00:00=140587916056220.0。
```

## 5. 测试结论

### 5.1 链路层面

本次真实端到端测试已经成功完成：

- `GET /health` 返回 `200`
- `POST /api/v1/chat` 返回 `200`
- 成功经过 `query_database -> insight -> anomaly -> format_answer`
- 未再出现 `context_length_exceeded`

### 5.2 业务结论

系统最终给出的结论是：

```text
在该数据范围内，没有明显的日周期或周周期重复波动。
```

### 5.3 关键风险与缺陷

这份成功响应**不应直接视为“Bitcoin USD 已被严格正确分析”**，原因如下：

1. 查询语句只过滤了：
   - `_measurement == "coindesk"`
   - `_field == "price"`

2. 但返回行中可以看到标签列包含：
   - `code = "EUR"`
   - `crypto = "bitcoin"`

3. 说明当前真实查询**没有明确过滤 `code = "USD"`**，而是把 `coindesk` 下多个币种的 `price` 数据混在一起分析。

4. 异常检测里出现的极大数值：
   - `163899553938400.0`
   - `162569929683130.0`
   - `140587916056220.0`

   也说明当前序列中存在明显异常或异构数据，进一步削弱了“Bitcoin USD 周期性结论”的可信度。

因此，严格表述应为：

> 当前链路已经能完成真实数据库查询、分析和回答装配；  
> 但这次成功回答分析的是 `coindesk.price` 的混合序列，而不是被严格限定为 `Bitcoin USD` 的纯净 USD 序列。

## 6. 建议的后续修复

优先级最高的修复项：

1. 在 Influx 查询规划中显式加入 `code == "USD"`
2. 必要时同时约束 `crypto == "bitcoin"`
3. 在 seasonality 分析前增加标签过滤校验，避免混合序列直接进入 `insight`
4. 对异常极值增加数据质量检查，避免错误量纲污染周期判断

## 7. 附件

- 成功响应原文: [cache_data/bitcoin_e2e_attempt_1.json](/home/feilvvl/TSPilot-v0.2/cache_data/bitcoin_e2e_attempt_1.json)
- 之前失败样例: [cache_data/bitcoin_e2e_response_2026-07-15.json](/home/feilvvl/TSPilot-v0.2/cache_data/bitcoin_e2e_response_2026-07-15.json)
