# 端到端测试报告（Bitcoin USD 周期性分析，真实 HTTP 请求）

## 1. 测试目标

验证 `TSPilot-v0.1` 在真实服务进程、真实外部 HTTP 请求、真实模型配置下，是否能够基于数据库证据判断以下问题：

```text
请判断 Bitcoin USD 在 2023 年 1 月 4 日 23:04:00 UTC 到 2023 年 2 月 3 日 22:47:00 UTC 这个历史数据集内有没有明显每天或每周重复的周期性波动。请严格基于数据库数据分析，并展示执行过程。
```

本次测试重点观察：

1. 真实服务是否能接收并执行该请求
2. `query_database` 是否能在指定时间范围内取回 Bitcoin USD 时序证据
3. 若不能完成分析，系统是否会进入受控失败终态

## 2. 测试环境

- 服务启动方式：
  - `/home/feilvvl/TSPilot/tspilot_env/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 5680`
- 服务地址：
  - `http://127.0.0.1:5680`
- 请求方式：
  - 独立 Python 进程通过 `httpx` 发起外部 HTTP 请求
- 请求时间：
  - 2026-07-15

## 3. 健康检查

请求：

```http
GET /health HTTP/1.1
Host: 127.0.0.1:5680
```

响应：

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"status":"ok"}
```

## 4. 主请求

请求：

```http
POST /api/v1/chat HTTP/1.1
Host: 127.0.0.1:5680
Content-Type: application/json
```

请求体：

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

响应总览：

```json
{
  "http_status": 200,
  "conversation_id": "conv_c39111917378",
  "request_id": "req_b4a6ccef1337",
  "status": "failed",
  "response_kind": "error",
  "used_tools": [
    "query_database",
    "query_database",
    "query_database",
    "query_database"
  ]
}
```

最终失败消息：

```text
我目前无法继续做出基于数据库证据的周期性判断，因为现有针对 Bitcoin USD 的查询都返回空结果。请确认：1) 该数据源中 Bitcoin USD 的正确 measurement / field / bucket 是否为当前使用的 `bitcoin / coindesk / _value`；2) 是否需要我改用其他时间粒度或标签条件重新查询。确认后我可以继续严格基于数据库数据分析。
```

## 5. 逐步输入输出

### Step 1: `query_database`

输入：

```json
{
  "message": "请从 influxdb2-bitcoin-sample 中查询 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 的历史时间序列数据，用于判断是否存在明显的每天或每周重复周期性波动。请尽量在 max_points=240 限制内返回可用于周期性分析的采样数据，并保留时间戳与价格值。",
  "database_context": {
    "database_id": "influxdb2-bitcoin-sample",
    "database_type": "influxdb",
    "display_name": null,
    "connection_hint": null,
    "selected_at": null
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

输出：

```json
{
  "success": true,
  "summary": "The query completed but returned no rows.",
  "result_type": "table",
  "query_language": "influxdb",
  "query": "from(bucket: \"bitcoin\") |> range(start: -7d) |> filter(fn: (r) => r._measurement == \"coindesk\") |> filter(fn: (r) => r._field == \"_value\") |> mean()",
  "data": {
    "rows": []
  },
  "columns": []
}
```

### Step 2: `query_database`

输入：

```json
{
  "message": "请检索 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 之间的历史价格数据，用于判断是否存在明显的每天或每周重复周期性波动。请尽量返回适合季节性分析的时间序列点，且总点数不超过 240。",
  "database_context": {
    "database_id": "influxdb2-bitcoin-sample",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2023-01-04T23:04:00Z",
    "end": "2023-02-03T22:47:00Z"
  },
  "constraints": {
    "max_points": 240,
    "requested_insight_types": ["seasonality"],
    "focus": "判断是否存在明显每天或每周重复的周期性波动"
  }
}
```

输出：

```json
{
  "success": true,
  "summary": "The query completed but returned no rows.",
  "result_type": "table",
  "query_language": "influxdb",
  "query": "from(bucket: \"bitcoin\") |> range(start: -7d) |> filter(fn: (r) => r._measurement == \"coindesk\") |> filter(fn: (r) => r._field == \"_value\")",
  "data": {
    "rows": []
  },
  "columns": []
}
```

### Step 3: `query_database`

输入：

```json
{
  "message": "Retrieval needed to assess daily/weekly seasonality of Bitcoin USD strictly from database data for the specified historical window. Please return the time-series rows within the target range with enough resolution to analyze repeated daily or weekly patterns.",
  "database_context": {
    "database_id": "influxdb2-bitcoin-sample",
    "database_type": "influxdb",
    "display_name": null,
    "connection_hint": null,
    "selected_at": null
  },
  "time_range": {
    "start": "2023-01-04T23:04:00Z",
    "end": "2023-02-03T22:47:00Z"
  },
  "constraints": {
    "max_points": 240,
    "requested_insight_types": ["seasonality"],
    "focus": "请判断 Bitcoin USD 在 2023 年 1 月 4 日 23:04:00 UTC 到 2023 年 2 月 3 日 22:47:00 UTC这个历史数据集内有没有明显每天或每周重复的周期性波动。请严格基于数据库数据分析，并展示执行过程。"
  }
}
```

输出：

```json
{
  "success": true,
  "summary": "The query completed but returned no rows.",
  "result_type": "table",
  "query_language": "influxdb",
  "query": "from(bucket: \"bitcoin\") |> range(start: -7d) |> filter(fn: (r) => r._measurement == \"coindesk\") |> filter(fn: (r) => r._field == \"_value\")",
  "data": {
    "rows": []
  },
  "columns": []
}
```

### Step 4: `query_database`

输入：

```json
{
  "message": "请从数据库 influxdb2-bitcoin-sample 中取回 Bitcoin USD 在 2023-01-04T23:04:00Z 到 2023-02-03T22:47:00Z 的历史时间序列，用于判断是否存在明显的每日或每周周期性波动。请尽量控制返回点数不超过 240，并返回足够用于后续周期性分析的数据。",
  "database_context": {
    "database_id": "influxdb2-bitcoin-sample",
    "database_type": "influxdb",
    "display_name": null,
    "connection_hint": null,
    "selected_at": null
  },
  "time_range": {
    "start": "2023-01-04T23:04:00Z",
    "end": "2023-02-03T22:47:00Z"
  },
  "constraints": {
    "max_points": 240,
    "metric": "Bitcoin USD",
    "analysis_target": "seasonality",
    "desired_granularity": "as available"
  }
}
```

输出：

```json
{
  "success": true,
  "summary": "The query completed but returned no rows.",
  "result_type": "table",
  "query_language": "influxdb",
  "query": "from(bucket: \"bitcoin\") |> range(start: -7d) |> filter(fn: (r) => r._measurement == \"coindesk\") |> filter(fn: (r) => r._field == \"_value\")",
  "data": {
    "rows": []
  },
  "columns": []
}
```

### Step 5: 终止判断

输出：

```json
{
  "message": "我目前无法继续做出基于数据库证据的周期性判断，因为现有针对 Bitcoin USD 的查询都返回空结果。请确认：1) 该数据源中 Bitcoin USD 的正确 measurement / field / bucket 是否为当前使用的 `bitcoin / coindesk / _value`；2) 是否需要我改用其他时间粒度或标签条件重新查询。确认后我可以继续严格基于数据库数据分析。"
}
```

## 6. 关键观察

### 6.1 服务层面

- 服务启动正常
- `/health` 返回 `200`
- 主请求返回 `200`
- 业务终态不是 `completed`，而是受控失败

### 6.2 查询层面

这次真实运行中，4 次 `query_database` 都返回空结果。

共同点：

- 数据源：`influxdb2-bitcoin-sample`
- measurement 被草拟为：`coindesk`
- field 被草拟为：`_value`

明显问题：

- 查询草稿中固定使用了 `range(start: -7d)`，没有采用用户请求里的绝对时间范围 `2023-01-04T23:04:00Z` 到 `2023-02-03T22:47:00Z`
- 第 1 次查询还因为 message 包含“判断/周期性分析”，被错误路由成了聚合查询并附加了 `|> mean()`

这说明当前这条链路在 Bitcoin USD 历史样本数据上存在两个现实缺口：

1. `query_database` 没有稳定把这类自然语言请求映射到正确的 measurement / field / label 条件
2. InfluxDB 通用查询草稿没有把用户提供的绝对时间范围真正下推到查询中

### 6.3 分析层面

因为没有拿到任何有效数据库时序证据：

- `insight` 没有被调用
- `anomaly` 没有被调用
- `forecast` 没有被调用

因此本次请求**没有实际进入季节性/周期性分析阶段**。

## 7. 测试结论

这次真实 E2E 测试完成了以下验证：

- 外部 HTTP 请求链路正常
- `data_agent -> query_database -> error` 的控制流可被受控收束
- 系统能在没有证据时进入受控失败终态，而不是伪造分析结论

但这次测试也明确表明，当前版本**还不能可靠完成这条 Bitcoin USD 周期性分析请求**，原因在于：

- 查询草稿未正确落实绝对时间范围
- 数据字段/measurement 推断不足
- 因为空结果，周期性判断没有进入 `insight` 分析阶段

本次真实请求标识：

- `conversation_id=conv_c39111917378`
- `request_id=req_b4a6ccef1337`
