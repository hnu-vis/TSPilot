# 端到端测试报告（真实 HTTP 请求）

## 1. 测试目标

验证 `TSPilot-v0.2` 在真实服务进程、真实模型配置、真实外部 HTTP 请求下，是否能完成以下完整链路：

1. 接收用户问题
2. 规划任务
3. 查询时序数据
4. 提取趋势事实
5. 检查异常
6. 生成短期预测
7. 组装最终回答

本报告对应的是一次**真实执行**，不是 `TestClient`，也不是 fake LLM。

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
  "message": "先规划一下，然后分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势，检查异常，再给一个短期预测，最后总结结论。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T23:00:00"
  },
  "constraints": {
    "max_points": 48
  }
}
```

响应总览：

```json
{
  "http_status": 200,
  "conversation_id": "conv_765b07ba9a1b",
  "request_id": "req_bb00f33367bc",
  "used_tools": [
    "todowrite",
    "query_database",
    "insight",
    "anomaly",
    "forecast",
    "format_answer"
  ]
}
```

最终摘要：

```text
appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。 appliances_energy_wh 的最高值为 430.00，最低值为 30.00。 异常检测发现 4 个异常点，典型异常包括 2016-01-11T19:00:00=430.0, 2016-01-11T22:30:00=390.0, 2016-01-12T01:00:00=380.0。 appliances_energy_wh 的短期预测共 12 个点，预测区间内整体下降，从 71.01 变化到 59.68。
```

## 5. 逐步输入输出

### Step 1: `todowrite`

输入：

```json
{
  "message": "先规划：1) 查询 appliances_energy_wh 在 2016-01-11 17:00 到 2016-01-12 23:00 的时间序列数据；2) 基于已取回数据分析趋势、变化幅度和极值；3) 检查异常点；4) 做短期预测；5) 汇总结论。",
  "current_intent": "分析 appliances_energy_wh 在指定时间范围内的趋势、异常与短期预测并总结结论",
  "requested_fact_types": ["trend", "change_percent", "extrema"],
  "focus": "appliances_energy_wh 2016-01-11 到 2016-01-12",
  "todos": [
    {"content": "查询 appliances_energy_wh 指标在指定时间范围内的数据", "status": "pending", "priority": 1},
    {"content": "分析趋势、变化百分比和极值", "status": "pending", "priority": 2},
    {"content": "检查异常", "status": "pending", "priority": 3},
    {"content": "进行短期预测", "status": "pending", "priority": 4},
    {"content": "总结结论", "status": "pending", "priority": 5}
  ],
  "evidence_summary": "用户要求先规划，再对 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势、异常和短期预测进行分析。"
}
```

输出：

```json
{
  "success": true,
  "summary": "Todo plan updated with 5 steps. Current step: 查询 appliances_energy_wh 指标在指定时间范围内的数据. Completed 0, remaining 5.",
  "todo_total": 5,
  "in_progress": {
    "content": "查询 appliances_energy_wh 指标在指定时间范围内的数据",
    "status": "in_progress",
    "priority": 1,
    "notes": null
  },
  "completed_count": 0,
  "pending_count": 5
}
```

### Step 2: `query_database`

输入：

```json
{
  "message": "查询 appliances_energy_wh 在 2016-01-11T17:00:00 到 2016-01-12T23:00:00 的时间序列数据，用于后续趋势分析、异常检测和短期预测。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T23:00:00"
  },
  "constraints": {
    "max_points": 48,
    "metric": "appliances_energy_wh"
  }
}
```

输出：

```json
{
  "success": true,
  "summary": "Loaded 48 points for appliances_energy_wh from the configured reference dataset.",
  "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
  "result_type": "timeseries",
  "query": "reference_dataset:appliances_energy_wh",
  "point_count": 48,
  "head_2": [
    {"timestamp": "2016-01-11T17:00:00", "value": 60.0},
    {"timestamp": "2016-01-11T17:30:00", "value": 50.0}
  ],
  "tail_2": [
    {"timestamp": "2016-01-12T16:00:00", "value": 30.0},
    {"timestamp": "2016-01-12T16:30:00", "value": 40.0}
  ]
}
```

### Step 3: `insight`

输入：

```json
{
  "message": "基于已加载的 appliances_energy_wh 时间序列数据，提炼可验证事实：整体趋势、起止变化百分比、局部峰值/谷值及其时间点，并尽量标注明显的波动特征。",
  "evidence_ids": [
    "evi_influxdb2-energydata_appliances_energy_wh"
  ],
  "requested_fact_types": [
    "trend",
    "change_percent",
    "extrema"
  ],
  "focus": "appliances_energy_wh 在 2016-01-11 17:00 到 2016-01-12 16:30 的趋势分析"
}
```

输出：

```json
[
  {
    "fact_id": "fact_evi_influxdb2-energydata_appliances_energy_wh_trend",
    "fact_type": "trend",
    "statement": "appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。",
    "confidence": 0.95,
    "evidence": {
      "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
      "start_value": 60.0,
      "end_value": 40.0,
      "change_percent": -33.33
    },
    "verification_status": "verified"
  },
  {
    "fact_id": "fact_evi_influxdb2-energydata_appliances_energy_wh_extrema",
    "fact_type": "extrema",
    "statement": "appliances_energy_wh 的最高值为 430.00，最低值为 30.00。",
    "confidence": 0.93,
    "evidence": {
      "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
      "max_point": {
        "timestamp": "2016-01-11T19:00:00",
        "value": 430.0
      },
      "min_point": {
        "timestamp": "2016-01-12T03:00:00",
        "value": 30.0
      }
    },
    "verification_status": "verified"
  }
]
```

### Step 4: `anomaly`

输入：

```json
{
  "message": "检查 appliances_energy_wh 在 2016-01-11T17:00:00 到 2016-01-12T23:00:00 的异常点，并结合现有 48 个时序点判断是否存在显著尖峰或异常波动。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb",
    "display_name": null,
    "connection_hint": null,
    "selected_at": null
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T23:00:00"
  },
  "constraints": {
    "max_points": 48
  },
  "evidence_refs": [
    "evi_influxdb2-energydata_appliances_energy_wh"
  ]
}
```

输出：

```json
{
  "success": true,
  "summary": "AnomalyTool",
  "anomaly_id": "anomaly_evi_influxdb2-energydata_appliances_energy_wh",
  "detector_name": "zscore",
  "anomaly_count": 4,
  "anomaly_points": [
    {
      "timestamp": "2016-01-11T19:00:00",
      "value": 430.0,
      "score": 3.3356763005546712
    },
    {
      "timestamp": "2016-01-11T22:30:00",
      "value": 390.0,
      "score": 2.935894871274711
    },
    {
      "timestamp": "2016-01-12T01:00:00",
      "value": 380.0,
      "score": 2.8359495139547204
    },
    {
      "timestamp": "2016-01-12T14:30:00",
      "value": 370.0,
      "score": 2.7360041566347304
    }
  ]
}
```

### Step 5: `forecast`

输入：

```json
{
  "message": "基于已获取的 appliances_energy_wh 时序数据，为 2016-01-12 16:30 之后做一个短期预测。",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb",
    "display_name": null,
    "connection_hint": null,
    "selected_at": null
  },
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T16:30:00"
  },
  "constraints": {
    "max_points": 48
  },
  "focus": "短期预测 appliances_energy_wh",
  "series_name": "appliances_energy_wh"
}
```

输出：

```json
{
  "success": true,
  "summary": "ForecastTool",
  "forecast_id": "forecast_evi_influxdb2-energydata_appliances_energy_wh",
  "model_name": "linear_regression",
  "horizon": 12,
  "head_3": [
    {"timestamp": "2016-01-12T17:00:00", "value": 71.01063829787228},
    {"timestamp": "2016-01-12T17:30:00", "value": 69.980460269214},
    {"timestamp": "2016-01-12T18:00:00", "value": 68.95028224055574}
  ],
  "tail_3": [
    {"timestamp": "2016-01-12T21:30:00", "value": 61.739036039947806},
    {"timestamp": "2016-01-12T22:00:00", "value": 60.708858011289536},
    {"timestamp": "2016-01-12T22:30:00", "value": 59.67867998263126}
  ]
}
```

### Step 6: `format_answer`

输入：

```json
{
  "message": "请基于已验证事实，输出包含趋势、异常、短期预测和简要结论的中文总结。"
}
```

输出：

```json
{
  "success": true,
  "summary": "appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。 appliances_energy_wh 的最高值为 430.00，最低值为 30.00。 异常检测发现 4 个异常点，典型异常包括 2016-01-11T19:00:00=430.0, 2016-01-11T22:30:00=390.0, 2016-01-12T01:00:00=380.0。 appliances_energy_wh 的短期预测共 12 个点，预测区间内整体下降，从 71.01 变化到 59.68。",
  "title": "TSPilot v0.2 Analysis",
  "section_types": [
    "summary",
    "plan",
    "facts",
    "anomaly",
    "forecast",
    "conclusion"
  ],
  "reference_types": [
    "query",
    "fact",
    "fact",
    "forecast",
    "anomaly"
  ]
}
```

## 6. 结果结论

- 服务启动正常，`/health` 返回 `200`
- 主链路 `POST /api/v1/chat` 返回 `200`
- 实际走通了 6 个工具：`todowrite -> query_database -> insight -> anomaly -> forecast -> format_answer`
- 最终回答成功包含规划、事实、异常、预测和结论
- 本次真实请求标识：
  - `conversation_id=conv_765b07ba9a1b`
  - `request_id=req_bb00f33367bc`
