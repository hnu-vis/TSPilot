# 真实 HTTP 端到端测试文档

## 1. 范围

这份文档记录的是**真实服务进程 + 真实端口 + 外部 HTTP 请求**的端到端测试。

不是：

- fake agent
- `FastAPI TestClient`
- 直接调用内部函数

而是：

1. 启动真实 `uvicorn` 服务
2. 从另一个外部进程发 HTTP 请求
3. 验证普通 JSON 与 SSE 两条链路

## 2. 测试方式

服务启动命令：

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 5680
```

说明：

- 服务绑定到 `127.0.0.1:5680`
- 请求方是另一个独立 Python 进程
- 请求使用 `httpx.Client(trust_env=False)`

`trust_env=False` 是必要的，因为当前环境里代理变量会干扰本地请求。  
不关闭代理时，本地请求会返回 `502`，这不是应用本身的错误。

## 3. 服务健康检查

请求：

```http
GET http://127.0.0.1:5680/health
```

实际结果：

- HTTP `200`
- body:

```json
{"status":"ok"}
```

## 4. 真实复杂 JSON 请求

### 请求地址

```http
POST http://127.0.0.1:5680/api/v1/chat
```

### 请求体

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

### 实际响应结果

- HTTP `200`
- `conversation_id=conv_34b88aad3f7e`
- `request_id=req_deb609e460a6`
- `status=completed`
- `response_kind=final_answer`

### 实际 `used_tools`

1. `todowrite`
2. `query_database`
3. `insight`
4. `anomaly`
5. `forecast`
6. `format_answer`

### 最终摘要

```text
appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。
```

### 最终回答结构

`sections` 包含：

- `insights`
- `forecast`
- `anomaly`

`references` 包含：

- `query`
- `insight`
- `forecast`
- `anomaly`

### 逐轮 ReAct 过程

这条真实 JSON 请求的 `trace` 中一共走了 6 轮，每轮结构都是：

1. `thought`
2. `action`
3. `observation`

最终再追加：

4. `final_answer`
5. `terminate`

#### 第 1 轮

Thought：

```text
先建一个简短任务计划，并明确需要先取数，再做趋势、异常和预测分析。
```

Action：

- `todowrite`

Action Input：

```json
{
  "message": "先规划任务：1) 拉取 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的时间序列数据；2) 基于数据分析趋势、涨跌幅和极值；3) 检查异常点；4) 做短期预测；5) 汇总结论。",
  "current_intent": "分析 appliances_energy_wh 在指定时间范围内的趋势、异常与短期预测，并输出结论",
  "requested_insight_types": ["trend", "change_percent", "extrema", "anomaly", "forecast"],
  "focus": "appliances_energy_wh, 2016-01-11 到 2016-01-12",
  "todos": [
    {"content": "查询 appliances_energy_wh 指定时间范围内的时间序列数据", "status": "pending", "priority": 1},
    {"content": "分析趋势、变化百分比与极值", "status": "pending", "priority": 2},
    {"content": "检查异常点", "status": "pending", "priority": 3},
    {"content": "做短期预测", "status": "pending", "priority": 4},
    {"content": "总结结论", "status": "pending", "priority": 5}
  ],
  "evidence_summary": "当前尚无数据库证据，需先取数后分析。"
}
```

Observation：

- `tool_name=todowrite`
- `success=true`
- `summary=Todo plan updated with 5 steps. Current step: 查询 appliances_energy_wh 指定时间范围内的时间序列数据. Completed 0, remaining 5.`

Observation Payload 摘要：

```json
{
  "todo_total": 5,
  "in_progress": "查询 appliances_energy_wh 指定时间范围内的时间序列数据",
  "completed_count": 0,
  "pending_count": 5
}
```

#### 第 2 轮

Thought：

```text
需要先获取 appliances_energy_wh 在指定时间范围内的时间序列证据，才能继续趋势、异常和预测分析。
```

Action：

- `query_database`

Action Input：

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
    "max_points": 48
  },
  "metric": "appliances_energy_wh"
}
```

Observation：

- `tool_name=query_database`
- `success=true`
- `summary=Loaded 48 points for appliances_energy_wh from the configured reference dataset.`

Observation Payload 关键字段：

```json
{
  "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
  "result_type": "timeseries",
  "query_language": "reference_dataset",
  "query": "reference_dataset:appliances_energy_wh",
  "time_field": "timestamp",
  "value_field": "appliances_energy_wh",
  "point_count": 48
}
```

#### 第 3 轮

Thought：

```text
已有时间序列证据，先把趋势、极值和变化幅度转成可核实事实。
```

Action：

- `insight`

Action Input：

```json
{
  "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
  "requested_insight_types": ["trend", "change_percent", "extrema"],
  "focus": "分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的整体趋势、日内波动、峰谷与主要变化幅度",
  "database_context": {
    "database_id": "influxdb2-energydata",
    "database_type": "influxdb"
  }
}
```

Observation：

- `tool_name=insight`
- `success=true`
- `summary=InsightTool`

Observation Payload 关键字段：

```json
{
  "insight_id": "ins_evi_influxdb2-energydata_appliances_energy_wh",
  "verified_insight_count": 2,
  "verified_insights": [
    "appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。",
    "appliances_energy_wh 的最高值为 430.00，最低值为 30.00。"
  ],
  "visualization_count": 1
}
```

#### 第 4 轮

Thought：

```text
已有时序与趋势/极值事实，当前缺口是异常检测与短期预测
```

Action：

- `anomaly`

Action Input：

Action Input 在 trace 中非常长，核心字段是：

```json
{
  "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
  "series_name": "appliances_energy_wh",
  "time_range": {
    "start": "2016-01-11T17:00:00",
    "end": "2016-01-12T23:00:00"
  },
  "constraints": {
    "max_points": 48
  },
  "focus": "检查 appliances_energy_wh 的异常点和异常波动"
}
```

Observation：

- `tool_name=anomaly`
- `success=true`
- `summary=AnomalyTool`

Observation Payload 关键字段：

```json
{
  "anomaly_id": "anomaly_evi_influxdb2-energydata_appliances_energy_wh",
  "detector_name": "zscore",
  "anomaly_point_count": 4,
  "anomaly_points": [
    {"timestamp": "2016-01-11T19:00:00", "value": 430.0},
    {"timestamp": "2016-01-11T22:30:00", "value": 390.0},
    {"timestamp": "2016-01-12T01:00:00", "value": 380.0},
    {"timestamp": "2016-01-12T14:30:00", "value": 370.0}
  ]
}
```

#### 第 5 轮

Thought：

trace 中这一轮的思路是：当前已有时序和异常证据，下一步补齐短期预测。

Action：

- `forecast`

Action Input：

Action Input 在 trace 中同样很长，核心字段是：

```json
{
  "evidence_id": "evi_influxdb2-energydata_appliances_energy_wh",
  "series_name": "appliances_energy_wh",
  "constraints": {
    "max_points": 48
  },
  "focus": "对 appliances_energy_wh 做短期预测"
}
```

Observation：

- `tool_name=forecast`
- `success=true`
- `summary=ForecastTool`

Observation Payload 关键字段：

```json
{
  "forecast_id": "forecast_evi_influxdb2-energydata_appliances_energy_wh",
  "model_name": "linear_regression",
  "horizon": 12,
  "forecast_preview": [
    {"timestamp": "2016-01-12T17:00:00", "value": 71.01063829787228},
    {"timestamp": "2016-01-12T17:30:00", "value": 69.980460269214},
    {"timestamp": "2016-01-12T18:00:00", "value": 68.95028224055574}
  ]
}
```

#### 第 6 轮

Thought：

```text
已有时序、异常和预测证据，适合直接整合为最终答复。
```

Action：

- `format_answer`

Action Input：

```json
{
  "message": "请基于已验证结果，输出 appliances_energy_wh 在 2016-01-11 17:00 到 2016-01-12 16:30 的趋势、异常、短期预测与总结结论。",
  "current_intent": "总结分析结果并回答用户",
  "requested_insight_types": ["trend", "change_percent", "extrema", "anomaly", "forecast"],
  "focus": "先规划一下，然后分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势，检查异常，再给一个短期预测，最后总结结论。",
  "evidence_summary": "已获得并验证趋势、极值、异常点和12步短期预测结果，可直接汇总输出。",
  "todos": [
    {"content": "查询 appliances_energy_wh 指定时间范围内的时间序列数据", "status": "completed", "priority": 1},
    {"content": "分析趋势、变化百分比与极值", "status": "completed", "priority": 2},
    {"content": "检查异常点", "status": "completed", "priority": 3},
    {"content": "做短期预测", "status": "completed", "priority": 4},
    {"content": "总结结论", "status": "completed", "priority": 5}
  ]
}
```

Observation：

- `tool_name=format_answer`
- `success=true`
- `summary=appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。`

Observation Payload 关键字段：

```json
{
  "title": "TSPilot v0.2 Analysis",
  "summary": "appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。",
  "section_types": ["insights", "forecast", "anomaly"],
  "reference_types": ["query", "insight", "forecast", "anomaly"],
  "visualization_count": 3
}
```

#### 终态

`final_answer`：

```json
{
  "summary": "appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。"
}
```

`terminate`：

```json
{
  "request_id": "req_deb609e460a6",
  "status": "completed"
}
```

## 5. 真实复杂 SSE 请求

### 请求地址

```http
POST http://127.0.0.1:5680/api/v1/chat
```

### 请求体

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
  },
  "stream": true
}
```

### 实际响应结果

- HTTP `200`
- `content-type=text/event-stream; charset=utf-8`
- `conversation_id=conv_27aeb02530d3`
- `request_id=req_1fc41743dc8a`

### 实际 SSE 事件顺序

1. `conversation_id`
2. `agent_step`
3. `tool_call`
4. `tool_result`
5. `agent_step`
6. `tool_call`
7. `tool_result`
8. `agent_step`
9. `tool_call`
10. `tool_result`
11. `agent_step`
12. `tool_call`
13. `tool_result`
14. `agent_step`
15. `tool_call`
16. `tool_result`
17. `agent_step`
18. `tool_call`
19. `tool_result`
20. `agent_step`
21. `final_answer`
22. `terminate`

### 实际 tool 顺序

1. `todowrite`
2. `query_database`
3. `insight`
4. `anomaly`
5. `forecast`
6. `format_answer`

### SSE 与 ReAct 的对应关系

真实 SSE 不直接暴露内部 `thought/action/observation`，而是映射为：

- `thought/action` -> `agent_step` + `tool_call`
- `observation` -> `tool_result`
- `final_answer` -> `final_answer`
- `terminate` -> `terminate`

对应关系如下：

#### 第 1 轮

- `agent_step.phase=intent`
- `tool_call.tool=todowrite`
- `tool_result.tool=todowrite`

#### 第 2 轮

- `agent_step.phase=tool_selection`
- `tool_call.tool=query_database`
- `tool_result.tool=query_database`

#### 第 3 轮

- `agent_step.phase=analysis`
- `tool_call.tool=insight`
- `tool_result.tool=insight`

#### 第 4 轮

- `agent_step.phase=analysis`
- `tool_call.tool=anomaly`
- `tool_result.tool=anomaly`

#### 第 5 轮

- `agent_step.phase=analysis`
- `tool_call.tool=forecast`
- `tool_result.tool=forecast`

#### 第 6 轮

- `agent_step.phase=answer_assembly`
- `tool_call.tool=format_answer`
- `tool_result.tool=format_answer`
- `final_answer`
- `terminate`

### 关键 SSE 片段

`todowrite`:

```text
event: tool_call
data: {"tool":"todowrite",...}
```

`query_database`:

```text
event: tool_result
data: {"tool":"query_database","success":true,...}
```

`forecast`:

```text
event: tool_result
data: {"tool":"forecast","success":true,...}
```

`final_answer`:

```text
event: final_answer
data: {"conversation_id":"conv_27aeb02530d3","request_id":"req_1fc41743dc8a","answer":{...}}
```

## 6. 这次测试证明了什么

这次测试证明的是部署级链路：

1. 真实 `uvicorn` 服务可启动
2. 外部进程可以通过真实端口访问服务
3. `/health` 可用
4. `/api/v1/chat` 的 JSON 路径可用
5. `/api/v1/chat` 的 SSE 路径可用
6. 真实模型可以驱动完整多步 ReAct
7. 真实 HTTP 请求下已经跑通：
   - `todowrite -> query_database -> insight -> anomaly -> forecast -> format_answer`

## 7. 与其他文档的关系

如果要看：

- fake 和真实混合的总测试说明：
  - [react-e2e-test-report.md](/home/feilvvl/TSPilot-v0.2/docs/react-e2e-test-report.md)
- 仅真实模型请求记录：
  - [real-e2e-request.md](/home/feilvvl/TSPilot-v0.2/docs/real-e2e-request.md)

这份文档的定位是：

- **真实 HTTP 端口测试记录**
