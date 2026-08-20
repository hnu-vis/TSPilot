# 真实端到端请求记录

## 1. 目的

这份文档只记录一条**真实模型**驱动的端到端请求。

不包含 fake agent，不包含测试桩推演，只保留：

- 真实请求输入
- 真实执行链
- 真实返回结果
- 真实 ReAct 逐轮过程摘要

运行环境：

- 仓库：`TSPilot-v0.1`
- 模型：`gpt-5.4-mini`
- 配置来源：本仓库 `.env`

接口入口：

- [app/routes/chat.py](../app/routes/chat.py)

执行编排：

- [runtime/react_loop.py](../runtime/react_loop.py)
- [runtime/action_policy.py](../runtime/action_policy.py)

## 2. 真实请求

请求方式：

- `POST /api/v1/chat`

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

本次真实运行返回的标识：

- `conversation_id=conv_91cecb67adf5`
- `request_id=req_7a604b55db38`

## 3. 真实终态结果

HTTP 状态：

- `200`

业务状态：

- `status=completed`
- `response_kind=final_answer`

真实 `used_tools`：

1. `todowrite`
2. `query_database`
3. `insight`
4. `anomaly`
5. `forecast`
6. `format_answer`

最终摘要：

```text
appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。
```

最终回答包含：

- `insights`
- `forecast`
- `anomaly`

引用包含：

- `query`
- `insight`
- `forecast`
- `anomaly`

## 4. 真实 ReAct 过程

### 第 1 轮

Thought：

- 先补齐数据证据，后续才能做趋势、异常和预测分析。

Action：

- `todowrite`

Action Input：

```json
{
  "message": "为 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的分析建立任务清单：先获取时序数据，再做趋势与变化幅度分析、异常检查、短期预测，最后汇总结论。",
  "current_intent": "分析 appliances_energy_wh 的趋势、异常和短期预测并总结",
  "requested_insight_types": ["trend", "change_percent", "extrema"],
  "focus": "appliances_energy_wh, 2016-01-11 至 2016-01-12",
  "todos": [
    {"content": "获取 appliances_energy_wh 指定时间范围内的时序数据", "status": "pending", "priority": 1},
    {"content": "分析趋势、变化百分比与极值", "status": "pending", "priority": 2},
    {"content": "检查异常点", "status": "pending", "priority": 2},
    {"content": "进行短期预测", "status": "pending", "priority": 3},
    {"content": "汇总结论", "status": "pending", "priority": 4}
  ],
  "evidence_summary": "当前没有数据库证据；需要先查询时序数据。"
}
```

Observation 摘要：

- `Todo plan updated with 5 steps.`
- 当前 `in_progress`：
  - `获取 appliances_energy_wh 指定时间范围内的时序数据`

### 第 2 轮

Thought：

- 需要先获取目标时间范围内的时序证据，才能继续做趋势、异常和预测分析。

Action：

- `query_database`

Action Input：

```json
{
  "message": "请查询 appliances_energy_wh 在 2016-01-11T17:00:00 到 2016-01-12T23:00:00 的时序数据，用于后续趋势分析、异常检测和短期预测。",
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

Observation 关键输出：

- `result_type=timeseries`
- `evidence_id=evi_influxdb2-energydata_appliances_energy_wh`
- `query_language=reference_dataset`
- `point_count=48`
- `value_field=appliances_energy_wh`

Observation 摘要：

```text
Loaded 48 points for appliances_energy_wh from the configured reference dataset.
```

### 第 3 轮

Thought：

- 已有时序数据，下一步应先把趋势与极值等事实转成可验证结论。

Action：

- `insight`

Action Input：

```json
{
  "message": "基于已加载的 appliances_energy_wh 时序数据，请提取并验证该区间的趋势、变化百分比、极值及分段特征，尽量给出简洁结论。",
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
  "focus": "appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势分析",
  "requested_insight_types": ["trend", "change_percent", "extrema"]
}
```

Observation 关键输出：

- `insight_id=ins_evi_influxdb2-energydata_appliances_energy_wh`
- verified insight 1：
  - `appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。`
- verified insight 2：
  - `appliances_energy_wh 的最高值为 430.00，最低值为 30.00。`

### 第 4 轮

Thought：

- 已有趋势与极值证据，当前缺口是基于时序做异常检测，再完成短期预测，之后即可总结。

Action：

- `anomaly`

Action Input：

输入很长，核心意图是：

- 基于当前 `timeseries` 证据做异常点检测
- 输出异常点和判定依据

Observation 关键输出：

- `anomaly_id=anomaly_evi_influxdb2-energydata_appliances_energy_wh`
- 检测到 4 个异常点：
  - `2016-01-11T19:00:00`, `430.0`
  - `2016-01-11T22:30:00`, `390.0`
  - `2016-01-12T01:00:00`, `380.0`
  - `2016-01-12T14:30:00`, `370.0`

### 第 5 轮

Action：

- `forecast`

Action Input：

输入很长，核心意图是：

- 基于当前 `timeseries` 证据给出短期预测

Observation 关键输出：

- `forecast_id=forecast_evi_influxdb2-energydata_appliances_energy_wh`
- `model_name=linear_regression`
- `horizon=12`

前几个预测点：

```json
[
  {"timestamp": "2016-01-12T17:00:00", "value": 71.01063829787228},
  {"timestamp": "2016-01-12T17:30:00", "value": 69.980460269214},
  {"timestamp": "2016-01-12T18:00:00", "value": 68.95028224055574}
]
```

### 第 6 轮

Thought：

- 已有趋势、异常和预测的证据，下一步直接汇总成答案。

Action：

- `format_answer`

Action Input：

```json
{
  "message": "请基于已验证证据，按用户要求先给出简短规划/结果，再分析 appliances_energy_wh 在 2016-01-11 到 2016-01-12 的趋势、异常、短期预测，并总结结论。",
  "current_intent": "总结已完成的趋势、异常和预测分析结果",
  "requested_insight_types": ["trend", "change_percent", "extrema", "anomaly", "forecast"],
  "focus": "appliances_energy_wh 在 2016-01-11 到 2016-01-12 的时序分析",
  "todos": [
    {"content": "获取 appliances_energy_wh 指定时间范围内的时序数据", "status": "completed", "priority": 1},
    {"content": "分析趋势、变化百分比与极值", "status": "completed", "priority": 2},
    {"content": "检查异常点", "status": "completed", "priority": 2},
    {"content": "进行短期预测", "status": "completed", "priority": 3},
    {"content": "汇总结论", "status": "completed", "priority": 4}
  ]
}
```

Observation 关键输出：

- 成功产出 `FinalAnswer`
- sections：
  - `insights`
  - `forecast`
  - `anomaly`

随后 runtime 追加：

- `final_answer`
- `terminate`

## 5. 真实最终响应摘要

响应的核心字段如下：

```json
{
  "status": "completed",
  "response_kind": "final_answer",
  "used_tools": [
    "todowrite",
    "query_database",
    "insight",
    "anomaly",
    "forecast",
    "format_answer"
  ],
  "answer": {
    "summary": "appliances_energy_wh 在所选时间范围内整体下降，变化幅度约为 -33.33%。"
  }
}
```

## 6. 这条真实请求证明了什么

这条记录已经证明：

1. 真实模型可以驱动多轮 outer ReAct
2. 系统不是单步 demo，而是能完成多步分析链
3. 真实链路已经实际覆盖：
   - 规划
   - 查库
   - 洞察
   - 异常检测
   - 预测
   - 最终组装
4. 最终能稳定返回 `final_answer`

## 7. 补充说明

这份文档只记录真实请求，不包含 fake 测试过程。

如果要看完整测试矩阵和 fake/real 的对照说明，可以看：

- [react-e2e-test-report.md](react-e2e-test-report.md)
