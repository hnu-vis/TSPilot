# 真实端到端复杂 Tool Calling 测试报告（2026-07-31）

## 测试目标

验证当前系统在真实 HTTP SSE 链路下，对复杂 ReAct 流程的执行能力与前端可展示的 tool timing 字段覆盖情况。

本轮重点覆盖：

- `todowrite`
- `sql_query`
- `code_interpreter`
- `anomaly`
- `forecast`
- `terminate`
- SSE `tool_call` / `tool_result` / `step.done`
- 每个 tool 的 `duration_ms` 与 `elapsed_seconds`

## 测试环境

- 项目路径：`/home/feilvvl/TSPilot-v0.1`
- Python 环境：`/home/feilvvl/TSPilot/tspilot_env`
- 服务命令：`/home/feilvvl/TSPilot/tspilot_env/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 18084`
- 请求方式：真实 HTTP POST `/api/v1/chat`，`stream=true`
- 数据库：`influxdb2-bitcoin-sample`
- 数据库类型：`influxdb`
- 时间说明：测试在 Asia/Shanghai 的 2026-07-31 执行；日志 timestamp 使用 UTC，显示为 2026-07-30。

## 总体结论

1. `tool_result` 已稳定携带后端真实 tool 执行耗时。
   - 所有测试中的 `sql_query`、`forecast`、`anomaly`、`code_interpreter`、`terminate`、policy 拒绝型 `todowrite` 都有 `elapsed_seconds` / `duration_ms`。

2. `tiktoken` 阻塞问题已消除。
   - 首个 `tool_call` 不再出现 120s 级等待。
   - 本轮复杂测试的首个 `tool_call` 通常在 2.6s - 10.9s 到达。

3. 复杂 ReAct 链路仍存在鲁棒性问题。
   - `todowrite` 最明显：模型经常输出不符合 `TodoWriteInput` 的结构，或在 policy 不允许时继续调用 `todowrite`。
   - `forecast` 成功后仍有多余重复调用，说明 completion/state transition 对“已有可用 artifact”约束不够强。
   - `code_interpreter` 偶发生成依赖未定义变量的代码，例如 `df` / `data` 未定义。
   - `anomaly` 偶发生成不存在的 detector 名称，例如 `bitcoin_price_anomaly_detector`，而实际可用 detector 是 `zscore`。

## 测试用例汇总

| 用例 | 测试问题 | 结果 | 总耗时 | Tool 链路 |
|---|---|---:|---:|---|
| 1 | 请先写 todo list，然后查询、统计、异常检测、预测、解释 | 失败，达到 30 轮上限 | 244.2s | `todowrite` → `todowrite` → `sql_query` → `sql_query` → `todowrite` → `code_interpreter` → `code_interpreter` → 多次 `todowrite/sql_query` → `anomaly` → `todowrite` |
| 2 | 查询最近 Bitcoin USD 数据，预测未来 5 个点并解释 | 成功 | 86.5s | `sql_query` → `forecast`失败 → `forecast`成功 → `forecast`成功 → `todowrite`拒绝×3 → `sql_query`×2 → `forecast`×2 → `terminate` |
| 3 | 查询 Bitcoin USD 时间序列，检测异常点并解释 | 成功 | 19.1s | `sql_query` → `anomaly` → `terminate` |
| 4 | 查询完整价格序列，用分析工具计算统计量和最大回撤 | 成功 | 36.2s | `sql_query` → `code_interpreter`×3 → `terminate` |
| 5 | 先制定 todo list，再查询最大值和对应时间并回答 | 成功，但 todo 被拒绝 | 38.4s | `todowrite`拒绝 → `sql_query`成功 → `sql_query`拒绝 → `code_interpreter`×2 → `terminate` |

## 详细结果

### 用例 1：todo + 统计 + 异常 + 预测 + 解释

测试问题：

```text
请先写一个todo list，然后基于 bitcoin USD 数据完成：查询价格数据，计算最大值、最小值、平均值和最新值，做异常检测，预测未来5个点，并用中文解释关键结论。
```

请求日志：

```text
cache_data/conversation_logs/2026-07-31_01-22-41_conv_31db2e1a659a/requests/req_12c873f1cbc9
```

执行结果：

- 最终状态：失败
- 失败原因：达到最大 ReAct 轮数 30，未生成 final answer
- 总耗时：244.2s
- `tool_call` 数量：30
- `tool_result` 数量：30
- `agent_step` 数量：251

实际链路：

```text
todowrite(false)
todowrite(false)
sql_query(true)
sql_query(true)
todowrite(false)
code_interpreter(true)
code_interpreter(true)
todowrite(false)
todowrite(false)
todowrite(false)
todowrite(false)
sql_query(true)
sql_query(true)
todowrite(false)
todowrite(false)
todowrite(false)
todowrite(false)
sql_query(true)
todowrite(false)
code_interpreter(false)
code_interpreter(false)
todowrite(false)
sql_query(true)
sql_query(true)
todowrite(false)
todowrite(false)
todowrite(false)
anomaly(false)
todowrite(false)
todowrite(false)
```

关键 tool 耗时：

| Tool | 成功情况 | 典型耗时 |
|---|---:|---:|
| `todowrite` | 全部失败 | 0.0s |
| `sql_query` | 成功多次 | 8.9s - 22.9s |
| `code_interpreter` | 成功/失败均出现 | 0.0s - 0.5s |
| `anomaly` | 失败 | 0.3s |

观察到的问题：

- 第一次 `todowrite` 输入为字符串列表，不符合 `TodoWriteInput` 所需的对象列表。
- 后续 `todowrite` 被 policy 拒绝：`todowrite is only valid before evidence or analysis work starts`。
- 模型没有稳定吸收 policy observation，反复选择被拒绝动作。
- `code_interpreter` 出现未定义变量：
  - `df is not defined`
  - `data is not defined`
- `anomaly` 生成了不存在的 detector：
  - `bitcoin_price_anomaly_detector`
  - 实际可用：`zscore`

结论：

该用例暴露的是系统层面的 ReAct 状态转移/工具契约对齐问题，不是单个 tool 执行慢。

### 用例 2：Forecast 复杂链路

测试问题：

```text
基于 bitcoin USD 最近价格数据，查询足够的数据，预测未来5个点，并用中文解释预测依据和不确定性。
```

请求日志：

```text
cache_data/conversation_logs/2026-07-31_01-27-08_conv_558558f8eb1c/requests/req_3d7eca2f6c8f
```

执行结果：

- 最终状态：成功
- 总耗时：86.5s
- `tool_call` 数量：12
- 首个 tool：`sql_query`
- 最终 tool：`terminate`

实际链路：

```text
sql_query(true)
forecast(false)
forecast(true)
forecast(true)
todowrite(false)
todowrite(false)
todowrite(false)
sql_query(true)
sql_query(true)
forecast(true)
forecast(true)
terminate(true)
```

关键 tool 耗时：

| Tool | 成功情况 | 耗时 |
|---|---:|---:|
| `sql_query` | 成功 | 10.8s / 9.6s / 10.1s |
| `forecast` | 1 次失败，多次成功 | 0.1s |
| `todowrite` | 被 policy 拒绝 | 0.0s |
| `terminate` | 成功 | 0.1s |

观察到的问题：

- 第一次 `forecast` 失败，因为模型选择了不存在的模型：
  - `arima`
  - 实际可用：`linear_regression`
- 后续 `forecast` 可以恢复成功。
- forecast 已成功后，模型仍多次重复调用 `forecast` / `sql_query`，并插入无效 `todowrite`。
- 最终能完成，但比理想链路多 7-8 轮。

结论：

功能上可完成，但 action policy 与完成判断还不够收敛；成功 artifact 没有足够强地引导模型进入 `terminate`。

### 用例 3：Anomaly 检测链路

测试问题：

```text
基于 bitcoin USD 价格数据，查询时间序列，检测异常点，列出异常时间和值，并解释可能原因。
```

请求日志：

```text
cache_data/conversation_logs/2026-07-31_01-28-34_conv_5e3ff956f06c/requests/req_e49565ed51aa
```

执行结果：

- 最终状态：成功
- 总耗时：19.1s
- `tool_call` 数量：3

实际链路：

```text
sql_query(true)
anomaly(true)
terminate(true)
```

关键 tool 耗时：

| Tool | 成功情况 | 耗时 |
|---|---:|---:|
| `sql_query` | 成功 | 8.3s |
| `anomaly` | 成功 | 0.1s |
| `terminate` | 成功 | 0.1s |

结论：

这是本轮最干净的复杂链路。说明 `sql_query -> anomaly -> terminate` 在明确任务下可以稳定完成。

### 用例 4：Code Interpreter 多统计链路

测试问题：

```text
基于 bitcoin USD 数据，查询完整价格序列，用分析工具计算最大值、最小值、平均值、标准差和最大回撤，并给出中文结论。
```

请求日志：

```text
cache_data/conversation_logs/2026-07-31_01-28-53_conv_92d6c6656980/requests/req_ecabbe54a51e
```

执行结果：

- 最终状态：成功
- 总耗时：36.2s
- `tool_call` 数量：5

实际链路：

```text
sql_query(true)
code_interpreter(true)
code_interpreter(true)
code_interpreter(true)
terminate(true)
```

关键 tool 耗时：

| Tool | 成功情况 | 耗时 |
|---|---:|---:|
| `sql_query` | 成功 | 8.8s |
| `code_interpreter` | 成功 3 次 | 0.0s / 0.2s / 0.0s |
| `terminate` | 成功 | 0.1s |

观察到的问题：

- 功能完成。
- 但 `code_interpreter` 被重复调用 3 次，说明模型没有一次性整合全部统计需求，或 completion 判断没有及时收敛。

结论：

统计分析能力可用，但仍有多余分析动作。

### 用例 5：Todo + 查询最大值

测试问题：

```text
请先制定一个todo list：1 查询 bitcoin USD 价格数据；2 计算最大值和对应时间；3 给出中文结论。然后按这个计划执行并回答。
```

请求日志：

```text
cache_data/conversation_logs/2026-07-31_01-29-51_conv_62e333a3ad84/requests/req_dfac451613bb
```

执行结果：

- 最终状态：成功
- 总耗时：38.4s
- `tool_call` 数量：6
- 注意：`todowrite` 未成功，最终是绕过 todo 后完成任务。

实际链路：

```text
todowrite(false)
sql_query(true)
sql_query(false)
code_interpreter(true)
code_interpreter(true)
terminate(true)
```

关键 tool 耗时：

| Tool | 成功情况 | 耗时 |
|---|---:|---:|
| `todowrite` | policy 拒绝 | 0.0s |
| `sql_query` | 1 次成功，1 次 policy 拒绝 | 10.1s / 0.0s |
| `code_interpreter` | 成功 2 次 | 0.0s / 0.1s |
| `terminate` | 成功 | 0.0s |

观察到的问题：

- 用户明确要求先制定 todo，但 policy 要求先满足 `sql_query`，所以 `todowrite` 被拒绝：
  - `Required specialized tool output is missing for the requested analysis. Required actions: ['sql_query']`
- 这说明当前 action policy 将“计划动作”和“证据动作”的优先级绑定过紧。

结论：

任务可完成，但用户显式要求的 todo planning 没有被满足。

## Tool timing 验证

本轮所有 `tool_result` 都包含：

```json
{
  "started_at": "...",
  "completed_at": "...",
  "duration_ms": 8837,
  "elapsed_seconds": 8.8
}
```

因此前端 timeline 可以展示每个 tool 的后端真实耗时，小数点后一位。

## 主要问题清单

### P0/P1：todowrite 契约和 policy 不一致

现象：

- 用户显式要求 todo list，但 `todowrite` 常被 policy 拒绝。
- 模型输出的 `todos` 有时是字符串列表，而工具 schema 需要对象列表。
- 被拒绝后，模型仍反复调用 `todowrite`。

影响：

- 复杂任务容易进入无效循环。
- 用户显式要求的 planning 无法稳定执行。
- 最大迭代数耗尽，最终无答案。

### P1：成功 artifact 后 completion 不够强

现象：

- `forecast` 成功后仍重复 forecast/sql_query。
- `code_interpreter` 完成统计后仍重复 code_interpreter。

影响：

- 延迟增加。
- ReAct 轮数不稳定。
- 模型可能在后续重复动作中引入新错误。

### P1：工具参数空间没有充分约束给模型

现象：

- `forecast` 选择不存在的 `arima`。
- `anomaly` 选择不存在的 `bitcoin_price_anomaly_detector`。
- `code_interpreter` 生成依赖不存在变量 `df` / `data` 的代码。

影响：

- 增加恢复轮次。
- 复杂任务中更容易达到最大迭代上限。

### 已验证改善：tiktoken 不再阻塞

现象：

- 本轮首个 `tool_call` 未出现 120s 级等待。
- 复杂用例首个 tool 到达时间：
  - 10.9s
  - 3.4s
  - 2.7s
  - 2.6s
  - 4.9s

结论：

之前的首轮 126s 等待已消除。

## 建议后续修复方向

1. 解耦 planning 与 evidence policy。
   - `todowrite` 应该作为任务管理动作，不应被“缺少 sql evidence”的规则拦截。
   - 如果用户显式要求先计划，第一轮应允许 `todowrite`。

2. 将每个 tool 的 action input schema 明确暴露为参数契约。
   - 特别是 `todowrite.todos` 必须是对象列表。
   - `forecast.model` 只允许当前 registry 里的模型。
   - `anomaly.detector` 只允许当前 registry 里的 detector。

3. 增强 observation 的“下一步约束”。
   - 当 policy 拒绝某动作后，下一轮 prompt 需要更强地表达 prohibited/required action。
   - 对重复失败签名应强制改变 action，而不是只给自然语言 hint。

4. 增强 completion 判断。
   - 如果 required outputs 已由 artifacts 覆盖，应提示/约束模型进入 `terminate`。
   - 对 forecast/anomaly/code_interpreter 成功 artifact，应避免重复调用同类 tool。

5. code_interpreter 输入上下文需要更结构化。
   - 不应让模型猜变量名 `df` / `data`。
   - 应明确可用对象、artifact ref、数据加载方式或提供标准输入模板。

