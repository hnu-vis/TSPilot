# Bitcoin 数据库端到端鲁棒性复测记录

## 测试范围

本次测试走真实后端 HTTP 链路：

- 服务：`http://127.0.0.1:5680`
- 接口：`POST /api/v1/chat`
- 数据库：`influxdb2-bitcoin-sample`
- 数据库类型：`influxdb`
- 测试产物目录：`cache_data/e2e_runs/bitcoin_robustness_2026-07-28_rerun/`
- 汇总文件：`cache_data/e2e_runs/bitcoin_robustness_2026-07-28_rerun/summary.json`

这不是单元测试，也不是 fake agent；每条请求都经过真实 ReAct 决策、数据库查询、code_interpreter 和最终回答生成。

## 总体结论

本轮共跑 9 条 Bitcoin 端到端用例：

- 8 条业务完成，1 条业务失败。
- 正常有数据的统计类问题，最终结果基本都能落到 `code_interpreter`，避免了最终回答阶段由 LLM 自己做高精度计算。
- `price > 1000000` 显式异常剔除场景，结果正确，并且 answer sections 中已经包含异常规则、阈值、原因、剔除明细、原始指标、调整后指标。
- 空数据场景仍不稳定：同类无数据请求中，一条最终失败，一条最终完成但经历大量 terminate block。
- code 失败后的行为比之前更安全：不合规 code artifact 会被 gate 拦住，并触发重试；但重试次数和耗时仍偏高，说明生成合规 analysis artifact 的稳定性还不够。

## 用例结果

| Case | 请求重点 | 终态 | 耗时 | 工具链 | 结果/问题 |
| --- | --- | --- | ---: | --- | --- |
| 01_raw_full_range_metrics | 原始数据，不剔除异常，算起止/涨跌/最高最低 | completed | 67.75s | `sql_query`, `code_interpreter` x3 | 最终数值正确；前两次 code 因缺少透明异常字段被拒绝，第三次成功。 |
| 02_threshold_exclude_gt_1000000 | 显式剔除 `price > 1000000` | completed | 79.99s | `sql_query` x2, `code_interpreter` | 剔除 2 条，剔除后起始 16858.2362，结束 23428.6802，涨跌幅 38.97468229802119%，最高 24104.6943，最低 16702.3044。 |
| 03_auto_outlier_transparency | 自动识别明显异常并输出透明规则 | completed | 50.96s | `sql_query`, `anomaly`, `code_interpreter` x2 | 输出了规则和剔除前后指标，但自动规则过敏感，剔除 42 条，其中很多点的 ratio 接近 1，不应被称为明显异常。 |
| 04_empty_range_unavailable | 2022-01-01 到 2022-01-02 无数据 | failed | 188.24s | `sql_query` x2 | 明确问题：SQL 返回空后，terminate 被 completion gate 反复阻塞 28 次，最后 `Request did not reach a final answer`。 |
| 05_future_range_unavailable | 2030-01-01 到 2030-01-02 无数据 | completed | 152.83s | `sql_query` | 最终正确说明不可用，但 terminate 仍被阻塞 21 次后才放行，效率和稳定性不足。 |
| 06_single_day_metrics | 单日区间统计 | completed | 26.45s | `sql_query`, `code_interpreter` | 正常完成：90 条记录，起始 17189.4086，结束 17435.7808，涨跌幅 1.4333%，最高 17469.8737，最低 17167.0285。 |
| 07_distribution_stats | 分布统计与异常影响 | completed | 60.36s | `sql_query`, `code_interpreter` | 正常完成：2680 条，均值 125050035469.19478，中位数 21257.103949999997；code 判断 2 个极端值显著影响均值。 |
| 08_low_max_points_range_metrics | `max_points=20`，要求不要因采样点限制影响范围统计 | completed | 33.95s | `sql_query`, `code_interpreter` | 查询仍返回 2680 行，未被 max_points 截断；但用户没有要求剔除异常，最终默认剔除了 2 条异常值，语义偏离。 |
| 09_invalid_asset_unavailable | Bitcoin EUR 字段/数据存在性与统计 | completed | 184.5s | `sql_query` x2, `code_interpreter` x7, `anomaly` | 数据库实际有 `code=EUR`，不是无数据；但过程非常不稳，多次 code schema 失败和 terminate block 后才完成。 |

## 关键数值核对

### 原始 Bitcoin USD 全区间

请求：`Bitcoin USD 2023-01-04 到 2023-02-03，原始数据不剔除异常值`

- 行数：2680
- 起始值：168249475888010.0，时间 `2023-01-04T23:04:00+00:00`
- 结束值：23428.6802，时间 `2023-02-03T22:47:00+00:00`
- 涨跌幅：-99.99999998607503%
- 最高值：168249475888010.0
- 最低值：16702.3044，时间 `2023-01-06T13:10:00+00:00`
- 分析证据：`analysis:ana_bitcoin_usd_6957fe65a3d5`

### 显式剔除 `price > 1000000`

请求：`把价格大于 1000000 的记录视为异常并剔除`

被剔除记录：

| timestamp | price |
| --- | ---: |
| `2023-01-04T23:04:00+00:00` | 168249475888010.0 |
| `2023-01-04T23:21:00+00:00` | 166884563179570.0 |

剔除后：

- clean_count：2678
- 起始值：16858.2362
- 结束值：23428.6802
- 涨跌幅：38.97468229802119%
- 最高值：24104.6943
- 最低值：16702.3044
- 分析证据：`analysis:ana_2023_01_04_2023_02_03_bitcoin_u_71334bbed4f7`

### 单日 Bitcoin USD

请求：`2023-01-10 这一天`

- 行数：90
- 起始值：17189.4086
- 结束值：17435.7808
- 涨跌幅：1.4333%
- 最高值：17469.8737
- 最低值：17167.0285

## 发现的问题

### P0：空数据终止策略仍会卡死

`04_empty_range_unavailable` 中，SQL 已经两次返回空结果，但最终没有给出“不可用”回答，而是返回：

```text
Request did not reach a final answer.
```

trace 中反复出现：

```text
Final answer is blocked because the current goal is not complete:
Task contract required outputs are not fully covered by the latest ReAct gap assessment.
```

根因判断：这是系统层 completion gate 与 task_contract 覆盖判断的设计缺陷，不是 React 前端问题。空数据证据已经足以覆盖“无法计算数值”的替代结论，但 gate 仍把 `start/end/pct/high/low` 当作必须产出的 analysis output，导致 terminate 被重复阻塞。

需要的通用修复方向：task_contract 应支持“required output 的 unavailable fulfillment”。当数据库证据证明无数据时，派生指标输出应被标记为 `unavailable` 且由 `unavailable_reason` 覆盖，而不是继续要求 code_interpreter 产出不存在的数值。

### P1：空数据同类请求表现不一致

`05_future_range_unavailable` 最终能完成，但经历 21 次 terminate block，耗时 152.83s。`04_empty_range_unavailable` 则最终失败。

根因判断：同类空结果路径受到 LLM gap assessment 表述影响，completion gate 对“空结果是否足以回答”的接受条件不稳定。

需要的通用修复方向：把“空查询结果 + 用户允许/要求说明不可用 + 无可分析 rows”建模为明确的 evidence state，并在 action_policy/completion 层统一处理，而不是依赖 LLM 每次说服 gate。

### P1：自动异常识别规则过敏感

`03_auto_outlier_transparency` 自动剔除了 42 条记录。前两条 1e14 量级尖峰明显应剔除，但后续多条 reason 中 ratio 接近 1，例如 `ratio=1.00208`、`ratio=0.997874`，这些不应被描述为“价格相对局部中位数偏离过大”。

根因判断：这是 analysis artifact 语义校验不足。当前 schema 校验要求有规则、阈值、原因、剔除明细，但没有验证“规则描述”和“被剔除记录”是否在数值上自洽。

需要的通用修复方向：code_interpreter 的 result contract 应增加可机检的 outlier decision fields，例如 `score`、`threshold`、`predicate_passed`、`predicate_components`。后端可以用通用 validator 检查每个 excluded row 是否真的满足其声明规则。

### P1：未要求剔除异常时，模型可能主动剔除

`08_low_max_points_range_metrics` 的用户重点是“不要因为采样点限制导致范围统计错误”，并没有要求剔除异常。但最终回答默认“已按明显脏数据剔除后再做范围统计”，输出的是调整后指标。

根因判断：这是意图约束问题。模型把“避免采样错误”泛化成“清洗异常值”，改变了统计口径。

需要的通用修复方向：task_contract 需要显式区分 `raw_metric`、`adjusted_metric`、`outlier_detection_only`、`outlier_excluded_metric`。除非用户明确要求剔除，最终主指标应默认使用原始口径；如果模型认为存在脏数据，应同时展示原始和调整后，而不能替换用户请求的主结果。

### P2：code artifact 生成稳定性不足

`01_raw_full_range_metrics` 的前两次 code_interpreter 被拒绝：

- 缺少 `details.outlier_rule`
- 缺少 object 类型的 `details.adjusted_metrics`

`09_invalid_asset_unavailable` 中也出现多次类似失败：

- 缺少 `adjusted_metrics`
- 缺少 `raw_metrics`
- terminate 因缺少 specialized analysis output 被阻塞

根因判断：当前 gate 能挡住坏 artifact，这是正确的；但 prompt/contract 到代码输出 schema 的对齐仍不够强，导致模型需要多轮 trial-and-error。

需要的通用修复方向：给 code_interpreter 输入增加更严格的 machine-readable output schema，并在提示中要求先构造 result object 再返回；必要时把 schema 作为工具参数传入 sandbox，而不是只靠自然语言约束。

### P2：查询策略有重复和效率问题

`02_threshold_exclude_gt_1000000` 先返回了一个缺少 price 列的 table evidence，随后又重新查询完整 timeseries。最终正确，但多了一轮 SQL。

根因判断：query planner 对“后续 code 分析需要哪些列”的预判不足。

需要的通用修复方向：当 task_contract 包含计算、异常剔除、分布统计时，查询阶段应优先保留 value column、timestamp、必要 tags，并把 evidence shape 设置为 code_interpreter 可直接消费的 timeseries/table。

## 前端可见性观察

显式阈值剔除 case 的最终 `response.answer.sections` 中已经包含：

- `section_type=analysis`
- `outlier_rule`
- `threshold_or_formula`
- `rationale`
- `excluded_rows`
- `raw_metrics`
- `adjusted_metrics`

如果前端只展示 `answer.summary`，用户会看不到“为什么剔除 1000000”。这不是 React 本身计算错误，而是前端渲染层如果没有展示 analysis section 或 structured payload，就会丢掉后端已经生成的证据解释。

## 建议后续调整顺序

1. 先修空结果的 unavailable fulfillment，让无数据/错误查询不会卡死在 terminate。
2. 再强化 outlier result contract，让每个剔除点都有可机检的判定分数和阈值。
3. 然后修 task_contract 的统计口径区分，避免未要求清洗时自动替换原始指标。
4. 最后优化 query planner 的 evidence shape，减少重复 SQL 和重复 code_interpreter。

## 修复后复测

针对 P0 空数据终止问题做了通用修复后，重新执行 `04_empty_range_unavailable` 同类请求：

- 临时服务：`http://127.0.0.1:5681`
- 产物：`cache_data/e2e_runs/bitcoin_robustness_2026-07-28_empty_fix/04_empty_range_unavailable_after_fix.json`
- HTTP 状态：200
- 业务状态：`completed`
- response_kind：`final_answer`
- used_tools：`sql_query`
- 耗时：20.86s
- 最终摘要：`该时间段查询结果为空（0 行），没有可用的 Bitcoin USD 数据，因此无法计算起始值、结束值、涨跌幅、最高值和最低值。`

结论：空数据库证据现在可以通过 `unavailable_outputs` + `unavailable_reason` 覆盖缺失的派生指标，不再要求 code_interpreter 产出不存在的数值，也不再卡在 terminate block 循环。

## Data Profile 补充

当前结构 schema 已能提供：

- measurement：`coindesk`
- field：`price`
- tags：`code`, `crypto`, `description`, `symbol`
- tag value domains：`USD`, `EUR`, `GBP` 和 `bitcoin`

但结构 schema 不能证明每个时间段都有数据。因此新增 InfluxDB `metadata.data_profile`，用于表达每个 measurement/field/tag series 的时间覆盖范围和点数。

Bitcoin preview 现在包含：

| code | crypto | start | end | point_count |
| --- | --- | --- | --- | ---: |
| EUR | bitcoin | `2023-01-04T23:04:00Z` | `2023-02-03T22:47:00Z` | 2680 |
| GBP | bitcoin | `2023-01-04T23:04:00Z` | `2023-02-03T22:47:00Z` | 2680 |
| USD | bitcoin | `2023-01-04T23:04:00Z` | `2023-02-03T22:47:00Z` | 2680 |

复测：

- 临时服务：`http://127.0.0.1:5681`
- preview 检查：`/api/v1/resources/databases/influxdb2-bitcoin-sample/preview`
- E2E 产物：`cache_data/e2e_runs/bitcoin_data_profile_2026-07-28/empty_range_with_data_profile.json`
- 请求：`Bitcoin USD 2022-01-01 到 2022-01-02`
- 结果：`completed`
- used_tools：`sql_query`
- 耗时：29.7s
- 摘要：`该时间范围内未查到 Bitcoin USD 数据，原因是数据库查询返回 0 行；因此起始值、结束值、涨跌幅、最高最低值均无法计算。`

## Fact Memory / DataFact 链路复测

本轮复测时间：`2026-07-28 20:59-21:09 Asia/Shanghai`

测试目标：

- 验证 Bitcoin 数据库上的常见任务是否能完成完整 ReAct/tool/final answer 链路。
- 验证 `fact_requests` 是否进入 tool action input。
- 验证 `produced_facts`、`fact_coverage`、`fact references` 是否进入 observation/final answer。
- 验证 long-term fact memory 是否可读、是否会沉淀 fact definition / recipe。

测试接口：

- `POST http://127.0.0.1:5680/api/v1/chat`
- `GET http://127.0.0.1:5680/api/v1/resources/fact-memory`
- `GET http://127.0.0.1:5680/api/v1/resources/databases/influxdb2-bitcoin-sample/fact-memory`

测试产物：

- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/summary.json`
- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/01_range_metrics.json`
- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/02_empty_range.json`
- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/03_exclude_threshold.json`
- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/04_periodicity_custom.json`
- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/05_raw_no_exclusion.json`
- `cache_data/e2e_runs/bitcoin_fact_memory_2026-07-28/06_single_day_fact_trace_confirm.json`

### 结果汇总

| Case | 请求重点 | 终态 | 耗时 | 工具链 | 结论 |
| --- | --- | --- | ---: | --- | --- |
| 01_range_metrics | 2023-01-04 到 2023-02-03 起止/涨跌/最高最低 | completed | 31.68s | `sql_query`, `code_interpreter` | 业务完成；LLM 在 `sql_query.action_input` 中带了 `fact_requests`；但 final answer 没有 fact references。 |
| 02_empty_range | 2022-01-01 到 2022-01-02 空数据 | completed | 22.04s | `sql_query` | 正确返回无数据不可计算；空数据路径本轮可用。 |
| 03_exclude_threshold | 显式剔除 `price > 1000000` | completed | 27.59s | `sql_query`, `code_interpreter` | 业务完成并解释剔除原因；但结束值为 `23491.0923`，与早前复测的 `23428.6802` 不一致，需要核查时间边界/排序/字段口径。 |
| 04_periodicity_custom | 周期性/custom fact 风格分析 | completed | 192.36s | `sql_query` x4, `code_interpreter` x6 | 最终回答“未可靠确认周期性”；但 61 个 trace event，耗时过长，存在多轮重试。 |
| 05_raw_no_exclusion | 明确要求原始数据不剔除 | completed | 29.70s | `sql_query`, `code_interpreter` | 正确使用原始异常值口径：起始/最高为 `168249475888010.0`，涨跌幅 `-99.99999998607503%`。 |
| 06_single_day_fact_trace_confirm | 单日短区间复测 DataFact trace | completed | 109.24s | `sql_query` x3, `code_interpreter` x6 | 严重错误：最终给出全 0 指标；9 次工具调用后 completed，说明当前链路仍会接受错误 analysis artifact。 |

### DataFact 观测

正向信号：

- `01_range_metrics` 的第一轮 `sql_query.action_input` 已包含 `fact_requests`：
  - `start_value`
  - `end_value`
  - `highest_value`
  - `lowest_value`
  - `percentage_change`
- 这说明 ReAct prompt 已经能让 LLM 用 fact request 表达“需要哪些事实”。

问题：

- 本轮所有 chat response 中：
  - `answer.references` 的 `source_type=fact` 数量均为 0。
  - `trace observation.payload.produced_facts` 均为空数组。
  - `fact_coverage.requested` 在 observation 中为空。
- 但本地用同一条 SQL observation 复放 `register_data_facts_from_payload()` 可以生成：
  - `data_coverage`
  - `start_value`
  - `end_value`
  - `highest_value`
  - `lowest_value`

初步判断：

- 这不是 fact runtime 完全不可用，而是当前 5680 运行链路中 tool input 的 `fact_requests` 没有稳定传到 `ToolCall.tool_input` / fact registration 阶段，或运行进程存在 reload 后模块版本不一致。
- 需要进一步检查运行时 worker 是否加载了最新 `SqlQueryInput.fact_requests`、`CodeInterpreterInput.fact_requests` 和 `register_data_facts_from_payload()`。

### Long-Term Fact Memory 观测

API 可用：

- `GET /api/v1/resources/fact-memory`
  - definitions：6
  - recipes：7
  - storage path：`cache_data/database/fact_memory/global.json`
- `GET /api/v1/resources/databases/influxdb2-bitcoin-sample/fact-memory`
  - definitions：5
  - recipes：1
  - storage path：`cache_data/database/fact_memory/influxdb2-bitcoin-sample.json`

当前 global memory 中包含：

- system definitions：
  - `point_value`
  - `extreme`
  - `change`
  - `seasonality`
  - `custom`
- observed definition：
  - `data_coverage`
- observed recipes：
  - `start_value`
  - `end_value`
  - `highest_value`
  - `lowest_value`
  - `percentage_change`

注意：

- long-term memory 当前沉淀的是 fact definition / recipe，不是具体数值结果。
- 也就是说它会记住“`percentage_change` 这类 fact 通常需要 code_interpreter，要求 start/end/current evidence 和 calculation trace”，但不会把“某个日期区间的 Bitcoin 涨跌幅”当作以后可直接复用的答案。

### 新发现的问题

#### P0：单日短区间可 completed 但返回全 0 错误指标

`06_single_day_fact_trace_confirm` 返回：

```text
起始值：0.0；结束值：0.0；涨跌幅：0.00%；最高值：0.0；最低值：0.0。
```

但 query reference 显示 SQL 实际返回了 90 行。analysis artifact 里也显示：

```json
{
  "input_row_count": 90,
  "result": {
    "metrics": {
      "start_value": 0.0,
      "end_value": 0.0,
      "high_value": 0.0,
      "low_value": 0.0
    },
    "details": {
      "row_count": 0
    }
  }
}
```

根因判断：

- SQL query 使用 `keep(columns: ["_time", "price"])`，返回 preview 只有 `timestamp` 列，没有数值列。
- code_interpreter 基于缺失 value 的 rows 运行，最终为了满足 expected schema，用 `0.0` 填充了空数据结果。
- 当前 validator 只检查字段类型，不检查 `input_row_count=90` 与 `result.details.row_count=0` 的矛盾，也不检查数值列缺失时不应返回 verified numeric metrics。

通用修复方向：

- code_interpreter validator 应增加 evidence/result consistency check：
  - 如果 `input_row_count > 0` 且任务要求数值指标，但 normalized numeric row count 为 0，则不能返回全 0 指标。
  - 这类情况应返回 failed/partial/unavailable，并让 ReAct 修正 SQL evidence shape。
- sql_query planner/generator 应避免生成只保留 `_time` 而丢失 `_value` 的 Flux query；如果用户要求计算，query evidence 必须包含 time + value。
- final answer 对 analysis artifact 需要做合理性 gate：`row_count=0` 但 answer 输出具体数值时应阻止 terminate。

#### P1：DataFact 没有进入最终 answer references

虽然 action input 已出现 `fact_requests`，但最终 answer 仍只引用 query/analysis，不引用 fact。

影响：

- 前端 Data facts 面板在本轮 E2E 中无法展示真实 produced facts。
- final answer 仍主要基于 analysis artifact，而不是 DataFact references。
- long-term fact memory 会沉淀 recipe，但当前 request 的 DataFact instance 没有形成稳定闭环。

通用修复方向：

- 确认运行时 worker 加载最新 tool input schema。
- 在 `ToolExecutor` 记录 `ToolCall.tool_input` 后增加 debug/trace preview，明确显示 validated input 是否保留 `fact_requests`。
- 在 `apply_observation` 后增加最小 telemetry：
  - `fact_registration.requested_count`
  - `fact_registration.produced_count`
  - `fact_registration.coverage`
- `terminate` 阶段若当前 request 有 `fact_set.facts`，应自动纳入 fact references，而不完全依赖 LLM 手写 `include_fact_ids`。

#### P1：周期性/custom fact 分析可完成但重试过多

`04_periodicity_custom`：

- 耗时：192.36s
- trace events：61
- 工具链：10 次 tool call

根因判断：

- `seasonality/custom` 的 fact definition 已在 long-term memory 中，但还没有强约束 code_interpreter 的 output schema。
- LLM 仍在多次尝试 SQL/code，而不是按 recipe 一次性生成合规 analysis。

通用修复方向：

- 从 long-term fact recipe 生成 `expected_result_schema`，而不是只把 recipe 放进 prompt。
- 对 custom/seasonality analysis 要求输出 `method`、`period`、`strength`、`limitations`、`evidence_refs`。

#### P2：显式阈值剔除结果与历史复测不一致

本轮 `03_exclude_threshold` 给出：

- end_value：`23491.0923`
- percentage_change：`39.3449%`

早前同类复测记录为：

- end_value：`23428.6802`
- percentage_change：`38.97468229802119%`

需要核查：

- 是否本轮 query stop/time boundary 与早前不同。
- 是否 code 使用了不同排序或末尾行。
- 是否数据源有新增或重复点。

### 本轮结论

任务完成率表面上较高：6 条复测均 `completed`，但鲁棒性仍不足。

最关键的问题不是 React 前端，而是后端证据链：

1. DataFact request 已进入 action input，但 DataFact instance 没有稳定进入 observation/final answer。
2. code_interpreter 可以在输入 evidence 缺少数值列时用 `0.0` 填充并通过 schema，导致错误答案 completed。
3. long-term memory 已能沉淀 fact definition/recipe，但还没有反向约束当前 request 的 DataFact production 和 final answer gate。

下一步优先级：

1. 修复 query evidence shape：计算任务必须返回 time + value，不允许只有 timestamp。
2. 修复 code_interpreter consistency validator：禁止用全 0/空 rows 伪造数值指标。
3. 修复 DataFact registration trace：确保 `fact_requests` 经 validated tool input 保留，并产出 `produced_facts`。
4. 让 final answer 自动引用当前 request 的 DataFact，不完全依赖 LLM 填 `include_fact_ids`。

结论：schema 现在不只是结构信息，也带有轻量 data profile。模型可以区分“字段/tag 存在但请求时间范围没有覆盖数据”和“字段或过滤条件不存在”。当前实现仍保留一次真实查询作为证据，不会仅凭 schema profile 伪造结果。

## Persistent Profile Cache 补充

Data profile 已改为持久缓存形态：

- 缓存目录：`cache_data/database/profiles/`
- Bitcoin 缓存文件：`cache_data/database/profiles/influxdb2-bitcoin-sample.json`
- 默认 TTL：900 秒
- 强制刷新：`GET /api/v1/resources/databases/{database_id}/preview?refresh=true`
- 普通读取：`GET /api/v1/resources/databases/{database_id}/preview`
- 新增/更新数据库配置后，会尝试刷新 profile；刷新失败不阻断配置保存。
- 测试连接成功后，也会尝试刷新 profile。

验证结果：

1. 强制刷新 preview：
   - `profile_cache.source=live_refresh_persisted`
   - 写入路径：`cache_data/database/profiles/influxdb2-bitcoin-sample.json`
   - profile sources：3 条，分别为 `EUR/GBP/USD`。

2. 普通 preview：
   - `profile_cache.source=persistent_cache`
   - `source_count=3`

3. 空区间对话复测：
   - 产物：`cache_data/e2e_runs/bitcoin_profile_cache_2026-07-28/empty_range_with_profile_cache.json`
   - 请求：`Bitcoin USD 2022-01-01 到 2022-01-02`
   - 结果：`completed`
   - used_tools：`sql_query`
   - 耗时：15.28s
   - 摘要：`查询结果为空：在指定时间范围内没有找到 Bitcoin USD 记录，所以无法提供起始值、结束值、涨跌幅、最高值和最低值。原因是数据库在该条件下返回 0 行。`

结论：对话前不再需要每次动态扫描 data profile。系统默认使用持久 profile cache，过期或用户点击数据库页面 Refresh 时再刷新。

## Structured Action 批量复测

在 DataAgent 改为结构化 function/tool calling 后，使用已重启的 `5680` 后端执行批量 E2E：

- 产物目录：`cache_data/e2e_runs/structured_action_batch_2026-07-28/`
- 共同数据库：`influxdb2-bitcoin-sample`
- 共同结果：HTTP/SSE 正常，无 error event，`llm_diagnostics=null`，未触发 ReAct JSON repair。

| case | 请求 | 状态 | 首个 tool_call | 总耗时 | 工具链 | 结果/问题 |
| --- | --- | --- | ---: | ---: | --- | --- |
| `normal_metrics` | Bitcoin USD 2023-01-04 到 2023-02-03 起始/结束/涨跌幅/最高最低 | completed | 8.10s | 33.63s | `sql_query -> code_interpreter -> code_interpreter -> terminate` | 有问题：第一次 code 因 outlier 透明字段缺失失败；第二次改为“不剔除任何异常值”，最终被 1e14 脏值污染。 |
| `empty_range` | Bitcoin USD 2022-01-01 到 2022-01-02 同类指标 | completed | 2.38s | 11.25s | `sql_query -> terminate` | 正确：返回 0 行并说明起始值、结束值、涨跌幅、最高最低无法计算。 |
| `gbp_metrics` | Bitcoin GBP 2023-01-04 到 2023-02-03 同类指标 | completed | 3.86s | 18.53s | `sql_query -> code_interpreter -> terminate` | 有问题：未识别/剔除 1e14 量级异常值，起始值和最高值被污染。 |
| `simple_count` | Bitcoin USD 2023-01-04 到 2023-02-03 条数 | completed | 2.27s | 8.68s | `sql_query -> terminate` | 正确：返回 `2680`。 |
| `invalid_asset` | Ethereum USD 同时间段同类指标 | completed | 2.76s | 27.48s | `sql_query -> sql_query -> sql_query -> terminate` | 可接受但偏慢：多次尝试后说明数据源中无 Ethereum USD 记录。 |

本轮结论：

1. 结构化 action 输出已经解决原先的非法 JSON 和 repair 延迟问题。
2. 首个业务工具调用稳定提前到 2.27s-8.10s，明显优于 repair 时代的 30s 级等待。
3. 新的主要风险转移到 code_interpreter 的异常值策略：当第一次 outlier 处理因透明字段校验失败后，模型可能改成“保留全部数据”来通过 schema，导致结果被明显脏值污染。

后续需做系统性修复：当数据中出现极端值且指标会被极端值支配时，code_interpreter 不能通过“未剔除任何异常值”绕过透明异常处理要求；至少必须同时返回 raw_metrics 与 adjusted_metrics，并在 final answer 中明确区分原始结果和清洗后结果。
