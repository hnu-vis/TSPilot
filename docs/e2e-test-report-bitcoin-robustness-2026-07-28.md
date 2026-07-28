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
