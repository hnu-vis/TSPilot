# Artifact、Forecast 与 Visualization 端到端修订报告

日期：2026-08-18  
范围：forecast/anomaly 产物传递、code 数据绑定、LLM 可视化字段匹配、Todo 推进、真实 HTTP E2E。

## 结论

问题是系统层面的产物传递与语义所有权缺陷，不是某个字段名缺失的单点问题。

修订后：

- forecast 与 anomaly 结果会作为完整、带类型和 lineage 的 artifact source 暴露给 code 与 visualization。
- code 通过 LLM 从多个候选 artifact 中选择语义主数据源；不会再默认退回数据库历史序列，也不会重新实现 forecast/anomaly。
- visualization 由 LLM 完成两阶段语义投影与图表规划，不做固定字段抽取或字段名特判。
- forecast 图默认同时包含历史实际值与未来预测值；历史上下文缺失时返回结构化 `needs_sources`，不静默生成纯预测图。
- Todo 由 LLM 按任务语义映射 owner tool，并在下一轮依据真实 Observation 推进；result_ref 指向实际所属 artifact。

## 原链路的根本问题

### 1. Observation 有结果，但下游没有绑定到结果

Observation 能描述工具成功与否，也持有 artifact ref；问题不在“forecast 成功信息完全不可见”，而在消费端：

1. ToolExecutor 会丢失 code 的显式 `source_refs`，并注入 latest database evidence。
2. code 的 canonical `df/rows` 绑定数据库祖先，而不是 forecast/anomaly owner artifact。
3. visualization 看到的是离散 inventory，缺少统一的完整 dataset 视图与明确 lineage。
4. visualization 的 dependency 在一次失败动作后会丢失，下一轮上下文发生错位。

因此，`forecast succeeded` 与“code/visualization 实际使用 forecast points”过去是两件不同的事。

### 2. `Visualization requires calculated semantic evidence`

旧实现把“能够画完整 forecast series”和“必须已有 calculated insight”混为一体。即使 forecast 已成功，visualization 也可能因为找不到计算型 insight 直接报错；同时 LLM 无法可靠定位完整 forecast dataset 的 record path。

修订后，完整 forecast/anomaly/evidence series 可直接作为可视化语义 owner。只有用户明确要求的派生结论（例如方向、绝对变化、百分比变化）才由 code 计算并形成 verified insight；图本身不再把 calculated insight 当成所有场景的前置条件。

### 3. Forecast 图缺少历史基线

只画未来 forecast 虽然可渲染，但缺少最近真实值基线，无法核验预测边界与走势合理性。现在 LLM 语义投影被要求：存在 forecast 的历史 evidence ancestor 时，同时规划历史实际序列和预测序列；source preference 只是偏好，不能用于省略历史上下文。

### 4. Todo 信息错位

旧 Todo 逻辑按 task contract 的位置映射 task type，导致 query/anomaly/forecast/visualization owner 错位；成功 Observation 还可能自动完成当前 Todo。现在改为 LLM 逐项语义匹配，并由下一轮 assessment 依据 tool 类型、成功状态和 artifact refs 推进。

## 实现设计

### 统一 artifact source plane

新增 `core/artifact_sources.py`，把下列 artifact 解析为一致的 source：

- database evidence
- forecast points / confidence intervals / quality
- anomaly points / status / spans / scores
- analysis insight / derived evidence

每个 source 都包含稳定 `source_ref`、完整 datasets 和 lineage。dataset 同时可按名称访问，但字段含义仍由 LLM 根据 schema、样例、statement 和 lineage 判断。

### Code Interpreter

- `CodeInterpreterInput` 支持多个 `source_refs`。
- sandbox 暴露 `sources` 与 `source_by_ref`。
- 新增 LLM source selection 阶段，为当前 analysis goal 选择语义 owner。
- canonical `df/rows` 绑定被选 owner 的完整主 dataset。
- specialized artifact 不可被 code 重新 forecast、重新 anomaly detection 或伪造区间。

### Visualization

- 阶段一：LLM semantic projection，从 grounded source 中选择 record grain、路径和字段语义。
- 阶段二：LLM chart planning，组合多 view、多 layer。
- 缺少源时返回 `status=needs_sources` 和结构化 dependency，可请求 sql_query/anomaly/forecast/code_interpreter。
- dependency 会跨失败 Observation 保留，直到对应动作成功。
- forecast 图默认加入历史 actual layer；缺少历史时请求 SQL context。
- 既有 `visualization:<id>` 若再次被作为 source preference，会解析回其 grounded source lineage，避免把展示 artifact 当数据 artifact。

### Forecast 输入质量

- LLM quality gate 只负责识别原始 forecast 输入是否缺少 anomaly 前置处理。
- edge samples 明确分成两个不相邻的局部窗口，避免把月初与月末价格差误判为相邻突变。
- 匹配 anomaly artifact 已应用后，由 anomaly 的过滤结果与 forecast 工具契约接管，不再重复触发相同 gate，避免 anomaly/forecast 循环。

### Task contract 与 Todo

- forecast series 不再自动代表方向、端点变化和百分比变化已经完成。
- 每个显式派生结论单列为 calculated analysis output，由 code 基于 owner artifact 计算。
- Todo 的 task type 由 LLM 按语义匹配，而非位置对齐。
- Todo 完成后 result_ref 分别指向 evidence/anomaly/forecast/visualization/final_answer。

## 真实 E2E 结果

### Case A：BTC/USD 未来一周走势、涨幅与预测图

- Conversation：`conv_b72cb3af933c`
- Request：`req_bd3125202e63`
- 状态：completed
- Tool 链路：`sql_query → forecast(质量门拒绝污染输入) → anomaly → forecast → code_interpreter → visualization → terminate`

产物结果：

- anomaly 排除 2 个不可能量级的污染点。
- forecast 使用 2677 个过滤后历史点，生成 672 个未来点。
- forecast 起点：24920.161219851227。
- forecast 终点：26931.5381094505。
- code 主数据源：`forecast:forecast_evi_influxdb2-bitcoin-sample_timeseries_2ca320702522`。
- code input row count：672；计算上涨 2011.3768895992725 美元，涨幅 8.07128361592229%。
- 主图：`viz_74928d1ce7b1_0`。
- 主图历史 actual dataset：2679 行。
- 主图 forecast dataset：672 行。
- 主图包含独立“历史实际价格”和“未来7天预测均值”图层。
- 最终回答：未来 7 天上涨，约 +2011.38 美元 / +8.07%。

该结果证明 forecast artifact 同时被 code 和 visualization 消费；code 没有基于历史数据重做 forecast。

### Case B：显式 5 步 Todo，多工具综合图

- Conversation：`conv_e8a0891871f3`
- Request：`req_e2bf9a320412`
- 状态：completed
- 主链路：`todowrite → sql_query → anomaly → forecast → visualization → terminate`

Todo 最终状态：

| Todo | Owner | 状态 | result_ref 类型 |
|---|---|---|---|
| 查询完整时间序列 | query | completed | evidence |
| 检测异常点 | anomaly | completed | anomaly |
| 基于异常结果预测 1 小时 | forecast | completed | forecast |
| 历史、异常、预测综合图 | visualization | completed | visualization |
| 总结结论 | answer | completed | final_answer |

综合图 `viz_e7120d52c1d4_0` 同时消费：

- 42 行历史 evidence
- 1 个 anomaly point
- 6 个 forecast points
- anomaly status

本轮曾出现一次把已生成 `visualization:<id>` 当作数据 source 再次请求的失败；已通过 presentation lineage dereference 修复，既有 visualization ref 会解析回原始 grounded view refs。

### 失败案例与修订验证

| 失败现象 | 根因 | 修订 |
|---|---|---|
| forecast 成功但 code 重做预测 | code canonical df 绑定 DB ancestor | LLM 选择 owner source，df 绑定 forecast points |
| visualization 报 calculated semantic evidence 缺失 | series 与 calculated insight 职责混淆 | series 可直接成图；派生数值单独由 code 产出 |
| visualization 第一次看不到 forecast 内容 | inventory/path/lineage 不统一 | 完整 dataset source plane + LLM semantic projection |
| dependency 在失败动作后丢失 | 只查看最后一条 Observation | 扫描依赖链直到成功 fulfillment |
| BTC forecast 被巨大脏点污染 | forecast 缺少语义输入质量路由 | LLM gate → anomaly → forecast |
| gate 在 anomaly 后反复否决 | 重复验证同一个已满足 dependency | gate 仅负责原始输入依赖路由 |
| 长 forecast insight 导致超大 prompt | 每个 Insight item 都发布为 planner source | parent Insight 暴露完整 items，默认不展开 item source |
| Todo task type/result_ref 错位 | 位置映射与第一 evidence ref | LLM 语义 owner 映射 + owning artifact ref |

## 自动化测试

重点回归覆盖：

- 完整 forecast/anomaly datasets 与 lineage
- code 多 source LLM owner selection
- ToolExecutor source_refs 保留
- forecast quality gate 与 anomaly 后不重复 gate
- visualization 两阶段 LLM 投影、needs_sources、dependency 持久化
- forecast 图历史上下文提示与多 layer materialization
- planner inventory 的 item 数量不随 insight rows 线性膨胀
- Todo 语义映射、assessment 推进与 artifact result_ref

测试命令使用 `/home/feilvvl/TSPilot/tspilot_env`。未执行前端编译。

最终重点 Python 回归：113 passed。另一次 visualization/Todo 子集：71 passed。`git diff --check` 通过。

说明：仓库原本存在其他未提交修改；本次未清理或覆盖无关改动，也未宣称整个历史测试全集全部通过。
