# Bitcoin LineChart E2E 测试报告

- 执行日期：2026-08-21（Asia/Shanghai）
- 数据源：`influxdb2-bitcoin-sample`
- 接口：`POST /api/v1/chat`
- 执行方式：10 条问题严格串行，每条使用独立 conversation，真实调用 LLM、数据库和 ReAct tools
- 总耗时：1093.7 秒（约 18 分 14 秒）

## 成功口径

- **接口完成**：响应状态为 `completed` 且有 FinalAnswer。
- **分析完成**：SQL/Anomaly/Code Interpreter 产出足以回答问题的 grounded 结果。
- **完整成功**：分析完成，并且被调用的 Visualization 成功创建至少一个 V4 LineChart artifact。
- Visualization 返回 `unavailable` 时，即使文字答案存在，也不计为完整成功。

## 汇总

| 指标 | 结果 |
| --- | ---: |
| 测试总数 | 10 |
| 接口 `completed` | 4 |
| 接口 `partial` | 6 |
| 分析结果成功产出 | 10 |
| V4 LineChart 创建成功 | 2 |
| 完整成功 | 2（20%） |
| Visualization `unavailable` | 8（80%） |
| Provider schema 400 | 0 |
| HTTP/传输失败 | 0 |

## 逐项结果

| # | 测试主题 | API 状态 | 完整结果 | 执行链路 | Visualization / 失败原因 | 请求 ID | 耗时 |
| ---: | --- | --- | --- | --- | --- | --- | ---: |
| 1 | 2 小时内最大跌幅窗口 | `completed` | **成功** | `sql_query(fail) → sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | V4 `viz_96e918bfe8d2` | `req_0c471625565b` | 104.8s |
| 2 | 1 月平均日波幅 | `completed` | **成功** | `sql_query(ok) → code_interpreter(fail) → code_interpreter(ok) → visualization(created) → terminate(ok)` | V4 `viz_7d9806f27a08` | `req_9e9856360e88` | 105.6s |
| 3 | 排除异常后的最低点最大反弹 | `partial` | **失败：无图** | `sql_query(ok) → anomaly(ok) → code_interpreter(fail) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `line_anomaly_score` 的 source 不属于其 content item | `req_83e6e363a919` | 150.9s |
| 4 | 2 小时内最大涨幅窗口 | `partial` | **失败：无图** | `sql_query(ok) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `pt1` 的 source 不属于其 content item | `req_b41de1d2046a` | 88.6s |
| 5 | 连续每天创新高的最长区段 | `completed` | **分析成功 / 无图** | `sql_query(fail) → sql_query(ok) → code_interpreter(fail) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `view_2` 缺少 `statement` 字段 | `req_ef2b727f2365` | 136.8s |
| 6 | 日内波动最剧烈日期 | `completed` | **分析成功 / 无图** | `sql_query(ok) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `view_1` 缺少 `daily_range`、`date` 字段 | `req_5c7fde06eaa5` | 89.3s |
| 7 | 上下半月均价比较 | `partial` | **失败：无图** | `sql_query(ok) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `line_first_half` 的 source 不属于其 content item | `req_b802b9f74681` | 89.8s |
| 8 | 最低点后反弹 20% 所需时间 | `partial` | **失败：无图** | `sql_query(ok) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `ref_min_price` 的 source 不属于其 content item | `req_703a83f74bea` | 95.4s |
| 9 | 月末持续稳定在 23000 以上的区段 | `partial` | **失败：无图** | `sql_query(ok) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `ann_longest_interval` 的 source 不属于其 content item | `req_6f57fa03784e` | 91.1s |
| 10 | 月初到月末净变化百分比 | `partial` | **失败：无图** | `sql_query(fail) → sql_query(fail) → sql_query(ok) → sql_query(ok) → code_interpreter(ok) → visualization(unavailable) → terminate(ok)` | `ann_direction` 的 source 不属于其 content item | `req_f94d92a0c2d0` | 141.4s |

## 分析结论

1. Structured-output schema 修复有效：10 条请求均未再出现 `additionalProperties` 相关 400。
2. SQL、异常检测和派生分析链路具备 ReAct 恢复能力：所有问题最终都产出了可回答的分析结果。
3. 当前瓶颈集中在 Visualization 第二阶段组合：
   - 6/8 的失败是 LineChart component 使用了不属于对应 ContentItem 的 source；
   - 2/8 的失败是 LLM 选择了 source view 中不存在的字段。
4. 失败均被 validator 正确拦截，并在两次 LLM repair 耗尽后返回 `unavailable`；系统没有生成确定性 fallback 图。

## 修复后增量回归

本节记录上述失败后的结构性修复及真实复测。它与前面的首次基线结果分开保留，避免覆盖失败证据。

### 修复内容

1. 组件只选择 `content_id`，source 由 compiler 从 ContentItem 唯一派生，彻底消除组件/source 所属关系错配。
2. 第二阶段 response schema 按实际 inventory 动态生成字段枚举；未知字段无法进入 provider 返回值。
3. 动态 schema 的空组件数组使用封闭 Pydantic model，不再产生 `items.additionalProperties` 400。
4. 每张图使用必填 `host_line`；该字段的枚举只包含目标 goal 的 host source，host 覆盖由结构保证。
5. line/band 仅允许绑定至少两行且具有真实时间字段的 source；单记录 Insight 只能进入 point、interval、reference line 或 annotation。
6. component ID 由 compiler 生成，LLM 不再负责全局唯一性。

### 复测结果

| # | 结果 | 实际执行链路 | V4 artifact | 请求 ID |
| ---: | --- | --- | --- | --- |
| 1 | 成功（基线保持） | `sql_query(fail) → sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_96e918bfe8d2` | `req_0c471625565b` |
| 2 | 成功（基线保持） | `sql_query(ok) → code_interpreter(fail) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_7d9806f27a08` | `req_9e9856360e88` |
| 3 | 成功 | `sql_query(ok) → anomaly(ok) → code_interpreter(fail) → code_interpreter(fail) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_8cd0807eb0ad` | `req_d8ca043ebaea` |
| 4 | 成功 | `sql_query(ok) → code_interpreter(fail) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_89d050dfea3f` | `req_a6d9aa0888cd` |
| 5 | 成功 | `sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_bba0e0035c19` | `req_9ae9c498f5b2` |
| 6 | 成功 | `sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_69400986daeb` | `req_555bf3686d56` |
| 7 | 成功 | `sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_570c76d2934d` | `req_bd4481f890de` |
| 8 | 成功（ReAct 重试） | `sql_query(ok) → code_interpreter(fail) → code_interpreter(ok) → visualization(fail) → visualization(created) → terminate(ok)` | `viz_ce653a00c3f8` | `req_61f1e4100aa8` |
| 9 | 成功 | `sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_aec28931945d` | `req_c72037cfb94c` |
| 10 | 成功 | `sql_query(ok) → code_interpreter(ok) → visualization(created) → terminate(ok)` | `viz_50890804cd30` | `req_e40a74872c91` |

复测覆盖的 10 个问题均已有成功创建 V4 LineChart 的真实请求证据。最终一轮针对新 `host_line` 和单记录 Insight 约束重新执行了 #4、#5、#7，三项均为 `completed`；后端完整回归为 `346 passed`。未执行前端全量编译。

## 测试问题

1. 这一个月比特币美元价格哪一次跌得最急？找出 2 小时内跌幅最大的那段，给出起止时间。只统计价格在 1 万到 10 万美元之间的数据。
2. 1月比特币美元价格波动大吗？用每天的最高最低价差（日波幅）的平均来衡量。只统计价格在 1 万到 10 万美元之间的数据。
3. 从月内最低点算起，价格最多反弹了多少？低点和反弹高点分别在何时？（排除异常点）
4. 1 月里比特币美元价格涨得最猛的是哪一段？找出 2 小时内涨幅最大的时间窗口。只统计价格在 1 万到 10 万美元之间的数据。
5. 这个月有没有一段时间价格连续每天都创新高？持续了几天？只统计价格在 1 万到 10 万美元之间的数据。
6. 这个月哪一天价格波动最剧烈？当天最高最低差多少？只统计价格在 1 万到 10 万美元之间的数据。
7. 比特币美元价格上半月（1/4–1/19）和下半月（1/20–2/3）的平均价各是多少？下半月比上半月高多少？只统计价格在 1 万到 10 万美元之间的数据。
8. 从这个月的最低点算起，价格花了多久才反弹 20%？只统计价格在 1 万到 10 万美元之间的数据。
9. 月末价格是不是一直站在高位？找出价格持续稳定在 23000 以上的那段时间。只统计价格在 1 万到 10 万美元之间的数据。
10. 这个月比特币美元价格从月初到月末，整体是涨还是跌？净变化了百分之几？只统计价格在 1 万到 10 万美元之间的数据。
