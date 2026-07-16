# Bitcoin USD 周期性 E2E 与 DB-GPT 对比报告

日期: 2026-07-16

## 请求

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

## TSPilot 修复后 E2E

服务入口: `http://127.0.0.1:5681`

完整响应:

- `cache_data/bitcoin_e2e_after_seasonality_fix_2026-07-16.json`

执行链路:

```text
query_database -> insight -> format_answer
```

真实 Flux:

```flux
from(bucket: "bitcoin")
  |> range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)
  |> filter(fn: (r) => r._measurement == "coindesk")
  |> filter(fn: (r) => r._field == "price")
  |> filter(fn: (r) => r.code == "USD")
  |> filter(fn: (r) => r.crypto == "bitcoin")
```

查询结果:

- `row_count = 2679`
- `series_count = 1`
- `outlier_count = 2`
- IQR 清洗边界: `[10985.9243, 30219.2541]`

周期性分析结果:

```json
{
  "method": "group_by_timestamp_profile",
  "sample_count": 2679,
  "clean_sample_count": 2677,
  "daily": {
    "bucket_count": 24,
    "amplitude": 263.8323,
    "relative_amplitude": 0.01262,
    "strength": 0.000812,
    "has_periodicity": false
  },
  "weekly": {
    "bucket_count": 7,
    "amplitude": 545.3349,
    "relative_amplitude": 0.026085,
    "strength": 0.006123,
    "has_periodicity": false
  },
  "has_seasonality": false
}
```

结论:

```text
value 在该时间范围内没有明显每天或每周重复的周期性波动；
日内相对振幅 1.26%、强度 0.0008，
周内相对振幅 2.61%、强度 0.0061。
```

## DB-GPT 对比

DB-GPT 服务 `10.110.1.71:5660` 当前从本机直连失败，错误为:

```text
curl: (7) Failed to connect to 10.110.1.71 port 5660
```

本次对比基于本地 DB-GPT 同题报告和日志:

- `/home/feilvvl/DB-GPT/比特币周期性波动_端到端测试报告_v1.md`
- `/home/feilvvl/DB-GPT/logs/dbgpt_webserver.log`

DB-GPT 可以得到较好结果的原因:

1. DB-GPT 在 prompt 中暴露了表结构和样例行:
   `time, value, field, measurement, code, crypto, description, symbol`。
2. LLM 直接生成语义完整的 SQL:
   `crypto='bitcoin' AND measurement='coindesk' AND code='USD' AND field='price'`。
3. DB-GPT 不是只做一次抽象 insight，而是多轮 SQL:
   原始序列、按日/星期聚合、按小时/星期小时聚合。
4. DB-GPT 在看到前两条异常大值后，后续 SQL 主动加入 `value < 1000000`，使周期性判断不被明显异常值主导。

TSPilot 原问题:

1. 旧报告中的 Flux 没有 `code="USD"` 和 `crypto="bitcoin"`，会把 EUR/GBP/USD 等数据混在一起。
2. `insight` 工具不接受 evidence id 字符串，模型传引用时会触发 Pydantic 校验失败。
3. seasonality 验证基于点序号自相关，只扫描 2..96 个点周期，不理解真实时间戳上的 24 小时或 7 天周期。
4. 最终回答只输出 `autocorrelation=0.0, strength=0.0`，没有展示可审计的数据库分析过程。

已修复:

1. `InsightInput.database_evidence` 支持 `DatabaseEvidence | dict | str | None`。
2. seasonality 改为时间戳驱动的日/周剖面分析，并在 evidence 中展示 bucket 均值、bucket 计数、振幅、相对振幅、强度、异常值数量和清洗边界。
3. 保留原点序号自相关作为没有可解析时间戳时的 fallback。

验证:

```text
/home/feilvvl/TSPilot/tspilot_env/bin/python -m py_compile tools/insight.py core/insight/verification.py
/home/feilvvl/TSPilot/tspilot_env/bin/python -m pytest tests/test_insight_fact_types.py tests/test_tool_evidence_refs.py tests/test_query_flow.py -q
```

结果:

```text
11 passed, 3 warnings
```
