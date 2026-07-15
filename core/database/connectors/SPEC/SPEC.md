# Core/Database/Connectors - 数据库连接器实现

## 1. 模块概述

Connectors 是 `core/database` 的具体数据库实现，每个连接器负责一种时序数据库的连接和查询执行。

## 2. 连接器列表

| 连接器 | 文件 | 支持的数据库 |
|--------|------|-------------|
| InfluxDB | `influxdb.py` | InfluxDB 1.x / 2.x |
| TimescaleDB | `timescaledb.py` | TimescaleDB (PostgreSQL) |
| Prometheus | `prometheus.py` | Prometheus |
| IoTDB | `iotdb.py` | Apache IoTDB |
| QuestDB | `questdb.py` | QuestDB |
| ClickHouse | `clickhouse.py` | ClickHouse HTTP SQL |

## 3. InfluxDB 连接器

```python
# core/database/connectors/influxdb.py
class InfluxDBConnector(BaseTSDatabase):
    """InfluxDB 连接器"""

    def __init__(self, config: TSDatabaseConfig):
        self._client: InfluxDBClient | None = None
        self._org: str | None = None  # InfluxDB 2.x 组织
        self._bucket: str | None = None

    async def connect(self) -> None:
        """建立 InfluxDB 连接"""
        # v1: username + password
        # v2: token + org

    async def execute(self, query: str) -> QueryResult:
        """执行 InfluxQL/Flux 查询"""
        # 使用 influxdb-client-python
        # 返回标准化 QueryResult

    async def get_schema(self, measurement: str) -> TableSchema:
        """获取 Measurement 结构"""
        # SHOW FIELD KEYS FROM measurement
        # SHOW TAG KEYS FROM measurement

    def translate_query(self, generic_sql: str) -> str:
        """转换为 InfluxQL"""
        # 处理 time() 函数
        # 处理 GROUP BY time()
```

### InfluxDB 特殊处理

```python
# 时间函数映射
INFLUX_FUNCTIONS = {
    "time_bucket": "time({interval})",
    "now": "now()",
    "last": "last",
    "first": "first",
    "difference": "difference",
    "non_negative_difference": "non_negative_difference",
    "moving_average": "moving_average",
    "timedifference": "timedelta",
}

# v1 vs v2 差异
def _get_query_api(self):
    if self._version == 2:
        return self._client.get_query_api()
    else:
        return self._client.query()
```

## 4. TimescaleDB 连接器

```python
# core/database/connectors/timescale.py
class TimescaleDBConnector(BaseTSDatabase):
    """TimescaleDB 连接器"""

    def __init__(self, config: TSDatabaseConfig):
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """建立 PostgreSQL 连接"""
        self._pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            user=self.config.username,
            password=self.config.password,
            database=self.config.database,
        )

    async def execute(self, query: str) -> QueryResult:
        """执行 SQL 查询"""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query)
            return self._to_query_result(rows)

    async def get_schema(self, table: str) -> TableSchema:
        """获取表结构"""
        # 查询 information_schema
        # 获取列信息、主键、时间列
```

### TimescaleDB 特殊函数

```python
# time_bucket 是 TimescaleDB 的核心功能
TIMESCALE_FUNCTIONS = {
    "time_bucket": "time_bucket({interval}, timestamp)",
    "last": "last(value, timestamp)",
    "first": "first(value, timestamp)",
    "gapfill": "time_bucket_gapfill({interval}, timestamp)",
    "interpolate": "interpolate(value)",
    "locf": "locf(value)",
}
```

## 5. Prometheus 连接器

```python
# core/database/connectors/prometheus.py
class PrometheusConnector(BaseTSDatabase):
    """Prometheus 连接器"""

    def __init__(self, config: TSDatabaseConfig):
        self._url: str = config.host  # Prometheus server URL
        self._client: prometheus_client.HttpClient | None = None

    async def connect(self) -> None:
        """初始化 Prometheus 客户端"""

    async def execute(self, query: str) -> QueryResult:
        """执行 PromQL 查询"""
        # 使用 prometheus_client
        # /api/v1/query 或 /api/v1/query_range

    async def get_schema(self, metric: str) -> TableSchema:
        """获取指标结构"""
        # 查询 metadata API

    def translate_query(self, generic_sql: str) -> str:
        """转换为 PromQL"""
        # 映射 SQL 函数到 PromQL 函数
```

### PromQL 翻译规则

```python
# SQL to PromQL 映射
PROMQL_FUNCTIONS = {
    "time_bucket": "rate",  # 转换为 rate() 聚合
    "last": "last_over_time",
    "avg": "avg_over_time",
    "max": "max_over_time",
    "min": "min_over_time",
    "sum": "sum",
}

# 时间字面量
PROMQL_TIME = {
    "1h": "1h",
    "1d": "1d",
    "now": "time()",
}
```

## 6. IoTDB 连接器

```python
# core/database/connectors/iotdb.py
class IoTDBConnector(BaseTSDatabase):
    """Apache IoTDB 连接器"""

    def __init__(self, config: TSDatabaseConfig):
        self._session: Session | None = None

    async def connect(self) -> None:
        """建立 IoTDB Session"""
        self._session = Session(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
        )
        await self._session.open()

    async def execute(self, query: str) -> QueryResult:
        """执行 IoTDB SQL (SHOW TIMESERIES, SELECT 等)"""
        # 使用 Session.execute_statement()

    async def get_schema(self, path: str) -> TableSchema:
        """获取时间序列结构"""
        # SHOW TIMESERIES path.**
```

## 7. QuestDB 连接器

```python
# core/database/connectors/questdb.py
class QuestDBConnector(BaseTSDatabase):
    """QuestDB 连接器"""

    def __init__(self, config: TSDatabaseConfig):
        self._conn: asyncpg.Connection | None = None

    async def connect(self) -> None:
        """建立 QuestDB 连接"""
        # QuestDB 使用 PostgreSQL 协议

    async def execute(self, query: str) -> QueryResult:
        """执行 SQL 查询"""
        # 使用 asyncpg 执行

    def translate_query(self, generic_sql: str) -> str:
        """转换为 QuestDB 特定 SQL"""
        # QuestDB 使用 SAMPLE BY 进行采样
```

## 8. 连接器工厂

```python
# core/database/connectors/registry.py
class ConnectorRegistry:
    """连接器注册表"""

    _connectors: dict[TSDatabaseType, type[BaseTSDatabase]] = {}

    @classmethod
    def register(cls, db_type: TSDatabaseType, connector: type[BaseTSDatabase]):
        """注册连接器"""
        cls._connectors[db_type] = connector

    @classmethod
    def create(cls, config: TSDatabaseConfig) -> BaseTSDatabase:
        """创建连接器实例"""
        connector_cls = cls._connectors.get(config.db_type)
        if not connector_cls:
            raise UnsupportedDialectError(config.db_type)
        return connector_cls(config)

# 注册内置连接器
ConnectorRegistry.register(TSDatabaseType.INFLUXDB, InfluxDBConnector)
ConnectorRegistry.register(TSDatabaseType.TIMESCALEDB, TimescaleDBConnector)
ConnectorRegistry.register(TSDatabaseType.PROMETHEUS, PrometheusConnector)
ConnectorRegistry.register(TSDatabaseType.IOTDB, IoTDBConnector)
ConnectorRegistry.register(TSDatabaseType.QUESTDB, QuestDBConnector)
ConnectorRegistry.register(TSDatabaseType.CLICKHOUSE, ClickHouseConnector)
```

## 9. 健康检查

```python
async def health_check(connector: BaseTSDatabase) -> HealthStatus:
    """检查数据库连接健康状态"""
    try:
        start = time.time()
        await connector.test_connection()
        return HealthStatus(
            status="healthy",
            latency_ms=(time.time() - start) * 1000
        )
    except Exception as e:
        return HealthStatus(
            status="unhealthy",
            error=str(e)
        )
```

## 10. 依赖关系

```
connectors/
├── registry.py           # 连接器注册表
├── influxdb.py           # ← influxdb-client
├── timescaledb.py       # ← psycopg2/PostgreSQL 协议
├── prometheus.py        # ← prometheus-client
├── iotdb.py             # ← iotdb-client
├── questdb.py           # ← requests 或 psycopg2
└── clickhouse.py        # ← requests HTTP SQL
```
