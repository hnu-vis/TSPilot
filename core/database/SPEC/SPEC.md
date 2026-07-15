# Core/Database - 时序数据库连接层

## 1. 模块职责

Core/Database 是 TSPilot 的时序数据库抽象层，提供统一的数据库连接和查询接口。支持多种主流时序数据库：
- **InfluxDB** - 云原生时序数据库
- **TimescaleDB** - 基于 PostgreSQL 的时序扩展
- **Prometheus** - 监控时序数据库
- **IoTDB** - 物联网时序数据库
- **QuestDB** - 高性能时序数据库
- **OpenTSDB** - HBase 上的时序数据库

当前推荐的解耦查询架构见：

- [query_architecture_SPEC.md](/home/feilvvl/TSPilot-v0.2/core/database/SPEC/query_architecture_SPEC.md)

## 2. 核心接口 / 关键类

| 文件 | 类/接口 | 职责 |
|------|---------|------|
| `base.py` | `BaseTSDatabase`, `TSDatabaseConfig` | 抽象基类和配置 |
| `influxdb.py` | `InfluxDBDatabase` | InfluxDB 实现 |
| `timescale.py` | `TimescaleDBDatabase` | TimescaleDB 实现 |
| `prometheus.py` | `PrometheusDatabase` | Prometheus 实现 |
| `iotdb.py` | `IoTDBDatabase` | IoTDB 实现 |
| `questdb.py` | `QuestDBDatabase` | QuestDB 实现 |
| `connection_pool.py` | `ConnectionPool` | 连接池管理 |
| `query_builder.py` | `QueryBuilder` | 跨数据库查询构建 |
| `dialect.py` | `DialectRegistry` | SQL 方言注册 |

### BaseTSDatabase 抽象接口

```python
class BaseTSDatabase(ABC):
    config: TSDatabaseConfig
    connection_pool: ConnectionPool

    @abstractmethod
    async def connect(self) -> None:
        """建立数据库连接"""

    @abstractmethod
    async def disconnect(self) -> None:
        """关闭数据库连接"""

    @abstractmethod
    async def execute(self, query: str) -> QueryResult:
        """执行查询"""

    @abstractmethod
    async def get_schema(self, table: str) -> TableSchema:
        """获取表结构"""

    @abstractmethod
    def translate_query(self, generic_query: str) -> str:
        """将通用查询翻译为数据库特定 SQL"""

    async def test_connection(self) -> bool:
        """测试连接是否正常"""

    async def get_metrics(self) -> list[str]:
        """获取可用指标列表"""
```

### TSDatabaseConfig 配置

```python
class TSDatabaseConfig(BaseModel):
    db_type: TSDatabaseType              # 数据库类型
    host: str                             # 主机地址
    port: int                             # 端口
    username: str | None                 # 用户名
    password: str | None                 # 密码
    database: str                         # 数据库名
    ssl_enabled: bool = False             # SSL 启用
    timeout: int = 30                     # 超时时间（秒）
    max_connections: int = 10            # 最大连接数
```

## 3. 连接管理

### ConnectionPool

```python
class ConnectionPool:
    def __init__(self, config: TSDatabaseConfig):
        self._pool: asyncpg.Pool | None    # PostgreSQL 连接池
        self._influx_client: InfluxDBClient | None  # InfluxDB 客户端

    async def acquire(self) -> Connection:
        """获取连接"""

    async def release(self, conn: Connection) -> None:
        """释放连接"""

    async def close(self) -> None:
        """关闭连接池"""
```

### 多数据库支持

```python
class DatabaseFactory:
    @staticmethod
    def create(config: TSDatabaseConfig) -> BaseTSDatabase:
        """根据配置创建对应的数据库实例"""
        ...

    @staticmethod
    def register_dialect(db_type: TSDatabaseType, impl: type[BaseTSDatabase]) -> None:
        """注册新的数据库方言"""
```

## 4. SQL 方言处理

不同数据库的 SQL 语法差异通过 `DialectRegistry` 统一管理：

```python
# 时序函数映射
DIALECT_FUNCTIONS = {
    "influxdb": {
        "time_bucket": 'time(timestamp, interval)',
        "last_value": "last",
        "difference": "difference",
    },
    "timescale": {
        "time_bucket": "time_bucket(interval, timestamp)",
        "last_value": "last(value, timestamp)",
        "difference": "value - lag(value)",
    },
    # ...
}
```

## 5. 查询构建器

### QueryBuilder

```python
class QueryBuilder:
    def __init__(self, target_db: TSDatabaseType):
        self._db_type = target_db
        self._query = QueryState()

    def select(self, *columns) -> QueryBuilder:
        """SELECT 子句"""

    def from_table(self, table: str) -> QueryBuilder:
        """FROM 子句"""

    def where(self, condition: Condition) -> QueryBuilder:
        """WHERE 子句"""

    def time_range(self, start: datetime, end: datetime) -> QueryBuilder:
        """时间范围过滤"""

    def group_by_time(self, interval: str) -> QueryBuilder:
        """时间聚合"""

    def build(self) -> str:
        """构建最终 SQL"""
```

## 6. 数据模型

### QueryResult

```python
class QueryResult(BaseModel):
    columns: list[str]                    # 列名
    rows: list[list[Any]]                 # 数据行
    types: list[ColumnType]               # 列类型
    row_count: int                        # 行数
    execution_time: float                  # 执行时间（毫秒）

class TableSchema(BaseModel):
    table_name: str
    columns: list[ColumnInfo]
    primary_key: str | None
    time_column: str | None
    indexes: list[str]
```

## 7. 错误处理

| 异常类 | 说明 |
|--------|------|
| `DatabaseConnectionError` | 数据库连接失败 |
| `QueryExecutionError` | SQL 执行错误 |
| `TranslationError` | SQL 翻译错误 |
| `SchemaNotFoundError` | 表/库不存在 |
| `AuthenticationError` | 认证失败 |
| `TimeoutError` | 查询超时 |

## 8. 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `DB_POOL_SIZE` | 10 | 连接池大小 |
| `DB_POOL_TIMEOUT` | 30 | 连接获取超时（秒） |
| `DB_QUERY_TIMEOUT` | 60 | 查询超时（秒） |
| `DB_RETRY_COUNT` | 3 | 失败重试次数 |

## 9. 依赖关系

- `influxdb-client` - InfluxDB 连接
- `asyncpg` - PostgreSQL/TimescaleDB 连接
- `prometheus-client` - Prometheus 连接
- `iotdb-client` - IoTDB 连接
- `questdb-client` - QuestDB 连接
- `sqlparse` - SQL 解析

## 10. 使用示例

```python
from core.database import DatabaseFactory

# 创建数据库实例
config = TSDatabaseConfig(
    db_type=TSDatabaseType.INFLUXDB,
    host="localhost",
    port=8086,
    database="metrics",
    username="admin",
    password="password"
)
db = DatabaseFactory.create(config)

# 连接并查询
await db.connect()
result = await db.execute("SELECT * FROM cpu WHERE time > now() - 1h")
print(result.rows)

# 获取模式
schema = await db.get_schema("cpu")
print(schema.columns)
```
