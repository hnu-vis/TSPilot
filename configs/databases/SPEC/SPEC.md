# Configs/Databases - 数据库配置文件

## 1. 模块概述

Configs/Databases 目录包含各种时序数据库的连接配置模板。

## 2. 配置文件列表

| 文件 | 数据库 | 说明 |
|------|--------|------|
| `influxdb.yaml` | InfluxDB 1.x/2.x | InfluxDB 连接配置 |
| `influxdb2.yaml` | InfluxDB 2.x | InfluxDB 2.x 专用配置 |
| `influxdb/influxdb2_energydata.yaml` | InfluxDB 2.x | EnergyData 演示数据库 |
| `influxdb/influxdb2_bitcoin_sample.yaml` | InfluxDB 2.x | Bitcoin sample data 演示数据库 |
| `timescaledb.yaml` | TimescaleDB | TimescaleDB/PostgreSQL 配置 |
| `prometheus/prometheus_local.yaml` | Prometheus | Prometheus 连接配置 |
| `prometheus_xiaoming_scores.yaml` | Prometheus | 小明成绩 CSV 演示数据库 |
| `iotdb.yaml` | Apache IoTDB | IoTDB 连接配置 |
| `questdb.yaml` | QuestDB | QuestDB 连接配置 |
| `clickhouse.yaml` | ClickHouse | ClickHouse HTTP SQL 配置 |

数据库连接配置支持放在 `configs/databases/` 根目录，也支持按数据库类型放在子目录中。
加载器会递归扫描 `*.yaml` 和 `*.yml` 文件，并忽略不满足数据库连接配置格式的扩展配置。
运行时缓存中的项目配置会按 `config_source` 与当前文件同步；删除项目配置文件后，对应数据库会从运行时缓存中移除。
Prometheus 的 scrape 配置属于服务自身配置，位于 `configs/services/prometheus/prometheus.yml`，不属于数据库连接配置。

## 3. influxdb.yaml

```yaml
# InfluxDB 1.x 配置
type: influxdb
version: 1  # 1 或 2

# 连接信息
host: localhost
port: 8086
database: metrics
username: admin
password: ${INFLUXDB_PASSWORD}

# SSL
ssl_enabled: false

# 连接池
max_connections: 10
timeout: 30

# 查询配置
query_timeout: 60
max_rows: 10000

# 认证（InfluxDB 1.x）
auth_type: auth
```

## 4. influxdb2.yaml

```yaml
# InfluxDB 2.x 配置
type: influxdb
version: 2

# 连接信息
host: localhost
port: 8086
org: myorg
bucket: mybucket
token: ${INFLUXDB_TOKEN}

# SSL
ssl_enabled: true

# 超时
timeout: 30
query_timeout: 60

# 连接池
max_connections: 10

# InfluxDB 2.x task（可选）
# 使用脚本同步：
#   python scripts/sync_influxdb_tasks.py --database influxdb2-bitcoin-sample
# 如果上游 sample.data 在当前 InfluxDB 环境失败，可先导入 annotated CSV：
#   python scripts/import_influxdb_annotated_csv.py --bucket bitcoin --replace-measurement
influxdb_tasks:
  - name: Collect Bitcoin sample data
    every: 15m
    description: Continually imports the InfluxDB Bitcoin sample dataset.
    sample:
      set: bitcoin
      target_bucket: bitcoin

  - name: Custom Flux ingestion task
    every: 1h
    flux: |
      from(bucket: "source")
        |> range(start: -1h)
        |> to(bucket: "target")
```

## 5. timescaledb.yaml

```yaml
# TimescaleDB / PostgreSQL 配置
type: timescaledb  # 兼容旧别名 timescale

# 连接信息
host: localhost
port: 5432
database: metrics
username: postgres
password: ${TIMESCALE_PASSWORD}

# SSL
ssl_enabled: false
ssl_mode: prefer  # disable / allow / prefer / require / verify-full

# 连接池
max_connections: 20
min_connections: 5
idle_timeout: 300

# TimescaleDB 特定配置
timescale_version: "2.10"
chunk_interval: 1d  # 压缩块间隔
compression: true   # 是否启用压缩
```

## 6. prometheus/prometheus_local.yaml

```yaml
# Prometheus 配置
type: prometheus

# 连接信息
host: localhost
port: 9090
query_url: /api/v1

# 认证（可选）
auth:
  username: ${PROM_USERNAME}
  password: ${PROM_PASSWORD}

# 查询配置
query_timeout: 60
max_samples: 10000

# 认证类型
auth_type: basic  # none / basic / bearer

# 代理（可选）
use_env_proxy: false
```

### configs/services/prometheus/prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['prometheus:9090']
```

## 7. iotdb.yaml

```yaml
# Apache IoTDB 配置
type: iotdb

# 连接信息
host: localhost
port: 6667
username: root
password: root

# SSL
ssl_enabled: false

# 连接池
max_connections: 10
timeout: 30

# IoTDB 特定配置
zone_id: default  # 时区
```

## 8. questdb.yaml

```yaml
# QuestDB 配置
type: questdb

# 连接信息
host: localhost
port: 8812
database: qdb
username: admin
password: quest

# SSL
ssl_enabled: false

# 连接池
max_connections: 10
timeout: 30

# QuestDB 特定配置
# QuestDB 使用 PostgreSQL 协议，所以大部分配置与 timescale 相同
```

## 9. clickhouse.yaml

```yaml
# ClickHouse 配置
type: clickhouse

# 连接信息
host: localhost
port: 8123
database: default
username: default
password: ${CLICKHOUSE_PASSWORD}

# SSL
ssl_enabled: false

# 连接池
max_connections: 10
timeout: 30
```

## 10. 配置模板使用

```python
# configs/databases/loader.py
import yaml
from pathlib import Path

def load_database_config(name: str) -> dict:
    """加载数据库配置"""
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{name}.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 替换环境变量
    config = substitute_env_vars(config)

    return config

def substitute_env_vars(config: dict) -> dict:
    """替换配置中的环境变量 ${VAR_NAME}"""
    for key, value in config.items():
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            config[key] = os.getenv(env_var, '')
        elif isinstance(value, dict):
            config[key] = substitute_env_vars(value)
    return config
```

## 11. 连接示例

```yaml
# 示例：配置多个数据库
databases:
  - name: influxdb-prod
    type: influxdb
    host: prod-influxdb.example.com
    port: 8086
    database: production
    username: ${INFLUX_USER}
    password: ${INFLUX_PASSWORD}
    ssl_enabled: true

  - name: timescale-dw
    type: timescaledb
    host: dw.example.com
    port: 5432
    database: metrics
    username: ${TS_USER}
    password: ${TS_PASSWORD}
    ssl_enabled: true

  - name: prometheus-monitoring
    type: prometheus
    url: http://prometheus:9090

  - name: clickhouse-events
    type: clickhouse
    host: clickhouse.example.com
    port: 8123
    database: observability
    username: default
    password: ${CLICKHOUSE_PASSWORD}
```

## 12. 配置验证

```python
# configs/databases/validator.py
from pydantic import BaseModel, validator

class DatabaseConfig(BaseModel):
    type: str
    host: str
    port: int
    database: str
    username: str | None = None
    password: str | None = None
    ssl_enabled: bool = False
    timeout: int = 30

    @validator('port')
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError('Invalid port number')
        return v

    @validator('type')
    def validate_type(cls, v):
        supported = ['influxdb', 'timescaledb', 'timescale', 'prometheus', 'iotdb', 'questdb', 'clickhouse']
        if v not in supported:
            raise ValueError(f'Unsupported database type: {v}')
        return v
```
