# Database Connector Specification

## 1. Project Overview

**Project Name:** TSPilot
**Module Name:** Connector
**Type:** Core Module - Database (Python)
**File Path:** `core/database/connector.py`
**Core Functionality:** Unified database connector interface for time-series databases.
**Target Users:** Query agent, schema agent.

---

## 2. Functionality Specification

### 2.1 Core Features

| Feature | Description |
|---------|-------------|
| Unified Interface | Single interface for all databases |
| Connection Management | Pooling, retry, health checks |
| Query Execution | Execute SQL queries |
| Schema Operations | Get tables, columns, indexes |
| Transaction Support | ACID transactions where supported |

### 2.2 Connector Interface

```python
from abc import ABC, abstractmethod

class DBConnector(ABC):
    def __init__(self, config: DBConfig):
        self.config = config
        self.pool: ConnectionPool | None = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def execute(self, query: str, params: dict | None = None) -> QueryResult:
        """Execute query and return results."""
        pass

    @abstractmethod
    async def get_schema(self) -> DatabaseSchema:
        """Get database schema."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check connection health."""
        pass

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Return SQL dialect name."""
        pass
```

### 2.3 Connection Pool

```python
class ConnectionPool:
    def __init__(
        self,
        connector: DBConnector,
        min_size: int = 5,
        max_size: int = 20,
        max_idle_time: int = 300,
    ):
        pass

    async def acquire(self) -> Connection:
        """Get connection from pool."""
        pass

    async def release(self, conn: Connection) -> None:
        """Return connection to pool."""
        pass
```

### 2.4 Query Result

```python
@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int
    execution_time_ms: int
    truncated: bool = False
    cursor: Any | None = None
```

---

## 3. Technical Specification

### 3.1 Supported Databases

- InfluxDB (InfluxQL)
- TimescaleDB (PostgreSQL)
- Prometheus (PromQL)
- Apache IoTDB
- QuestDB (QuestDB SQL)

### 3.2 Error Handling

| Error | Handling |
|-------|----------|
| ConnectionError | Retry with backoff |
| QueryTimeout | Cancel and return partial |
| SyntaxError | Return error immediately |
| AuthError | Fail fast, notify user |

---

## 4. Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | All database types connect |
| 2 | Query execution works |
| 3 | Connection pooling works |
| 4 | Schema retrieval correct |
| 5 | Health checks work |
| 6 | Errors handled gracefully |
