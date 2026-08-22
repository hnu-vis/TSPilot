"""Database connector interface and base classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DBConfig:
    """Database configuration base class."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 0,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.timeout = timeout
        self.extra = kwargs


class DatabaseType(str, Enum):
    """Supported database types."""
    INFLUXDB = "influxdb"
    TIMESCALEDB = "timescaledb"
    PROMETHEUS = "prometheus"
    IOTDB = "iotdb"
    QUESTDB = "questdb"
    OPENMLDB = "openmldb"
    VICTORIAMETRICS = "victoriametrics"
    M3DB = "m3db"
    GREPTIMEDB = "greptimedb"
    TDENGINE = "tdengine"
    CNOSDB = "cnosdb"
    ARCADEDB = "arcadedb"
    CRATEDB = "cratedb"
    DRUID = "druid"
    INFLUXDB3 = "influxdb3"
    GRIDDB = "griddb"
    MACHBASE = "machbase"
    OPENGEMINI = "opengemini"
    DB2 = "db2"
    RIAK_TS = "riak_ts"
    DOLPHINDB = "dolphindb"
    KDB = "kdb"
    BANGDB = "bangdb"
    ARC = "arc"


@dataclass
class ColumnSchema:
    """Column schema definition."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    default_value: Any = None
    description: str | None = None
    unit: str | None = None


@dataclass
class TableSchema:
    """Table schema definition."""
    name: str
    schema: str = ""
    type: str = "table"
    columns: list[ColumnSchema] = field(default_factory=list)
    row_count: int | None = None
    size_bytes: int | None = None


@dataclass
class DatabaseSchema:
    """Complete database schema."""
    database: str
    tables: list[TableSchema] = field(default_factory=list)
    views: list[TableSchema] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Query execution result."""
    columns: list[str]
    rows: list[dict]
    row_count: int
    execution_time_ms: int
    truncated: bool = False
    cursor: Any = None


class DBConnector(ABC):
    """Abstract base class for database connectors.

    Provides unified interface for all time-series databases.
    """

    def __init__(self, config: DBConfig):
        self.config = config
        self._connected = False

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Return SQL dialect name."""
        pass

    @property
    def database_type(self) -> DatabaseType:
        """Return database type."""
        raise NotImplementedError("database connector must declare its database_type")

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
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

    async def test_connection(self) -> dict:
        """Test database connection and return result."""
        try:
            await self.connect()
            is_healthy = await self.health_check()
            await self.disconnect()
            return {
                "success": is_healthy,
                "latency_ms": None,
                "version": None,
            }
        except Exception as e:
            return {
                "success": False,
                "latency_ms": None,
                "version": None,
                "error": str(e),
            }

    async def __aenter__(self) -> "DBConnector":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def _format_value(self, value: Any) -> Any:
        """Format value for display."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def _row_to_dict(self, columns: list[str], row: tuple) -> dict:
        """Convert row tuple to dictionary."""
        return {col: self._format_value(val) for col, val in zip(columns, row)}
