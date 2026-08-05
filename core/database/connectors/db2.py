"""IBM Db2 connector implementation.

Db2 is a SQL RDBMS with time-series extensions. Access is via the official
``ibm_db`` driver (DRDA protocol). The driver is imported lazily so the type
registers even where ``ibm_db`` is not installed; it is required at connect time.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class DB2Config(DBConfig):
    """IBM Db2 connection configuration (default port 50000)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50000,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 50000,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        self.schema = str(kwargs.get("schema") or "")


class DB2Connector(DBConnector):
    """Connector for IBM Db2 over the ibm_db driver."""

    def __init__(self, config: DB2Config):
        super().__init__(config)
        self.config: DB2Config = config
        self._conn: Any = None

    @property
    def dialect(self) -> str:
        return "db2"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.DB2

    def _dsn(self) -> str:
        parts = [
            f"DATABASE={self.config.database}",
            f"HOSTNAME={self.config.host}",
            f"PORT={self.config.port}",
            "PROTOCOL=TCPIP",
            f"UID={self.config.username}",
            f"PWD={self.config.password}",
        ]
        if self.config.ssl:
            parts.append("SECURITY=SSL")
        return ";".join(parts) + ";"

    async def connect(self) -> None:
        try:
            import ibm_db
        except ImportError:
            raise ImportError("ibm_db package required for Db2. Install with: pip install ibm_db")
        try:
            self._conn = ibm_db.connect(self._dsn(), "", "")
            self._connected = bool(self._conn)
        except Exception as exc:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Db2: {exc}")

    async def disconnect(self) -> None:
        if self._conn is not None:
            try:
                import ibm_db
                ibm_db.close(self._conn)
            except Exception:
                pass
            self._conn = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        if not self._connected or self._conn is None:
            raise ConnectionError("Not connected to Db2")
        import ibm_db
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            stmt = ibm_db.exec_immediate(self._conn, sql)
            rows: list[dict] = []
            row = ibm_db.fetch_assoc(stmt)
            while row:
                rows.append({str(k): self._format_value(v) for k, v in row.items()})
                row = ibm_db.fetch_assoc(stmt)
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        columns = list(rows[0].keys()) if rows else []
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            where = f"WHERE TABSCHEMA = '{self.config.schema}'" if self.config.schema else "WHERE TABSCHEMA NOT LIKE 'SYS%'"
            result = await self.execute(f"SELECT TABNAME FROM SYSCAT.TABLES {where} FETCH FIRST 200 ROWS ONLY")
            for row in result.rows:
                name = str(next(iter(row.values())) if row else "")
                if name:
                    schema.tables.append(TableSchema(name=name, type="table", columns=[]))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            result = await self.execute("SELECT 1 FROM SYSIBM.SYSDUMMY1")
            return result.row_count >= 0
        except Exception:
            return False

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
