"""Shared ODBC connector base.

Several embedded/edge time-series databases (RaimaDB, eXtremeDB, ITTIA DB) are
reachable from Python only through their ODBC driver in server/SQL editions.
This base implements a generic SQL-over-ODBC connector via ``pyodbc`` (imported
lazily so the type registers without pyodbc/the vendor driver installed).

Concrete connectors set ``ODBC_DRIVER`` (the installed ODBC driver name), a
default port, ``dialect`` and ``database_type``.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class ODBCConfig(DBConfig):
    """Generic ODBC connection configuration."""

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
        super().__init__(
            host=host,
            port=port,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        # Allow overriding the ODBC driver name per connection.
        self.driver = str(kwargs.get("driver") or "")


class ODBCConnector(DBConnector):
    """Generic SQL-over-ODBC connector (pyodbc)."""

    ODBC_DRIVER = ""

    def __init__(self, config: ODBCConfig):
        super().__init__(config)
        self.config: ODBCConfig = config
        self._conn: Any = None

    @property
    def dialect(self) -> str:
        return "odbc"

    def _dsn(self) -> str:
        driver = self.config.driver or self.ODBC_DRIVER
        parts = [f"DRIVER={{{driver}}}", f"SERVER={self.config.host}"]
        if self.config.port:
            parts.append(f"PORT={self.config.port}")
        if self.config.database:
            parts.append(f"DATABASE={self.config.database}")
        if self.config.username:
            parts.append(f"UID={self.config.username}")
        if self.config.password:
            parts.append(f"PWD={self.config.password}")
        return ";".join(parts) + ";"

    async def connect(self) -> None:
        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc package required for ODBC connectors. Install with: pip install pyodbc")
        try:
            self._conn = pyodbc.connect(self._dsn(), timeout=self.config.timeout)
            self._connected = True
        except Exception as exc:
            self._connected = False
            raise ConnectionError(f"Failed to connect via ODBC ({self.ODBC_DRIVER}): {exc}")

    async def disconnect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
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
            raise ConnectionError("Not connected (ODBC)")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            cursor = self._conn.cursor()
            cursor.execute(sql)
            columns = [str(desc[0]) for desc in (cursor.description or [])]
            rows = [
                {columns[idx]: self._format_value(value) for idx, value in enumerate(record) if idx < len(columns)}
                for record in cursor.fetchall()
            ]
            cursor.close()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            cursor = self._conn.cursor()
            for row in cursor.tables(tableType="TABLE"):
                name = str(getattr(row, "table_name", "") or "")
                if name:
                    schema.tables.append(TableSchema(name=name, type="table", columns=[]))
            cursor.close()
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            result = await self.execute("SELECT 1")
            return result.row_count >= 0
        except Exception:
            return False

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
