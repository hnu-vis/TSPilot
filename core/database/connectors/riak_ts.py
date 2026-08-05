"""Riak TS connector implementation.

Riak TS (Basho) is a distributed time-series store that speaks ANSI SQL over the
Protocol Buffers interface via the official ``riak`` Python client. The client
is imported lazily so the type registers without ``riak`` installed.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class RiakTSConfig(DBConfig):
    """Riak TS connection configuration (default PB port 8087)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8087,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8087,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )


class RiakTSConnector(DBConnector):
    """Connector for Riak TS over the riak Python client (ts_query)."""

    def __init__(self, config: RiakTSConfig):
        super().__init__(config)
        self.config: RiakTSConfig = config
        self._client: Any = None

    @property
    def dialect(self) -> str:
        return "riak_ts"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.RIAK_TS

    async def connect(self) -> None:
        try:
            import riak
        except ImportError:
            raise ImportError("riak package required for Riak TS. Install with: pip install riak")
        try:
            self._client = riak.RiakClient(host=self.config.host, pb_port=self.config.port)
            self._client.ping()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._client = None
            raise ConnectionError(f"Failed to connect to Riak TS: {exc}")

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False

    @staticmethod
    def _table_from_sql(sql: str) -> str:
        match = re.search(r"\bfrom\s+([A-Za-z_][\w]*)", sql, re.IGNORECASE)
        return match.group(1) if match else ""

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        if not self._connected or self._client is None:
            raise ConnectionError("Not connected to Riak TS")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            ts_obj = self._client.ts_query(self._table_from_sql(sql), sql)
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        columns = [str(c) for c in (getattr(ts_obj, "columns", None) or [])]
        raw_rows = getattr(ts_obj, "rows", None) or []
        if not columns and raw_rows:
            columns = [f"col{i}" for i in range(len(raw_rows[0]))]
        rows = [
            {columns[idx]: self._format_value(value) for idx, value in enumerate(row) if idx < len(columns)}
            for row in raw_rows
        ]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database or "riak_ts")
        return schema

    async def health_check(self) -> bool:
        try:
            return bool(self._client and self._client.ping())
        except Exception:
            return False

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
