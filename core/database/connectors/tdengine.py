"""TDengine connector implementation.

TDengine (via taosAdapter) answers SQL over an HTTP REST endpoint
(``POST /rest/sql``) with the raw SQL in the request body and HTTP Basic auth.
The response carries column metadata (names at ``column_meta[i][0]``) plus
positional rows in ``data``.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class TDengineConfig(DBConfig):
    """TDengine REST (taosAdapter) configuration (default port 6041)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6041,
        database: str = "",
        username: str = "root",
        password: str = "taosdata",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 6041,
            database=database or "",
            username=username or "root",
            password=password or "taosdata",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class TDengineConnector(DBConnector):
    """Connector for TDengine over its taosAdapter REST endpoint."""

    def __init__(self, config: TDengineConfig):
        super().__init__(config)
        self.config: TDengineConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "tdengine"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.TDENGINE

    def _sql_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        base = f"{protocol}://{self.config.host}:{self.config.port}/rest/sql"
        return f"{base}/{self.config.database}" if self.config.database else base

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                self._sql_url(),
                data="SELECT SERVER_VERSION()".encode("utf-8"),
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to TDengine: {exc}")

    async def disconnect(self) -> None:
        self._close_session()
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        if not self._connected or self._session is None:
            raise ConnectionError("Not connected to TDengine")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._sql_url(),
                data=sql.encode("utf-8"),
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        # TDengine returns code==0 (newer) or status=="succ" (older) on success.
        ok = payload.get("code", 0) in (0, None) and payload.get("status", "succ") != "error"
        if not isinstance(payload, dict) or not ok:
            raise RuntimeError(f"TDengine query error: {payload.get('desc') or payload.get('error') or payload}")

        column_meta = payload.get("column_meta") or []
        columns = [str(meta[0]) for meta in column_meta if isinstance(meta, (list, tuple)) and meta]
        raw_rows = payload.get("data") or []
        if not columns and raw_rows:
            columns = [f"col{i}" for i in range(len(raw_rows[0]))]

        rows = [
            {columns[idx]: self._format_value(value) for idx, value in enumerate(row) if idx < len(columns)}
            for row in raw_rows
            if isinstance(row, (list, tuple))
        ]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        if not self.config.database:
            schema.metadata["error"] = "no database configured"
            return schema
        try:
            result = await self.execute(
                "SELECT table_name, col_name, col_type FROM information_schema.ins_columns "
                f"WHERE db_name = '{self.config.database}'"
            )
            by_table: dict[str, list[ColumnSchema]] = {}
            for row in result.rows:
                table_name = str(row.get("table_name") or "")
                column_name = str(row.get("col_name") or "")
                if not table_name or not column_name:
                    continue
                by_table.setdefault(table_name, []).append(
                    ColumnSchema(name=column_name, data_type=str(row.get("col_type") or "unknown").lower())
                )
            for table_name, columns in by_table.items():
                schema.tables.append(TableSchema(name=table_name, type="table", columns=columns))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            result = await self.execute("SELECT SERVER_VERSION()")
            return result.row_count >= 0
        except Exception:
            return False

    def _auth(self) -> tuple[str, str] | None:
        if not self.config.username and not self.config.password:
            return None
        return (self.config.username, self.config.password)

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
