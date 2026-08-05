"""Machbase Neo connector implementation.

Machbase Neo answers SQL over its HTTP query endpoint (``POST /db/query``) with
a JSON body ``{"q": "...", "format": "json"}``. With ``format=json`` the
response carries ``data.columns`` (names) and positional ``data.rows``.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class MachbaseConfig(DBConfig):
    """Machbase Neo HTTP SQL configuration (default port 5654)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5654,
        database: str = "",
        username: str = "sys",
        password: str = "manager",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 5654,
            database=database or "",
            username=username or "sys",
            password=password or "manager",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class MachbaseConnector(DBConnector):
    """Connector for Machbase Neo over its HTTP query endpoint."""

    def __init__(self, config: MachbaseConfig):
        super().__init__(config)
        self.config: MachbaseConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "machbase"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.MACHBASE

    def _query_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/db/query"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                self._query_url(),
                json={"q": "SELECT 1", "format": "json"},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to Machbase Neo: {exc}")

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
            raise ConnectionError("Not connected to Machbase Neo")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                json={"q": sql, "format": "json"},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"Machbase query error: {payload.get('reason')}")

        data = payload.get("data") or {} if isinstance(payload, dict) else {}
        columns = [str(col) for col in (data.get("columns") or [])]
        raw_rows = data.get("rows") or []
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
        schema = DatabaseSchema(database=self.config.database or "machbase")
        try:
            result = await self.execute("SELECT NAME FROM M$SYS_TABLES")
            for row in result.rows:
                name = str(next(iter(row.values())) if row else "")
                if name:
                    schema.tables.append(TableSchema(name=name, type="table", columns=[]))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            result = await self.execute("SELECT 1")
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
