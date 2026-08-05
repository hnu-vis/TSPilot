"""Arc (Basekick Labs) connector implementation.

Arc is a DuckDB/Parquet-based time-series data warehouse. It answers SQL over
its HTTP API (``POST /api/v1/query``) with a JSON body ``{"sql": ..., "format":
"json"}`` and a Bearer token. The JSON response is parsed defensively across the
common shapes (array of row objects, or a columns/rows envelope).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class ArcConfig(DBConfig):
    """Arc HTTP SQL configuration (default port 8000, Bearer token)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        database: str = "default",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8000,
            database=database or "default",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        self.token = str(kwargs.get("token") or kwargs.get("api_token") or password or "")


class ArcConnector(DBConnector):
    """Connector for Arc over its HTTP SQL query API."""

    def __init__(self, config: ArcConfig):
        super().__init__(config)
        self.config: ArcConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "arc"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.ARC

    def _query_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/api/v1/query"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return headers

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        self._session = requests.Session()
        try:
            response = self._session.post(
                self._query_url(),
                json={"sql": "SELECT 1", "format": "json"},
                headers=self._headers(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to Arc: {exc}")

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
            raise ConnectionError("Not connected to Arc")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                json={"sql": sql, "format": "json"},
                headers=self._headers(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Arc query error: {payload.get('error')}")

        columns, rows = _parse_arc_payload(payload)
        rows = [{col: self._format_value(val) for col, val in row.items()} for row in rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            result = await self.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "ORDER BY table_name, ordinal_position"
            )
            by_table: dict[str, list[ColumnSchema]] = {}
            for row in result.rows:
                table_name = str(row.get("table_name") or "")
                column_name = str(row.get("column_name") or "")
                if not table_name or not column_name:
                    continue
                by_table.setdefault(table_name, []).append(
                    ColumnSchema(name=column_name, data_type=str(row.get("data_type") or "unknown").lower())
                )
            for table_name, columns in by_table.items():
                schema.tables.append(TableSchema(name=table_name, type="table", columns=columns))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            result = await self.execute("SELECT 1")
            return result.row_count >= 0
        except Exception:
            return False

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


def _parse_arc_payload(payload: Any) -> tuple[list[str], list[dict]]:
    """Parse an Arc JSON query payload into (columns, row dicts) defensively."""
    # Shape A: a bare list of row objects.
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
        return _columns(rows), rows
    if isinstance(payload, dict):
        # Shape B: {"data": [ {..}, .. ]} or {"rows": [ {..} ]}.
        for key in ("data", "rows", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                rows = [row for row in value if isinstance(row, dict)]
                return _columns(rows), rows
        # Shape C: {"columns": [...], "data"/"rows": [[...]]}.
        cols = payload.get("columns")
        data = payload.get("data") or payload.get("rows")
        if isinstance(cols, list) and isinstance(data, list):
            names = [str(c.get("name") if isinstance(c, dict) else c) for c in cols]
            rows = [
                {names[i]: v for i, v in enumerate(record) if i < len(names)}
                for record in data
                if isinstance(record, (list, tuple))
            ]
            return names, rows
    return [], []


def _columns(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    return columns
