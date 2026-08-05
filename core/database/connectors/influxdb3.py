"""InfluxDB 3.x (Core/Enterprise) connector implementation.

InfluxDB 3 answers SQL over the v3 HTTP query API (``POST /api/v3/query_sql``)
with a JSON body ``{"db": ..., "q": ..., "format": "json"}`` and a Bearer
token. With ``format: json`` the response is an array of row objects. This is
a different engine from InfluxDB 1.x/2.x (Flux), so it is registered as its own
``influxdb3`` SQL type rather than reusing the Flux dialect.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class InfluxDB3Config(DBConfig):
    """InfluxDB 3 HTTP SQL configuration (default port 8181, Bearer token)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8181,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8181,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        # InfluxDB 3 authenticates with a Bearer token; accept it from several keys.
        self.token = str(kwargs.get("token") or kwargs.get("auth_token") or password or "")


class InfluxDB3Connector(DBConnector):
    """Connector for InfluxDB 3.x over its v3 HTTP SQL query API."""

    def __init__(self, config: InfluxDB3Config):
        super().__init__(config)
        self.config: InfluxDB3Config = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "influxdb3"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.INFLUXDB3

    def _query_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/api/v3/query_sql"

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
                json={"db": self.config.database, "q": "SELECT 1", "format": "json"},
                headers=self._headers(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to InfluxDB 3: {exc}")

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
            raise ConnectionError("Not connected to InfluxDB 3")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                json={"db": self.config.database, "q": sql, "format": "json"},
                headers=self._headers(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"InfluxDB 3 query error: {payload.get('error')}")

        record_rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
        columns = _ordered_columns(record_rows)
        rows = [{col: self._format_value(row.get(col)) for col in columns} for row in record_rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            result = await self.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'iox' ORDER BY table_name, ordinal_position"
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


def _ordered_columns(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    return columns
