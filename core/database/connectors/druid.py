"""Apache Druid connector implementation.

Druid answers Druid SQL over an HTTP endpoint (``POST /druid/v2/sql/``) with a
JSON body ``{"query": "..."}``. With the default ``object`` result format the
response is an array of row objects (column name -> value).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class DruidConfig(DBConfig):
    """Apache Druid SQL HTTP configuration (default router port 8888)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        database: str = "druid",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8888,
            database=database or "druid",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class DruidConnector(DBConnector):
    """Connector for Apache Druid over its Druid SQL HTTP endpoint."""

    def __init__(self, config: DruidConfig):
        super().__init__(config)
        self.config: DruidConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "druid"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.DRUID

    def _sql_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/druid/v2/sql/"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                self._sql_url(),
                json={"query": "SELECT 1"},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to Druid: {exc}")

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
            raise ConnectionError("Not connected to Druid")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._sql_url(),
                json={"query": sql, "resultFormat": "object"},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Druid query error: {payload.get('errorMessage') or payload.get('error')}")

        record_rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
        columns = _ordered_columns(record_rows)
        rows = [{col: self._format_value(row.get(col)) for col in columns} for row in record_rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            result = await self.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_SCHEMA = '{self.config.database}' ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )
            by_table: dict[str, list[ColumnSchema]] = {}
            for row in result.rows:
                table_name = str(row.get("TABLE_NAME") or row.get("table_name") or "")
                column_name = str(row.get("COLUMN_NAME") or row.get("column_name") or "")
                if not table_name or not column_name:
                    continue
                by_table.setdefault(table_name, []).append(
                    ColumnSchema(
                        name=column_name,
                        data_type=str(row.get("DATA_TYPE") or row.get("data_type") or "unknown").lower(),
                    )
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


def _ordered_columns(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    return columns
