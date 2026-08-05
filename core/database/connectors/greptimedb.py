"""GreptimeDB connector implementation.

GreptimeDB answers SQL over an HTTP endpoint (``POST /v1/sql``). The response
carries both column names and types in ``output[].records.schema`` plus
positional rows, so no projection parsing is needed.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class GreptimeDBConfig(DBConfig):
    """GreptimeDB HTTP SQL configuration (default HTTP port 4000)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4000,
        database: str = "public",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 4000,
            database=database or "public",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class GreptimeDBConnector(DBConnector):
    """Connector for GreptimeDB over its HTTP SQL endpoint."""

    def __init__(self, config: GreptimeDBConfig):
        super().__init__(config)
        self.config: GreptimeDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "greptimedb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.GREPTIMEDB

    def _base_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                f"{self._base_url()}/v1/sql",
                params={"db": self.config.database},
                data={"sql": "SELECT 1"},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to GreptimeDB: {exc}")

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
            raise ConnectionError("Not connected to GreptimeDB")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                f"{self._base_url()}/v1/sql",
                params={"db": self.config.database},
                data={"sql": sql},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if not isinstance(payload, dict) or payload.get("code", 0) not in (0, None):
            raise RuntimeError(f"GreptimeDB query error: {payload.get('error') or payload}")

        records = {}
        for item in payload.get("output") or []:
            if isinstance(item, dict) and isinstance(item.get("records"), dict):
                records = item["records"]
                break
        column_schemas = (records.get("schema") or {}).get("column_schemas") or []
        columns = [str(col.get("name")) for col in column_schemas if isinstance(col, dict)]
        raw_rows = records.get("rows") or []
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
        try:
            result = await self.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                f"WHERE table_schema = '{self.config.database}' ORDER BY table_name, ordinal_position"
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
