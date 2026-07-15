"""ClickHouse connector implementation."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class ClickHouseConfig(DBConfig):
    """ClickHouse HTTP connection configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        username: str = "default",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port,
            database=database or "default",
            username=username or "default",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class ClickHouseConnector(DBConnector):
    """Connector for ClickHouse using the built-in HTTP SQL endpoint."""

    def __init__(self, config: ClickHouseConfig):
        super().__init__(config)
        self.config: ClickHouseConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "clickhouse"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.CLICKHOUSE

    def _base_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}"

    async def connect(self) -> None:
        """Establish a ClickHouse HTTP session."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.get(
                self._base_url(),
                params={
                    "database": self.config.database,
                    "query": "SELECT 1 FORMAT JSON",
                },
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to ClickHouse: {exc}")

    async def disconnect(self) -> None:
        """Close the ClickHouse HTTP session."""
        self._close_session()
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute a SQL query through ClickHouse HTTP JSON output."""
        if not self._connected or self._session is None:
            raise ConnectionError("Not connected to ClickHouse")

        start_time = time.time()
        sql = self._ensure_json_format(query)
        try:
            response = self._session.get(
                self._base_url(),
                params={
                    "database": self.config.database,
                    "query": sql,
                    **(params or {}),
                },
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            rows = [
                {
                    str(key): self._format_value(value)
                    for key, value in row.items()
                }
                for row in payload.get("data", [])
                if isinstance(row, dict)
            ]
            columns = [
                str(item.get("name"))
                for item in payload.get("meta", [])
                if isinstance(item, dict) and item.get("name")
            ]
            if not columns and rows:
                columns = list(rows[0].keys())
            execution_time_ms = int((time.time() - start_time) * 1000)
            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=int(payload.get("rows", len(rows)) or len(rows)),
                execution_time_ms=execution_time_ms,
                truncated=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

    async def get_schema(self) -> DatabaseSchema:
        """Get ClickHouse table and column schema for the configured database."""
        if not self._connected:
            raise ConnectionError("Not connected to ClickHouse")

        schema = DatabaseSchema(database=self.config.database)
        try:
            table_result = await self.execute(
                """
                SELECT
                    name,
                    engine,
                    total_rows
                FROM system.tables
                WHERE database = currentDatabase()
                ORDER BY name
                """
            )
            column_result = await self.execute(
                """
                SELECT
                    table,
                    name,
                    type,
                    default_expression
                FROM system.columns
                WHERE database = currentDatabase()
                ORDER BY table, position
                """
            )
            columns_by_table: dict[str, list[ColumnSchema]] = {}
            for row in column_result.rows:
                table_name = str(row.get("table") or "")
                column_name = str(row.get("name") or "")
                if not table_name or not column_name:
                    continue
                columns_by_table.setdefault(table_name, []).append(
                    ColumnSchema(
                        name=column_name,
                        data_type=str(row.get("type") or "unknown"),
                        default_value=row.get("default_expression") or None,
                    )
                )

            for row in table_result.rows:
                table_name = str(row.get("name") or "")
                if not table_name:
                    continue
                schema.tables.append(
                    TableSchema(
                        name=table_name,
                        type=str(row.get("engine") or "table"),
                        columns=columns_by_table.get(table_name, []),
                        row_count=self._coerce_int(row.get("total_rows")),
                    )
                )
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        """Check ClickHouse connection health."""
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

    def _ensure_json_format(self, query: str) -> str:
        text = str(query or "").strip().rstrip(";")
        if not text:
            return "SELECT 1 FORMAT JSON"
        if " format " in f" {text.lower()} ":
            return text
        return f"{text} FORMAT JSON"

    def _coerce_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
