"""CrateDB connector implementation.

CrateDB answers SQL over a simple HTTP endpoint (``POST /_sql``) with a JSON
body ``{"stmt": "..."}``. The response carries column names in ``cols`` and
positional rows in ``rows``.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class CrateDBConfig(DBConfig):
    """CrateDB HTTP SQL configuration (default port 4200)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4200,
        database: str = "doc",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 4200,
            database=database or "doc",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class CrateDBConnector(DBConnector):
    """Connector for CrateDB over its HTTP SQL endpoint."""

    def __init__(self, config: CrateDBConfig):
        super().__init__(config)
        self.config: CrateDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "cratedb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.CRATEDB

    def _sql_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/_sql"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                self._sql_url(),
                json={"stmt": "SELECT 1"},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to CrateDB: {exc}")

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
            raise ConnectionError("Not connected to CrateDB")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._sql_url(),
                json={"stmt": sql},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if not isinstance(payload, dict) or payload.get("error"):
            error = payload.get("error") if isinstance(payload, dict) else payload
            raise RuntimeError(f"CrateDB query error: {error}")

        columns = [str(col) for col in (payload.get("cols") or [])]
        raw_rows = payload.get("rows") or []
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
