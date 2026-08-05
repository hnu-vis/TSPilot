"""ArcadeDB connector implementation.

ArcadeDB exposes a native HTTP/JSON API on port 2480. SQL commands run via
``POST /api/v1/command/{database}`` with a JSON body ``{"language":"sql",
"command":"..."}`` and HTTP Basic auth. Results come back as an array of row
objects under the ``result`` key.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class ArcadeDBConfig(DBConfig):
    """ArcadeDB HTTP configuration (default port 2480)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2480,
        database: str = "",
        username: str = "root",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 2480,
            database=database or "",
            username=username or "root",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class ArcadeDBConnector(DBConnector):
    """Connector for ArcadeDB over its HTTP/JSON command endpoint."""

    def __init__(self, config: ArcadeDBConfig):
        super().__init__(config)
        self.config: ArcadeDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "arcadedb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.ARCADEDB

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
            # /api/v1/ready is a lightweight server-readiness probe.
            response = self._session.get(
                f"{self._base_url()}/api/v1/ready",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to ArcadeDB: {exc}")

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
            raise ConnectionError("Not connected to ArcadeDB")
        if not self.config.database:
            raise ValueError("ArcadeDB queries require a configured database name")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                f"{self._base_url()}/api/v1/command/{self.config.database}",
                json={"language": "sql", "command": sql},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if not isinstance(payload, dict):
            raise RuntimeError(f"ArcadeDB unexpected response: {payload}")
        record_rows = [row for row in (payload.get("result") or []) if isinstance(row, dict)]
        columns = _ordered_columns(record_rows)
        rows = [{col: self._format_value(row.get(col)) for col in columns} for row in record_rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            result = await self.execute("SELECT name FROM schema:types")
            for row in result.rows:
                name = str(row.get("name") or "")
                if name:
                    schema.tables.append(TableSchema(name=name, type="table", columns=[]))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            response = self._session.get(
                f"{self._base_url()}/api/v1/ready",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            return response.ok
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
