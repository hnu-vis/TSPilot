"""GridDB connector implementation.

GridDB exposes SQL over its WebAPI component
(``POST /griddb/v2/{cluster}/dbs/{database}/sql/dml/query``) with a JSON body
``[{"stmt": "..."}]`` and HTTP Basic auth. The response is a list whose first
element carries ``columns`` (name/type objects) and positional ``results``.

The cluster name is part of the WebAPI path; provide it via the ``cluster``
config key (default ``defaultCluster``).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class GridDBConfig(DBConfig):
    """GridDB WebAPI SQL configuration (default WebAPI port 8081)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8081,
        database: str = "public",
        username: str = "admin",
        password: str = "admin",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8081,
            database=database or "public",
            username=username or "admin",
            password=password or "admin",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        self.cluster = str(kwargs.get("cluster") or "defaultCluster")


class GridDBConnector(DBConnector):
    """Connector for GridDB over its WebAPI SQL endpoint."""

    def __init__(self, config: GridDBConfig):
        super().__init__(config)
        self.config: GridDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "griddb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.GRIDDB

    def _base(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/griddb/v2/{self.config.cluster}/dbs/{self.config.database}"

    def _query_url(self) -> str:
        return f"{self._base()}/sql/dml/query"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                self._query_url(),
                json=[{"stmt": "SELECT 1"}],
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to GridDB WebAPI: {exc}")

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
            raise ConnectionError("Not connected to GridDB")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                json=[{"stmt": sql}],
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        block = payload[0] if isinstance(payload, list) and payload else (payload if isinstance(payload, dict) else {})
        columns = [str(col.get("name")) for col in (block.get("columns") or []) if isinstance(col, dict)]
        raw_rows = block.get("results") or []
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
            response = self._session.get(
                f"{self._base()}/containers",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            names = payload.get("names") if isinstance(payload, dict) else payload
            for name in (names or []):
                schema.tables.append(TableSchema(name=str(name), type="table", columns=[]))
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
