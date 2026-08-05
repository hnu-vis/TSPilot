"""BangDB connector implementation.

BangDB exposes a REST API (``POST /rest/...``) and supports a SQL-like query
language. This connector posts the query to the REST query endpoint and parses
an array-of-row-objects response defensively.

Note: lowest-confidence connector — unverified against a live BangDB; the REST
query path/shape may need adjustment.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class BangDBConfig(DBConfig):
    """BangDB REST configuration (default port 10101)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 10101,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 10101,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class BangDBConnector(DBConnector):
    """Connector for BangDB over its REST query endpoint."""

    def __init__(self, config: BangDBConfig):
        super().__init__(config)
        self.config: BangDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "bangdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.BANGDB

    def _query_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/rest/query"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        self._session = requests.Session()
        try:
            response = self._session.post(
                self._query_url(),
                json={"query": "SELECT 1", "db": self.config.database},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            self._connected = response.status_code < 500
            if not self._connected:
                raise ConnectionError(f"HTTP {response.status_code}")
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to BangDB: {exc}")

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
            raise ConnectionError("Not connected to BangDB")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                json={"query": sql, "db": self.config.database},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        record_rows = _extract_rows(payload)
        columns: list[str] = []
        for row in record_rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(str(key))
        rows = [{col: self._format_value(row.get(col)) for col in columns} for row in record_rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        return DatabaseSchema(database=self.config.database or "bangdb")

    async def health_check(self) -> bool:
        try:
            response = self._session.post(
                self._query_url(),
                json={"query": "SELECT 1", "db": self.config.database},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            return response.status_code < 500
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


def _extract_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "result", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []
