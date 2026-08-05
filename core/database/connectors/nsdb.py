"""NSDb (Natural Series Database) connector implementation.

NSDb answers a SQL-like language over an HTTP endpoint (``POST /query``). Unlike
a plain SQL API, NSDb needs the query decomposed into db / namespace / metric,
so the connector parses the metric name from the query's FROM clause and sends
``{"db", "namespace", "metric", "queryString"}``. The response carries a
``records`` array of row objects.

Note: unverified against a live NSDb instance; based on NSDb REST docs.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class NSDbConfig(DBConfig):
    """NSDb HTTP query configuration (default HTTP port 9000)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        database: str = "root",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 9000,
            database=database or "root",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        self.namespace = str(kwargs.get("namespace") or "registry")


class NSDbConnector(DBConnector):
    """Connector for NSDb over its HTTP query endpoint."""

    def __init__(self, config: NSDbConfig):
        super().__init__(config)
        self.config: NSDbConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "nsdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.NSDB

    def _query_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/query"

    @staticmethod
    def _metric_from_sql(sql: str) -> str:
        match = re.search(r"\bfrom\s+([A-Za-z_][\w]*)", sql, re.IGNORECASE)
        return match.group(1) if match else ""

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.get(
                self._query_url().replace("/query", "/"),
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            # Any HTTP answer means the server is reachable.
            self._connected = response.status_code < 500
            if not self._connected:
                raise ConnectionError(f"HTTP {response.status_code}")
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to NSDb: {exc}")

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
            raise ConnectionError("Not connected to NSDb")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                json={
                    "db": self.config.database,
                    "namespace": self.config.namespace,
                    "metric": self._metric_from_sql(sql),
                    "queryString": sql,
                },
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        record_rows = _flatten_records(payload.get("records") if isinstance(payload, dict) else payload)
        columns = _ordered_columns(record_rows)
        rows = [{col: self._format_value(row.get(col)) for col in columns} for row in record_rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        schema.metadata["namespace"] = self.config.namespace
        return schema

    async def health_check(self) -> bool:
        try:
            response = self._session.get(
                self._query_url().replace("/query", "/"),
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


def _flatten_records(records: Any) -> list[dict]:
    """Flatten NSDb records (top-level scalars + merged dimensions/tags)."""
    out: list[dict] = []
    for record in (records or []):
        if not isinstance(record, dict):
            continue
        flat: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[str(sub_key)] = sub_value
            else:
                flat[str(key)] = value
        out.append(flat)
    return out


def _ordered_columns(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    return columns
