"""Axibase Time Series Database (ATSD) connector implementation.

ATSD answers SQL over an HTTP endpoint (``POST /api/sql``) with the query in the
form field ``q`` and HTTP Basic auth, over HTTPS (default port 8443). The JSON
output carries column metadata and positional data rows; the connector parses
defensively across the documented shapes.

Note: ATSD is effectively end-of-life; this connector is unverified against a
live instance and based on the ATSD SQL API docs.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class AxibaseConfig(DBConfig):
    """ATSD SQL API configuration (default HTTPS port 8443)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8443,
        database: str = "atsd",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8443,
            database=database or "atsd",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        # ATSD is HTTPS by default.
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", True)))


class AxibaseConnector(DBConnector):
    """Connector for Axibase ATSD over its SQL API endpoint."""

    def __init__(self, config: AxibaseConfig):
        super().__init__(config)
        self.config: AxibaseConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "axibase"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.AXIBASE

    def _sql_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/api/sql"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        self._session.verify = False
        try:
            response = self._session.post(
                self._sql_url(),
                data={"q": "SELECT 1", "outputFormat": "json"},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to ATSD: {exc}")

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
            raise ConnectionError("Not connected to ATSD")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._sql_url(),
                data={"q": sql, "outputFormat": "json"},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        columns, raw_rows = _parse_atsd_payload(payload)
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
            result = await self.execute("SELECT metric, tags FROM atsd_series LIMIT 0")
            if result.columns:
                schema.tables.append(TableSchema(name="atsd_series", type="table", columns=[
                    ColumnSchema(name=col, data_type="unknown") for col in result.columns
                ]))
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


def _parse_atsd_payload(payload: Any) -> tuple[list[str], list[list]]:
    """Parse an ATSD SQL JSON payload into (columns, rows) defensively."""
    if not isinstance(payload, dict):
        return [], []
    # Column names may live under metadata.tableSchema.columns[].name or columnNames.
    columns: list[str] = []
    metadata = payload.get("metadata") or {}
    table_schema = metadata.get("tableSchema") if isinstance(metadata, dict) else None
    if isinstance(table_schema, dict):
        for col in table_schema.get("columns") or []:
            if isinstance(col, dict) and col.get("name"):
                columns.append(str(col["name"]))
    if not columns and isinstance(payload.get("columnNames"), list):
        columns = [str(c) for c in payload["columnNames"]]
    data = payload.get("data")
    rows = [row for row in data if isinstance(row, (list, tuple))] if isinstance(data, list) else []
    if not columns and rows:
        columns = [f"col{i}" for i in range(len(rows[0]))]
    return columns, rows
