"""IRONdb (Circonus) connector implementation.

IRONdb's native analytics language is CAQL, queried over HTTP
(``GET /extension/lua/caql_v1``) with a time window. CAQL is not SQL, so a
dedicated CAQL dialect should be added in ``dialects.py`` for tailored query
generation; for now the connector forwards the query string as CAQL and parses
the response defensively.

Note: lowest-confidence connector — unverified against a live IRONdb and CAQL's
response shape varies; treat as a starting point.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class IRONdbConfig(DBConfig):
    """IRONdb HTTP configuration (default port 8112)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8112,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8112,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        self.lookback_seconds = int(kwargs.get("lookback_seconds", 3600))
        self.period_seconds = int(kwargs.get("period_seconds", 60))


class IRONdbConnector(DBConnector):
    """Connector for IRONdb over its CAQL HTTP endpoint."""

    def __init__(self, config: IRONdbConfig):
        super().__init__(config)
        self.config: IRONdbConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "irondb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.IRONDB

    def _caql_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/extension/lua/caql_v1"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        self._session = requests.Session()
        try:
            response = self._session.get(
                f"{self._caql_url().rsplit('/extension', 1)[0]}/",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            self._connected = response.status_code < 500
            if not self._connected:
                raise ConnectionError(f"HTTP {response.status_code}")
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to IRONdb: {exc}")

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
            raise ConnectionError("Not connected to IRONdb")
        caql = str(query or "").strip()
        end = int(datetime.now(timezone.utc).timestamp())
        start = end - self.config.lookback_seconds
        start_time = time.time()
        try:
            response = self._session.get(
                self._caql_url(),
                params={"query": caql, "start": start, "end": end, "period": self.config.period_seconds},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        columns, rows = _parse_caql(payload)
        rows = [{col: self._format_value(val) for col, val in row.items()} for row in rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        return DatabaseSchema(database=self.config.database or "irondb")

    async def health_check(self) -> bool:
        try:
            response = self._session.get(
                f"{self._caql_url().rsplit('/extension', 1)[0]}/",
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


def _parse_caql(payload: Any) -> tuple[list[str], list[dict]]:
    """Best-effort parse of an IRONdb CAQL response into timestamp/value rows."""
    rows: list[dict] = []
    # CAQL DF4 shape: {"head":{"count":..},"meta":[...],"data":[[series...]]} or
    # a list of [timestamp, {label: value}] pairs. Handle both defensively.
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                timestamp, value = entry[0], entry[1]
                if isinstance(value, dict):
                    for label, val in value.items():
                        rows.append({"timestamp": timestamp, "label": label, "value": val})
                else:
                    rows.append({"timestamp": timestamp, "value": value})
    columns = ["timestamp", "value"] + (["label"] if any("label" in r for r in rows) else [])
    return columns, rows
