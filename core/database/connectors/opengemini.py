"""openGemini connector implementation.

openGemini is InfluxDB-1.x compatible and answers InfluxQL over the HTTP
``/query`` endpoint (``GET/POST /query?db=&q=``). The response is the Influx v1
shape: ``{"results":[{"series":[{"name","columns":[...],"values":[[...]]}]}]}``.

InfluxQL is SELECT-based; the type is routed to the SQL-family dialect for query
generation. If tailored InfluxQL prompt rules are needed, add a dedicated
dialect in ``dialects.py``.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class OpenGeminiConfig(DBConfig):
    """openGemini InfluxQL HTTP configuration (default port 8086)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8086,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8086,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class OpenGeminiConnector(DBConnector):
    """Connector for openGemini over its InfluxQL HTTP query endpoint."""

    def __init__(self, config: OpenGeminiConfig):
        super().__init__(config)
        self.config: OpenGeminiConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "opengemini"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.OPENGEMINI

    def _query_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/query"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.post(
                self._query_url(),
                params={"q": "SHOW DATABASES"},
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to openGemini: {exc}")

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
            raise ConnectionError("Not connected to openGemini")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._query_url(),
                params={"db": self.config.database, "q": sql},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        columns, rows = _parse_influx_results(payload)
        rows = [{col: self._format_value(val) for col, val in row.items()} for row in rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            result = await self.execute("SHOW MEASUREMENTS")
            for row in result.rows:
                name = str(row.get("name") or (next(iter(row.values())) if row else ""))
                if name:
                    schema.tables.append(TableSchema(name=name, type="measurement", columns=[]))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            response = self._session.post(
                self._query_url(),
                params={"q": "SHOW DATABASES"},
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


def _parse_influx_results(payload: Any) -> tuple[list[str], list[dict]]:
    """Parse the InfluxDB v1 results/series shape into (columns, row dicts)."""
    columns: list[str] = []
    rows: list[dict] = []
    if not isinstance(payload, dict):
        return columns, rows
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        for series in result.get("series") or []:
            if not isinstance(series, dict):
                continue
            series_columns = [str(c) for c in (series.get("columns") or [])]
            for column in series_columns:
                if column not in columns:
                    columns.append(column)
            measurement = series.get("name")
            for values in series.get("values") or []:
                row = {col: values[idx] for idx, col in enumerate(series_columns) if idx < len(values)}
                if measurement and "name" not in row:
                    row.setdefault("_measurement", measurement)
                rows.append(row)
    if any("_measurement" in row for row in rows) and "_measurement" not in columns:
        columns.append("_measurement")
    return columns, rows
