"""OpenMLDB connector implementation.

Talks to the OpenMLDB APIServer over its HTTP REST interface, so it needs no
native SDK or ZooKeeper client. Two endpoints are used:

* ``POST /dbs/{db}``          -- run a read-only SQL query. The response only
  carries column *types* in ``data.schema`` and positional rows in
  ``data.data`` (no column names), so names are recovered from the SELECT
  projection with a positional fallback.
* ``GET  /dbs/{db}/tables``   -- list tables with full ``column_desc`` (name +
  data_type + not_null), used to build the schema.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class OpenMLDBConfig(DBConfig):
    """OpenMLDB APIServer HTTP connection configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9080,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 9080,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))
        # "online" is the interactive execute mode of the APIServer.
        self.mode = str(kwargs.get("mode", "online") or "online")


class OpenMLDBConnector(DBConnector):
    """Connector for OpenMLDB using the APIServer REST endpoints."""

    def __init__(self, config: OpenMLDBConfig):
        super().__init__(config)
        self.config: OpenMLDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "openmldb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.OPENMLDB

    def _base_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}"

    async def connect(self) -> None:
        """Open an HTTP session and confirm the APIServer is reachable."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        try:
            response = self._session.get(
                f"{self._base_url()}/dbs",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to OpenMLDB APIServer: {exc}")

    async def disconnect(self) -> None:
        """Close the HTTP session."""
        self._close_session()
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute a read-only SQL query through ``POST /dbs/{db}``."""
        if not self._connected or self._session is None:
            raise ConnectionError("Not connected to OpenMLDB")
        if not self.config.database:
            raise ValueError("OpenMLDB queries require a configured database (db) name")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                f"{self._base_url()}/dbs/{self.config.database}",
                json={"mode": self.config.mode, "sql": sql},
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        if not isinstance(payload, dict) or payload.get("code", 0) != 0:
            message = payload.get("msg") if isinstance(payload, dict) else str(payload)
            raise RuntimeError(f"OpenMLDB query error: {message}")

        data = payload.get("data") or {}
        raw_rows = data.get("data") or []
        type_schema = data.get("schema") or []
        column_count = len(type_schema) or (len(raw_rows[0]) if raw_rows else 0)
        columns = self._columns_from_sql(sql, column_count)

        rows = [
            {columns[idx]: self._format_value(value) for idx, value in enumerate(row) if idx < len(columns)}
            for row in raw_rows
            if isinstance(row, (list, tuple))
        ]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            execution_time_ms=execution_time_ms,
            truncated=False,
        )

    async def get_schema(self) -> DatabaseSchema:
        """Build schema from ``GET /dbs/{db}/tables`` (carries column names)."""
        if not self._connected or self._session is None:
            raise ConnectionError("Not connected to OpenMLDB")

        schema = DatabaseSchema(database=self.config.database)
        if not self.config.database:
            schema.metadata["error"] = "no database configured"
            return schema
        try:
            response = self._session.get(
                f"{self._base_url()}/dbs/{self.config.database}/tables",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("code", 0) != 0:
                raise RuntimeError(payload.get("msg") or "failed to list tables")
            for table in (payload.get("tables") or []):
                if not isinstance(table, dict):
                    continue
                table_name = str(table.get("name") or "")
                if not table_name:
                    continue
                columns = [
                    ColumnSchema(
                        name=str(col.get("name") or ""),
                        data_type=self._normalize_type(col.get("data_type")),
                        nullable=not bool(col.get("not_null", False)),
                    )
                    for col in (table.get("column_desc") or [])
                    if isinstance(col, dict) and col.get("name")
                ]
                schema.tables.append(
                    TableSchema(name=table_name, type="table", columns=columns)
                )
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        """Prove the query path when a db is set, else just server reachability."""
        try:
            if self.config.database:
                result = await self.execute("SELECT 1")
                return result.row_count >= 0
            response = self._session.get(
                f"{self._base_url()}/dbs",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            return response.ok
        except Exception:
            try:
                response = self._session.get(
                    f"{self._base_url()}/dbs",
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

    def _columns_from_sql(self, sql: str, count: int) -> list[str]:
        """Recover column names from the SELECT projection; positional fallback.

        The APIServer response omits column names, so we parse them from the
        query. Anything ambiguous (``*``, function/arithmetic expressions,
        count mismatch) degrades to ``col0..colN`` rather than guessing wrong.
        """
        names = self._parse_projection(sql)
        if len(names) == count and all(names):
            return self._dedupe(names)
        return [f"col{index}" for index in range(count)]

    @staticmethod
    def _parse_projection(sql: str) -> list[str]:
        match = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
        if match:
            projection = match.group(1)
        else:
            tail = re.search(r"\bselect\b(.*)$", sql, re.IGNORECASE | re.DOTALL)
            projection = tail.group(1) if tail else ""
        projection = re.sub(r"^\s*distinct\s+", "", projection.strip(), flags=re.IGNORECASE)
        if not projection:
            return []

        names: list[str] = []
        for item in OpenMLDBConnector._split_top_level(projection):
            item = item.strip()
            if not item or item == "*" or item.endswith(".*"):
                return []
            alias = re.search(r"\s+as\s+([A-Za-z_]\w*|`[^`]+`|\"[^\"]+\")\s*$", item, re.IGNORECASE)
            if alias:
                names.append(alias.group(1).strip('`"'))
                continue
            if re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)?", item):
                names.append(item.split(".")[-1])
                continue
            return []
        return names

    @staticmethod
    def _split_top_level(text: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        current = []
        for char in text:
            if char in "([":
                depth += 1
            elif char in ")]":
                depth = max(0, depth - 1)
            if char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        return parts

    @staticmethod
    def _dedupe(names: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        out: list[str] = []
        for name in names:
            if name in seen:
                seen[name] += 1
                out.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                out.append(name)
        return out

    @staticmethod
    def _normalize_type(data_type: Any) -> str:
        """Map OpenMLDB internal type names (kTimestamp, kDouble) to plain SQL types."""
        raw = str(data_type or "unknown")
        if len(raw) >= 2 and raw[0] == "k" and raw[1].isupper():
            return raw[1:].lower()
        return raw.lower()

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
