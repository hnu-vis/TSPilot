"""CnosDB connector implementation.

CnosDB answers SQL over an HTTP endpoint (``POST /api/v1/sql``) with the raw SQL
in the request body and HTTP Basic auth. With ``Accept: application/json`` the
result comes back as an array of row objects (column name -> value).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class CnosDBConfig(DBConfig):
    """CnosDB HTTP SQL configuration (default port 8902)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8902,
        database: str = "public",
        username: str = "root",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8902,
            database=database or "public",
            username=username or "root",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.ssl = bool(kwargs.get("ssl", kwargs.get("ssl_enabled", False)))


class CnosDBConnector(DBConnector):
    """Connector for CnosDB over its HTTP SQL endpoint."""

    def __init__(self, config: CnosDBConfig):
        super().__init__(config)
        self.config: CnosDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "cnosdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.CNOSDB

    def _sql_url(self) -> str:
        protocol = "https" if self.config.ssl else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}/api/v1/sql"

    async def connect(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        try:
            response = self._session.post(
                self._sql_url(),
                params={"db": self.config.database},
                data=b"SELECT 1",
                auth=self._auth(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._close_session()
            raise ConnectionError(f"Failed to connect to CnosDB: {exc}")

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
            raise ConnectionError("Not connected to CnosDB")

        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._session.post(
                self._sql_url(),
                params={"db": self.config.database},
                data=sql.encode("utf-8"),
                auth=self._auth(),
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")

        record_rows = _extract_row_dicts(payload)
        columns = _ordered_columns(record_rows)
        rows = [{col: self._format_value(row.get(col)) for col in columns} for row in record_rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            tables_result = await self.execute("SHOW TABLES")
            table_names = [
                str(next(iter(row.values())))
                for row in tables_result.rows
                if row
            ]
            for table_name in table_names:
                try:
                    desc = await self.execute(f"DESCRIBE TABLE {table_name}")
                    columns = [
                        ColumnSchema(
                            name=str(row.get("COLUMN_NAME") or row.get("column_name") or next(iter(row.values()))),
                            data_type=str(row.get("DATA_TYPE") or row.get("data_type") or "unknown").lower(),
                        )
                        for row in desc.rows
                        if row
                    ]
                except Exception:
                    columns = []
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


def _extract_row_dicts(payload: Any) -> list[dict]:
    """Normalize a CnosDB JSON payload into a list of row dicts."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "records", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _ordered_columns(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    return columns
