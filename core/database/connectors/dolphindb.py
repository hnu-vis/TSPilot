"""DolphinDB connector implementation.

DolphinDB is queried with SQL through the official ``dolphindb`` Python client
(binary protocol). Table results come back as pandas DataFrames, which are
normalized to rows here. The client is imported lazily so the type registers
without ``dolphindb`` installed.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class DolphinDBConfig(DBConfig):
    """DolphinDB connection configuration (default port 8848)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8848,
        database: str = "",
        username: str = "admin",
        password: str = "123456",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8848,
            database=database or "",
            username=username or "admin",
            password=password or "123456",
            timeout=timeout,
            **kwargs,
        )


class DolphinDBConnector(DBConnector):
    """Connector for DolphinDB over the dolphindb Python client."""

    def __init__(self, config: DolphinDBConfig):
        super().__init__(config)
        self.config: DolphinDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "dolphindb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.DOLPHINDB

    async def connect(self) -> None:
        try:
            import dolphindb as ddb
        except ImportError:
            raise ImportError("dolphindb package required. Install with: pip install dolphindb")
        try:
            self._session = ddb.session()
            self._session.connect(self.config.host, int(self.config.port), self.config.username, self.config.password)
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._session = None
            raise ConnectionError(f"Failed to connect to DolphinDB: {exc}")

    async def disconnect(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        if not self._connected or self._session is None:
            raise ConnectionError("Not connected to DolphinDB")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            result = self._session.run(sql)
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        columns, rows = _normalize_result(result)
        rows = [{col: self._format_value(val) for col, val in row.items()} for row in rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database or "dolphindb")
        return schema

    async def health_check(self) -> bool:
        try:
            return bool(self._session and self._session.run("1"))
        except Exception:
            return False

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value


def _normalize_result(result: Any) -> tuple[list[str], list[dict]]:
    """Normalize a DolphinDB run() result (pandas DataFrame or scalar) to rows."""
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        columns = [str(c) for c in result.columns]
        records = result.to_dict("records")
        return columns, [{str(k): v for k, v in rec.items()} for rec in records]
    if isinstance(result, list) and result and isinstance(result[0], dict):
        columns: list[str] = []
        for row in result:
            for key in row.keys():
                if key not in columns:
                    columns.append(str(key))
        return columns, [dict(row) for row in result]
    # Scalar or other: wrap as a single value.
    return ["value"], [{"value": result}]
