"""kdb+ connector implementation.

kdb+ is queried with q / qSQL through the official ``pykx`` client (IPC). Table
results are normalized to rows via pandas. The client is imported lazily so the
type registers without ``pykx`` installed.

kdb+'s native language is q (functional), not standard SQL. qSQL SELECT
statements are accepted; a dedicated q dialect can be added in ``dialects.py``
if tailored query generation is required.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class KdbConfig(DBConfig):
    """kdb+ connection configuration (default port 5000)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 5000,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )


class KdbConnector(DBConnector):
    """Connector for kdb+ over the pykx IPC client."""

    def __init__(self, config: KdbConfig):
        super().__init__(config)
        self.config: KdbConfig = config
        self._conn: Any = None

    @property
    def dialect(self) -> str:
        return "kdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.KDB

    def _open(self) -> Any:
        import pykx as kx
        kwargs: dict[str, Any] = {"host": self.config.host, "port": int(self.config.port)}
        if self.config.username:
            kwargs["username"] = self.config.username
        if self.config.password:
            kwargs["password"] = self.config.password
        return kx.SyncQConnection(**kwargs)

    async def connect(self) -> None:
        try:
            import pykx  # noqa: F401
        except ImportError:
            raise ImportError("pykx package required for kdb+. Install with: pip install pykx")
        try:
            self._conn = self._open()
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._conn = None
            raise ConnectionError(f"Failed to connect to kdb+: {exc}")

    async def disconnect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        if not self._connected or self._conn is None:
            raise ConnectionError("Not connected to kdb+")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            result = self._conn.sql(sql) if hasattr(self._conn, "sql") else self._conn(sql)
            frame = result.pd() if hasattr(result, "pd") else result
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        columns, rows = _normalize_frame(frame)
        rows = [{col: self._format_value(val) for col, val in row.items()} for row in rows]
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database or "kdb")
        try:
            result = self._conn("tables[]") if self._conn else None
            names = result.py() if hasattr(result, "py") else (list(result) if result is not None else [])
            for name in names or []:
                schema.tables.append(TableSchema(name=str(name), type="table", columns=[]))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            if self._conn is None:
                return False
            self._conn("1+1")
            return True
        except Exception:
            return False

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value


def _normalize_frame(frame: Any) -> tuple[list[str], list[dict]]:
    if hasattr(frame, "to_dict") and hasattr(frame, "columns"):
        columns = [str(c) for c in frame.columns]
        return columns, [{str(k): v for k, v in rec.items()} for rec in frame.to_dict("records")]
    if isinstance(frame, list) and frame and isinstance(frame[0], dict):
        columns: list[str] = []
        for row in frame:
            for key in row.keys():
                if key not in columns:
                    columns.append(str(key))
        return columns, [dict(row) for row in frame]
    return ["value"], [{"value": frame}]
