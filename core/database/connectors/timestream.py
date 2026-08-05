"""Amazon Timestream connector implementation.

Amazon Timestream answers SQL via the AWS ``timestream-query`` API (``boto3``).
Credentials/region come from the config (or the standard AWS environment). The
boto3 client is imported lazily so the type registers without boto3 installed.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..connector import ColumnSchema, DBConfig, DBConnector, DatabaseSchema, DatabaseType, QueryResult, TableSchema


class TimestreamConfig(DBConfig):
    """Amazon Timestream configuration (AWS region + credentials)."""

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port,
            database=database or "",
            username=username or "",
            password=password or "",
            timeout=timeout,
            **kwargs,
        )
        self.region = str(kwargs.get("region") or kwargs.get("aws_region") or host or "us-east-1")


class TimestreamConnector(DBConnector):
    """Connector for Amazon Timestream over the boto3 timestream-query client."""

    def __init__(self, config: TimestreamConfig):
        super().__init__(config)
        self.config: TimestreamConfig = config
        self._client: Any = None

    @property
    def dialect(self) -> str:
        return "timestream"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.TIMESTREAM

    def _make_client(self) -> Any:
        import boto3
        kwargs: dict[str, Any] = {"region_name": self.config.region}
        if self.config.username and self.config.password:
            kwargs["aws_access_key_id"] = self.config.username
            kwargs["aws_secret_access_key"] = self.config.password
        return boto3.client("timestream-query", **kwargs)

    async def connect(self) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise ImportError("boto3 package required for Timestream. Install with: pip install boto3")
        try:
            self._client = self._make_client()
            # A trivial query validates credentials + reachability.
            self._client.query(QueryString="SELECT 1")
            self._connected = True
        except Exception as exc:
            self._connected = False
            self._client = None
            raise ConnectionError(f"Failed to connect to Timestream: {exc}")

    async def disconnect(self) -> None:
        self._client = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        if not self._connected or self._client is None:
            raise ConnectionError("Not connected to Timestream")
        sql = str(query or "").strip().rstrip(";")
        start_time = time.time()
        try:
            response = self._client.query(QueryString=sql)
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        columns = [str(col.get("Name")) for col in response.get("ColumnInfo", [])]
        rows: list[dict] = []
        for record in response.get("Rows", []):
            values = [_scalar(datum) for datum in record.get("Data", [])]
            rows.append({columns[idx]: self._format_value(val) for idx, val in enumerate(values) if idx < len(columns)})
        execution_time_ms = int((time.time() - start_time) * 1000)
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), execution_time_ms=execution_time_ms)

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(database=self.config.database)
        try:
            where = f"WHERE database_name = '{self.config.database}'" if self.config.database else ""
            result = await self.execute(f"SELECT table_name FROM information_schema.tables {where}")
            for row in result.rows:
                name = str(next(iter(row.values())) if row else "")
                if name:
                    schema.tables.append(TableSchema(name=name, type="table", columns=[]))
        except Exception as exc:
            schema.metadata["error"] = str(exc)
        return schema

    async def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            self._client.query(QueryString="SELECT 1")
            return True
        except Exception:
            return False

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value


def _scalar(datum: Any) -> Any:
    if not isinstance(datum, dict):
        return datum
    if "ScalarValue" in datum:
        return datum["ScalarValue"]
    if datum.get("NullValue"):
        return None
    return datum.get("ScalarValue")
