"""QuestDB connector implementation."""
import time
from datetime import datetime
from typing import Any

from ..connector import DBConfig, DBConnector, QueryResult, DatabaseSchema, TableSchema, ColumnSchema, DatabaseType


class QuestDBConfig(DBConfig):
    """QuestDB specific configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8812,
        database: str = "qdb",
        username: str = "admin",
        password: str = "quest",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            timeout=timeout,
            **kwargs,
        )
        self.use_http = bool(kwargs.get("use_http", False))


class QuestDBConnector(DBConnector):
    """Connector for QuestDB.

    QuestDB is a high-performance time-series database.
    Supports both PostgreSQL wire protocol and HTTP REST API.
    """

    def __init__(self, config: QuestDBConfig):
        super().__init__(config)
        self.config: QuestDBConfig = config
        self._client: Any = None

    @property
    def dialect(self) -> str:
        return "questdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.QUESTDB

    def _get_http_url(self) -> str:
        """Get HTTP URL for REST API."""
        return f"http://{self.config.host}:{self.config.port}"

    async def connect(self) -> None:
        """Establish QuestDB connection."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package required. Install with: pip install requests"
            )

        if self.config.use_http:
            # Test HTTP connection
            try:
                response = requests.get(
                    f"{self._get_http_url()}/exec",
                    params={"query": "SELECT 1"},
                    auth=(self.config.username, self.config.password),
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                self._connected = True
            except Exception as e:
                self._connected = False
                raise ConnectionError(f"Failed to connect to QuestDB: {e}")
        else:
            # PostgreSQL wire protocol
            try:
                import psycopg2
            except ImportError:
                raise ImportError(
                    "psycopg2 required for PostgreSQL mode. Install with: pip install psycopg2-binary"
                )

            try:
                self._client = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    dbname=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.timeout,
                )
                self._client.autocommit = True
                self._connected = True
            except Exception as e:
                self._connected = False
                raise ConnectionError(f"Failed to connect to QuestDB: {e}")

    async def disconnect(self) -> None:
        """Close QuestDB connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute SQL query."""
        if not self._connected:
            raise ConnectionError("Not connected to QuestDB")

        start_time = time.time()

        try:
            if self.config.use_http:
                return await self._execute_http(query, start_time, timeout)
            else:
                return await self._execute_pg(query, start_time, timeout)

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            raise RuntimeError(f"Query execution failed: {e}")

    async def _execute_http(
        self,
        query: str,
        start_time: float,
        timeout: int | None,
    ) -> QueryResult:
        """Execute query via HTTP REST API."""
        import requests

        params = {"query": query}
        if timeout:
            params["timeout"] = str(timeout)

        response = requests.get(
            f"{self._get_http_url()}/exec",
            params=params,
            auth=(self.config.username, self.config.password),
            timeout=timeout or self.config.timeout,
        )
        response.raise_for_status()

        # QuestDB returns JSON with columns and rows
        data = response.json()

        columns = data.get("columns", [])
        dataset = data.get("dataset", [])

        rows = []
        for row_data in dataset:
            row_dict = {}
            for i, col in enumerate(columns):
                col_name = col.get("name", f"col_{i}")
                val = row_data[i] if i < len(row_data) else None
                row_dict[col_name] = self._format_value(val)
            rows.append(row_dict)

        column_names = [col.get("name", f"col_{i}") for i, col in enumerate(columns)]

        execution_time_ms = int((time.time() - start_time) * 1000)

        return QueryResult(
            columns=column_names,
            rows=rows,
            row_count=len(rows),
            execution_time_ms=execution_time_ms,
        )

    async def _execute_pg(
        self,
        query: str,
        start_time: float,
        timeout: int | None,
    ) -> QueryResult:
        """Execute query via PostgreSQL wire protocol."""
        if not self._client:
            raise ConnectionError("PostgreSQL client not initialized")

        cursor = self._client.cursor()

        try:
            if timeout:
                cursor.execute(f"SET statement_timeout = '{timeout}s'")

            cursor.execute(query)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = []

            for row in cursor.fetchall():
                rows.append(self._row_to_dict(columns, row))

            cursor.close()

            execution_time_ms = int((time.time() - start_time) * 1000)

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time_ms,
            )

        except Exception:
            cursor.close()
            raise

    async def get_schema(self) -> DatabaseSchema:
        """Get QuestDB schema."""
        if not self._connected:
            raise ConnectionError("Not connected to QuestDB")

        schema = DatabaseSchema(database=self.config.database)

        try:
            # Get all tables
            result = await self.execute("SHOW TABLES")

            for row in result.rows:
                table_name = row.get("table")
                if not table_name:
                    continue

                columns = await self._get_table_columns(table_name)

                # Get row count
                count_result = await self.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = self._first_count_value(count_result.rows[0]) if count_result.rows else 0

                schema.tables.append(TableSchema(
                    name=table_name,
                    type="table",
                    columns=columns,
                    row_count=row_count,
                ))

        except Exception as e:
            schema.metadata["error"] = str(e)

        return schema

    async def _get_table_columns(self, table_name: str) -> list[ColumnSchema]:
        """Return QuestDB columns using catalog metadata with a LIMIT 0 fallback."""
        try:
            result = await self.execute(f"SELECT * FROM table_columns('{table_name}')")
            columns = []
            for row in result.rows:
                name = row.get("column") or row.get("name") or row.get("column_name")
                if not name:
                    continue
                columns.append(ColumnSchema(
                    name=str(name),
                    data_type=str(row.get("type") or row.get("data_type") or "unknown"),
                ))
            if columns:
                return columns
        except Exception:
            pass

        col_result = await self.execute(f"SELECT * FROM {table_name} LIMIT 0")
        return [
            ColumnSchema(name=col_name, data_type="unknown")
            for col_name in col_result.columns
        ]

    def _first_count_value(self, row: dict[str, Any]) -> int:
        """Return COUNT(*) from whichever column name the dialect used."""
        for value in row.values():
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    async def health_check(self) -> bool:
        """Check QuestDB connection health."""
        try:
            result = await self.execute("SELECT 1")
            return result.row_count >= 0
        except Exception:
            return False

    def _row_to_dict(self, columns: list[str], row: tuple) -> dict:
        """Convert row tuple to dictionary."""
        return {col: self._format_value(val) for col, val in zip(columns, row)}

    def _format_value(self, value: Any) -> Any:
        """Format value for display."""
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        return value
