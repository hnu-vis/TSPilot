"""TimescaleDB connector implementation."""
import time
from datetime import datetime
from typing import Any

from ..connector import DBConfig, DBConnector, QueryResult, DatabaseSchema, TableSchema, ColumnSchema, DatabaseType


class TimescaleDBConfig(DBConfig):
    """TimescaleDB specific configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        username: str = "postgres",
        password: str = "",
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


class TimescaleDBConnector(DBConnector):
    """Connector for TimescaleDB.

    TimescaleDB is a PostgreSQL extension for time-series data.
    Uses standard PostgreSQL wire protocol.
    """

    def __init__(self, config: TimescaleDBConfig):
        super().__init__(config)
        self.config: TimescaleDBConfig = config
        self._client: Any = None

    @property
    def dialect(self) -> str:
        return "timescaledb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.TIMESCALEDB

    async def connect(self) -> None:
        """Establish TimescaleDB connection."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 package required. Install with: pip install psycopg2-binary"
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
            raise ConnectionError(f"Failed to connect to TimescaleDB: {e}")

    async def disconnect(self) -> None:
        """Close TimescaleDB connection."""
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
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to TimescaleDB")

        start_time = time.time()

        try:
            cursor = self._client.cursor()

            if timeout:
                cursor.execute(f"SET statement_timeout = '{timeout}ms'")

            cursor.execute(query, params or None)

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch all rows
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
                truncated=False,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            raise RuntimeError(f"Query execution failed: {e}")

    async def get_schema(self) -> DatabaseSchema:
        """Get TimescaleDB schema."""
        if not self._connected:
            raise ConnectionError("Not connected to TimescaleDB")

        schema = DatabaseSchema(database=self.config.database)

        try:
            cursor = self._client.cursor()

            # Get hypertables
            cursor.execute("""
                SELECT hypertable_name, num_rows
                FROM timescaledb_information.hypertables
            """)
            hypertables = cursor.fetchall()

            for ht_name, row_count in hypertables:
                schema.tables.append(TableSchema(
                    name=ht_name,
                    type="hypertable",
                    row_count=row_count or 0,
                ))

            # Get regular tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                AND NOT table_name IN (
                    SELECT hypertable_name FROM timescaledb_information.hypertables
                )
            """)
            tables = cursor.fetchall()

            for (table_name,) in tables:
                # Get columns
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                columns_info = cursor.fetchall()

                columns = [
                    ColumnSchema(
                        name=col_name,
                        data_type=data_type,
                        nullable=(is_nullable == "YES"),
                        default_value=default_val,
                    )
                    for col_name, data_type, is_nullable, default_val in columns_info
                ]

                schema.tables.append(TableSchema(
                    name=table_name,
                    type="table",
                    columns=columns,
                ))

            cursor.close()

        except Exception as e:
            schema.metadata["error"] = str(e)

        return schema

    async def health_check(self) -> bool:
        """Check TimescaleDB connection health."""
        if not self._client:
            return False

        try:
            cursor = self._client.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
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