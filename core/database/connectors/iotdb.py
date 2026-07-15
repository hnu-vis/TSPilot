"""IoTDB connector implementation."""
import time
from datetime import datetime
from typing import Any

from ..connector import DBConfig, DBConnector, QueryResult, DatabaseSchema, TableSchema, ColumnSchema, DatabaseType


class IoTDBConfig(DBConfig):
    """IoTDB specific configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6667,
        database: str = "root",
        username: str = "root",
        password: str = "root",
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


class IoTDBConnector(DBConnector):
    """Connector for Apache IoTDB.

    Uses IoTDB Session API for time-series queries.
    Supports SQL-like query language.
    """

    def __init__(self, config: IoTDBConfig):
        super().__init__(config)
        self.config: IoTDBConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "iotdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.IOTDB

    async def connect(self) -> None:
        """Establish IoTDB connection."""
        try:
            from iotdb.Session import Session
        except ImportError:
            raise ImportError(
                "iotdb package required. Install with: pip install iotdb"
            )

        try:
            self._session = Session(
                node_urls=[f"{self.config.host}:{self.config.port}"],
                username=self.config.username,
                password=self.config.password,
            )
            self._session.open(False)
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to IoTDB: {e}")

    async def disconnect(self) -> None:
        """Close IoTDB connection."""
        if self._session:
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
        """Execute IoTDB SQL query."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to IoTDB")

        start_time = time.time()

        try:
            # Handle SHOW queries
            if query.strip().upper().startswith("SHOW"):
                result = self._execute_show(query)
            else:
                # Execute INSERT or other statements
                self._session.execute_non_query_statement(query)
                result = {"columns": ["status"], "rows": [{"status": "success"}]}

            columns = result.get("columns", [])
            rows = result.get("rows", [])

            execution_time_ms = int((time.time() - start_time) * 1000)

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            raise RuntimeError(f"Query execution failed: {e}")

    def _execute_show(self, query: str) -> dict:
        """Execute SHOW query and format results."""
        query_upper = query.strip().upper()

        if "MEASUREMENTS" in query_upper or "TIMESERIES" in query_upper:
            # Show timeseries
            raw_result = self._session.execute_query_statement(query)
            return self._format_dataset(raw_result)

        elif "DEVICES" in query_upper:
            # Show devices
            raw_result = self._session.execute_query_statement(query)
            return self._format_dataset(raw_result)

        elif "STORAGE GROUP" in query_upper:
            # Show storage groups
            raw_result = self._session.execute_query_statement(query)
            return self._format_dataset(raw_result)

        elif "NODES" in query_upper:
            # Show cluster nodes
            raw_result = self._session.execute_query_statement(query)
            return self._format_dataset(raw_result)

        else:
            raw_result = self._session.execute_query_statement(query)
            return self._format_dataset(raw_result)

    def _format_dataset(self, raw_result: Any) -> dict:
        """Format IoTDB dataset to standard format."""
        columns = []
        rows = []

        if raw_result:
            column_names = raw_result.get_column_names()
            if column_names:
                columns = [str(c) for c in column_names]

            raw_rows = raw_result.get_values()
            if raw_rows:
                for row_values in raw_rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        val = row_values[i]
                        if isinstance(val, datetime):
                            row_dict[col] = val.isoformat()
                        elif hasattr(val, 'isoformat'):
                            row_dict[col] = str(val)
                        else:
                            row_dict[col] = val
                    rows.append(row_dict)

        return {"columns": columns, "rows": rows}

    async def get_schema(self) -> DatabaseSchema:
        """Get IoTDB schema."""
        if not self._connected:
            raise ConnectionError("Not connected to IoTDB")

        schema = DatabaseSchema(database=self.config.database)

        try:
            # Get storage groups
            sg_result = self._session.execute_query_statement("SHOW STORAGE GROUP")
            sg_data = self._format_dataset(sg_result)

            for row in sg_data.get("rows", []):
                storage_group = row.get("storage group", row.get("StorageGroup", ""))
                if storage_group:
                    schema.tables.append(TableSchema(
                        name=str(storage_group),
                        type="storage_group",
                    ))

            # Get timeseries
            ts_result = self._session.execute_query_statement("SHOW TIMESERIES")
            ts_data = self._format_dataset(ts_result)

            for row in ts_data.get("rows", []):
                timeseries = row.get("timeseries", row.get("Timeseries", ""))
                if timeseries:
                    columns = [
                        ColumnSchema(name="timestamp", data_type="int64"),
                        ColumnSchema(name="value", data_type="float"),
                    ]
                    # Add tags if present
                    for key in ["tag", "attribute", "description"]:
                        if key in row:
                            columns.append(ColumnSchema(
                                name=key,
                                data_type="string",
                            ))

                    schema.tables.append(TableSchema(
                        name=str(timeseries),
                        type="timeseries",
                        columns=columns,
                    ))

        except Exception as e:
            schema.metadata["error"] = str(e)

        return schema

    async def health_check(self) -> bool:
        """Check IoTDB connection health."""
        if not self._session:
            return False

        try:
            self._session.execute_query_statement("SELECT 1 FROM root")
            return True
        except Exception:
            return False

    async def insert_record(
        self,
        device_path: str,
        measurements: list[str],
        values: list[Any],
        timestamp: int | None = None,
    ) -> bool:
        """Insert a record into IoTDB.

        Args:
            device_path: Full path to the device (e.g., root.db.device1)
            measurements: List of measurement names
            values: List of values to insert
            timestamp: Optional timestamp (uses current time if None)

        Returns:
            True if successful
        """
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to IoTDB")

        try:
            self._session.insert_record(
                device_path=device_path,
                measurements=measurements,
                values=values,
                timestamp=timestamp or int(time.time() * 1000),
            )
            return True
        except Exception:
            return False

    async def insert_records(
        self,
        device_paths: list[str],
        measurements: list[str],
        values_list: list[list[Any]],
        timestamps: list[int] | None = None,
    ) -> bool:
        """Insert multiple records into IoTDB."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to IoTDB")

        try:
            if timestamps is None:
                timestamps = [int(time.time() * 1000)] * len(device_paths)

            self._session.insert_records(
                device_paths=device_paths,
                measurements=measurements,
                values=values_list,
                timestamps=timestamps,
            )
            return True
        except Exception:
            return False