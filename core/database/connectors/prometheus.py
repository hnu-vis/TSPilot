"""Prometheus connector implementation."""
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from ..connector import (
    ColumnSchema,
    DatabaseSchema,
    DatabaseType,
    DBConfig,
    DBConnector,
    QueryResult,
    TableSchema,
)


class PrometheusConfig(DBConfig):
    """Prometheus specific configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        database: str = "",
        username: str = "",
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
        self.query_url = kwargs.get("query_url", "/api/v1")
        self.use_env_proxy = bool(kwargs.get("use_env_proxy", False))


class PrometheusConnector(DBConnector):
    """Connector for Prometheus.

    Uses Prometheus HTTP API for queries (not SQL).
    Supports PromQL for time-series queries.
    """

    _HIDDEN_SCHEMA_LABELS = {
        "dataset",
        "exported_dataset",
        "instance",
        "job",
        "series",
        "service",
        "source",
        "storage",
    }

    def __init__(self, config: PrometheusConfig):
        super().__init__(config)
        self.config: PrometheusConfig = config
        self._session: Any = None

    @property
    def dialect(self) -> str:
        return "prometheus"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.PROMETHEUS

    def _get_base_url(self) -> str:
        """Get base URL for API calls."""
        protocol = "https" if self.config.extra.get("ssl") else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}"

    def _get_api_url(self, endpoint: str) -> str:
        """Get full API URL."""
        base = self._get_base_url()
        return urljoin(base, f"{self.config.query_url}/{endpoint}")

    async def connect(self) -> None:
        """Establish connection to Prometheus."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package required. Install with: pip install requests"
            )

        self._session = requests.Session()
        self._session.trust_env = self.config.use_env_proxy
        if self.config.username and self.config.password:
            self._session.auth = (self.config.username, self.config.password)

        # Test connection
        try:
            response = self._session.get(
                self._get_api_url("query"),
                params={"query": "up"},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Prometheus: {e}")

    async def disconnect(self) -> None:
        """Close Prometheus connection."""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute PromQL query."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to Prometheus")

        start_time = time.time()

        try:
            query_params = {"query": query}
            if timeout:
                query_params["timeout"] = f"{timeout}ms"

            response = self._session.get(
                self._get_api_url("query"),
                params=query_params,
                timeout=timeout or self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                raise RuntimeError(f"Prometheus query failed: {data.get('error', 'Unknown error')}")

            columns, rows = self._parse_query_payload(data.get("data", {}))

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

    async def get_schema(self) -> DatabaseSchema:
        """Get Prometheus metrics schema."""
        if not self._connected:
            raise ConnectionError("Not connected to Prometheus")

        schema = DatabaseSchema(database="prometheus")

        try:
            # Get all metric names
            response = self._session.get(
                self._get_api_url("label/__name__/values"),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                metric_names = list(dict.fromkeys(data.get("data", [])))
                excluded_metric_names = set(self._get_hidden_schema_metric_names())
                if excluded_metric_names:
                    metric_names = [
                        metric_name
                        for metric_name in metric_names
                        if metric_name not in excluded_metric_names
                    ]
                configured_metric_names = self._get_configured_schema_metric_names()
                if configured_metric_names:
                    available_metric_names = set(metric_names)
                    metric_names = [
                        metric_name
                        for metric_name in configured_metric_names
                        if metric_name in available_metric_names
                    ]
                metric_limit = self.config.extra.get("schema_metric_limit")
                if isinstance(metric_limit, int) and metric_limit > 0:
                    metric_names = metric_names[:metric_limit]

                # Get sample metadata for each metric
                for metric_name in metric_names:
                    cursor = self._session.get(
                        self._get_api_url("series"),
                        params={"match[]": metric_name},
                        timeout=self.config.timeout,
                    )
                    cursor.raise_for_status()
                    series_data = cursor.json()

                    if series_data.get("status") == "success":
                        for series in series_data.get("data", [])[:1]:
                            # Get labels as columns
                            labels = series.get("__name__", "")
                            columns = [
                                ColumnSchema(name="timestamp", data_type="datetime"),
                                ColumnSchema(name="value", data_type="float"),
                            ]
                            # Add only analysis-relevant labels as preview columns.
                            for key, val in series.items():
                                if key != "__name__" and self._is_schema_label_visible(key):
                                    columns.append(ColumnSchema(
                                        name=f"label_{key}",
                                        data_type="string",
                                    ))

                            schema.tables.append(TableSchema(
                                name=labels,
                                type="metric",
                                columns=columns,
                            ))

        except Exception as e:
            schema.metadata["error"] = str(e)

        reference_dataset = self._build_reference_dataset_metadata()
        if reference_dataset:
            self._inject_reference_dataset_schema(schema, reference_dataset)

        return schema

    async def health_check(self) -> bool:
        """Check Prometheus connection health."""
        if not self._session:
            return False

        try:
            response = self._session.get(
                self._get_api_url("query"),
                params={"query": "up"},
                timeout=self.config.timeout,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def get_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s",
    ) -> QueryResult:
        """Execute range query in PromQL."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to Prometheus")

        start_time = time.time()

        try:
            response = self._session.get(
                self._get_api_url("query_range"),
                params={
                    "query": query,
                    "start": start.timestamp(),
                    "end": end.timestamp(),
                    "step": step,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                raise RuntimeError(f"Prometheus range query failed: {data.get('error', 'Unknown error')}")

            columns, rows = self._parse_query_payload(data.get("data", {}))

            execution_time_ms = int((time.time() - start_time) * 1000)

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            raise RuntimeError(f"Range query execution failed: {e}")

    def _parse_query_payload(self, payload: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        """Parse Prometheus query/query_range payload into normalized rows."""
        result_type = payload.get("resultType", "vector")
        result = payload.get("result")

        if result_type == "vector":
            return self._parse_vector_result(result or [])
        if result_type == "matrix":
            return self._parse_matrix_result(result or [])
        if result_type == "scalar":
            return self._parse_scalar_result(result)
        if result_type == "string":
            return self._parse_string_result(result)

        raise RuntimeError(f"Unsupported Prometheus resultType: {result_type}")

    def _parse_vector_result(
        self,
        result: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Parse instant vector results."""
        label_columns = self._collect_label_columns(result)
        columns = [*label_columns, "timestamp", "value"]
        rows: list[dict[str, Any]] = []

        for item in result:
            metric = item.get("metric", {})
            value_pair = item.get("value", [])
            if len(value_pair) != 2:
                continue
            rows.append(self._build_sample_row(metric, value_pair[0], value_pair[1], label_columns))

        return columns, rows

    def _parse_matrix_result(
        self,
        result: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Parse range vector results."""
        label_columns = self._collect_label_columns(result)
        columns = [*label_columns, "timestamp", "value"]
        rows: list[dict[str, Any]] = []

        for item in result:
            metric = item.get("metric", {})
            for timestamp, value in item.get("values", []):
                rows.append(self._build_sample_row(metric, timestamp, value, label_columns))

        return columns, rows

    def _parse_scalar_result(self, result: list[Any] | None) -> tuple[list[str], list[dict[str, Any]]]:
        """Parse scalar results."""
        if not result or len(result) != 2:
            return ["timestamp", "value"], []
        return ["timestamp", "value"], [
            {
                "timestamp": self._format_timestamp(result[0]),
                "value": self._parse_sample_value(result[1]),
            }
        ]

    def _parse_string_result(self, result: list[Any] | None) -> tuple[list[str], list[dict[str, Any]]]:
        """Parse string results."""
        if not result or len(result) != 2:
            return ["timestamp", "value"], []
        return ["timestamp", "value"], [
            {
                "timestamp": self._format_timestamp(result[0]),
                "value": str(result[1]),
            }
        ]

    def _collect_label_columns(self, result: list[dict[str, Any]]) -> list[str]:
        """Collect all label columns from a Prometheus result set."""
        label_names: set[str] = set()
        for item in result:
            metric = item.get("metric", {})
            label_names.update(key for key in metric.keys() if key != "__name__")
        return sorted(label_names)

    def _build_sample_row(
        self,
        metric: dict[str, Any],
        timestamp: Any,
        value: Any,
        label_columns: list[str],
    ) -> dict[str, Any]:
        """Build a normalized query row from a single Prometheus sample."""
        row: dict[str, Any] = {
            "metric_name": metric.get("__name__", ""),
            "timestamp": self._format_timestamp(timestamp),
            "value": self._parse_sample_value(value),
        }
        for label in label_columns:
            row[label] = metric.get(label)
        return row

    def _format_timestamp(self, timestamp: Any) -> str:
        """Format a Prometheus timestamp as ISO8601."""
        return datetime.fromtimestamp(float(timestamp)).isoformat()

    def _parse_sample_value(self, value: Any) -> float | None:
        """Convert Prometheus sample values into numeric values."""
        if value in (None, "NaN"):
            return None
        return float(value)

    def _build_reference_dataset_metadata(self) -> dict[str, Any] | None:
        """Build metadata for the configured reference dataset, if any."""
        reference_dataset = self.config.extra.get("reference_dataset")
        if not isinstance(reference_dataset, dict):
            return None

        raw_dataset_path = reference_dataset.get("dataset_path")
        resolved_dataset_path = reference_dataset.get("resolved_dataset_path")
        row_count = self._count_reference_dataset_rows(resolved_dataset_path or raw_dataset_path)

        static_labels = reference_dataset.get("static_labels")
        if not isinstance(static_labels, dict):
            static_labels = {}

        dataset_name = ""
        if raw_dataset_path:
            dataset_name = Path(str(raw_dataset_path)).name

        labels = {
            "dataset": dataset_name,
            "series": reference_dataset.get("series_name"),
            "source": reference_dataset.get("source"),
            **static_labels,
        }

        return {
            **reference_dataset,
            "dataset_path": raw_dataset_path,
            "resolved_dataset_path": resolved_dataset_path,
            "row_count": row_count,
            "sample_rows": self._reference_dataset_sample_rows(resolved_dataset_path or raw_dataset_path, limit=3),
            "labels": {
                key: str(value)
                for key, value in labels.items()
                if value not in (None, "")
            },
            "value_domains": self._reference_dataset_value_domains(reference_dataset),
        }

    def _count_reference_dataset_rows(self, dataset_path: str | None) -> int | None:
        """Count CSV rows for the configured reference dataset."""
        if not dataset_path:
            return None

        resolved_path = Path(str(dataset_path))
        if not resolved_path.is_absolute():
            resolved_path = (Path(__file__).resolve().parents[3] / resolved_path).resolve()
        if not resolved_path.exists():
            return None

        try:
            with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return sum(1 for _ in csv.DictReader(handle))
        except Exception:
            return None

    def _reference_dataset_sample_rows(self, dataset_path: str | None, *, limit: int) -> list[dict[str, Any]]:
        """Read a bounded set of CSV rows for schema grounding."""
        if not dataset_path or limit <= 0:
            return []

        resolved_path = Path(str(dataset_path))
        if not resolved_path.is_absolute():
            resolved_path = (Path(__file__).resolve().parents[3] / resolved_path).resolve()
        if not resolved_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        try:
            with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append(dict(row))
                    if len(rows) >= limit:
                        break
        except Exception:
            return []
        return rows

    def _get_configured_schema_metric_names(self) -> list[str]:
        """Return schema metric names configured for this logical Prometheus database."""
        configured = self.config.extra.get("schema_metric_names")
        if isinstance(configured, str):
            return [configured]
        if isinstance(configured, list):
            return [str(item) for item in configured if item not in (None, "")]
        return []

    def _get_hidden_schema_metric_names(self) -> list[str]:
        """Return metric names that should not be exposed to schema prompts."""
        hidden = self.config.extra.get("hidden_schema_metric_names")
        if isinstance(hidden, str):
            return [hidden]
        if isinstance(hidden, list | tuple | set):
            return [str(item) for item in hidden if item not in (None, "")]
        return []

    def _is_schema_label_visible(self, label_name: str) -> bool:
        """Return whether a Prometheus label should appear as a preview column."""
        hidden_labels = self.config.extra.get("hidden_schema_labels", self._HIDDEN_SCHEMA_LABELS)
        if isinstance(hidden_labels, str):
            hidden_label_names = {hidden_labels}
        elif isinstance(hidden_labels, list | tuple | set):
            hidden_label_names = {str(label) for label in hidden_labels}
        else:
            hidden_label_names = set(self._HIDDEN_SCHEMA_LABELS)
        return label_name not in hidden_label_names

    def _inject_reference_dataset_schema(
        self,
        schema: DatabaseSchema,
        reference_dataset: dict[str, Any],
    ) -> None:
        """Attach reference dataset metadata and a synthetic metric schema."""
        metric_name = reference_dataset.get("metric_name")
        if not metric_name:
            return

        label_columns = [
            ColumnSchema(name=f"label_{key}", data_type="string")
            for key in sorted(reference_dataset.get("labels", {}))
            if self._is_schema_label_visible(key)
        ]
        reference_columns = [
            ColumnSchema(name="timestamp", data_type="datetime"),
            ColumnSchema(name="value", data_type="float"),
            *label_columns,
        ]

        table_index = next(
            (
                index
                for index, table in enumerate(schema.tables)
                if getattr(table, "name", "") == metric_name
            ),
            None,
        )

        if table_index is None:
            schema.tables.insert(
                0,
                TableSchema(
                    name=metric_name,
                    type="metric",
                    columns=reference_columns,
                    row_count=reference_dataset.get("row_count"),
                ),
            )
        else:
            table = schema.tables.pop(table_index)
            existing_columns = {column.name for column in table.columns}
            for column in reference_columns:
                if column.name not in existing_columns:
                    table.columns.append(column)
            if table.row_count is None:
                table.row_count = reference_dataset.get("row_count")
            schema.tables.insert(0, table)

        schema.metadata["reference_dataset"] = reference_dataset
        value_domains = reference_dataset.get("value_domains")
        if isinstance(value_domains, dict):
            existing_domains = schema.metadata.setdefault("value_domains", {})
            if isinstance(existing_domains, dict):
                existing_domains[metric_name] = value_domains

    def _reference_dataset_value_domains(self, reference_dataset: dict[str, Any]) -> dict[str, list[str]]:
        """Infer canonical label value domains from the configured CSV reference dataset."""
        dataset_path = reference_dataset.get("resolved_dataset_path") or reference_dataset.get("dataset_path")
        if not dataset_path:
            return {}
        resolved_path = Path(str(dataset_path))
        if not resolved_path.is_absolute():
            resolved_path = (Path(__file__).resolve().parents[3] / resolved_path).resolve()
        if not resolved_path.exists():
            return {}

        value_column = str(reference_dataset.get("value_column") or "")
        timestamp_column = str(reference_dataset.get("timestamp_column") or "")
        domains: dict[str, list[str]] = {}
        try:
            with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key, raw_value in row.items():
                        if key in {timestamp_column, value_column} or raw_value in (None, ""):
                            continue
                        values = domains.setdefault(key, [])
                        value = str(raw_value)
                        if value not in values:
                            values.append(value)
        except Exception:
            return {}

        static_labels = reference_dataset.get("static_labels")
        if not isinstance(static_labels, dict):
            static_labels = reference_dataset.get("static_tags")
        if isinstance(static_labels, dict):
            for key, raw_value in static_labels.items():
                if raw_value in (None, ""):
                    continue
                values = domains.setdefault(str(key), [])
                value = str(raw_value)
                if value not in values:
                    values.append(value)
        return domains


class DevMockPrometheusConnector(PrometheusConnector):
    """Development-only Prometheus connector that reliably exercises ReAct repair."""

    def __init__(self, config: PrometheusConfig):
        super().__init__(config)
        self._query_attempts = 0
        self._range_attempts = 0

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def health_check(self) -> bool:
        return True

    async def get_schema(self) -> DatabaseSchema:
        schema = DatabaseSchema(
            database=self.config.extra.get("name") or "prometheus-mock",
            metadata={
                "dev_mock_react": True,
                "description": "Development mock Prometheus schema for exercising ReAct repair.",
            },
        )
        for metric_name in ("dev_react_metric_broken", "dev_react_metric_fixed"):
            schema.tables.append(
                TableSchema(
                    name=metric_name,
                    type="metric",
                    columns=[
                        ColumnSchema(name="timestamp", data_type="datetime"),
                        ColumnSchema(name="value", data_type="float"),
                        ColumnSchema(name="label_source", data_type="string"),
                    ],
                )
            )
        return schema

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        self._query_attempts += 1
        normalized_query = query.strip()
        if "dev_react_metric_fixed" not in normalized_query:
            raise RuntimeError(
                "Synthetic Prometheus ReAct test failure: metric "
                "'dev_react_metric_broken' is unavailable. Repair the PromQL to query "
                "'dev_react_metric_fixed' instead."
            )

        return QueryResult(
            columns=["timestamp", "value", "label_source"],
            rows=[
                {
                    "metric_name": "dev_react_metric_fixed",
                    "timestamp": "2026-05-14T08:00:00",
                    "value": 8.5,
                    "label_source": "dev_mock",
                }
            ],
            row_count=1,
            execution_time_ms=1,
        )

    async def get_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s",
    ) -> QueryResult:
        self._range_attempts += 1
        normalized_query = query.strip()
        if "dev_react_metric_fixed" not in normalized_query:
            raise RuntimeError(
                "Synthetic Prometheus ReAct range failure: metric "
                "'dev_react_metric_broken' is unavailable. Repair the PromQL to query "
                "'dev_react_metric_fixed' instead."
            )

        return QueryResult(
            columns=["timestamp", "value", "label_source"],
            rows=[
                {
                    "metric_name": "dev_react_metric_fixed",
                    "timestamp": "2026-05-14T08:00:00",
                    "value": 8.1,
                    "label_source": "dev_mock",
                },
                {
                    "metric_name": "dev_react_metric_fixed",
                    "timestamp": "2026-05-14T08:01:00",
                    "value": 8.5,
                    "label_source": "dev_mock",
                },
                {
                    "metric_name": "dev_react_metric_fixed",
                    "timestamp": "2026-05-14T08:02:00",
                    "value": 8.9,
                    "label_source": "dev_mock",
                },
                {
                    "metric_name": "dev_react_metric_fixed",
                    "timestamp": "2026-05-14T08:03:00",
                    "value": 9.2,
                    "label_source": "dev_mock",
                },
            ],
            row_count=4,
            execution_time_ms=1,
        )
