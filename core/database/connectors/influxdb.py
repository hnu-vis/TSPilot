"""InfluxDB connector implementation."""
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from ..connector import DBConfig, DBConnector, QueryResult, DatabaseSchema, TableSchema, ColumnSchema, DatabaseType


class InfluxDBConfig(DBConfig):
    """InfluxDB specific configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8086,
        database: str = "",
        username: str = "root",
        password: str = "root",
        ssl: bool = False,
        timeout: int = 30,
        version: int | str = 1,
        org: str = "",
        bucket: str = "",
        token: str = "",
        url: str = "",
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
        self.ssl = ssl
        self.version = str(version or kwargs.get("influxdb_version") or "1")
        self.org = org or kwargs.get("organization", "")
        self.bucket = bucket or kwargs.get("default_bucket", "") or database
        self.token = token or kwargs.get("api_token", "")
        self.url = url
        self.influxdb_tasks = kwargs.get("influxdb_tasks", [])


class InfluxDBConnector(DBConnector):
    """Connector for InfluxDB 1.x and 2.x.

    Uses the InfluxDB 2.x Python client and Flux for version=2 configs, while
    keeping the legacy InfluxQL client for version=1 configs.
    """

    def __init__(self, config: InfluxDBConfig):
        super().__init__(config)
        self.config: InfluxDBConfig = config
        self._client: Any = None
        self._query_api: Any = None

    @property
    def dialect(self) -> str:
        return "flux" if self._is_v2 else "influxdb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.INFLUXDB

    @property
    def _is_v2(self) -> bool:
        return self.config.version.startswith("2")

    async def connect(self) -> None:
        """Establish InfluxDB connection."""
        if self._is_v2:
            await self._connect_v2()
            return

        await self._connect_v1()

    async def _connect_v2(self) -> None:
        """Establish InfluxDB 2.x connection."""
        try:
            from influxdb_client import InfluxDBClient
        except ImportError:
            raise ImportError(
                "influxdb-client package required for InfluxDB 2.x. "
                "Install with: pip install influxdb-client"
            )

        if not self.config.org:
            raise ValueError("InfluxDB 2.x config requires org")
        if not self.config.bucket:
            raise ValueError("InfluxDB 2.x config requires bucket")
        if not self.config.token:
            raise ValueError("InfluxDB 2.x config requires token")

        self._client = InfluxDBClient(
            url=self._build_v2_url(),
            token=self.config.token,
            org=self.config.org,
            timeout=self.config.timeout * 1000,
        )
        self._query_api = self._client.query_api()

        try:
            if not self._client.ping():
                raise ConnectionError("InfluxDB ping returned false")
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to InfluxDB 2.x: {e}")

    async def _connect_v1(self) -> None:
        """Establish InfluxDB 1.x connection."""
        try:
            from influxdb import InfluxDBClient
        except ImportError:
            raise ImportError(
                "influxdb package required. Install with: pip install influxdb"
            )

        self._client = InfluxDBClient(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            database=self.config.database,
            ssl=self.config.ssl,
            timeout=self.config.timeout,
        )

        # Test connection
        try:
            self._client.ping()
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to InfluxDB: {e}")

    async def disconnect(self) -> None:
        """Close InfluxDB connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._query_api = None
        self._connected = False

    async def execute(
        self,
        query: str,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute an InfluxDB query."""
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to InfluxDB")

        if self._is_v2:
            return await self._execute_flux(query=query, timeout=timeout)
        return await self._execute_influxql(query=query)

    async def _execute_flux(self, query: str, timeout: int | None = None) -> QueryResult:
        """Execute Flux query against InfluxDB 2.x."""
        if not self._query_api:
            raise ConnectionError("InfluxDB 2.x query API is not initialized")

        query = self._normalize_flux_bucket(query)
        start_time = time.time()

        try:
            tables = self._query_api.query(
                query=query,
                org=self.config.org,
                params={"bucket": self.config.bucket},
            )
            rows: list[dict[str, Any]] = []
            columns: list[str] = []
            seen_columns: set[str] = set()

            for table in tables or []:
                for record in getattr(table, "records", []) or []:
                    row = self._flux_record_to_row(record)
                    for column in row:
                        if column not in seen_columns:
                            seen_columns.add(column)
                            columns.append(column)
                    rows.append(row)

            execution_time_ms = int((time.time() - start_time) * 1000)
            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time_ms,
            )
        except Exception as e:
            raise RuntimeError(f"Flux query execution failed: {e}")

    async def _execute_influxql(self, query: str) -> QueryResult:
        """Execute InfluxQL query against InfluxDB 1.x."""
        start_time = time.time()

        try:
            result = self._client.query(query)

            # Convert result to QueryResult
            columns = []
            rows = []

            if result:
                for series in result:
                    if series.get("columns"):
                        columns = series["columns"]
                    if series.get("values"):
                        for values in series["values"]:
                            row = {
                                col: val
                                for col, val in zip(columns, values)
                            }
                            rows.append(row)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}")

    async def get_schema(self) -> DatabaseSchema:
        """Get InfluxDB schema."""
        if not self._connected:
            raise ConnectionError("Not connected to InfluxDB")

        if self._is_v2:
            return await self._get_schema_v2()
        return await self._get_schema_v1()

    async def probe_value_domains(
        self,
        *,
        source_name: str,
        columns: list[str],
        limit: int = 100,
    ) -> dict[str, list[str]]:
        """Probe candidate tag values for one measurement."""
        if not self._connected:
            raise ConnectionError("Not connected to InfluxDB")
        if not source_name or not columns:
            return {}
        if self._is_v2:
            return await self._probe_value_domains_v2(source_name=source_name, columns=columns, limit=limit)
        return await self._probe_value_domains_v1(source_name=source_name, columns=columns, limit=limit)

    async def _get_schema_v2(self) -> DatabaseSchema:
        """Get InfluxDB 2.x bucket schema through Flux schema helpers."""
        schema_start = "1970-01-01T00:00:00Z"
        schema = DatabaseSchema(
            database=self.config.bucket,
            metadata={
                "version": 2,
                "org": self.config.org,
                "bucket": self.config.bucket,
                "query_language": "flux",
            },
        )
        reference_dataset = self._build_reference_dataset_metadata()
        if reference_dataset:
            schema.metadata["reference_dataset"] = reference_dataset

        try:
            measurements = await self._query_schema_values(
                f'import "influxdata/influxdb/schema"\n'
                f'schema.measurements(bucket: "{self._escape_flux_string(self.config.bucket)}", '
                f'start: {schema_start})'
            )

            value_domains: dict[str, dict[str, list[str]]] = {}
            for measurement in measurements:
                fields = await self._query_schema_pairs(
                    f'import "influxdata/influxdb/schema"\n'
                    f'schema.fieldKeys(bucket: "{self._escape_flux_string(self.config.bucket)}", '
                    f'start: {schema_start}, '
                    f'predicate: (r) => r._measurement == "{self._escape_flux_string(measurement)}")'
                )
                tags = await self._query_schema_values(
                    f'import "influxdata/influxdb/schema"\n'
                    f'schema.tagKeys(bucket: "{self._escape_flux_string(self.config.bucket)}", '
                    f'start: {schema_start}, '
                    f'predicate: (r) => r._measurement == "{self._escape_flux_string(measurement)}")'
                )
                value_domains[measurement] = {
                    "_field": [name for name, _field_type in fields],
                }
                for tag in tags:
                    if tag in {"_measurement", "_field", "_start", "_stop", "result", "table"}:
                        continue
                    tag_values = await self._query_schema_values(
                        f'import "influxdata/influxdb/schema"\n'
                        f'schema.tagValues(bucket: "{self._escape_flux_string(self.config.bucket)}", '
                        f'start: {schema_start}, '
                        f'tag: "{self._escape_flux_string(tag)}", '
                        f'predicate: (r) => r._measurement == "{self._escape_flux_string(measurement)}")'
                    )
                    if tag_values:
                        value_domains[measurement][tag] = tag_values[:100]

                columns = [
                    ColumnSchema(name="_time", data_type="datetime"),
                    *[
                        ColumnSchema(
                            name=name,
                            data_type=self._influx_type_to_generic(field_type),
                        )
                        for name, field_type in fields
                    ],
                    *[
                        ColumnSchema(name=tag, data_type="string")
                        for tag in tags
                        if tag not in {"_measurement", "_field", "_start", "_stop", "result", "table"}
                    ],
                ]
                schema.tables.append(TableSchema(name=measurement, columns=columns))

            reference_domains = self._reference_dataset_value_domains(reference_dataset)
            for measurement, domains in reference_domains.items():
                merged_domains = value_domains.setdefault(measurement, {})
                for key, values in domains.items():
                    existing = merged_domains.setdefault(key, [])
                    for value in values:
                        if value not in existing:
                            existing.append(value)
            if value_domains:
                schema.metadata["value_domains"] = value_domains

        except Exception as e:
            schema.metadata["error"] = str(e)

        return schema

    async def _probe_value_domains_v2(
        self,
        *,
        source_name: str,
        columns: list[str],
        limit: int,
    ) -> dict[str, list[str]]:
        reference_domains = self._reference_dataset_value_domains(self._build_reference_dataset_metadata())
        if source_name in reference_domains:
            return {
                column: list(reference_domains[source_name].get(column, []))[:limit]
                for column in columns
                if reference_domains[source_name].get(column)
            }

        domains: dict[str, list[str]] = {}
        for column in columns:
            values = await self._query_schema_values(
                f'import "influxdata/influxdb/schema"\n'
                f'schema.tagValues(bucket: "{self._escape_flux_string(self.config.bucket)}", '
                f'start: 1970-01-01T00:00:00Z, '
                f'tag: "{self._escape_flux_string(column)}", '
                f'predicate: (r) => r._measurement == "{self._escape_flux_string(source_name)}")'
            )
            if values:
                domains[str(column)] = values[:limit]
        return domains

    async def _get_schema_v1(self) -> DatabaseSchema:
        """Get InfluxDB 1.x schema."""
        schema = DatabaseSchema(database=self.config.database, metadata={"version": 1})

        try:
            # Get measurements (tables)
            result = self._client.query("SHOW MEASUREMENTS")
            measurements = []
            if result:
                for series in result:
                    for values in series.get("values", []):
                        measurements.append(values[0])

            for measurement in measurements:
                # Get fields for each measurement
                field_result = self._client.query(f"SHOW FIELD KEYS FROM {measurement}")
                columns = []

                if field_result:
                    for series in field_result:
                        for values in series.get("values", []):
                            columns.append(ColumnSchema(
                                name=values.get("fieldKey", ""),
                                data_type=self._influx_type_to_generic(values.get("fieldType", "string")),
                            ))

                # Get tag keys
                tag_result = self._client.query(f"SHOW TAG KEYS FROM {measurement}")
                if tag_result:
                    for series in tag_result:
                        for values in series.get("values", []):
                            columns.append(ColumnSchema(
                                name=values.get("tagKey", ""),
                                data_type="string",
                            ))

                schema.tables.append(TableSchema(
                    name=measurement,
                    columns=columns,
                ))

        except Exception as e:
            schema.metadata["error"] = str(e)

        return schema

    async def _probe_value_domains_v1(
        self,
        *,
        source_name: str,
        columns: list[str],
        limit: int,
    ) -> dict[str, list[str]]:
        domains: dict[str, list[str]] = {}
        try:
            for column in columns:
                result = self._client.query(
                    f'SHOW TAG VALUES FROM "{source_name}" WITH KEY = "{column}"'
                )
                values: list[str] = []
                if result:
                    for series in result:
                        for row in series.get("values", [])[:limit]:
                            if len(row) > 1 and row[1] not in (None, ""):
                                values.append(str(row[1]))
                if values:
                    domains[str(column)] = values[:limit]
        except Exception:
            return {}
        return domains

    async def health_check(self) -> bool:
        """Check InfluxDB connection health."""
        if not self._client:
            return False

        try:
            return self._client.ping()
        except Exception:
            return False

    async def ensure_configured_buckets(self, dry_run: bool = False) -> list[dict[str, Any]]:
        """Create the connector bucket and configured task target buckets when missing."""
        if not self._is_v2:
            raise ValueError("InfluxDB buckets are only supported for InfluxDB 2.x")
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to InfluxDB")

        bucket_names = self._configured_bucket_names()
        if not bucket_names:
            return []

        buckets_api = self._client.buckets_api()
        results: list[dict[str, Any]] = []
        for bucket_name in bucket_names:
            existing = buckets_api.find_bucket_by_name(bucket_name)
            if existing:
                results.append({"bucket": bucket_name, "action": "unchanged", "bucket_id": getattr(existing, "id", None)})
                continue
            if dry_run:
                results.append({"bucket": bucket_name, "action": "created", "bucket_id": None})
                continue
            created = buckets_api.create_bucket(bucket_name=bucket_name, org=self.config.org)
            results.append({"bucket": bucket_name, "action": "created", "bucket_id": getattr(created, "id", None)})
        return results

    async def sync_configured_tasks(
        self,
        dry_run: bool = False,
        ensure_buckets: bool = True,
        run_now: bool = False,
    ) -> list[dict[str, Any]]:
        """Create or update configured InfluxDB 2.x tasks.

        Tasks are declared in database YAML under ``influxdb_tasks``. Each task
        can provide a complete ``flux`` program or a ``sample`` block such as
        ``{"set": "bitcoin", "target_bucket": "bitcoin"}``.
        """
        if not self._is_v2:
            raise ValueError("InfluxDB tasks are only supported for InfluxDB 2.x")
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to InfluxDB")

        task_configs = self._configured_task_configs()
        if not task_configs:
            return []

        if ensure_buckets:
            await self.ensure_configured_buckets(dry_run=dry_run)

        tasks_api = self._client.tasks_api()
        results: list[dict[str, Any]] = []
        for task_config in task_configs:
            desired = self._build_task_definition(task_config)
            existing = self._find_task_by_name(tasks_api, desired["name"])
            action = self._task_sync_action(existing, desired)

            if dry_run:
                results.append({**desired, "action": action, "task_id": getattr(existing, "id", None)})
                continue

            synced_task = existing
            if existing is None:
                task_payload = type("InfluxDBTaskPayload", (), {})()
                task_payload.org_id = None
                task_payload.org = self.config.org
                task_payload.status = desired["status"]
                task_payload.flux = desired["flux"]
                task_payload.description = desired.get("description")
                synced_task = tasks_api.create_task(
                    task=task_payload
                )
            elif action == "updated":
                existing.flux = desired["flux"]
                existing.status = desired["status"]
                existing.description = desired.get("description")
                synced_task = tasks_api.update_task(existing)

            task_id = getattr(synced_task, "id", None)
            ran_now = False
            if run_now and task_id:
                tasks_api.run_manually(task_id)
                ran_now = True
            results.append({**desired, "action": action, "task_id": task_id, "ran_now": ran_now})

        return results

    def _configured_bucket_names(self) -> list[str]:
        """Return unique bucket names needed by the connector and configured tasks."""
        buckets: list[str] = []
        for bucket in [self.config.bucket, *self._configured_task_target_buckets()]:
            bucket = str(bucket or "").strip()
            if bucket and bucket not in buckets:
                buckets.append(bucket)
        return buckets

    def _configured_task_target_buckets(self) -> list[str]:
        """Return bucket names explicitly targeted by configured tasks."""
        buckets: list[str] = []
        for task_config in self._configured_task_configs():
            explicit_buckets = task_config.get("target_buckets")
            if isinstance(explicit_buckets, list):
                buckets.extend(str(bucket) for bucket in explicit_buckets if bucket not in (None, ""))
            sample = task_config.get("sample")
            if isinstance(sample, dict):
                target_bucket = sample.get("target_bucket") or self.config.bucket
                if target_bucket:
                    buckets.append(str(target_bucket))
        return buckets

    def _configured_task_configs(self) -> list[dict[str, Any]]:
        """Return valid task config dictionaries."""
        tasks = self.config.influxdb_tasks
        if not isinstance(tasks, list):
            return []
        return [task for task in tasks if isinstance(task, dict)]

    def _find_task_by_name(self, tasks_api: Any, name: str) -> Any | None:
        """Find one task by name in the configured organization."""
        matches = tasks_api.find_tasks(name=name, org=self.config.org)
        for task in matches or []:
            if getattr(task, "name", None) == name:
                return task
        return None

    def _task_sync_action(self, existing: Any | None, desired: dict[str, Any]) -> str:
        """Return the action required to make one task match its config."""
        if existing is None:
            return "created"
        if (
            getattr(existing, "flux", None) != desired["flux"]
            or getattr(existing, "status", None) != desired["status"]
            or (getattr(existing, "description", None) or None) != desired.get("description")
        ):
            return "updated"
        return "unchanged"

    def _build_task_definition(self, task_config: dict[str, Any]) -> dict[str, Any]:
        """Build a normalized InfluxDB task definition from YAML config."""
        name = str(task_config.get("name") or "").strip()
        if not name:
            raise ValueError("InfluxDB task config requires name")

        raw_flux = task_config.get("flux")
        has_task_option = isinstance(raw_flux, str) and self._flux_has_task_option(raw_flux)
        every = str(task_config.get("every") or "").strip()
        cron = str(task_config.get("cron") or "").strip()
        if not has_task_option and not every and not cron:
            raise ValueError(f"InfluxDB task {name!r} requires every or cron")

        flux_body = self._build_task_flux_body(task_config)
        if has_task_option:
            task_flux = flux_body.strip()
        else:
            schedule = f'every: {every}' if every else f'cron: "{self._escape_flux_string(cron)}"'
            task_flux = self._compose_task_flux(
                task_option=f'option task = {{name: "{self._escape_flux_string(name)}", {schedule}}}',
                flux_body=flux_body,
            )
        status = str(task_config.get("status") or "active")
        description = task_config.get("description")
        if description is not None:
            description = str(description)
        return {
            "name": name,
            "status": status,
            "description": description,
            "flux": task_flux,
        }

    def _build_task_flux_body(self, task_config: dict[str, Any]) -> str:
        """Build the Flux body for direct Flux or sample-data task configs."""
        flux = task_config.get("flux")
        if isinstance(flux, str) and flux.strip():
            return flux

        sample = task_config.get("sample")
        if not isinstance(sample, dict):
            raise ValueError("InfluxDB task config requires flux or sample")
        sample_set = str(sample.get("set") or "").strip()
        if not sample_set:
            raise ValueError("InfluxDB sample task config requires sample.set")
        target_bucket = str(sample.get("target_bucket") or self.config.bucket).strip()
        if not target_bucket:
            raise ValueError("InfluxDB sample task config requires sample.target_bucket or connector bucket")
        return (
            'import "influxdata/influxdb/sample"\n\n'
            f'sample.data(set: "{self._escape_flux_string(sample_set)}")\n'
            f'    |> to(bucket: "{self._escape_flux_string(target_bucket)}")'
        )

    def _flux_has_task_option(self, flux: str) -> bool:
        """Return whether Flux already declares an InfluxDB task option."""
        return bool(re.search(r"\boption\s+task\s*=", flux))

    def _compose_task_flux(self, task_option: str, flux_body: str) -> str:
        """Compose task Flux while keeping import declarations first."""
        body = flux_body.strip()
        if not body:
            return task_option

        lines = body.splitlines()
        import_lines: list[str] = []
        body_start = 0
        seen_import = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                if seen_import:
                    import_lines.append(line)
                    continue
                body_start = index + 1
                continue
            if stripped.startswith("import "):
                seen_import = True
                import_lines.append(line)
                continue
            body_start = index
            break
        else:
            body_start = len(lines)

        remainder = "\n".join(lines[body_start:]).strip()
        if not import_lines:
            return f"{task_option}\n\n{body}"
        prefix = "\n".join(import_lines).strip()
        if not remainder:
            return f"{prefix}\n\n{task_option}"
        return f"{prefix}\n\n{task_option}\n\n{remainder}"

    def _build_v2_url(self) -> str:
        """Build InfluxDB 2.x base URL from config."""
        if self.config.url:
            return self.config.url
        scheme = "https" if self.config.ssl else "http"
        return f"{scheme}://{self.config.host}:{self.config.port}"

    def _normalize_flux_bucket(self, query: str) -> str:
        """Force Flux from(bucket: ...) calls to use the configured bucket.

        The UI and LLM prompts often refer to a saved database by id/name, but
        InfluxDB 2.x requires the actual bucket name. A connector instance is
        scoped to one bucket, so normalize generated Flux before execution.
        """
        if not self.config.bucket:
            return query
        bucket = self._escape_flux_string(self.config.bucket)
        return re.sub(
            r'from\s*\(\s*bucket\s*:\s*"[^"]+"\s*\)',
            f'from(bucket: "{bucket}")',
            query,
            flags=re.IGNORECASE,
        )

    def _flux_record_to_row(self, record: Any) -> dict[str, Any]:
        """Convert an influxdb-client FluxRecord into a display row."""
        values = dict(getattr(record, "values", {}) or {})
        row: dict[str, Any] = {}
        for key, value in values.items():
            if key in {"result", "table", "_start", "_stop"}:
                continue
            display_key = {
                "_time": "time",
                "_measurement": "measurement",
                "_field": "field",
                "_value": "value",
            }.get(key, key)
            row[display_key] = self._format_value(value)
        return row

    async def _query_schema_values(self, query: str) -> list[str]:
        """Run a Flux schema query and return unique string values."""
        if not self._query_api:
            return []

        values: list[str] = []
        seen: set[str] = set()
        for table in self._query_api.query(query=query, org=self.config.org) or []:
            for record in getattr(table, "records", []) or []:
                value = self._schema_record_value(record)
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        return values

    async def _query_schema_pairs(self, query: str) -> list[tuple[str, str]]:
        """Run a Flux schema query and return field name/type pairs."""
        if not self._query_api:
            return []

        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for table in self._query_api.query(query=query, org=self.config.org) or []:
            for record in getattr(table, "records", []) or []:
                values = getattr(record, "values", {}) or {}
                name = (
                    values.get("_value")
                    or values.get("fieldKey")
                    or values.get("_field")
                    or getattr(record, "get_value", lambda: None)()
                )
                field_type = values.get("fieldType") or values.get("type") or "float"
                if name and str(name) not in seen:
                    seen.add(str(name))
                    pairs.append((str(name), str(field_type)))
        return pairs

    def _schema_record_value(self, record: Any) -> str | None:
        """Extract a string value from a Flux schema record."""
        values = getattr(record, "values", {}) or {}
        value = values.get("_value")
        if value is None and hasattr(record, "get_value"):
            value = record.get_value()
        return str(value) if value not in (None, "") else None

    def _build_reference_dataset_metadata(self) -> dict[str, Any] | None:
        """Build metadata for the configured reference dataset, if any."""
        reference_dataset = self.config.extra.get("reference_dataset")
        if not isinstance(reference_dataset, dict):
            return None
        raw_dataset_path = reference_dataset.get("dataset_path")
        resolved_dataset_path = reference_dataset.get("resolved_dataset_path")
        dataset_path = resolved_dataset_path or raw_dataset_path
        row_count = self._count_reference_dataset_rows(dataset_path)
        time_range = self._reference_dataset_time_range(
            dataset_path=dataset_path,
            timestamp_column=reference_dataset.get("timestamp_column"),
        )
        return {
            **reference_dataset,
            "dataset_path": raw_dataset_path,
            "resolved_dataset_path": resolved_dataset_path,
            "row_count": row_count,
            "time_range": time_range,
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

    def _reference_dataset_time_range(
        self,
        *,
        dataset_path: str | None,
        timestamp_column: str | None,
    ) -> dict[str, str] | None:
        """Infer the timestamp bounds of the configured CSV reference dataset."""
        if not dataset_path or not timestamp_column:
            return None
        resolved_path = Path(str(dataset_path))
        if not resolved_path.is_absolute():
            resolved_path = (Path(__file__).resolve().parents[3] / resolved_path).resolve()
        if not resolved_path.exists():
            return None

        start: datetime | None = None
        stop: datetime | None = None
        try:
            with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    timestamp = self._parse_reference_timestamp(row.get(str(timestamp_column)))
                    if timestamp is None:
                        continue
                    start = timestamp if start is None or timestamp < start else start
                    stop = timestamp if stop is None or timestamp > stop else stop
        except Exception:
            return None
        if start is None or stop is None:
            return None
        return {
            "start": self._format_flux_bound_timestamp(start),
            "stop": self._format_flux_bound_timestamp(stop),
        }

    def _parse_reference_timestamp(self, raw_value: Any) -> datetime | None:
        """Parse common CSV timestamp shapes into a naive UTC-compatible datetime."""
        if raw_value in (None, ""):
            return None
        text = str(raw_value).strip().strip('"')
        if not text:
            return None
        normalized = text.replace("T", " ")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _format_flux_bound_timestamp(self, value: datetime) -> str:
        """Format a timestamp as a Flux RFC3339 bound."""
        return value.replace(microsecond=0).isoformat().replace("+00:00", "") + "Z"

    def _reference_dataset_value_domains(
        self,
        reference_dataset: dict[str, Any] | None,
    ) -> dict[str, dict[str, list[str]]]:
        """Infer canonical value domains from the configured CSV reference dataset."""
        if not isinstance(reference_dataset, dict):
            return {}
        measurement = (
            reference_dataset.get("measurement")
            or reference_dataset.get("metric_name")
            or reference_dataset.get("table")
        )
        dataset_path = reference_dataset.get("resolved_dataset_path") or reference_dataset.get("dataset_path")
        if not measurement or not dataset_path:
            return {}
        resolved_path = Path(str(dataset_path))
        if not resolved_path.is_absolute():
            resolved_path = (Path(__file__).resolve().parents[3] / resolved_path).resolve()
        if not resolved_path.exists():
            return {}

        field_columns = reference_dataset.get("field_columns")
        if isinstance(field_columns, list) and field_columns:
            domains: dict[str, list[str]] = {
                "_field": [str(column) for column in field_columns if column not in (None, "")]
            }
            static_tags = reference_dataset.get("static_tags")
            if isinstance(static_tags, dict):
                for key, raw_value in static_tags.items():
                    if raw_value in (None, ""):
                        continue
                    domains[str(key)] = [str(raw_value)]
            return {str(measurement): domains}

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

        static_tags = reference_dataset.get("static_tags")
        if isinstance(static_tags, dict):
            for key, raw_value in static_tags.items():
                if raw_value in (None, ""):
                    continue
                values = domains.setdefault(str(key), [])
                value = str(raw_value)
                if value not in values:
                    values.append(value)
        if value_column:
            domains.setdefault("_field", [])
            if value_column not in domains["_field"]:
                domains["_field"].append(value_column)
        return {str(measurement): domains}

    def _escape_flux_string(self, value: str) -> str:
        """Escape a Flux string literal value."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _influx_type_to_generic(self, influx_type: str) -> str:
        """Convert InfluxDB type to generic type."""
        type_map = {
            "float": "float",
            "double": "float",
            "integer": "integer",
            "int": "integer",
            "long": "integer",
            "unsignedLong": "integer",
            "string": "string",
            "boolean": "boolean",
            "bool": "boolean",
        }
        return type_map.get(influx_type, type_map.get(influx_type.lower(), "string"))
