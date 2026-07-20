"""Query database tool backed by core.database."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from pydantic import BaseModel, Field

from app.settings import Settings
from core.time_range import format_utc_rfc3339, parse_time_to_utc
from core.database import (
    DatabaseFactory,
    DatabaseSchema,
    build_reference_dataset_statistics_evidence,
    build_reference_dataset_timeseries_evidence,
    execute_query,
    execute_range_query,
    infer_evidence_family,
    infer_prometheus_metric,
    metric_list_preview,
    normalize_query_result,
    schema_preview,
)
from core.database.query_flow import DatabaseQueryFlow
from schemas.database import DatabaseEvidence
from schemas.database_context import DatabaseContext
from tools.base import BaseTool


class QueryDatabaseInput(BaseModel):
    message: str
    database_context: DatabaseContext
    time_range: dict | None = None
    constraints: dict = Field(default_factory=dict)
    intent_profile: dict = Field(default_factory=dict)
    selected_database: str | None = None
    selected_database_type: str | None = None
    history: list[dict] = Field(default_factory=list)


class QueryDatabaseTool(BaseTool):
    def __init__(self, settings: Settings):
        self._settings = settings

    async def execute(self, validated_input: QueryDatabaseInput, **kwargs) -> dict:
        config_path, config = await self._load_database_config(validated_input.database_context.database_id)
        evidence_family = str(validated_input.constraints.get("expected_result_type") or "").strip()
        if evidence_family not in {"schema", "metric_list", "statistics", "table", "timeseries"}:
            evidence_family = infer_evidence_family(validated_input.message)
        if evidence_family == "metric_list":
            return await self._metric_list_evidence(validated_input, config)
        if evidence_family == "schema":
            return await self._schema_evidence(validated_input, config)

        reference_dataset = config.get("reference_dataset")
        db_type = str(config.get("type", validated_input.database_context.database_type)).lower()
        if isinstance(reference_dataset, dict):
            if evidence_family == "statistics":
                return self._reference_dataset_statistics(validated_input, config_path, config, reference_dataset)
            return self._reference_dataset_timeseries(validated_input, config_path, config, reference_dataset)

        if db_type == "prometheus":
            return await self._prometheus_timeseries(validated_input, config)

        return await self._connector_query_evidence(validated_input, config, evidence_family)

    async def _load_database_config(self, database_id: str) -> tuple[Path, dict]:
        await DatabaseFactory.load_databases()
        config = await DatabaseFactory.get_database(database_id)
        if not config:
            raise FileNotFoundError(
                f"Database config for '{database_id}' was not found in {self._settings.resolved_database_config_dir}"
            )
        config_path = Path(str(config.get("config_source", "")))
        if not config_path.is_absolute():
            config_path = (Path(self._settings.tspilot_root) / config_path).resolve()
        return config_path, config

    async def _metric_list_evidence(self, validated_input: QueryDatabaseInput, config: dict) -> dict:
        connector = await DatabaseFactory.create_connector(**config)
        async with connector:
            schema = await connector.get_schema()
        preview = metric_list_preview(schema)
        metric_names = preview["metrics"]
        evidence = DatabaseEvidence(
            evidence_id=f"evi_{validated_input.database_context.database_id}_metrics",
            result_type="metric_list",
            database=validated_input.database_context.database_id,
            query_language=str(config.get("type", validated_input.database_context.database_type)),
            query=None,
            summary=f"Loaded {len(metric_names)} available metrics.",
            data={"metrics": metric_names},
            columns=["metric"],
            metadata={"database_type": config.get("type"), **preview.get("metadata", {})},
            diagnostics={},
        )
        return evidence.model_dump(mode="json")

    async def _schema_evidence(self, validated_input: QueryDatabaseInput, config: dict) -> dict:
        connector = await DatabaseFactory.create_connector(**config)
        async with connector:
            schema = await connector.get_schema()
        preview = schema_preview(schema)
        evidence = DatabaseEvidence(
            evidence_id=f"evi_{validated_input.database_context.database_id}_schema",
            result_type="schema",
            database=validated_input.database_context.database_id,
            query_language=str(config.get("type", validated_input.database_context.database_type)),
            query=None,
            summary=f"Loaded schema preview with {len(preview['tables_or_measurements'])} tables or metrics.",
            data={k: v for k, v in preview.items() if k != "metadata"},
            columns=["name", "data_type"],
            metadata=preview["metadata"],
            diagnostics={},
        )
        return evidence.model_dump(mode="json")

    def _reference_dataset_timeseries(
        self,
        validated_input: QueryDatabaseInput,
        config_path: Path,
        config: dict,
        reference_dataset: dict,
    ) -> dict:
        dataset_path = self._resolve_dataset_path(reference_dataset["dataset_path"], config_path)
        rows = self._read_rows(dataset_path)
        if not rows:
            raise ValueError(f"Reference dataset '{dataset_path}' is empty.")
        time_field = reference_dataset.get("timestamp_column", next(iter(rows[0].keys())))
        reference_dataset = self._reference_dataset_with_query_hints(reference_dataset, validated_input.constraints)
        value_fields = self._pick_value_fields(validated_input.message, reference_dataset, rows[0].keys())
        value_field = value_fields[0]
        filtered_rows = self._filter_rows(rows, time_field, validated_input.time_range)
        requested_max_points = validated_input.constraints.get("max_points")
        points = self._to_points(filtered_rows, time_field, value_field)
        evidence = build_reference_dataset_timeseries_evidence(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type")),
            config_path=config_path,
            dataset_path=dataset_path,
            value_field=value_field,
            value_fields=value_fields,
            time_field=time_field,
            rows=self._to_timeseries_rows(filtered_rows, time_field, value_fields),
            points=points,
            source=reference_dataset.get("source", "reference_dataset"),
        )
        evidence.diagnostics = {
            **evidence.diagnostics,
            "row_count_total": len(filtered_rows),
            "row_count_materialized": len(points),
            "is_full_fidelity": True,
            "sampling_policy": {
                "analysis_input": "full_filtered_reference_dataset",
                "prompt_preview": "runtime_prompt_safe_sampling",
                "requested_max_points": requested_max_points,
            },
        }
        return evidence.model_dump(mode="json")

    def _reference_dataset_statistics(
        self,
        validated_input: QueryDatabaseInput,
        config_path: Path,
        config: dict,
        reference_dataset: dict,
    ) -> dict:
        dataset_path = self._resolve_dataset_path(reference_dataset["dataset_path"], config_path)
        rows = self._read_rows(dataset_path)
        if not rows:
            raise ValueError(f"Reference dataset '{dataset_path}' is empty.")
        time_field = reference_dataset.get("timestamp_column", next(iter(rows[0].keys())))
        reference_dataset = self._reference_dataset_with_query_hints(reference_dataset, validated_input.constraints)
        value_field = self._pick_value_field(validated_input.message, reference_dataset, rows[0].keys())
        filtered_rows = self._filter_rows(rows, time_field, validated_input.time_range)
        values = []
        for row in filtered_rows:
            try:
                values.append(float(str(row[value_field]).strip()))
            except ValueError:
                continue
        if not values:
            raise ValueError(f"No numeric values could be extracted from '{value_field}'.")
        evidence = build_reference_dataset_statistics_evidence(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type")),
            config_path=config_path,
            dataset_path=dataset_path,
            value_field=value_field,
            time_field=time_field,
            values=values,
        )
        return evidence.model_dump(mode="json")

    async def _prometheus_timeseries(self, validated_input: QueryDatabaseInput, config: dict) -> dict:
        connector = await DatabaseFactory.create_connector(**config)
        async with connector:
            flow = DatabaseQueryFlow(
                connector=connector,
                config={
                    **config,
                    "snapshot_dir": str((Path(self._settings.tspilot_root) / "cache_data" / "query_snapshots").resolve()),
                },
            )
            evidence = await flow.run(
                context=self._build_query_context(validated_input, config),
                execute_range_query_fn=execute_range_query,
            )
        return evidence.model_dump(mode="json")

    async def _connector_query_evidence(
        self,
        validated_input: QueryDatabaseInput,
        config: dict,
        evidence_family: str,
    ) -> dict:
        connector = await DatabaseFactory.create_connector(**config)
        async with connector:
            flow = DatabaseQueryFlow(
                connector=connector,
                config={
                    **config,
                    "snapshot_dir": str((Path(self._settings.tspilot_root) / "cache_data" / "query_snapshots").resolve()),
                },
            )
            evidence = await flow.run(
                context=self._build_query_context(validated_input, config),
                execute_range_query_fn=execute_range_query,
            )
        return evidence.model_dump(mode="json")

    def _build_query_context(self, validated_input: QueryDatabaseInput, config: dict):
        from core.database.contracts import QueryRequestContext

        return QueryRequestContext(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type", validated_input.database_context.database_type)),
            message=validated_input.message,
            time_range=validated_input.time_range,
            constraints=validated_input.constraints,
            history=validated_input.history,
            intent_profile=validated_input.intent_profile,
        )

    def _prometheus_step(self, time_range: dict, constraints: dict) -> str:
        max_points = int(constraints.get("max_points", 240))
        start = self._parse_time(time_range["start"])
        end = self._parse_time(time_range["end"])
        total_seconds = max(1, int((end - start).total_seconds()))
        step_seconds = max(1, math.ceil(total_seconds / max_points))
        return f"{step_seconds}s"

    def _draft_generic_query(self, message: str, schema, config: dict, evidence_family: str) -> str:
        db_type = str(config.get("type", "")).lower()
        first_table = schema.tables[0].name if schema.tables else ""
        normalized = message.lower()
        if db_type == "influxdb":
            measurement = first_table or config.get("database") or "measurement"
            value_field = self._guess_field_from_message(normalized, schema) or "_value"
            if evidence_family == "statistics":
                return (
                    f'from(bucket: "{config.get("bucket", config.get("database", ""))}") '
                    f'|> range(start: -7d) |> filter(fn: (r) => r._measurement == "{measurement}") '
                    f'|> filter(fn: (r) => r._field == "{value_field}") |> mean()'
                )
            return (
                f'from(bucket: "{config.get("bucket", config.get("database", ""))}") '
                f'|> range(start: -7d) |> filter(fn: (r) => r._measurement == "{measurement}") '
                f'|> filter(fn: (r) => r._field == "{value_field}")'
            )
        return first_table or message

    def _guess_field_from_message(self, normalized_message: str, schema) -> str | None:
        for table in schema.tables:
            for column in table.columns:
                if column.name.lower() in normalized_message:
                    return column.name
        return None

    def _resolve_dataset_path(self, dataset_path: str, config_path: Path) -> Path:
        path = Path(dataset_path)
        if path.is_absolute():
            return path
        project_root = Path(self._settings.tspilot_root).resolve()
        if (project_root / path).exists():
            return (project_root / path).resolve()
        return (config_path.parent / path).resolve()

    def _read_rows(self, dataset_path: Path) -> list[dict]:
        with dataset_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    def _pick_value_field(self, message: str, reference_dataset: dict, columns) -> str:
        return self._pick_value_fields(message, reference_dataset, columns)[0]

    def _reference_dataset_with_query_hints(self, reference_dataset: dict, constraints: dict) -> dict:
        selected_fields = constraints.get("selected_fields")
        if not isinstance(selected_fields, list) or not selected_fields:
            return reference_dataset
        return {**reference_dataset, "selected_fields": selected_fields}

    def _pick_value_fields(self, message: str, reference_dataset: dict, columns) -> list[str]:
        selected_from_constraints = reference_dataset.get("selected_fields")
        if isinstance(selected_from_constraints, list):
            available = {str(column) for column in columns}
            selected = [
                str(field)
                for field in selected_from_constraints
                if field not in (None, "") and str(field) in available and str(field) != reference_dataset.get("timestamp_column")
            ]
            if selected:
                return selected
        text = message.lower()
        configured = list(reference_dataset.get("field_columns", []))
        selected: list[str] = []
        for field in configured:
            if str(field).lower() in text:
                selected.append(str(field))
        for field in columns:
            field = str(field)
            if field.lower() in text and field != reference_dataset.get("timestamp_column") and field not in selected:
                selected.append(field)
        if selected:
            return selected
        if configured:
            return [str(configured[0])]
        for field in columns:
            field = str(field)
            if field != reference_dataset.get("timestamp_column"):
                return [field]
        raise ValueError("Could not infer a value field from the dataset.")

    def _filter_rows(self, rows: list[dict], time_field: str, time_range: dict | None) -> list[dict]:
        if not time_range:
            return rows
        start = self._parse_time(time_range.get("start")) if time_range.get("start") else None
        end = self._parse_time(time_range.get("end")) if time_range.get("end") else None
        filtered = []
        for row in rows:
            timestamp = self._parse_time(row[time_field])
            if start and timestamp < start:
                continue
            if end and timestamp > end:
                continue
            filtered.append(row)
        return filtered or rows

    def _to_points(self, rows: list[dict], time_field: str, value_field: str) -> list[dict]:
        points = []
        for row in rows:
            raw_value = str(row[value_field]).strip()
            try:
                value = float(raw_value)
            except ValueError:
                continue
            points.append({"timestamp": format_utc_rfc3339(self._parse_time(row[time_field])), "value": value})
        if not points:
            raise ValueError(f"No numeric points could be extracted from '{value_field}'.")
        return points

    def _to_timeseries_rows(self, rows: list[dict], time_field: str, value_fields: list[str]) -> list[dict]:
        normalized_rows: list[dict] = []
        for row in rows:
            normalized = {time_field: format_utc_rfc3339(self._parse_time(row[time_field]))}
            has_numeric = False
            for field in value_fields:
                raw_value = row.get(field)
                if raw_value is None or str(raw_value).strip() == "":
                    normalized[field] = None
                    continue
                try:
                    normalized[field] = float(str(raw_value).strip())
                    has_numeric = True
                except ValueError:
                    normalized[field] = None
            if has_numeric:
                normalized_rows.append(normalized)
        return normalized_rows

    def _parse_time(self, value: str) -> datetime:
        return parse_time_to_utc(value)
