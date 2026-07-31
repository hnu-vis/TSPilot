"""Deterministic schema and catalog inspection helpers."""
from __future__ import annotations

from typing import Any

from .connector import DatabaseSchema


def schema_preview(schema: DatabaseSchema, *, dialect: Any | None = None) -> dict:
    metadata = schema.metadata or {}
    value_domains = metadata.get("value_domains") if isinstance(metadata.get("value_domains"), dict) else {}
    reference_dataset = (
        metadata.get("reference_dataset")
        if isinstance(metadata.get("reference_dataset"), dict)
        else None
    )
    tables = [
        _table_preview(
            table=table,
            table_domains=value_domains.get(table.name) if isinstance(value_domains, dict) else None,
            reference_dataset=reference_dataset,
        )
        for table in schema.tables
    ]
    fields = [
        {
            "table": table["name"],
            "name": column["name"],
            "data_type": column["data_type"],
            "nullable": column["nullable"],
        }
        for table in tables
        for column in table["columns"]
    ]
    time_columns = sorted(
        {
            column["name"]
            for column in fields
            if "time" in str(column["name"]).lower() or column["name"] == "_time"
        }
    )
    labels_or_tags = []
    if isinstance(value_domains, dict):
        for table_name, domains in value_domains.items():
            if not isinstance(domains, dict):
                continue
            for name, values in domains.items():
                if name == "_field":
                    continue
                labels_or_tags.append(
                    {
                        "table": table_name,
                        "name": name,
                        "values": list(values)[:20] if isinstance(values, list) else [],
                    }
                )
    preview = {
        "tables_or_measurements": tables,
        "fields": fields,
        "labels_or_tags": labels_or_tags,
        "time_columns": time_columns,
        "metadata": metadata,
    }
    if dialect is not None:
        extension_builder = getattr(dialect, "schema_preview_extensions", None)
        if callable(extension_builder):
            extensions = extension_builder(schema=schema, preview=preview)
            if isinstance(extensions, dict):
                preview.update(extensions)
    return preview


def metric_list_preview(schema: DatabaseSchema) -> dict:
    return {
        "metrics": [table.name for table in schema.tables],
        "metadata": schema.metadata,
    }


def _table_preview(*, table, table_domains: dict | None, reference_dataset: dict | None) -> dict:
    row_count = table.row_count
    reference_matches = _reference_matches_table(reference_dataset, table.name)
    if row_count is None and reference_matches:
        row_count = reference_dataset.get("row_count") if reference_dataset else None

    preview = {
        "name": table.name,
        "schema": table.schema,
        "type": table.type,
        "row_count": row_count,
        "columns": [
            {
                "name": column.name,
                "data_type": column.data_type,
                "nullable": column.nullable,
            }
            for column in table.columns
        ],
    }
    if isinstance(table_domains, dict):
        field_values = table_domains.get("_field")
        if isinstance(field_values, list):
            preview["field_values"] = field_values[:100]
    if reference_matches and reference_dataset:
        if isinstance(reference_dataset.get("time_range"), dict):
            preview["time_range"] = reference_dataset["time_range"]
        if isinstance(reference_dataset.get("sample_rows"), list):
            preview["sample_rows"] = reference_dataset["sample_rows"][:3]
    return preview


def _reference_matches_table(reference_dataset: dict | None, table_name: str) -> bool:
    if not isinstance(reference_dataset, dict):
        return False
    candidates = {
        reference_dataset.get("measurement"),
        reference_dataset.get("metric_name"),
        reference_dataset.get("table"),
        reference_dataset.get("series_name"),
    }
    return table_name in {str(value) for value in candidates if value not in (None, "")}
