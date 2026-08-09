"""Schema and catalog inspection helpers."""
from __future__ import annotations

from typing import Any

from .connector import DatabaseSchema


def schema_preview(schema: DatabaseSchema, *, dialect: Any | None = None) -> dict:
    metadata = schema.metadata or {}
    value_domains = metadata.get("value_domains") if isinstance(metadata.get("value_domains"), dict) else {}
    tables = [
        _table_preview(
            table=table,
            table_domains=value_domains.get(table.name) if isinstance(value_domains, dict) else None,
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


def _table_preview(*, table, table_domains: dict | None) -> dict:
    preview = {
        "name": table.name,
        "schema": table.schema,
        "type": table.type,
        "row_count": table.row_count,
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
    return preview
