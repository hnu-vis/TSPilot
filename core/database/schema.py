"""Deterministic schema and catalog inspection helpers."""
from __future__ import annotations

from .connector import DatabaseSchema


def schema_preview(schema: DatabaseSchema) -> dict:
    tables = [
        {
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
    return {
        "tables_or_measurements": tables,
        "fields": fields,
        "labels_or_tags": [],
        "time_columns": time_columns,
        "metadata": schema.metadata,
    }


def metric_list_preview(schema: DatabaseSchema) -> dict:
    return {
        "metrics": [table.name for table in schema.tables],
        "metadata": schema.metadata,
    }
