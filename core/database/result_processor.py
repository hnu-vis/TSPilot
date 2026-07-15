"""Result processor for query results."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .connector import QueryResult, TableSchema


@dataclass
class Column:
    """Processed column information."""
    name: str
    type: str
    nullable: bool = True


@dataclass
class ResultMetadata:
    """Metadata for processed results."""
    database: str = ""
    table: str = ""
    query_time_ms: int = 0
    total_rows: int = 0
    truncated: bool = False


@dataclass
class ProcessedResult:
    """Processed query result."""
    columns: list[Column]
    rows: list[dict]
    row_count: int
    execution_time_ms: int
    truncated: bool = False
    metadata: ResultMetadata = field(default_factory=ResultMetadata)


@dataclass
class Aggregation:
    """Aggregation definition."""
    column: str
    function: str  # SUM, AVG, COUNT, MIN, MAX
    alias: str | None = None


@dataclass
class AggregatedResult:
    """Aggregated query result."""
    columns: list[str]
    rows: list[dict]
    aggregations: list[Aggregation]


@dataclass
class PaginatedResult:
    """Paginated result."""
    page: int
    page_size: int
    total_rows: int
    total_pages: int
    data: list[dict]


class ResultProcessor:
    """Processes and transforms query results.

    Handles type normalization, timestamp conversion,
    aggregation, and pagination.
    """

    TYPE_MAPPING = {
        "INTEGER": "integer",
        "BIGINT": "integer",
        "SMALLINT": "integer",
        "FLOAT": "float",
        "DOUBLE": "float",
        "DECIMAL": "float",
        "NUMERIC": "float",
        "VARCHAR": "string",
        "TEXT": "string",
        "BOOLEAN": "boolean",
        "TIMESTAMP": "datetime",
        "DATETIME": "datetime",
        "TIME": "time",
    }

    def __init__(self, timezone: str = "UTC"):
        self._timezone = timezone

    def process(self, result: QueryResult, schema: TableSchema | None = None) -> ProcessedResult:
        """Process raw query result."""
        columns = self._extract_columns(result.columns, schema)
        rows = self._normalize_rows(result.rows, schema)
        metadata = ResultMetadata(
            total_rows=result.row_count,
            query_time_ms=result.execution_time_ms,
            truncated=result.truncated,
        )

        return ProcessedResult(
            columns=columns,
            rows=rows,
            row_count=result.row_count,
            execution_time_ms=result.execution_time_ms,
            truncated=result.truncated,
            metadata=metadata,
        )

    def _extract_columns(
        self,
        column_names: list[str],
        schema: TableSchema | None = None,
    ) -> list[Column]:
        """Extract column information."""
        if schema:
            schema_cols = {c.name: c for c in schema.columns}
            return [
                Column(
                    name=name,
                    type=schema_cols.get(name, Column(name=name, type="string")).type,
                    nullable=schema_cols.get(name, Column(name=name, type="string", nullable=True)).nullable,
                )
                for name in column_names
            ]

        return [Column(name=name, type="string") for name in column_names]

    def _normalize_rows(
        self,
        rows: list[dict],
        schema: TableSchema | None = None,
    ) -> list[dict]:
        """Normalize row values."""
        if not rows:
            return []

        normalized = []
        for row in rows:
            normalized_row = {}
            for key, value in row.items():
                normalized_row[key] = self._normalize_value(value, key, schema)
            normalized.append(normalized_row)

        return normalized

    def _normalize_value(
        self,
        value: Any,
        column_name: str,
        schema: TableSchema | None = None,
    ) -> Any:
        """Normalize a single value."""
        if value is None:
            return None

        if schema:
            for col in schema.columns:
                if col.name == column_name:
                    return self._cast_value(value, col.type)

        # Try to infer type
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, str):
            # Check if it's a datetime
            if "time" in column_name.lower() or "date" in column_name.lower():
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    pass
            return value
        elif isinstance(value, datetime):
            return value

        return value

    def _cast_value(self, value: Any, target_type: str) -> Any:
        """Cast value to target type."""
        if value is None:
            return None

        target_type = target_type.upper()
        if target_type not in self.TYPE_MAPPING:
            return value

        python_type = self.TYPE_MAPPING[target_type]

        try:
            if python_type == "integer":
                return int(value)
            elif python_type == "float":
                return float(value)
            elif python_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif python_type == "datetime":
                if isinstance(value, datetime):
                    return value
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        return value

    def normalize_types(
        self,
        rows: list[dict],
        schema: TableSchema,
    ) -> list[dict]:
        """Normalize types for rows based on schema."""
        return self._normalize_rows(rows, schema)

    def normalize_timestamps(
        self,
        rows: list[dict],
        timezone: str | None = None,
    ) -> list[dict]:
        """Normalize timestamp columns to specified timezone."""
        tz = timezone or self._timezone
        normalized = []

        for row in rows:
            new_row = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=None)
                    new_row[key] = value
                elif isinstance(value, str) and ("time" in key.lower() or "date" in key.lower()):
                    try:
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        new_row[key] = dt
                    except Exception:
                        new_row[key] = value
                else:
                    new_row[key] = value
            normalized.append(new_row)

        return normalized

    def aggregate(
        self,
        result: ProcessedResult,
        aggregations: list[Aggregation],
    ) -> AggregatedResult:
        """Perform aggregations on processed result."""
        if not result.rows:
            return AggregatedResult(
                columns=[a.alias or f"{a.function}({a.column})" for a in aggregations],
                rows=[],
                aggregations=aggregations,
            )

        # Group by unique combinations
        agg_results: dict[int, dict] = {}

        for i, row in enumerate(result.rows):
            key_values = tuple(row.get("host", ""))
            if key_values not in agg_results:
                agg_results[key_values] = {"_count": 0}
                for a in aggregations:
                    agg_results[key_values][a.alias or f"{a.function}({a.column})"] = []

            agg_results[key_values]["_count"] += 1
            for a in aggregations:
                col_name = a.alias or f"{a.function}({a.column})"
                val = row.get(a.column)
                if val is not None:
                    agg_results[key_values][col_name].append(val)

        # Compute aggregations
        output_rows = []
        for key_values, agg_data in agg_results.items():
            output_row = {}
            for a in aggregations:
                col_name = a.alias or f"{a.function}({a.column})"
                values = agg_data.get(col_name, [])

                if not values:
                    output_row[col_name] = None
                elif a.function.upper() == "SUM":
                    output_row[col_name] = sum(values)
                elif a.function.upper() == "AVG":
                    output_row[col_name] = sum(values) / len(values) if values else None
                elif a.function.upper() == "COUNT":
                    output_row[col_name] = len(values)
                elif a.function.upper() == "MIN":
                    output_row[col_name] = min(values)
                elif a.function.upper() == "MAX":
                    output_row[col_name] = max(values)

            output_rows.append(output_row)

        return AggregatedResult(
            columns=[a.alias or f"{a.function}({a.column})" for a in aggregations],
            rows=output_rows,
            aggregations=aggregations,
        )

    def paginate(
        self,
        result: ProcessedResult,
        page: int = 1,
        page_size: int = 100,
    ) -> PaginatedResult:
        """Paginate results."""
        total_rows = result.row_count
        total_pages = (total_rows + page_size - 1) // page_size if page_size > 0 else 0

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        return PaginatedResult(
            page=page,
            page_size=page_size,
            total_rows=total_rows,
            total_pages=total_pages,
            data=result.rows[start_idx:end_idx],
        )
