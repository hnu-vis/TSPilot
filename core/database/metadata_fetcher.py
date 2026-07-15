"""Metadata fetcher for database schema and statistics."""
from dataclasses import dataclass, field
from typing import Any

from .connector import DBConnector, DatabaseSchema, TableSchema, ColumnSchema


@dataclass
class TableMetadata:
    """Table metadata information."""
    name: str
    schema: str = ""
    type: str = "table"
    row_count: int | None = None
    size_bytes: int | None = None
    columns: list[ColumnSchema] = field(default_factory=list)


@dataclass
class ColumnMetadata:
    """Column metadata information."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    default_value: Any = None
    description: str | None = None
    unit: str | None = None


@dataclass
class MetricMetadata:
    """Metric metadata for time-series databases."""
    name: str
    description: str | None = None
    unit: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TableSizeInfo:
    """Table size information."""
    table_name: str
    row_count: int
    size_bytes: int
    size_human: str = ""


class MetadataFetcher:
    """Fetches database metadata including tables, columns, and metrics.

    Uses caching to reduce database load.
    """

    def __init__(
        self,
        connectors: dict[str, DBConnector],
        cache: Any | None = None,
        cache_ttl: int = 300,
    ):
        self._connectors = connectors
        self._cache = cache
        self._cache_ttl = cache_ttl

    async def get_tables(
        self,
        database: str,
        include_views: bool = True,
    ) -> list[TableMetadata]:
        """Get list of tables in database."""
        cache_key = f"tables:{database}"
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        connector = self._connectors.get(database)
        if not connector:
            return []

        schema = await connector.get_schema()
        tables = []

        for table in schema.tables:
            tables.append(TableMetadata(
                name=table.name,
                schema=table.schema,
                type=table.type,
                row_count=table.row_count,
                size_bytes=table.size_bytes,
                columns=[ColumnMetadata(
                    name=c.name,
                    data_type=c.data_type,
                    nullable=c.nullable,
                    is_primary_key=c.is_primary_key,
                    is_foreign_key=c.is_foreign_key,
                    description=c.description,
                    unit=getattr(c, "unit", None),
                ) for c in table.columns],
            ))

        if include_views:
            for view in schema.views:
                tables.append(TableMetadata(
                    name=view.name,
                    schema=view.schema,
                    type="view",
                    row_count=None,
                    size_bytes=None,
                    columns=[ColumnMetadata(
                        name=c.name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                    ) for c in view.columns],
                ))

        if self._cache and (cache_ttl := self._cache_ttl):
            await self._cache.set(cache_key, tables, ttl=cache_ttl)

        return tables

    async def get_columns(
        self,
        database: str,
        table: str,
    ) -> list[ColumnMetadata]:
        """Get column information for a table."""
        cache_key = f"columns:{database}:{table}"
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        connector = self._connectors.get(database)
        if not connector:
            return []

        schema = await connector.get_schema()

        for t in schema.tables:
            if t.name == table:
                columns = [
                    ColumnMetadata(
                        name=c.name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                        is_primary_key=c.is_primary_key,
                        is_foreign_key=c.is_foreign_key,
                        description=c.description,
                        unit=getattr(c, "unit", None),
                    )
                    for c in t.columns
                ]

                if self._cache:
                    await self._cache.set(cache_key, columns, ttl=self._cache_ttl)

                return columns

        return []

    async def get_metrics(self, database: str) -> list[MetricMetadata]:
        """Get list of metrics for time-series databases."""
        cache_key = f"metrics:{database}"
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        connector = self._connectors.get(database)
        if not connector:
            return []

        # Try to get metrics from InfluxDB
        if hasattr(connector, "dialect") and connector.dialect == "influxdb":
            try:
                result = await connector.execute("SHOW MEASUREMENTS")
                metrics = []
                for row in result.rows:
                    name = row.get("name") or row.get("measurement")
                    if name:
                        metrics.append(MetricMetadata(name=name))
                return metrics
            except Exception:
                pass

        # Fallback: try SHOW TABLES or similar
        try:
            result = await connector.execute("SHOW TABLES")
            metrics = [
                MetricMetadata(name=row.get("TableName") or row.get("table_name") or list(row.values())[0])
                for row in result.rows
            ]
            return metrics
        except Exception:
            pass

        return []

    async def get_tags(
        self,
        database: str,
        metric: str | None = None,
    ) -> list[str]:
        """Get available tags for a metric."""
        connector = self._connectors.get(database)
        if not connector:
            return []

        if hasattr(connector, "dialect") and connector.dialect == "influxdb":
            try:
                query = f"SHOW TAG KEYS"
                if metric:
                    query += f" FROM {metric}"
                result = await connector.execute(query)
                tags = []
                for row in result.rows:
                    for key in row.keys():
                        if key not in ("time", "hostname"):
                            tags.append(key)
                return list(set(tags))
            except Exception:
                pass

        return []

    async def get_table_size(
        self,
        database: str,
        table: str,
    ) -> TableSizeInfo | None:
        """Get size information for a table."""
        connector = self._connectors.get(database)
        if not connector:
            return None

        # Try database-specific queries
        if hasattr(connector, "dialect"):
            dialect = connector.dialect

            if dialect == "influxdb":
                try:
                    result = await connector.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    )
                    if result.rows:
                        row = result.rows[0]
                        count = row.get("count", row.get("value", 0))
                        return TableSizeInfo(
                            table_name=table,
                            row_count=count,
                            size_bytes=0,
                            size_human="Unknown (InfluxDB)",
                        )
                except Exception:
                    pass

            elif dialect in ("postgresql", "timescaledb"):
                try:
                    result = await connector.execute("""
                        SELECT
                            pg_size_pretty(pg_total_relation_size('{table}')) as size,
                            pg_total_relation_size('{table}') as size_bytes
                    """.format(table=table))
                    if result.rows:
                        row = result.rows[0]
                        return TableSizeInfo(
                            table_name=table,
                            row_count=0,
                            size_bytes=row.get("size_bytes", 0),
                            size_human=row.get("size", "Unknown"),
                        )
                except Exception:
                    pass

            elif dialect == "questdb":
                try:
                    result = await connector.execute(
                        f"SELECT * FROM '{table}' LIMIT 0"
                    )
                except Exception:
                    pass

        # Fallback: try information_schema
        try:
            result = await connector.execute(f"""
                SELECT
                    TABLE_ROWS,
                    DATA_LENGTH
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME = '{table}'
            """)
            if result.rows:
                row = result.rows[0]
                size_bytes = row.get("DATA_LENGTH", 0) or 0
                return TableSizeInfo(
                    table_name=table,
                    row_count=row.get("TABLE_ROWS", 0) or 0,
                    size_bytes=size_bytes,
                    size_human=self._bytes_to_human(size_bytes),
                )
        except Exception:
            pass

        return None

    def _bytes_to_human(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    async def refresh_cache(self, database: str | None = None) -> None:
        """Refresh cached metadata."""
        if database:
            keys_to_delete = [f"tables:{database}", f"metrics:{database}"]
            if self._cache:
                for key in keys_to_delete:
                    await self._cache.delete(key)
        else:
            # Refresh all databases
            for db in self._connectors.keys():
                await self.refresh_cache(db)
