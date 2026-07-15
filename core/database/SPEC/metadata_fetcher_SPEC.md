# Metadata Fetcher Specification

## 1. Project Overview

**Project Name:** TSPilot
**Module Name:** MetadataFetcher
**Type:** Core Module - Database (Python)
**File Path:** `core/database/metadata_fetcher.py`
**Core Functionality:** Fetches and caches database metadata (tables, columns, metrics, tags).
**Target Users:** Schema agent, query agent.

---

## 2. Functionality Specification

### 2.1 Core Features

| Feature | Description |
|---------|-------------|
| Table Discovery | List all tables/views |
| Column Fetching | Get column details |
| Index Information | Get index and key info |
| Metric Discovery | Discover time-series metrics |
| Tag Extraction | Extract tags/labels |
| Caching | Cache metadata for performance |

### 2.2 Fetcher Interface

```python
class MetadataFetcher:
    def __init__(
        self,
        connectors: dict[str, DBConnector],
        cache: CacheManager,
    ):
        self.connectors = connectors
        self.cache = cache

    async def get_tables(
        self,
        database: str,
        include_views: bool = True,
    ) -> list[TableMetadata]:
        """Get all tables in database."""
        pass

    async def get_columns(
        self,
        database: str,
        table: str,
    ) -> list[ColumnMetadata]:
        """Get columns for a table."""
        pass

    async def get_metrics(
        self,
        database: str,
    ) -> list[MetricMetadata]:
        """Get time-series metrics."""
        pass

    async def get_tags(
        self,
        database: str,
        metric: str | None = None,
    ) -> list[str]:
        """Get available tags/labels."""
        pass

    async def get_table_size(
        self,
        database: str,
        table: str,
    ) -> TableSizeInfo:
        """Get table size information."""
        pass
```

### 2.3 Metadata Structures

```python
@dataclass
class TableMetadata:
    name: str
    schema: str
    type: Literal["table", "view", "materialized"]
    row_count: int | None
    size_bytes: int | None
    created_at: datetime | None
    columns: list[ColumnMetadata]

@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool
    default: Any | None
    is_primary_key: bool
    is_foreign_key: bool
    references: str | None

@dataclass
class MetricMetadata:
    name: str
    description: str | None
    tags: list[str]
    columns: list[ColumnMetadata]

@dataclass
class TableSizeInfo:
    total_bytes: int
    data_bytes: int
    index_bytes: int
    row_count: int
    last_updated: datetime | None
```

---

## 3. Technical Specification

### 3.1 Caching Strategy

- TTL: 10 minutes for tables/columns
- Invalidate on schema changes
- Manual refresh available

### 3.2 Performance

- Parallel fetching for multiple databases
- Lazy column loading
- Incremental updates

---

## 4. Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | Tables listed correctly |
| 2 | Columns fetched accurately |
| 3 | Metrics discovered |
| 4 | Tags extracted |
| 5 | Caching works |
| 6 | Cache invalidation works |
