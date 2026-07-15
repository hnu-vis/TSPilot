# Result Processor Specification

## 1. Project Overview

**Project Name:** TSPilot
**Module Name:** ResultProcessor
**Type:** Core Module - Database (Python)
**File Path:** `core/database/result_processor.py`
**Core Functionality:** Standardizes and transforms query results for consistent output.
**Target Users:** Query agent, API layer.

---

## 2. Functionality Specification

### 2.1 Core Features

| Feature | Description |
|---------|-------------|
| Result Normalization | Convert to standard format |
| Type Coercion | Convert string types to appropriate types |
| Timezone Handling | Normalize timestamps |
| Aggregation | Apply client-side aggregations |
| Filtering | Apply result filters |
| Pagination | Paginate large results |

### 2.2 Processor Interface

```python
class ResultProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config

    def process(self, result: QueryResult) -> ProcessedResult:
        """Process query result."""
        pass

    def normalize_types(
        self,
        rows: list[dict],
        schema: TableSchema,
    ) -> list[dict]:
        """Convert string types to proper types."""
        pass

    def normalize_timestamps(
        self,
        rows: list[dict],
        timezone: str = "UTC",
    ) -> list[dict]:
        """Normalize all timestamps to UTC."""
        pass

    def aggregate(
        self,
        result: ProcessedResult,
        aggregations: list[Aggregation],
    ) -> AggregatedResult:
        """Apply aggregations to result."""
        pass

    def paginate(
        self,
        result: ProcessedResult,
        page: int,
        page_size: int,
    ) -> PaginatedResult:
        """Paginate results."""
        pass
```

### 2.3 Processed Result

```python
@dataclass
class ProcessedResult:
    columns: list[Column]
    rows: list[dict]
    row_count: int
    execution_time_ms: int
    truncated: bool
    metadata: ResultMetadata

@dataclass
class Column:
    name: str
    type: str  # python type name
    nullable: bool

@dataclass
class ResultMetadata:
    database: str
    table: str | None
    query_id: str
    cached: bool
    truncated_rows: int | None
```

### 2.4 Type Mapping

| Database Type | Python Type |
|---------------|-------------|
| INTEGER | int |
| BIGINT | int |
| DOUBLE | float |
| VARCHAR | str |
| TIMESTAMP | datetime |
| BOOLEAN | bool |
| JSON | dict |

---

## 3. Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | Types correctly coerced |
| 2 | Timestamps normalized |
| 3 | Aggregations work |
| 4 | Pagination correct |
| 5 | Large results truncated |
