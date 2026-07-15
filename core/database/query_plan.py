"""Structured query planning primitives for time-series database queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PlanSourceKind = Literal["table", "metric", "measurement", "view", "unknown"]
ExecutionStrategy = Literal["single_query", "multi_query_merge", "unknown"]
OutputShape = Literal["table", "scalar", "long_series", "wide_series", "unknown"]


@dataclass
class LinkedColumn:
    """A schema column/tag/label linked to the user's request."""

    name: str
    data_type: str = "unknown"
    role: str = "unknown"
    source: str | None = None
    confidence: float = 0.0


@dataclass
class LinkedSource:
    """A table, metric, measurement, or view linked to the user's request."""

    name: str
    kind: PlanSourceKind = "unknown"
    schema: str = ""
    time_column: str | None = None
    value_columns: list[str] = field(default_factory=list)
    dimension_columns: list[str] = field(default_factory=list)
    columns: list[LinkedColumn] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SchemaLinkingResult:
    """Natural-language references grounded against a concrete database schema."""

    contract_version: str = "schema_linking.v1"
    sources: list[LinkedSource] = field(default_factory=list)
    join_keys: list[str] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    value_columns: list[str] = field(default_factory=list)
    ambiguous_terms: dict[str, list[str]] = field(default_factory=dict)
    confidence: Literal["high", "medium", "low"] = "low"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class QuerySource:
    """One physical source referenced by a database query plan."""

    name: str
    kind: PlanSourceKind = "unknown"
    schema: str = ""
    alias: str | None = None
    time_column: str | None = None
    value_columns: list[str] = field(default_factory=list)


@dataclass
class QueryJoin:
    """A relationship between two sources in a query plan."""

    left: str
    right: str
    keys: list[str] = field(default_factory=list)
    type: Literal["inner", "left", "right", "full", "time_alignment"] = "inner"
    time_bucket: str | None = None


@dataclass
class QueryAlignment:
    """Time-series alignment required before returning or analyzing results."""

    method: Literal["none", "exact_time", "time_bucket", "asof", "application_merge"] = "none"
    time_column: str | None = None
    bucket: str | None = None
    keys: list[str] = field(default_factory=list)


@dataclass
class QueryProjection:
    """A selected expression or field in a query plan."""

    source: str | None = None
    column: str = ""
    alias: str | None = None
    aggregation: str | None = None


@dataclass
class QueryFilter:
    """A normalized filter predicate."""

    source: str | None = None
    column: str = ""
    operator: str = "="
    value: Any = None


@dataclass
class TimeRangePlan:
    """Time-window semantics for a query plan."""

    start: str | None = None
    end: str | None = None
    lookback: str | None = None
    timezone: str | None = None


@dataclass
class QueryBatchItem:
    """One labeled query candidate in a batch evidence request."""

    id: str
    label: str
    query: str = ""
    language: str | None = None
    measure: str | None = None
    aggregation: str | None = None
    time_range: TimeRangePlan = field(default_factory=TimeRangePlan)
    semantic_role: str = "comparison_candidate"


@dataclass
class QueryBatchMerge:
    """How query batch results should be normalized for downstream evidence."""

    mode: Literal["stack_rows", "unknown"] = "stack_rows"
    label_column: str = "candidate"
    value_column: str = "value"


@dataclass
class DatabaseQueryPlan:
    """Database-independent query intent grounded in linked schema."""

    contract_version: str = "database_query_plan.v1"
    sources: list[QuerySource] = field(default_factory=list)
    projections: list[QueryProjection] = field(default_factory=list)
    filters: list[QueryFilter] = field(default_factory=list)
    joins: list[QueryJoin] = field(default_factory=list)
    alignment: QueryAlignment = field(default_factory=QueryAlignment)
    time_range: TimeRangePlan = field(default_factory=TimeRangePlan)
    output_shape: OutputShape = "unknown"
    execution_strategy: ExecutionStrategy = "unknown"
    subqueries: list[QueryBatchItem] = field(default_factory=list)
    merge: QueryBatchMerge = field(default_factory=QueryBatchMerge)
    schema_linking: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        if not payload.get("schema_linking"):
            payload.pop("schema_linking", None)
        return payload


def query_plan_from_dict(payload: dict[str, Any] | None) -> DatabaseQueryPlan | None:
    """Best-effort conversion from a JSON payload to a query plan."""
    if not isinstance(payload, dict):
        return None
    try:
        return DatabaseQueryPlan(
            contract_version=str(payload.get("contract_version") or "database_query_plan.v1"),
            sources=[
                QuerySource(
                    name=str(item.get("name") or ""),
                    kind=item.get("kind") or "unknown",
                    schema=str(item.get("schema") or ""),
                    alias=item.get("alias"),
                    time_column=item.get("time_column"),
                    value_columns=list(item.get("value_columns") or []),
                )
                for item in list(payload.get("sources") or [])
                if isinstance(item, dict) and item.get("name")
            ],
            projections=[
                QueryProjection(
                    source=item.get("source"),
                    column=str(item.get("column") or ""),
                    alias=item.get("alias"),
                    aggregation=item.get("aggregation"),
                )
                for item in list(payload.get("projections") or [])
                if isinstance(item, dict) and item.get("column")
            ],
            filters=[
                QueryFilter(
                    source=item.get("source"),
                    column=str(item.get("column") or ""),
                    operator=str(item.get("operator") or "="),
                    value=item.get("value"),
                )
                for item in list(payload.get("filters") or [])
                if isinstance(item, dict) and item.get("column")
            ],
            joins=[
                QueryJoin(
                    left=str(item.get("left") or ""),
                    right=str(item.get("right") or ""),
                    keys=list(item.get("keys") or []),
                    type=item.get("type") or "inner",
                    time_bucket=item.get("time_bucket"),
                )
                for item in list(payload.get("joins") or [])
                if isinstance(item, dict) and item.get("left") and item.get("right")
            ],
            alignment=QueryAlignment(
                method=(payload.get("alignment") or {}).get("method") or "none",
                time_column=(payload.get("alignment") or {}).get("time_column"),
                bucket=(payload.get("alignment") or {}).get("bucket"),
                keys=list((payload.get("alignment") or {}).get("keys") or []),
            ),
            time_range=TimeRangePlan(
                start=(payload.get("time_range") or {}).get("start"),
                end=(payload.get("time_range") or {}).get("end"),
                lookback=(payload.get("time_range") or {}).get("lookback"),
                timezone=(payload.get("time_range") or {}).get("timezone"),
            ),
            output_shape=payload.get("output_shape") or "unknown",
            execution_strategy=payload.get("execution_strategy") or "unknown",
            subqueries=[
                QueryBatchItem(
                    id=str(item.get("id") or item.get("label") or f"query_{index + 1}"),
                    label=str(item.get("label") or item.get("id") or f"Query {index + 1}"),
                    query=str(item.get("query") or ""),
                    language=item.get("language"),
                    measure=item.get("measure"),
                    aggregation=item.get("aggregation"),
                    time_range=TimeRangePlan(
                        start=(item.get("time_range") or {}).get("start"),
                        end=(item.get("time_range") or {}).get("end"),
                        lookback=(item.get("time_range") or {}).get("lookback"),
                        timezone=(item.get("time_range") or {}).get("timezone"),
                    ),
                    semantic_role=str(item.get("semantic_role") or "comparison_candidate"),
                )
                for index, item in enumerate(list(payload.get("subqueries") or []))
                if isinstance(item, dict) and (item.get("id") or item.get("label") or item.get("query"))
            ],
            merge=QueryBatchMerge(
                mode=(payload.get("merge") or {}).get("mode") or "stack_rows",
                label_column=(payload.get("merge") or {}).get("label_column") or "candidate",
                value_column=(payload.get("merge") or {}).get("value_column") or "value",
            ),
            schema_linking=payload.get("schema_linking") if isinstance(payload.get("schema_linking"), dict) else None,
            notes=list(payload.get("notes") or []),
        )
    except Exception:
        return None


def schema_linking_from_dict(payload: dict[str, Any] | None) -> SchemaLinkingResult | None:
    """Best-effort conversion from a JSON payload to schema linking results."""
    if not isinstance(payload, dict):
        return None
    try:
        sources = []
        for item in list(payload.get("sources") or []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            sources.append(
                LinkedSource(
                    name=str(item.get("name") or ""),
                    kind=item.get("kind") or "unknown",
                    schema=str(item.get("schema") or ""),
                    time_column=item.get("time_column"),
                    value_columns=list(item.get("value_columns") or []),
                    dimension_columns=list(item.get("dimension_columns") or []),
                    columns=[
                        LinkedColumn(
                            name=str(column.get("name") or ""),
                            data_type=str(column.get("data_type") or "unknown"),
                            role=str(column.get("role") or "unknown"),
                            source=column.get("source"),
                            confidence=float(column.get("confidence") or 0.0),
                        )
                        for column in list(item.get("columns") or [])
                        if isinstance(column, dict) and column.get("name")
                    ],
                    confidence=float(item.get("confidence") or 0.0),
                )
            )
        return SchemaLinkingResult(
            contract_version=str(payload.get("contract_version") or "schema_linking.v1"),
            sources=sources,
            join_keys=list(payload.get("join_keys") or []),
            time_columns=list(payload.get("time_columns") or []),
            value_columns=list(payload.get("value_columns") or []),
            ambiguous_terms=dict(payload.get("ambiguous_terms") or {}),
            confidence=payload.get("confidence") or "low",
            evidence=list(payload.get("evidence") or []),
        )
    except Exception:
        return None
