"""Decoupled contracts for backend-agnostic database query planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from schemas.database import DatabaseEvidence

from .connector import DatabaseSchema, QueryResult
from .query_plan import DatabaseQueryPlan


@dataclass
class QueryIntent:
    """Backend-agnostic interpretation of a user database request."""

    requested_fact_families: list[str] = field(default_factory=list)
    evidence_family: str = "timeseries"
    query_shape: str = "raw_timeseries"
    focus: str | None = None
    metrics: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    requires_raw_points: bool = False
    requires_full_fidelity: bool = False


@dataclass
class FieldMappingCandidate:
    """Grounded mapping from a user-facing concept to a backend object."""

    user_term: str
    source_name: str
    field_name: str
    role: str = "value"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class QueryRequestContext:
    """All runtime context needed for query planning."""

    database_id: str
    database_type: str
    message: str
    time_range: dict[str, Any] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    intent_profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedQuery:
    """Backend-specific physical query or request payload."""

    query_text: str = ""
    query_language: str = ""
    structured_request: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """One validation or repair issue for a rendered query."""

    code: str
    message: str
    severity: str = "error"


@dataclass
class QueryValidationReport:
    """Deterministic validation result for a rendered query."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    safe_to_repair: bool = False


@dataclass
class QueryRepairDecision:
    """Safe repair decision for a rendered query."""

    should_retry: bool = False
    reason: str = ""
    replacement_query: RenderedQuery | None = None


@dataclass
class QueryExecutionTrace:
    """Trace payload that makes database execution inspectable in E2E tests."""

    adapter_type: str
    logical_plan: dict[str, Any]
    rendered_query: dict[str, Any]
    schema_summary: dict[str, Any] = field(default_factory=dict)
    field_mappings: list[dict[str, Any]] = field(default_factory=list)
    snapshot_ref: dict[str, Any] | None = None
    repaired: bool = False
    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    raw_result_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryArtifactRef:
    """Stable reference to a stored full query result or execution snapshot."""

    artifact_id: str
    artifact_kind: str
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SchemaCatalog(Protocol):
    """Load and expose grounded datasource schema metadata."""

    async def load_schema(self, *, context: QueryRequestContext) -> DatabaseSchema:
        """Return the datasource schema or metric catalog."""


class IntentInterpreter(Protocol):
    """Interpret the user request into backend-agnostic intent."""

    def interpret(self, *, context: QueryRequestContext) -> QueryIntent:
        """Return deterministic intent metadata for planning."""


class FieldMapper(Protocol):
    """Resolve user-facing concepts onto grounded schema fields."""

    def map_fields(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
    ) -> list[FieldMappingCandidate]:
        """Return grounded field candidates."""


class LogicalQueryPlanner(Protocol):
    """Build a backend-agnostic database query plan."""

    def build_plan(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
        field_mappings: list[FieldMappingCandidate],
    ) -> DatabaseQueryPlan:
        """Return a logical query plan."""


class DialectRenderer(Protocol):
    """Render logical plans into backend-specific query text or requests."""

    def render(
        self,
        *,
        context: QueryRequestContext,
        plan: DatabaseQueryPlan,
    ) -> RenderedQuery:
        """Return a backend-specific query representation."""


class QueryExecutor(Protocol):
    """Execute one rendered database query."""

    async def execute(
        self,
        *,
        context: QueryRequestContext,
        rendered_query: RenderedQuery,
    ) -> QueryResult:
        """Run the backend query and return raw results."""


class QueryValidator(Protocol):
    """Validate whether the rendered query matches the logical plan."""

    def validate(
        self,
        *,
        context: QueryRequestContext,
        plan: DatabaseQueryPlan,
        rendered_query: RenderedQuery,
        result: QueryResult | None = None,
    ) -> QueryValidationReport:
        """Return deterministic validation feedback."""


class QueryRepairPolicy(Protocol):
    """Decide whether a rendered query can be safely repaired."""

    def decide(
        self,
        *,
        context: QueryRequestContext,
        plan: DatabaseQueryPlan,
        rendered_query: RenderedQuery,
        validation: QueryValidationReport,
    ) -> QueryRepairDecision:
        """Return a safe repair decision."""


class QueryResultSnapshotStore(Protocol):
    """Persist and retrieve full query results outside the model prompt."""

    def store(
        self,
        *,
        context: QueryRequestContext,
        rendered_query: RenderedQuery,
        result: QueryResult,
    ) -> QueryArtifactRef:
        """Persist one full query result and return a stable reference."""


@runtime_checkable
class ValueDomainProbe(Protocol):
    """Optional connector capability for probing dimension/tag value domains."""

    async def probe_value_domains(
        self,
        *,
        source_name: str,
        columns: list[str],
        limit: int = 100,
    ) -> dict[str, list[str]]:
        """Return candidate values for the requested source columns."""


class EvidenceNormalizer(Protocol):
    """Convert raw backend results into DatabaseEvidence."""

    def normalize(
        self,
        *,
        context: QueryRequestContext,
        rendered_query: RenderedQuery,
        result: QueryResult,
        plan: DatabaseQueryPlan,
    ) -> DatabaseEvidence:
        """Return normalized evidence."""
