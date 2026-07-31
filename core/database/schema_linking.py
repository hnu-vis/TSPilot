"""Unified schema linking pipeline for grounded query planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.time_range import normalize_time_range

from .connector import DatabaseSchema
from .contracts import FieldMappingCandidate, QueryIntent, QueryRequestContext
from .query_plan import DatabaseQueryPlan, QueryFilter, QueryProjection, SchemaLinkingResult, TimeRangePlan
from .schema_linker import SchemaLinker


@dataclass
class SchemaLinkingPipelineResult:
    """Grounded schema linking output used by query planning and validation."""

    linking: SchemaLinkingResult
    field_mappings: list[FieldMappingCandidate]
    plan: DatabaseQueryPlan
    required_filters: list[QueryFilter]
    candidate_filters: list[dict[str, Any]] = field(default_factory=list)
    measure_mappings: list[dict[str, Any]] = field(default_factory=list)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_linking": self.linking.to_dict(),
            "field_mappings": [
                {
                    "user_term": item.user_term,
                    "source_name": item.source_name,
                    "field_name": item.field_name,
                    "role": item.role,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in self.field_mappings
            ],
            "required_filters": [
                {
                    "source": item.source,
                    "column": item.column,
                    "operator": item.operator,
                    "value": item.value,
                }
                for item in self.required_filters
            ],
            "candidate_filters": self.candidate_filters or [],
            **({"measure_mappings": self.measure_mappings} if self.measure_mappings else {}),
        }


class SchemaLinkingPipeline:
    """Single entrypoint for linking schema, fields, filters, and plan shape."""

    _NON_VALUE_DOMAIN_FILTER_COLUMNS = {"_measurement", "_field", "_start", "_stop", "result", "table"}
    _NON_REQUIRED_FILTER_COLUMNS = {
        "_measurement",
        "_field",
        "_time",
        "time",
        "timestamp",
        "_start",
        "_stop",
        "result",
        "table",
    }

    def __init__(self, linker: SchemaLinker | None = None):
        self._linker = linker or SchemaLinker()

    def ground(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
    ) -> SchemaLinkingPipelineResult:
        linking = self._linker.link(
            user_message=context.message,
            schema=schema,
            db_type=context.database_type,
            dialect=context.database_type,
        )
        field_mappings = self.map_fields_from_linking(
            context=context,
            schema=schema,
            linking=linking,
        )
        plan = self.build_plan_from_linking(
            context=context,
            schema=schema,
            intent=intent,
            linking=linking,
            field_mappings=field_mappings,
        )
        required_filters = self.required_filters(plan)
        candidate_filters = self.candidate_filters(schema=schema, plan=plan)
        measure_mappings = self.measure_mappings(schema=schema, linking=linking)
        return SchemaLinkingPipelineResult(
            linking=linking,
            field_mappings=field_mappings,
            plan=plan,
            required_filters=required_filters,
            candidate_filters=candidate_filters,
            measure_mappings=measure_mappings,
        )

    def map_fields(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
    ) -> list[FieldMappingCandidate]:
        linking = self._linker.link(
            user_message=context.message,
            schema=schema,
            db_type=context.database_type,
            dialect=context.database_type,
        )
        return self.map_fields_from_linking(context=context, schema=schema, linking=linking)

    def map_fields_from_linking(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        linking: SchemaLinkingResult,
    ) -> list[FieldMappingCandidate]:
        candidates: list[FieldMappingCandidate] = []
        for source in linking.sources:
            linked_columns = sorted(
                source.columns,
                key=lambda column: (0 if column.role == "value" else 1 if column.role == "dimension" else 2, -column.confidence),
            )
            for column in linked_columns:
                candidates.append(
                    FieldMappingCandidate(
                        user_term=column.name,
                        source_name=source.name,
                        field_name=column.name,
                        role=column.role,
                        confidence=column.confidence,
                        evidence=["linked_from_request"],
                    )
                )
            if not source.columns and source.value_columns:
                candidates.append(
                    FieldMappingCandidate(
                        user_term=context.message,
                        source_name=source.name,
                        field_name=source.value_columns[0],
                        role="value",
                        confidence=source.confidence,
                        evidence=["fallback_first_value_column"],
                    )
                )

        if not candidates and schema.tables:
            first = schema.tables[0]
            numeric = [
                column.name
                for column in first.columns
                if str(column.data_type).lower() in {"float", "double", "integer", "int", "numeric"}
            ]
            if not numeric:
                numeric = [column.name for column in first.columns if column.name not in {"_time", "time", "timestamp"}][:1]
            for column_name in numeric[:1]:
                candidates.append(
                    FieldMappingCandidate(
                        user_term=context.message,
                        source_name=first.name,
                        field_name=column_name,
                        role="value",
                        confidence=0.35,
                        evidence=["fallback_first_schema_source"],
                    )
                )
        return candidates

    def build_plan(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
        field_mappings: list[FieldMappingCandidate],
    ) -> DatabaseQueryPlan:
        linking = self._linker.link(
            user_message=context.message,
            schema=schema,
            db_type=context.database_type,
            dialect=context.database_type,
        )
        return self.build_plan_from_linking(
            context=context,
            schema=schema,
            intent=intent,
            linking=linking,
            field_mappings=field_mappings,
        )

    def build_plan_from_linking(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
        linking: SchemaLinkingResult,
        field_mappings: list[FieldMappingCandidate],
    ) -> DatabaseQueryPlan:
        output_shape = "long_series" if intent.query_shape == "raw_timeseries" else "scalar" if intent.query_shape == "scalar_aggregate" else "table"
        plan = self._linker.build_plan(linking=linking, output_shape=output_shape)
        if not plan.sources and schema.tables:
            first = schema.tables[0]
            fallback_linking = self._linker.link(
                user_message=first.name,
                schema=schema,
                db_type=context.database_type,
                dialect=context.database_type,
            )
            plan = self._linker.build_plan(linking=fallback_linking, output_shape=output_shape)
        self._apply_time_range(plan, context.time_range)
        self._apply_projections(plan, intent, field_mappings)
        self._apply_value_filters(plan, context, schema)
        plan.notes.extend(intent.notes)
        if context.constraints.get("max_points"):
            plan.notes.append(f"max_points={int(context.constraints['max_points'])}")
        return plan

    def required_filters(self, plan: DatabaseQueryPlan) -> list[QueryFilter]:
        return [
            item for item in plan.filters
            if item.column not in self._NON_REQUIRED_FILTER_COLUMNS
        ]

    def candidate_filters(self, *, schema: DatabaseSchema, plan: DatabaseQueryPlan) -> list[dict[str, Any]]:
        value_domains = schema.metadata.get("value_domains")
        if not isinstance(value_domains, dict):
            return []
        source_names = [source.name for source in plan.sources] or [table.name for table in schema.tables[:1]]
        candidates = []
        for source_name in source_names:
            domains = value_domains.get(source_name)
            if not isinstance(domains, dict):
                continue
            for column_name, values in domains.items():
                if column_name in self._NON_VALUE_DOMAIN_FILTER_COLUMNS:
                    continue
                if not isinstance(values, list) or not values:
                    continue
                candidates.append(
                    {
                        "source": source_name,
                        "column": str(column_name),
                        "operator": "=",
                        "values": [str(value) for value in values[:20] if value not in (None, "")],
                    }
                )
        return candidates

    def measure_mappings(self, *, schema: DatabaseSchema, linking: SchemaLinkingResult) -> list[dict[str, Any]]:
        value_domains = schema.metadata.get("value_domains")
        if not isinstance(value_domains, dict):
            return []
        mappings: list[dict[str, Any]] = []
        linked_source_names = {source.name for source in linking.sources}
        for source_name, domains in value_domains.items():
            if linked_source_names and source_name not in linked_source_names:
                continue
            if not isinstance(domains, dict):
                continue
            field_values = domains.get("_field")
            if not isinstance(field_values, list) or not field_values:
                continue
            for logical_measure in field_values:
                if logical_measure in (None, ""):
                    continue
                mappings.append(
                    {
                        "source": source_name,
                        "logical_measure": str(logical_measure),
                        "selector_column": "_field",
                        "selector_value": str(logical_measure),
                        "physical_value_column": "_value",
                        "aggregate_column": "_value",
                        "time_column": "_time",
                    }
                )
        return mappings

    def _apply_time_range(self, plan: DatabaseQueryPlan, time_range: dict[str, Any] | None) -> None:
        if not time_range:
            return
        normalized = normalize_time_range(time_range) or {}
        plan.time_range = TimeRangePlan(
            start=normalized.get("start"),
            end=normalized.get("end"),
            timezone=normalized.get("timezone"),
        )

    def _apply_projections(
        self,
        plan: DatabaseQueryPlan,
        intent: QueryIntent,
        field_mappings: list[FieldMappingCandidate],
    ) -> None:
        if not plan.sources:
            return
        source = plan.sources[0]
        value_candidates = [item for item in field_mappings if item.role == "value"]
        chosen_field = (
            value_candidates[0].field_name
            if value_candidates
            else field_mappings[0].field_name
            if field_mappings
            else (source.value_columns[0] if source.value_columns else "")
        )
        time_column = source.time_column or "timestamp"
        alias = source.alias or source.name
        aggregation = str(intent.filters.get("aggregation") or "")
        if intent.query_shape == "scalar_aggregate" and chosen_field:
            plan.projections = [
                QueryProjection(
                    source=alias,
                    column=chosen_field,
                    alias=f"{aggregation or 'value'}_{chosen_field}",
                    aggregation=aggregation or "avg",
                )
            ]
            return

        projections: list[QueryProjection] = []
        if time_column:
            projections.append(QueryProjection(source=alias, column=time_column, alias="timestamp"))
        if chosen_field:
            projections.append(QueryProjection(source=alias, column=chosen_field, alias="value"))
        elif source.value_columns:
            projections.append(QueryProjection(source=alias, column=source.value_columns[0], alias="value"))
        plan.projections = projections
        plan.alignment.time_column = time_column

    def _apply_value_filters(
        self,
        plan: DatabaseQueryPlan,
        context: QueryRequestContext,
        schema: DatabaseSchema,
    ) -> None:
        if not plan.sources:
            return
        value_domains = schema.metadata.get("value_domains")
        if not isinstance(value_domains, dict):
            return
        source = plan.sources[0]
        domains = value_domains.get(source.name)
        if not isinstance(domains, dict):
            return
        message_lower = context.message.lower()
        existing = {(item.column, str(item.value).lower()) for item in plan.filters}
        source_alias = source.alias or source.name
        for column_name, values in domains.items():
            if column_name in self._NON_VALUE_DOMAIN_FILTER_COLUMNS:
                continue
            matched_value = self._match_domain_value(message_lower, values)
            if not matched_value:
                continue
            key = (str(column_name), matched_value.lower())
            if key in existing:
                continue
            plan.filters.append(
                QueryFilter(
                    source=source_alias,
                    column=str(column_name),
                    operator="=",
                    value=matched_value,
                )
            )
            existing.add(key)

    def _match_domain_value(self, message_lower: str, values: Any) -> str | None:
        if not isinstance(values, list):
            return None
        normalized_values = [
            str(value).strip()
            for value in values
            if value not in (None, "")
        ]
        for candidate in sorted(normalized_values, key=len, reverse=True):
            if candidate.lower() in message_lower:
                return candidate
        return None
