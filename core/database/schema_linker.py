"""Schema linking for grounding user requests before query generation."""

from __future__ import annotations

import re
from typing import Any

from .query_plan import (
    DatabaseQueryPlan,
    LinkedColumn,
    LinkedSource,
    QueryAlignment,
    QueryJoin,
    QueryProjection,
    QuerySource,
    SchemaLinkingResult,
)


class SchemaLinker:
    """Ground natural-language query terms against a database schema."""

    TIME_HINTS = ("time", "timestamp", "date", "datetime", "_time")
    VALUE_HINTS = ("value", "_value", "score", "rate", "amount", "count", "temperature", "energy")
    DIMENSION_HINTS = ("id", "host", "device", "instance", "job", "region", "label", "tag")

    def link(
        self,
        *,
        user_message: str,
        schema: Any,
        db_type: str = "",
        dialect: str = "",
    ) -> SchemaLinkingResult:
        """Return schema objects that are relevant to the user request."""
        message = user_message or ""
        message_lower = message.lower()
        tables = list(getattr(schema, "tables", []) or [])
        linked_sources = []
        ambiguous_terms: dict[str, list[str]] = {}

        for table in tables:
            source = self._link_table(table, message_lower, db_type=db_type, dialect=dialect)
            if source:
                linked_sources.append(source)

        if not linked_sources and len(tables) == 1:
            source = self._source_from_table(tables[0], confidence=0.45, db_type=db_type, dialect=dialect)
            source.columns = self._link_columns(tables[0], message_lower, source.name)
            linked_sources.append(source)

        if not linked_sources:
            candidates = [str(getattr(table, "name", "")) for table in tables[:10] if getattr(table, "name", "")]
            if candidates:
                ambiguous_terms["source"] = candidates

        join_keys = self._common_join_keys(linked_sources)
        time_columns = self._unique(
            column
            for source in linked_sources
            for column in ([source.time_column] if source.time_column else [])
        )
        value_columns = self._unique(
            column
            for source in linked_sources
            for column in source.value_columns
        )
        evidence = []
        if linked_sources:
            evidence.append("sources linked from schema names, columns, and request terms")
        if join_keys:
            evidence.append("join key candidates inferred from shared dimension columns")
        if time_columns:
            evidence.append("time columns inferred from schema column names and types")

        confidence = "high" if linked_sources and not ambiguous_terms else "medium" if linked_sources else "low"
        return SchemaLinkingResult(
            sources=linked_sources,
            join_keys=join_keys,
            time_columns=time_columns,
            value_columns=value_columns,
            ambiguous_terms=ambiguous_terms,
            confidence=confidence,
            evidence=evidence,
        )

    def build_plan(
        self,
        *,
        linking: SchemaLinkingResult,
        output_shape: str = "unknown",
    ) -> DatabaseQueryPlan:
        """Build a conservative query plan from linked schema."""
        sources = [
            QuerySource(
                name=source.name,
                kind=source.kind,
                schema=source.schema,
                alias=self._alias_for_source(source.name, index),
                time_column=source.time_column,
                value_columns=source.value_columns,
            )
            for index, source in enumerate(linking.sources)
        ]
        projections = []
        for source in sources:
            if source.time_column:
                projections.append(QueryProjection(source=source.alias or source.name, column=source.time_column, alias="time"))
            for column in source.value_columns[:3]:
                projections.append(QueryProjection(source=source.alias or source.name, column=column))

        joins = []
        if len(sources) > 1:
            base = sources[0]
            for source in sources[1:]:
                joins.append(
                    QueryJoin(
                        left=base.alias or base.name,
                        right=source.alias or source.name,
                        keys=linking.join_keys,
                        type="inner" if linking.join_keys else "time_alignment",
                    )
                )

        alignment_method = "none"
        if len(sources) > 1:
            alignment_method = "exact_time" if linking.join_keys else "application_merge"

        execution_strategy = "single_query"
        if len(sources) > 1 and not linking.join_keys:
            execution_strategy = "multi_query_merge"
        elif not sources:
            execution_strategy = "unknown"

        return DatabaseQueryPlan(
            sources=sources,
            projections=projections,
            joins=joins,
            alignment=QueryAlignment(
                method=alignment_method,
                time_column=linking.time_columns[0] if linking.time_columns else None,
                keys=linking.join_keys,
            ),
            output_shape=output_shape if output_shape in {"table", "scalar", "long_series", "wide_series"} else "unknown",
            execution_strategy=execution_strategy,
            schema_linking=linking.to_dict(),
            notes=["Plan is grounded from schema linking and may be refined by dialect compiler."],
        )

    def _link_table(self, table: Any, message_lower: str, *, db_type: str, dialect: str) -> LinkedSource | None:
        name = str(getattr(table, "name", "") or "")
        if not name:
            return None
        columns = self._link_columns(table, message_lower, name)
        name_hit = self._contains_term(message_lower, name)
        column_hit = bool(columns)
        if not name_hit and not column_hit:
            return None
        confidence = 0.9 if name_hit else 0.65
        source = self._source_from_table(table, confidence=confidence, db_type=db_type, dialect=dialect)
        source.columns = columns
        return source

    def _source_from_table(self, table: Any, *, confidence: float, db_type: str, dialect: str) -> LinkedSource:
        columns = list(getattr(table, "columns", []) or [])
        time_column = self._detect_time_column(columns)
        value_columns = self._detect_value_columns(columns)
        dimension_columns = self._detect_dimension_columns(columns, time_column, value_columns)
        kind = self._source_kind(getattr(table, "type", ""), db_type=db_type, dialect=dialect)
        return LinkedSource(
            name=str(getattr(table, "name", "") or ""),
            kind=kind,
            schema=str(getattr(table, "schema", "") or ""),
            time_column=time_column,
            value_columns=value_columns,
            dimension_columns=dimension_columns,
            confidence=confidence,
        )

    def _link_columns(self, table: Any, message_lower: str, source_name: str) -> list[LinkedColumn]:
        linked = []
        columns = list(getattr(table, "columns", []) or [])
        for column in columns:
            name = str(getattr(column, "name", "") or "")
            if not name:
                continue
            if self._contains_term(message_lower, name):
                linked.append(
                    LinkedColumn(
                        name=name,
                        data_type=str(getattr(column, "data_type", "") or "unknown"),
                        role=self._column_role(column),
                        source=source_name,
                        confidence=0.8,
                    )
                )
        return linked

    def _source_kind(self, table_type: str, *, db_type: str, dialect: str) -> str:
        normalized = str(table_type or "").lower()
        if db_type == "prometheus" or dialect == "prometheus":
            return "metric"
        if dialect == "flux":
            return "measurement"
        if normalized in {"table", "view"}:
            return normalized
        return "unknown"

    def _detect_time_column(self, columns: list[Any]) -> str | None:
        for column in columns:
            name = str(getattr(column, "name", "") or "")
            data_type = str(getattr(column, "data_type", "") or "").lower()
            if self._is_time_column(name, data_type):
                return name
        return None

    def _detect_value_columns(self, columns: list[Any]) -> list[str]:
        values = []
        for column in columns:
            name = str(getattr(column, "name", "") or "")
            data_type = str(getattr(column, "data_type", "") or "").lower()
            if self._is_time_column(name, data_type):
                continue
            if self._is_numeric_type(data_type) or any(hint in name.lower() for hint in self.VALUE_HINTS):
                values.append(name)
        return values

    def _detect_dimension_columns(
        self,
        columns: list[Any],
        time_column: str | None,
        value_columns: list[str],
    ) -> list[str]:
        dimensions = []
        value_set = set(value_columns)
        for column in columns:
            name = str(getattr(column, "name", "") or "")
            data_type = str(getattr(column, "data_type", "") or "").lower()
            if name == time_column or name in value_set:
                continue
            if (
                not self._is_numeric_type(data_type)
                or any(hint in name.lower() for hint in self.DIMENSION_HINTS)
                or name.startswith("label_")
            ):
                dimensions.append(name.removeprefix("label_"))
        return dimensions

    def _common_join_keys(self, sources: list[LinkedSource]) -> list[str]:
        if len(sources) < 2:
            return []
        common = set(sources[0].dimension_columns)
        for source in sources[1:]:
            common &= set(source.dimension_columns)
        return sorted(common)

    def _column_role(self, column: Any) -> str:
        name = str(getattr(column, "name", "") or "")
        data_type = str(getattr(column, "data_type", "") or "").lower()
        if self._is_time_column(name, data_type):
            return "time"
        if self._is_numeric_type(data_type) or any(hint in name.lower() for hint in self.VALUE_HINTS):
            return "value"
        return "dimension"

    def _is_time_column(self, name: str, data_type: str) -> bool:
        normalized = name.lower()
        return any(hint == normalized or hint in normalized for hint in self.TIME_HINTS) or "time" in data_type

    def _is_numeric_type(self, data_type: str) -> bool:
        return any(token in data_type for token in ("int", "float", "double", "decimal", "numeric", "real"))

    def _contains_term(self, message_lower: str, term: str) -> bool:
        normalized = str(term or "").strip().lower()
        if not normalized:
            return False
        if normalized in message_lower:
            return True
        compact = normalized.replace("_", " ").replace("-", " ")
        return compact != normalized and compact in message_lower

    def _alias_for_source(self, name: str, index: int) -> str:
        words = re.findall(r"[A-Za-z0-9]+", name)
        if words:
            alias = "".join(word[0].lower() for word in words[:3])
            if alias:
                return f"{alias}{index + 1}"
        return f"s{index + 1}"

    def _unique(self, values: Any) -> list[str]:
        result = []
        for value in values:
            if value and value not in result:
                result.append(str(value))
        return result

