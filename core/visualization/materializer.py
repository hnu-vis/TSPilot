"""Grounded visualization catalog, semantic projection executor, and materializer."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from schemas.key_insight import KeyInsight, InsightItem
from schemas.output import VisualGoal, VisualLayerPlan
from schemas.state import RequestStateModel
from schemas.visual_verification import VisualizationVerification
from schemas.visualization import (
    VisualizationAccessibility,
    VisualizationBinding,
    VisualizationDataset,
    VisualizationDimension,
    VisualizationLayer,
    VisualizationPayload,
    VisualizationPoint,
    VisualizationSeries,
)


_DATA_BEARING_PRESENTATION_KEYS = {
    "data", "source", "dataset", "datasetid", "encode", "dimensions", "series",
}


@dataclass(frozen=True)
class DataViewValue:
    rows: list[dict]
    scalar: dict | None
    shape: str
    schema_fields: list[dict]
    lineage: list[str]
    name: str
    field_semantics: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, VisualizationBinding] = field(default_factory=dict)


@dataclass(frozen=True)
class PresentationSource:
    ref: str
    kind: str
    value: Any
    reference: "ReferencePresentation | None" = None


@dataclass(frozen=True)
class ReferencePresentation:
    """Storage-independent, bounded projection for final-answer references."""

    source_type: str
    source_id: str
    label: str
    evidence: dict | None


class InvalidPresentationLineageError(ValueError):
    def __init__(self, *, view_ref: str, lineage_ref: str):
        self.view_ref = view_ref
        self.lineage_ref = lineage_ref
        super().__init__(f"data view '{view_ref}' contains unknown lineage source '{lineage_ref}'")


class PresentationCatalog:
    """Index artifacts, insights, insight items, and explicit typed data views."""

    def __init__(self, request_state: RequestStateModel):
        self.request_state = request_state
        self._sources: dict[str, PresentationSource] = {}
        for evidence_id, evidence in request_state.database_evidence_artifacts.items():
            self._register_artifact(
                "evidence", evidence_id, evidence,
                source_type="query", label=evidence.summary or evidence.result_type,
            )
            rows = _rows_from_evidence(evidence)
            self._register_view(
                f"view:evidence:{evidence_id}:default",
                name=evidence.summary or evidence_id,
                shape="timeseries" if _candidate_fields(rows, "time") else "records",
                rows=rows,
                scalar=None,
                lineage=[f"evidence:{evidence_id}"],
            )
        for analysis_id, analysis in request_state.analysis_artifacts.items():
            self._register_artifact(
                "analysis", analysis_id, analysis,
                source_type="analysis", label=analysis.analysis_goal,
            )
        for evidence_id, evidence in request_state.derived_evidence_artifacts.items():
            self._register_artifact(
                "derived_evidence", evidence_id, evidence,
                source_type="derived_evidence", label=evidence.name,
            )
            self._register_view(
                f"view:derived_evidence:{evidence_id}", name=evidence.name, shape=evidence.shape,
                rows=[dict(row) for row in evidence.rows], scalar=evidence.scalar,
                lineage=[f"derived_evidence:{evidence_id}", *evidence.lineage],
            )
        for forecast_id, forecast in request_state.forecast_artifacts.items():
            self._register_artifact(
                "forecast", forecast_id, forecast,
                source_type="forecast", label=forecast.model_name,
            )
            rows = [{"timestamp": point.timestamp, "value": point.value} for point in forecast.forecast_points]
            self._register_view(
                f"view:forecast:{forecast_id}:points", name="Forecast", shape="timeseries", rows=rows,
                scalar=None, lineage=[f"forecast:{forecast_id}", *_forecast_input_refs(forecast)],
            )
            interval_rows = [
                {
                    "timestamp": _object_value(item, "timestamp"),
                    "lower": _object_value(item, "lower"),
                    "upper": _object_value(item, "upper"),
                }
                for item in forecast.confidence_interval
            ]
            if interval_rows:
                self._register_view(
                    f"view:forecast:{forecast_id}:interval", name="Confidence interval", shape="intervals",
                    rows=interval_rows, scalar=None, lineage=[f"forecast:{forecast_id}"],
                )
            diagnostics = forecast.diagnostics if isinstance(forecast.diagnostics, dict) else {}
            quality = diagnostics.get("input_quality") if isinstance(diagnostics.get("input_quality"), dict) else None
            if quality:
                self._register_view(
                    f"view:forecast:{forecast_id}:quality", name="Forecast input quality", shape="scalar",
                    rows=[], scalar=dict(quality), lineage=[f"forecast:{forecast_id}"],
                )
        for anomaly_id, anomaly in request_state.anomaly_artifacts.items():
            self._register_artifact(
                "anomaly", anomaly_id, anomaly,
                source_type="anomaly", label=anomaly.detector_name,
            )
            rows = [dict(item) for item in anomaly.anomaly_points if isinstance(item, dict)]
            evidence_ref = _anomaly_evidence_ref(anomaly)
            lineage = [f"anomaly:{anomaly_id}", *([evidence_ref] if evidence_ref else [])]
            self._register_view(
                f"view:anomaly:{anomaly_id}:status", name="Anomaly detection status", shape="scalar",
                rows=[], scalar={"detected_count": len(rows)}, lineage=lineage,
            )
            if rows:
                self._register_view(
                    f"view:anomaly:{anomaly_id}:points", name="Anomaly points", shape="records",
                    rows=rows, scalar=None, lineage=lineage,
                )
            spans = [dict(item) for item in anomaly.anomaly_spans if isinstance(item, dict)]
            if spans:
                self._register_view(
                    f"view:anomaly:{anomaly_id}:spans", name="Anomaly spans", shape="intervals",
                    rows=spans, scalar=None, lineage=lineage,
                )
            scores = [dict(item) for item in anomaly.scores if isinstance(item, dict)]
            if scores:
                self._register_view(
                    f"view:anomaly:{anomaly_id}:scores", name="Anomaly scores", shape="timeseries",
                    rows=scores, scalar=None, lineage=lineage,
                )
        for insight in request_state.insight_set.insights:
            self._register_artifact(
                "insight", insight.insight_id, insight,
                source_type="insight", label=insight.name,
            )
            insight_source = self._sources[f"insight:{insight.insight_id}"]
            # The outer ReAct model reasons with semantic insight keys while the
            # presentation layer persists canonical insight ids. Resolve either
            # form to the same canonical source without exposing more paths.
            if insight.insight_key:
                self._sources.setdefault(f"insight:{insight.insight_key}", insight_source)
                self._sources.setdefault(insight.insight_key, insight_source)
            for item in insight.items:
                ref = f"insight:{insight.insight_id}#{item.item_id}"
                self._sources[ref] = PresentationSource(
                    ref,
                    "insight_item",
                    (insight, item),
                    ReferencePresentation(
                        source_type="insight",
                        source_id=insight.insight_id,
                        label=item.label or insight.name,
                        evidence=_bounded_presentation_value({
                            "parent_ref": f"insight:{insight.insight_id}",
                            "parent": insight.model_dump(mode="json", exclude={"items"}),
                            "item": item.model_dump(mode="json"),
                        }),
                    ),
                )

    def _register_artifact(
        self,
        kind: str,
        source_id: str,
        value: Any,
        *,
        source_type: str,
        label: str,
    ) -> None:
        ref = f"{kind}:{source_id}"
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        source = PresentationSource(
            ref,
            kind,
            value,
            ReferencePresentation(
                source_type=source_type,
                source_id=source_id,
                label=str(label or ref),
                evidence=_bounded_presentation_value(payload),
            ),
        )
        self._sources[ref] = source
        self._sources.setdefault(source_id, source)

    def _register_view(
        self, ref: str, *, name: str, shape: str, rows: list[dict], scalar: dict | None,
        lineage: list[str], schema_fields: list[dict] | None = None,
        field_semantics: dict[str, str] | None = None,
        bindings: dict[str, VisualizationBinding] | None = None,
    ) -> None:
        fields = schema_fields or _schema_fields(rows or ([scalar] if scalar else []))
        self._sources[ref] = PresentationSource(
            ref,
            "view",
            DataViewValue(
                rows,
                scalar,
                shape,
                fields,
                list(dict.fromkeys(lineage)),
                name,
                dict(field_semantics or {}),
                dict(bindings or {}),
            ),
        )

    def resolve(self, ref: str) -> PresentationSource:
        source = self._sources.get(str(ref or "").strip())
        if source is None:
            available = sorted(key for key in self._sources if ":" in key)
            raise ValueError(f"unknown presentation source '{ref}'. Available source_refs: {available}")
        return source

    def reference_presentation(self, ref_or_source: str | PresentationSource) -> ReferencePresentation:
        source = self.resolve(ref_or_source) if isinstance(ref_or_source, str) else ref_or_source
        if source.reference is None:
            raise ValueError(f"presentation source '{source.ref}' is not an answer reference")
        return source.reference

    def analysis_input_evidence_id(self, ref: str | None) -> str | None:
        """Resolve a presentation ref to the one database evidence id it derives from."""
        if not ref:
            return None
        source = self.resolve(ref)
        candidates: set[str] = set()
        pending = [source]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current.ref in visited:
                continue
            visited.add(current.ref)
            if current.kind == "evidence":
                evidence_id = str(getattr(current.value, "evidence_id", "") or "")
                if evidence_id:
                    candidates.add(evidence_id)
                continue
            if current.kind == "view":
                pending.extend(self.resolve_lineage(current))
        candidates.discard("")
        if len(candidates) != 1:
            raise ValueError(
                f"presentation source '{source.ref}' does not resolve to exactly one database evidence input"
            )
        return next(iter(candidates))

    def analysis_input_source_refs(self, refs: list[str]) -> list[str]:
        """Resolve presentation refs to authoritative analysis artifact refs."""

        result: list[str] = []
        pending = [self.resolve(ref) for ref in refs if ref]
        visited: set[str] = set()
        while pending:
            source = pending.pop()
            if source.ref in visited:
                continue
            visited.add(source.ref)
            if source.kind == "view":
                pending.extend(reversed(self.resolve_lineage(source)))
                continue
            if source.kind in {"evidence", "forecast", "anomaly", "derived_evidence", "insight"}:
                result.append(source.ref)
        return list(dict.fromkeys(result))

    def canonical_refs(self) -> list[str]:
        return sorted(key for key, source in self._sources.items() if key == source.ref and ":" in key)

    def renderable_refs(self) -> list[str]:
        return sorted(
            ref
            for ref, source in self._sources.items()
            if ref == source.ref and self._is_renderable_source(source)
        )

    def projection_refs(self) -> list[str]:
        """Return grounded sources that the semantic projection LLM may interpret."""
        return sorted(
            ref
            for ref, source in self._sources.items()
            if ref == source.ref and self._is_projection_source(source)
        )

    @staticmethod
    def _is_projection_source(source: PresentationSource) -> bool:
        if source.kind == "insight":
            return source.value.status == "verified"
        if source.kind == "insight_item":
            return source.value[0].status == "verified"
        return source.kind == "view"

    @staticmethod
    def _is_renderable_source(source: PresentationSource) -> bool:
        if source.kind == "insight" and source.value.status != "verified":
            return False
        if source.kind == "insight_item" and source.value[0].status != "verified":
            return False
        if source.kind not in {"view", "insight", "insight_item"}:
            return False
        rows, scalar = _source_data(source)
        fields = _schema_fields(rows or ([scalar] if scalar else []))
        return bool(_render_capabilities(fields, scalar=not bool(rows))["renderable"])

    def expand_preferences(self, refs: list[str]) -> tuple[set[str], set[str]]:
        """Expand stable artifact refs into renderable sources without exposing storage refs."""

        renderable = set(self.projection_refs())
        expanded: set[str] = set()
        unknown: set[str] = set()
        for raw_ref in refs:
            ref = str(raw_ref or "").strip()
            if not ref:
                continue
            source = self._sources.get(ref)
            if source is None:
                unknown.add(ref)
                continue
            if source.ref in renderable:
                expanded.add(source.ref)
                if source.kind == "insight":
                    expanded.update(self._insight_context_refs(source.value, renderable))
                continue
            expanded.update(self._views_for_lineage_ref(source.ref, renderable))
            if source.kind == "analysis":
                expanded.update(self._analysis_renderable_refs(source.value, renderable))
        return expanded, unknown

    def _insight_context_refs(self, insight: KeyInsight, renderable: set[str]) -> set[str]:
        """Resolve an upstream Insight's lineage into its related visual data."""

        refs: set[str] = set()
        for evidence_ref in insight.evidence_refs:
            source_type = str(evidence_ref.source_type or "").strip()
            source_id = str(evidence_ref.source_id or "").strip()
            if not source_id:
                continue
            kind = "evidence" if source_type in {"query", "database"} else source_type
            artifact_ref = f"{kind}:{source_id}"
            source = self._sources.get(artifact_ref)
            if source is None:
                continue
            refs.update(self._views_for_lineage_ref(source.ref, renderable))
            if source.kind == "analysis":
                refs.update(self._analysis_context_view_refs(source.value, renderable))
        return refs

    def _views_for_lineage_ref(self, artifact_ref: str, renderable: set[str]) -> set[str]:
        return {
            ref
            for ref in renderable
            if (source := self._sources.get(ref)) is not None
            and source.kind == "view"
            and artifact_ref in source.value.lineage
        }

    def _analysis_renderable_refs(self, analysis, renderable: set[str]) -> set[str]:
        refs = self._analysis_context_view_refs(analysis, renderable)
        for insight in getattr(analysis, "produced_insights", []) or []:
            candidate = f"insight:{getattr(insight, 'insight_id', '')}"
            if candidate in renderable:
                refs.add(candidate)
            refs.update(ref for ref in renderable if ref.startswith(f"{candidate}#"))
        return refs

    def _analysis_context_view_refs(self, analysis, renderable: set[str]) -> set[str]:
        refs: set[str] = set()
        evidence_id = str(getattr(analysis, "input_evidence_id", "") or "").removeprefix("evidence:")
        if evidence_id:
            refs.update(self._views_for_lineage_ref(f"evidence:{evidence_id}", renderable))
        for derived in getattr(analysis, "derived_evidence", []) or []:
            derived_id = str(getattr(derived, "evidence_id", "") or "")
            if derived_id:
                refs.update(self._views_for_lineage_ref(f"derived_evidence:{derived_id}", renderable))
        return refs

    def resolve_lineage(self, source: PresentationSource) -> list[PresentationSource]:
        if source.kind != "view":
            return []
        resolved: list[PresentationSource] = []
        for lineage_ref in source.value.lineage:
            lineage = self._sources.get(str(lineage_ref or "").strip())
            if lineage is None:
                raise InvalidPresentationLineageError(view_ref=source.ref, lineage_ref=str(lineage_ref))
            resolved.append(lineage)
        return resolved

    def planner_inventory(self, preferred_refs: set[str] | None = None) -> dict:
        """Describe grounded sources for LLM-authored semantic projection."""
        preferred = preferred_refs or set()
        sources = []
        for ref, source in sorted(
            self._sources.items(), key=lambda item: (item[0] not in preferred, item[0]),
        ):
            if ref != source.ref or not self._is_projection_source(source):
                continue
            # The parent Insight already exposes the complete item collection
            # through $.items. Publishing every item as another planner source
            # scales the prompt with row count; retain item sources only when
            # the caller explicitly requested that exact located item.
            if source.kind == "insight_item" and ref not in preferred:
                continue
            description = self._source_inventory(source)
            description["preferred_by_caller"] = ref in preferred
            sources.append(description)
        return {
            "schema_version": "semantic-source-v1",
            "sources": sources,
            "rules": [
                "Interpret source meaning from its evidence, insight statement, item semantics, structure, and lineage.",
                "Semantic views may select, rename, and reorganize existing values but may not calculate new business values.",
                "Artifact refs are lineage only; choose the grounded view or verified Insight that actually contains the values.",
            ],
        }

    def targeted_planner_inventory(self, preferred_refs: set[str]) -> dict:
        """Expose only explicit Insight targets and data resolved from their lineage."""

        inventory = self.planner_inventory(preferred_refs)
        inventory["sources"] = [
            source
            for source in inventory["sources"]
            if source["source_ref"] in preferred_refs
        ]
        return inventory

    def semantic_inventory(self, refs: list[str]) -> dict:
        """Describe materialized semantic views for the chart-planning LLM."""
        return {
            "schema_version": "semantic-view-v1",
            "renderer": "echarts",
            "series_type": "open renderer-native string",
            "views": [self._source_inventory(self.resolve(ref)) for ref in refs],
            "rules": [
                "Compose chart layers from semantic views according to their purpose and field semantics.",
                "Use semantic column names exactly as exposed by each view.",
                "Choose renderer-native series types and presentation options that best express the analytical goal.",
                "Do not calculate, aggregate, or invent values in chart planning.",
            ],
        }

    def materialize_semantic_views(self, plans: Iterable[Any]) -> list[str]:
        """Execute LLM-authored semantic projections without interpreting business meaning."""
        refs: list[str] = []
        try:
            for plan in plans:
                source_ref = str(getattr(plan, "source_ref", "") or "").strip()
                source = self.resolve(source_ref)
                record_path = getattr(plan, "record_path", None)
                if record_path is not None:
                    projection_root = _projection_root(source)
                    records = _records_at_path(projection_root, str(record_path))
                else:
                    rows, scalar = _source_data(source)
                    records = rows or ([dict(scalar)] if scalar else [])
                if not records:
                    raise ValueError(f"semantic projection source '{source_ref}' contains no records")
                projected: list[dict] = []
                source_bindings = _projection_bindings(source)
                bindings: dict[str, VisualizationBinding] = {}
                field_semantics: dict[str, str] = {}
                projected_columns: dict[str, list[Any]] = {}
                for mapping in getattr(plan, "fields", []) or []:
                    name = str(getattr(mapping, "name", "") or "").strip()
                    path = str(getattr(mapping, "source_path", "") or "").strip()
                    values: list[Any] = []
                    resolved_count = 0
                    for record in records:
                        try:
                            values.append(_value_at_path(record, path))
                            resolved_count += 1
                        except ValueError:
                            values.append(None)
                    if resolved_count == 0:
                        structures = [_structure_outline([record]) for record in records[:4]]
                        raise ValueError(
                            f"semantic source path '{path}' is unavailable in every record within "
                            f"record_path {record_path!r}; available record paths: "
                            f"{_record_path_candidates(_projection_root(source))}; "
                            f"representative record structures: {structures}"
                        )
                    projected_columns[name] = values
                    field_semantics[name] = str(getattr(mapping, "semantic_role", "") or name)
                for row_index, record in enumerate(records):
                    output = {
                        name: values[row_index]
                        for name, values in projected_columns.items()
                    }
                    binding = source_bindings.get(str(record.get("item_id") or "")) or source_bindings.get("")
                    if binding is not None:
                        binding_id = f"semantic:{getattr(plan, 'view_id', 'view')}:{row_index}"
                        bindings[binding_id] = binding.model_copy(update={"binding_id": binding_id})
                        output["__binding_id"] = binding_id
                    projected.append(output)
                view_id = str(getattr(plan, "view_id", "") or "").strip()
                ref = f"semantic:{view_id}"
                if ref in self._sources:
                    raise ValueError(f"duplicate semantic view id '{view_id}'")
                self._register_view(
                    ref,
                    name=str(getattr(plan, "name", "") or view_id),
                    shape=str(getattr(plan, "grain", "") or "records"),
                    rows=projected,
                    scalar=None,
                    lineage=[source.ref],
                    field_semantics=field_semantics,
                    bindings=bindings,
                )
                refs.append(ref)
        except Exception:
            for ref in refs:
                self._sources.pop(ref, None)
            raise
        return refs

    def _source_inventory(self, source: PresentationSource) -> dict:
        if source.kind == "view":
            value: DataViewValue = source.value
            records = value.rows or ([value.scalar] if value.scalar else [])
            lineage_sources = self._resolved_lineage_sources(source)
            materialization_complete = _full_fidelity_status(lineage_sources)
            capabilities = _render_capabilities(value.schema_fields, scalar=value.scalar is not None)
            result = {
                "source_ref": source.ref, "kind": "data_view", "name": value.name, "shape": value.shape,
                "row_count": len(value.rows) or int(value.scalar is not None), "schema_fields": value.schema_fields,
                "render_capabilities": capabilities,
                "lineage": value.lineage,
                "time_range": _row_time_range(value.rows),
                "materialization_complete": materialization_complete,
                "query_context": [
                    context
                    for item in lineage_sources
                    if (context := _query_context(item)) is not None
                ],
            }
            result["data_structure"] = _structure_outline(records)
            full_projection_root = _projection_root(source)
            result["projection_root"] = {
                "data_structure": _structure_outline([full_projection_root]),
                "record_path_candidates": _record_path_candidates(full_projection_root),
            }
            if value.field_semantics:
                result["field_semantics"] = value.field_semantics
            return result
        if source.kind == "insight":
            insight: KeyInsight = source.value
            locator_row = _insight_locator_row(insight)
            _rows, scalar = _source_data(source)
            fields = _schema_fields(
                [_insight_item_row(item) for item in insight.items]
                or ([locator_row] if locator_row else [])
                or ([scalar] if scalar else [])
            )
            records = [_insight_item_row(item) for item in insight.items]
            if not records:
                records = [locator_row] if locator_row else ([scalar] if scalar else [])
            result = {
                "source_ref": source.ref, "kind": "insight", "status": insight.status, "insight_type": insight.insight_type,
                "insight_key": insight.insight_key, "name": insight.name,
                "semantic_class": insight.semantic_class, "statement": insight.statement, "value_shape": insight.value_shape,
                "value": _bounded_value(insight.value),
                "derived_from": insight.derived_from,
                "evidence_refs": [f"{ref.source_type}:{ref.source_id}" for ref in insight.evidence_refs[:6]],
                "item_refs": [f"{source.ref}#{item.item_id}" for item in insight.items[:12]],
                "items": [_bounded_value(_insight_item_row(item)) for item in insight.items[:12]],
                "item_count": len(insight.items),
                "schema_fields": fields,
                "locator_fields": list(locator_row) if locator_row else [],
                "locator": _bounded_row(locator_row) if locator_row else None,
                "render_capabilities": _render_capabilities(fields, scalar=not bool(insight.items or locator_row)),
                "data_structure": _structure_outline(records),
            }
            full_projection_root = _projection_root(source)
            result["projection_root"] = {
                "data_structure": _structure_outline([full_projection_root]),
                "record_path_candidates": _record_path_candidates(full_projection_root),
            }
            return result
        if source.kind == "insight_item":
            insight, item = source.value
            fields = _schema_fields([_insight_item_row(item)])
            result = {
                "source_ref": source.ref, "kind": "insight_item", "status": insight.status,
                "insight_key": insight.insight_key, "insight_name": insight.name,
                "label": item.label or insight.name, "timestamp": item.timestamp, "value": item.value,
                "item": _bounded_value(_insight_item_row(item)),
                "schema_fields": fields,
                "render_capabilities": _render_capabilities(fields, scalar=False),
                "data_structure": _structure_outline([_insight_item_row(item)]),
            }
            full_projection_root = _projection_root(source)
            result["projection_root"] = {
                "data_structure": _structure_outline([full_projection_root]),
                "record_path_candidates": _record_path_candidates(full_projection_root),
            }
            return result
        value = source.value
        return {
            "source_ref": source.ref, "kind": source.kind,
            "summary": getattr(value, "summary", None) or getattr(value, "analysis_goal", None),
        }

    def _resolved_lineage_sources(self, source: PresentationSource) -> list[PresentationSource]:
        """Resolve semantic lineage through views, Insights, and analysis artifacts."""
        resolved: list[PresentationSource] = []
        pending = list(getattr(source.value, "lineage", None) or [])
        visited: set[str] = set()
        while pending:
            ref = str(pending.pop(0) or "").strip()
            if not ref or ref in visited:
                continue
            visited.add(ref)
            item = self._sources.get(ref)
            if item is None:
                continue
            resolved.append(item)
            if item.kind == "view":
                pending.extend(item.value.lineage)
            elif item.kind == "insight":
                pending.extend(_insight_evidence_lineage(item.value))
            elif item.kind == "insight_item":
                insight, insight_item = item.value
                pending.extend(_insight_evidence_lineage(insight_item))
                pending.extend(_insight_evidence_lineage(insight))
            elif item.kind == "analysis":
                pending.extend(getattr(item.value, "input_source_refs", None) or [])
                input_evidence_id = str(getattr(item.value, "input_evidence_id", "") or "").strip()
                if input_evidence_id:
                    pending.append(
                        input_evidence_id
                        if input_evidence_id.startswith("evidence:")
                        else f"evidence:{input_evidence_id}"
                    )
        return resolved


class VisualizationMaterializer:
    """Compile LLM-planned semantic views into grounded V3 renderer payloads."""

    def __init__(
        self,
        request_state: RequestStateModel,
        *,
        catalog: PresentationCatalog | None = None,
        visual_constraints: dict | None = None,
    ):
        self.request_state = request_state
        self.catalog = catalog or PresentationCatalog(request_state)

    def materialize_all(
        self,
        goals: list[VisualGoal],
        *,
        verification: VisualizationVerification | None = None,
    ) -> list[VisualizationPayload]:
        output: list[VisualizationPayload] = []
        purposes: set[str] = set()
        for index, goal in enumerate(goals):
            key = goal.purpose.strip().casefold()
            if goal.priority == "primary" and key in purposes:
                raise ValueError(f"multiple primary visualizations cover the same purpose: {goal.purpose}")
            if goal.priority == "primary":
                purposes.add(key)
            output.append(self.materialize(goal, index=index, verification=verification))
        return output

    def materialize(
        self,
        goal: VisualGoal,
        *,
        index: int = 0,
        verification: VisualizationVerification | None = None,
    ) -> VisualizationPayload:
        if not goal.layers:
            raise ValueError(f"visual goal '{goal.purpose}' requires at least one layer")
        datasets: list[VisualizationDataset] = []
        layers: list[VisualizationLayer] = []
        bindings: dict[str, VisualizationBinding] = {}
        source_refs: list[str] = []
        for layer_index, plan in enumerate(goal.layers):
            source = self.catalog.resolve(plan.source_ref)
            dataset, layer, layer_bindings = self._materialize_layer(plan, source, layer_index)
            datasets.append(dataset)
            layers.append(layer)
            if source.ref.startswith("semantic:") and source.kind == "view":
                source_refs.extend(source.value.lineage)
            else:
                source_refs.append(source.ref)
            for binding in layer_bindings:
                bindings[binding.binding_id] = binding
        _require_compatible_shared_x_domain(goal, datasets)
        identity = {
            "goal": goal.model_dump(mode="json"),
            "source_refs": source_refs,
            "verification": verification.model_dump(mode="json") if verification else None,
        }
        digest = hashlib.sha1(
            json.dumps(identity, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        payload = VisualizationPayload(
            visualization_id=f"viz_{digest}_{index}", purpose=goal.purpose, priority=goal.priority,
            title=goal.title, summary=goal.summary, verification=verification,
            source_refs=list(dict.fromkeys(source_refs)),
            required_roles=list(dict.fromkeys(goal.required_roles)), datasets=datasets, layers=layers,
            bindings=list(bindings.values()), layout=_layout_from_chart_plan(goal),
            presentation=_presentation_options(goal.presentation),
            accessibility=_accessibility(goal, datasets),
        )
        return payload

    def _materialize_layer(
        self, plan: VisualLayerPlan, source: PresentationSource, index: int,
    ) -> tuple[VisualizationDataset, VisualizationLayer, list[VisualizationBinding]]:
        rows, scalar = _source_data(source)
        encoding = _encoding_fields(plan.encoding)
        _validate_layer_encoding(plan.mark, encoding, source)
        presentation = _presentation_options(plan.presentation)
        rows = _apply_presentation_transforms(rows, plan.transform, source)
        if plan.transform:
            scalar = None
        points, bindings, x_field, y_field = _points_for_source(source, rows, scalar, encoding)
        if not points:
            raise ValueError(f"layer role '{plan.role}' produced no renderable points from {source.ref}")
        dataset_id = f"dataset_{index}"
        group_field = _first_encoding_field(encoding.get("series"))
        grouped_points: dict[str, list[VisualizationPoint]] = {}
        if group_field:
            for point in points:
                group = str(point.metadata.get(group_field, ""))
                grouped_points.setdefault(group, []).append(point)
        else:
            grouped_points[""] = points
        series = [
            VisualizationSeries(
                series_id=f"series_{index}_{group_index}",
                name=(f"{plan.label or plan.role}: {group}" if group else plan.label or plan.role),
                role=(f"{plan.role}:{group}" if group else plan.role),
                points=group_points,
            )
            for group_index, (group, group_points) in enumerate(grouped_points.items())
        ]
        dataset = VisualizationDataset(
            dataset_id=dataset_id, source_ref=source.ref,
            dimensions=_dimensions(x_field, y_field, rows, scalar), series=series,
        )
        layer = VisualizationLayer(
            layer_id=f"layer_{index}", mark=plan.mark, role=plan.role, source_ref=source.ref,
            encoding=encoding, transform=[item.model_dump(mode="json") for item in plan.transform],
            presentation=presentation,
            dataset_id=dataset_id, series_id=series[0].series_id if len(series) == 1 else None,
            points=points if plan.mark in {"point", "rule", "rect"} else [], label=plan.label,
        )
        return dataset, layer, bindings


class VisualizationSemanticValidator:
    """Validate grounded invariants independently of the chart-planning LLM."""

    def __init__(self, catalog: PresentationCatalog, *, required_located_roles: set[str] | None = None):
        self.catalog = catalog
        self.required_located_roles = required_located_roles or set()

    def validate(self, goal: VisualGoal, payload: VisualizationPayload) -> None:
        materialized_roles = {layer.role.strip().casefold() for layer in payload.layers}
        missing_roles = [
            role for role in goal.required_roles
            if role.strip().casefold() not in materialized_roles
        ]
        if missing_roles:
            raise ValueError(f"required visual roles were not materialized: {missing_roles}")
        if payload.verification is not None:
            verified_ids = {
                insight.insight_id
                for insight in self.catalog.request_state.insight_set.insights
                if insight.status == "verified"
            }
            unknown = set(payload.verification.target_insight_ids) - verified_ids
            if unknown:
                raise ValueError(
                    f"visual verification targets are not verified Key Insights: {sorted(unknown)}"
                )
        for layer in payload.layers:
            source = self.catalog.resolve(layer.source_ref)
            _validate_layer_encoding(layer.mark, layer.encoding, source)
            if layer.mark.strip().casefold() in {"line", "area"}:
                dataset = next(
                    (item for item in payload.datasets if item.dataset_id == layer.dataset_id),
                    None,
                )
                plotted_series = (
                    [item for item in dataset.series if item.series_id == layer.series_id]
                    if dataset is not None and layer.series_id
                    else list(dataset.series) if dataset is not None else []
                )
                if not plotted_series or any(len(series.points) < 2 for series in plotted_series):
                    raise ValueError(
                        f"{layer.mark} layer '{layer.role}' requires at least two grounded points per series"
                    )


def _constraint_role_candidates(constraints: dict) -> set[str]:
    """Collect role labels explicitly supplied as members of constraint arrays."""
    roles: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                if isinstance(nested, str) and nested.strip():
                    roles.add(_semantic_role_key(nested))
                else:
                    visit(nested)

    visit(constraints)
    return roles


def _semantic_role_key(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()


def _apply_presentation_transforms(rows: list[dict], transforms, source: PresentationSource) -> list[dict]:
    """Apply selection-only transforms to copies of grounded rows without changing values."""
    selected = [dict(row) for row in rows]
    available = {field["name"] for field in _source_schema(source)}
    for transform in transforms:
        field = transform.field.strip()
        if field not in available:
            raise ValueError(
                f"visual filter references unavailable field '{field}'; available fields: {sorted(available)}"
            )
        operator = transform.operator
        expected = transform.value
        if operator == "between" and _ordered_filter_value(expected[0]) > _ordered_filter_value(expected[1]):
            raise ValueError(f"visual filter field '{field}' has reversed between boundaries")

        def keep(row: dict) -> bool:
            actual = row.get(field)
            if operator == "exists":
                return actual is not None
            if operator == "not_exists":
                return actual is None
            if operator == "eq":
                return actual == expected
            if operator == "neq":
                return actual != expected
            if operator == "in":
                return actual in expected
            if operator == "not_in":
                return actual not in expected
            if actual is None:
                return False
            if operator == "between":
                return _ordered_filter_value(expected[0]) <= _ordered_filter_value(actual) <= _ordered_filter_value(expected[1])
            left = _ordered_filter_value(actual)
            right = _ordered_filter_value(expected)
            if operator == "gt":
                return left > right
            if operator == "gte":
                return left >= right
            if operator == "lt":
                return left < right
            return left <= right

        selected = [row for row in selected if keep(row)]
    return selected


def _ordered_filter_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return 0, int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return 1, float(value)
    if isinstance(value, (str, datetime)):
        parsed = _time_value(value)
        if parsed is not None:
            return 2, parsed.timestamp()
    return 3, str(value)


def _source_data(source: PresentationSource) -> tuple[list[dict], dict | None]:
    if source.kind == "view":
        return [dict(row) for row in source.value.rows], dict(source.value.scalar) if source.value.scalar else None
    if source.kind == "insight_item":
        _insight, item = source.value
        return [_insight_item_row(item)], None
    if source.kind == "insight":
        insight: KeyInsight = source.value
        if insight.items:
            return [_insight_item_row(item) for item in insight.items], None
        locator_row = _insight_locator_row(insight)
        if locator_row:
            metric = dict(insight.value) if isinstance(insight.value, dict) else None
            return [locator_row], metric
        if isinstance(insight.value, list):
            return [dict(item) if isinstance(item, dict) else {"value": item} for item in insight.value], None
        if isinstance(insight.value, dict):
            return [], dict(insight.value)
        return [], {"label": insight.name, "value": insight.value}
    if source.kind == "forecast":
        return [{"timestamp": item.timestamp, "value": item.value} for item in source.value.forecast_points], None
    if source.kind == "anomaly":
        return [dict(item) for item in source.value.anomaly_points if isinstance(item, dict)], None
    if source.kind == "evidence":
        return _rows_from_evidence(source.value), None
    if source.kind == "analysis":
        raise ValueError(f"analysis artifact {source.ref} must be referenced through an explicit view:analysis source")
    return [], None


def _insight_locator_row(insight: KeyInsight) -> dict:
    trace = insight.calculation_trace if isinstance(insight.calculation_trace, dict) else {}
    row = trace.get("row")
    if isinstance(row, dict):
        candidate = {str(key): value for key, value in row.items() if value is not None}
        if _candidate_fields([candidate], "time") and _candidate_fields([candidate], "number"):
            return candidate
    if isinstance(insight.value, dict):
        candidate = {str(key): value for key, value in insight.value.items() if value is not None}
        if _candidate_fields([candidate], "time") and _candidate_fields([candidate], "number"):
            return candidate
    return {}


def _encoding_fields(encoding: dict) -> dict[str, str | list[str]]:
    result: dict[str, str | list[str]] = {}
    channel_aliases = {
        "time": "x",
        "timestamp": "x",
        "date": "x",
        "category": "x",
        "value": "y",
        "measure": "y",
        "metric": "y",
    }
    for channel, value in (encoding or {}).items():
        normalized_channel = channel_aliases.get(str(channel).strip().lower(), channel)
        if isinstance(value, str):
            if channel != "columns":
                result[normalized_channel] = value
            continue
        if isinstance(value, list):
            fields = []
            for item in value:
                field = item if isinstance(item, str) else getattr(item, "field", None)
                if isinstance(field, str) and field.strip():
                    fields.append(field.strip())
            if fields:
                result[normalized_channel] = fields
            continue
        field = getattr(value, "field", None)
        if isinstance(field, str) and field.strip():
            result[normalized_channel] = field.strip()
    return result


def _presentation_options(value: dict) -> dict:
    """Accept renderer presentation JSON while preventing it from carrying data.

    The renderer may evolve independently from this materializer.  Data-bearing
    properties stay owned by the grounded dataset compiler and are injected by
    the frontend after presentation options are applied.
    """
    def copy_presentation(item: Any, path: tuple[str, ...] = ()) -> Any:
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("visual presentation contains a non-finite number")
            return item
        if isinstance(item, list):
            return [copy_presentation(nested, path) for nested in item]
        if isinstance(item, dict):
            copied = {}
            for raw_key, nested in item.items():
                key = str(raw_key)
                if key.casefold() in _DATA_BEARING_PRESENTATION_KEYS:
                    location = ".".join((*path, key))
                    raise ValueError(
                        f"visual presentation property '{location}' may carry data; "
                        "bind it through source_ref and encoding instead"
                    )
                copied[key] = copy_presentation(nested, (*path, key))
            return copied
        raise ValueError(f"visual presentation contains unsupported JSON value {type(item).__name__}")

    return copy_presentation(value or {})


def _validate_layer_encoding(
    mark: str,
    encoding: dict[str, str | list[str]],
    source: PresentationSource,
) -> None:
    """Reject ambiguous or ungrounded encodings instead of guessing fields."""
    available = {str(item.get("name")) for item in _source_schema(source) if item.get("name")}
    referenced = {
        field
        for value in encoding.values()
        for field in (value if isinstance(value, list) else [value])
        if isinstance(field, str) and field
    }
    unknown = referenced - available
    if unknown:
        raise ValueError(
            f"visual encoding references unavailable fields {sorted(unknown)} from {source.ref}; "
            f"available fields: {sorted(available)}"
        )
    normalized_mark = str(mark or "").strip().casefold()
    x_field = _first_encoding_field(encoding.get("x"))
    y_field = _first_encoding_field(encoding.get("y") or encoding.get("value"))
    if normalized_mark in {"line", "area", "bar", "point", "boxplot"}:
        if not x_field or not y_field:
            raise ValueError(
                f"{mark} layer over {source.ref} requires explicit x and y/value encodings; "
                "automatic field selection is not allowed"
            )
    elif normalized_mark == "band":
        if not x_field or not _first_encoding_field(encoding.get("lower")) or not _first_encoding_field(encoding.get("upper")):
            raise ValueError("band layer requires explicit x, lower, and upper encodings")
    elif normalized_mark in {"rule", "rect"}:
        if not x_field and not y_field:
            raise ValueError(f"{mark} layer requires an explicit x or y/value encoding")
    elif not referenced:
        raise ValueError(f"renderer-native mark '{mark}' requires at least one grounded field encoding")


def _points_for_source(source, rows, scalar, encoding):
    bindings: list[VisualizationBinding] = []
    if source.kind == "insight_item":
        insight, item = source.value
        binding = _insight_item_binding(source.ref, insight, item)
        bindings.append(binding)
    elif source.kind == "insight":
        insight = source.value
        if insight.items:
            bindings.extend(
                _insight_item_binding(f"insight:{insight.insight_id}#{item.item_id}", insight, item)
                for item in insight.items
            )
        else:
            bindings.append(_insight_binding(source.ref, insight))
    elif source.kind == "view" and source.value.bindings:
        bindings.extend(source.value.bindings.values())
    x_field = _first_encoding_field(encoding.get("x"))
    y_field = _first_encoding_field(encoding.get("y")) or _first_encoding_field(encoding.get("value"))
    lower_field = _first_encoding_field(encoding.get("lower"))
    upper_field = _first_encoding_field(encoding.get("upper"))
    label_field = _first_encoding_field(encoding.get("label"))
    points: list[VisualizationPoint] = []
    binding_by_item = {binding.item_id: binding.binding_id for binding in bindings if binding.item_id}
    for row in rows:
        y = _number(row.get(y_field)) if y_field else None
        lower = _number(row.get(lower_field)) if lower_field else None
        upper = _number(row.get(upper_field)) if upper_field else None
        item_id = str(row.get("item_id") or "")
        point = VisualizationPoint(
            x=row.get(x_field) if x_field else row.get("label") or row.get("name"), y=y,
            lower=lower, upper=upper,
            label=str(row.get(label_field) or row.get("label") or row.get("type") or "") or None,
            binding_id=str(row.get("__binding_id") or binding_by_item.get(item_id) or "") or None,
            # Keep an immutable copy of every grounded field so renderer-native
            # encodings can address shapes beyond the normalized x/y pair.
            metadata=dict(row),
        )
        if point.x is not None or point.y is not None or point.lower is not None or point.upper is not None:
            points.append(point)
    if scalar and not points:
        y_key = y_field
        x_key = x_field
        binding_id = bindings[0].binding_id if bindings else None
        point = VisualizationPoint(
            x=scalar.get(x_key) if x_key else scalar.get("label"),
            y=_number(scalar.get(y_key)) if y_key else None,
            label=str(scalar.get("label") or "") or None, binding_id=binding_id,
            metadata=dict(scalar),
        )
        if point.x is not None or point.y is not None:
            points.append(point)
        x_field = x_key or "label"
        y_field = y_key or "value"
    return points, bindings, x_field or "x", y_field or "value"


def _first_encoding_field(value: str | list[str] | None) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, str) and item), None)
    return None


def _insight_item_binding(ref: str, insight: KeyInsight, item: InsightItem) -> VisualizationBinding:
    evidence_id = next((e.source_id for e in item.evidence_refs or insight.evidence_refs if e.source_type in {"query", "database_evidence"}), None)
    return VisualizationBinding(
        binding_id=ref, source_type="insight_item", insight_id=insight.insight_id, item_id=item.item_id,
        evidence_id=evidence_id, source_ref=ref, locator=item.locator,
    )


def _insight_binding(ref: str, insight: KeyInsight) -> VisualizationBinding:
    evidence_id = next((e.source_id for e in insight.evidence_refs if e.source_type in {"query", "database_evidence"}), None)
    return VisualizationBinding(
        binding_id=ref, source_type="insight", insight_id=insight.insight_id, evidence_id=evidence_id,
        source_ref=ref, locator={"time_range": insight.time_range} if insight.time_range else {},
    )


def _source_schema(source: PresentationSource) -> list[dict]:
    if source.kind == "view":
        return source.value.schema_fields
    rows, scalar = _source_data(source)
    return _schema_fields(rows or ([scalar] if scalar else []))


def _schema_fields(rows: list[dict]) -> list[dict]:
    result = []
    for name in (field for field in _row_fields(rows) if not field.startswith("__")):
        values = [row.get(name) for row in rows if row.get(name) is not None]
        kind = "string"
        if any(isinstance(value, bool) for value in values):
            kind = "boolean"
        elif any(_number(value) is not None for value in values):
            kind = "number"
        elif any(_time_value(value) is not None for value in values):
            kind = "time"
        elif any(isinstance(value, (dict, list)) for value in values):
            kind = "object"
        elif len({str(value) for value in values}) <= 12:
            kind = "category"
        result.append({"name": name, "data_type": kind})
    return result


def _dimensions(x_field, y_field, rows, scalar):
    schema = {item["name"]: item["data_type"] for item in _schema_fields(rows or ([scalar] if scalar else []))}
    x_type = schema.get(x_field, "string")
    if x_type not in {"time", "number", "category", "string"}:
        x_type = "string"
    return [
        VisualizationDimension(name=x_field, data_type=x_type, role="x"),
        VisualizationDimension(name=y_field, data_type="number", role="y"),
    ]


def _accessibility(goal, datasets):
    rows: list[dict] = []
    for dataset in datasets:
        for series in dataset.series:
            rows.extend(
                {"role": series.role, "x": point.x, "y": point.y, "lower": point.lower, "upper": point.upper}
                for point in series.points[:12]
            )
    columns = _row_fields(rows)
    return VisualizationAccessibility(
        description=goal.summary or goal.purpose, table_columns=columns, table_rows=rows[:24],
    )


def _layout_from_chart_plan(goal):
    requested_indices = [
        layer.presentation.get("xAxisIndex")
        for layer in goal.layers
        if isinstance(layer.presentation, dict)
        and isinstance(layer.presentation.get("xAxisIndex"), int)
    ]
    if requested_indices and set(requested_indices) == {0}:
        return "overlay"
    grids = goal.presentation.get("grid") if isinstance(goal.presentation, dict) else None
    return "facets" if isinstance(grids, list) and len(grids) > 1 else "overlay"


def _require_compatible_shared_x_domain(goal, datasets):
    if _layout_from_chart_plan(goal) != "overlay":
        return
    domain_types = {
        dimension.data_type
        for dataset in datasets
        for dimension in dataset.dimensions
        if dimension.role == "x"
    }
    normalized = {"category" if item in {"category", "string"} else item for item in domain_types}
    if len(normalized) > 1:
        raise ValueError(
            "one shared chart requires compatible x-domain semantics across all layers; "
            "keep time-aligned layers in the primary visual goal and leave scalar summaries in answer text "
            "or an explicitly requested supporting chart"
        )


def _forecast_input_refs(forecast) -> list[str]:
    coverage = forecast.diagnostics.get("coverage") if isinstance(forecast.diagnostics, dict) else {}
    return [str(ref) if ":" in str(ref) else f"evidence:{ref}" for ref in (coverage or {}).get("input_evidence_refs", [])]


def _anomaly_evidence_ref(anomaly) -> str | None:
    diagnostics = anomaly.diagnostics if isinstance(anomaly.diagnostics, dict) else {}
    for key in ("resolved_evidence_id", "selected_evidence_id", "input_evidence_id"):
        if diagnostics.get(key):
            value = str(diagnostics[key])
            return value if value.startswith("evidence:") else f"evidence:{value}"
    return None


def _insight_evidence_lineage(value: KeyInsight | InsightItem) -> list[str]:
    """Translate typed Insight provenance into catalog refs without guessing sources."""
    refs: list[str] = []
    for evidence_ref in value.evidence_refs:
        source_type = str(evidence_ref.source_type or "").strip()
        source_id = str(evidence_ref.source_id or "").strip()
        if not source_id:
            continue
        if source_type in {"query", "database_evidence", "evidence"}:
            source_type = "evidence"
        refs.append(f"{source_type}:{source_id}" if source_type else source_id)
    return list(dict.fromkeys(refs))


def _full_fidelity_status(sources: list[PresentationSource]) -> bool | None:
    values: list[bool] = []
    for source in sources:
        diagnostics = getattr(source.value, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            continue
        if isinstance(diagnostics.get("is_full_fidelity"), bool):
            values.append(diagnostics["is_full_fidelity"])
        sampling = diagnostics.get("prompt_sampling")
        if isinstance(sampling, dict) and isinstance(sampling.get("is_full_fidelity"), bool):
            values.append(sampling["is_full_fidelity"])
    if not values:
        return None
    return all(values)


def _render_capabilities(schema_fields: list[dict], *, scalar: bool) -> dict:
    data_types = {str(item.get("data_type") or "") for item in schema_fields if isinstance(item, dict)}
    timestamped_numeric = "time" in data_types and "number" in data_types
    renderable = bool(data_types - {"object"}) and (not scalar or "number" in data_types)
    return {
        "timestamped_numeric": timestamped_numeric,
        "scalar_only": scalar,
        "renderer_series_type": "open",
        "renderable": renderable,
    }


def _bounded_presentation_value(value: Any, *, depth: int = 0) -> Any:
    """Bound arbitrary typed artifact payloads without interpreting their fields."""

    if depth >= 6:
        return "[depth limit]"
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            str(key): _bounded_presentation_value(item, depth=depth + 1)
            for key, item in items[:40]
        }
        if len(items) > 40:
            bounded["_truncated_field_count"] = len(items) - 40
        return bounded
    if isinstance(value, (list, tuple)):
        bounded = [_bounded_presentation_value(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            bounded.append({"_truncated_item_count": len(value) - 20})
        return bounded
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + f"… [truncated {len(value) - 2000} chars]"
    return value


def _query_context(source: PresentationSource) -> dict | None:
    if source.kind != "evidence":
        return None
    evidence = source.value
    diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
    metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
    coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    result = {
        "evidence_ref": source.ref,
        "result_type": evidence.result_type,
        "summary": evidence.summary,
        "query_language": evidence.query_language,
        "query": str(evidence.query or "")[:1600] or None,
        "columns": list(evidence.columns or [])[:40],
        "query_task_contract": coverage.get("query_task_contract"),
        "query_execution": metadata.get("query_execution"),
        "satisfied": coverage.get("satisfied"),
        "missing": coverage.get("missing"),
        "result_summary": coverage.get("result_summary"),
        "materialization_complete": diagnostics.get("is_full_fidelity"),
        "truncated": diagnostics.get("truncated"),
    }
    return {key: _bounded_value(value) for key, value in result.items() if value not in (None, "", [], {})}


def _row_time_range(rows: list[dict]) -> dict | None:
    time_fields = _candidate_fields(rows, "time")
    if not time_fields:
        return None
    field = time_fields[0]
    values = [row.get(field) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return None
    ordered = sorted(values, key=_x_sort_key)
    return {"field": field, "start": ordered[0], "end": ordered[-1]}


def _rows_from_evidence(evidence) -> list[dict]:
    data = evidence.data if isinstance(evidence.data, dict) else {}
    for key in ("rows", "points"):
        if isinstance(data.get(key), list):
            return [dict(row) for row in data[key] if isinstance(row, dict)]
    series = data.get("series")
    if isinstance(series, list):
        rows = []
        for item in series:
            if not isinstance(item, dict):
                continue
            name = item.get("series_name") or item.get("value_field") or "series"
            rows.extend({**point, "series": name} for point in item.get("points", []) if isinstance(point, dict))
        return rows
    statistics = data.get("statistics")
    if isinstance(statistics, dict):
        return [{"metric": key, "value": value} for key, value in statistics.items()]
    return []


def _insight_item_row(item: InsightItem) -> dict:
    row = {
        "item_id": item.item_id, "value": item.value, "label": item.label, "rank": item.rank,
        "timestamp": item.timestamp, **item.dimensions, **item.locator,
    }
    return {key: value for key, value in row.items() if value is not None}


def _candidate_fields(rows: list[dict], expected: str) -> list[str]:
    fields = []
    for field in _row_fields(rows):
        values = [row.get(field) for row in rows[:40] if row.get(field) is not None]
        if not values:
            continue
        if expected == "time" and any(_time_value(value) is not None for value in values):
            fields.append(field)
        elif expected == "number" and any(_number(value) is not None for value in values):
            fields.append(field)
        elif expected == "category" and not any(_number(value) is not None or _time_value(value) is not None for value in values):
            fields.append(field)
    return fields


def _first_field(rows: list[dict], expected: str) -> str | None:
    values = _candidate_fields(rows, expected)
    return values[0] if values else None


def _row_fields(rows: Iterable[dict]) -> list[str]:
    result: list[str] = []
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def _projection_bindings(source: PresentationSource) -> dict[str, VisualizationBinding]:
    if source.kind == "insight_item":
        insight, item = source.value
        return {item.item_id: _insight_item_binding(source.ref, insight, item)}
    if source.kind == "insight":
        insight = source.value
        bindings = {"": _insight_binding(source.ref, insight)}
        if insight.items:
            bindings.update({
                item.item_id: _insight_item_binding(
                    f"insight:{insight.insight_id}#{item.item_id}", insight, item,
                )
                for item in insight.items
            })
        return bindings
    if source.kind == "view":
        return dict(source.value.bindings)
    return {}


def _projection_root(source: PresentationSource) -> dict:
    """Expose a source document whose nested grains can be selected by the LLM."""
    if source.kind == "view":
        value: DataViewValue = source.value
        return {"records": value.rows, "scalar": value.scalar}
    if source.kind == "insight":
        insight: KeyInsight = source.value
        return {
            **insight.model_dump(mode="json", exclude={"items"}),
            "items": [_insight_item_row(item) for item in insight.items],
        }
    if source.kind == "insight_item":
        insight, item = source.value
        return {
            "insight": insight.model_dump(mode="json", exclude={"items"}),
            "item": _insight_item_row(item),
        }
    raise ValueError(f"presentation source '{source.ref}' cannot be semantically projected")


def _object_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _value_at_path(record: dict, path: str) -> Any:
    """Read one value selected by the LLM from an existing grounded record."""
    current: Any = record
    for token in _path_tokens(path):
        if token is _PATH_WILDCARD:
            raise ValueError(
                f"semantic source path '{path}' cannot expand records; use [*] only in record_path"
            )
        if isinstance(token, int):
            if not isinstance(current, (list, tuple)) or token >= len(current):
                raise ValueError(f"semantic source path '{path}' is unavailable")
            current = current[token]
            continue
        if not isinstance(current, dict) or token not in current:
            raise ValueError(f"semantic source path '{path}' is unavailable")
        current = current[token]
    return current


_PATH_WILDCARD = object()


def _records_at_path(root: dict, path: str) -> list[dict]:
    """Select and flatten an LLM-declared nested record grain."""

    selected: list[Any] = [root]
    for token in _path_tokens(path):
        next_values: list[Any] = []
        for current in selected:
            if token is _PATH_WILDCARD:
                if isinstance(current, (list, tuple)):
                    next_values.extend(current)
            elif isinstance(token, int):
                if isinstance(current, (list, tuple)) and token < len(current):
                    next_values.append(current[token])
            elif isinstance(current, dict) and token in current:
                next_values.append(current[token])
        if not next_values:
            raise ValueError(f"semantic record path '{path}' is unavailable")
        selected = next_values

    records: list[dict] = []
    for value in selected:
        if isinstance(value, dict):
            records.append(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, dict) for item in value):
            records.extend(value)
        else:
            raise ValueError(
                f"semantic record path '{path}' must resolve to an object or array of objects"
            )
    return records


def _path_tokens(path: str) -> list[Any]:
    normalized = path.strip()
    if normalized in {"", "$"}:
        return []
    if normalized.startswith("$"):
        normalized = normalized[1:]
    tokens: list[Any] = []
    cursor = 0
    while cursor < len(normalized):
        if normalized[cursor] == ".":
            cursor += 1
            if cursor >= len(normalized):
                raise ValueError(f"semantic source path '{path}' is invalid")
        if normalized[cursor] == "[":
            closing = normalized.find("]", cursor + 1)
            if closing < 0:
                raise ValueError(f"semantic source path '{path}' is invalid")
            content = normalized[cursor + 1:closing]
            if content == "*":
                tokens.append(_PATH_WILDCARD)
            elif content.isdigit():
                tokens.append(int(content))
            elif (
                len(content) >= 2
                and content[0] in {"'", '"'}
                and content[-1] == content[0]
            ):
                tokens.append(content[1:-1])
            else:
                raise ValueError(f"semantic source path '{path}' is invalid")
            cursor = closing + 1
            if cursor < len(normalized) and normalized[cursor] not in ".[":
                raise ValueError(f"semantic source path '{path}' is invalid")
            continue
        match = re.match(r"[^.\[\]]+", normalized[cursor:])
        if match is None:
            raise ValueError(f"semantic source path '{path}' is invalid")
        tokens.append(match.group(0))
        cursor += len(match.group(0))
    if not tokens:
        raise ValueError(f"semantic source path '{path}' is invalid")
    return tokens


def _record_path_candidates(root: dict, *, max_depth: int = 6) -> list[str]:
    """Describe array-backed record grains without assigning business semantics."""

    candidates: list[str] = []
    seen_candidates: set[str] = set()

    def child_path(path: str, key: Any) -> str:
        name = str(key)
        if re.fullmatch(r"[^.\[\]]+", name):
            return f"{path}.{name}"
        escaped = name.replace("'", "\\'")
        return f"{path}['{escaped}']"

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, child_path(path, key), depth + 1)
            return
        if not isinstance(value, list):
            return
        object_items = [item for item in value if isinstance(item, dict)]
        if object_items:
            if path not in seen_candidates:
                candidates.append(path)
                seen_candidates.add(path)
            for item in object_items:
                visit(item, f"{path}[*]", depth + 1)

    visit(root, "$", 0)
    return list(dict.fromkeys(candidates))


def _structure_outline(records: list[dict]) -> dict:
    """Expose complete nested field structure and types without copying record values."""
    samples = [item for item in records if isinstance(item, dict)]

    def describe(values: list[Any], depth: int = 0) -> Any:
        present = [value for value in values if value is not None]
        if not present:
            return {"type": "null"}
        if depth >= 6:
            return {"type": "nested"}
        if any(isinstance(value, dict) for value in present):
            keys = []
            for value in present:
                if isinstance(value, dict):
                    for key in value:
                        if key not in keys:
                            keys.append(key)
            return {
                "type": "object",
                "fields": {
                    str(key): describe(
                        [value.get(key) for value in present if isinstance(value, dict)], depth + 1,
                    )
                    for key in keys[:40]
                },
            }
        if any(isinstance(value, list) for value in present):
            items = [item for value in present if isinstance(value, list) for item in value]
            return {"type": "array", "items": describe(items, depth + 1)}
        if any(isinstance(value, bool) for value in present):
            kind = "boolean"
        elif any(_number(value) is not None for value in present):
            kind = "number"
        elif any(_time_value(value) is not None for value in present):
            kind = "time"
        else:
            kind = "string"
        return {"type": kind}

    return describe(samples)


def _bounded_row(row: dict) -> dict:
    visible = [(key, value) for key, value in row.items() if not str(key).startswith("__")]
    return {str(key): _bounded_value(value) for key, value in visible[:12]}


def _bounded_value(value):
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, list):
        return [_bounded_value(item) for item in value[:6]]
    if isinstance(value, dict):
        return {str(key): _bounded_value(item) for key, item in list(value.items())[:8]}
    return value


def _number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _time_value(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _sample_points(points, limit, preserve):
    if len(points) <= limit:
        return points
    preserved = [point for point in points if point.x in preserve]
    remaining = max(2, limit - len(preserved))
    step = max(1, math.ceil(len(points) / remaining))
    sampled = [points[index] for index in range(0, len(points), step)][:remaining]
    by_key = {(str(point.x), point.y, point.binding_id): point for point in [*sampled, *preserved]}
    return sorted(by_key.values(), key=lambda point: _x_sort_key(point.x))[:limit]


def _x_sort_key(value):
    parsed = _time_value(value)
    if parsed is not None:
        return (0, parsed.timestamp())
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))
