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
from schemas.output import VisualGoal, VisualLayerPlan, VisualSemanticPlan
from schemas.state import RequestStateModel
from schemas.visual_verification import VisualizationVerification
from schemas.visualization import (
    VisualizationAccessibility,
    VisualizationBinding,
    ChartSemanticTarget,
    IntervalSemanticTarget,
    ReferenceSemanticTarget,
    RelationSemanticTarget,
    VisualizationChannelBinding,
    VisualizationCoordinateSpace,
    VisualizationDataMark,
    VisualizationDataView,
    VisualizationField,
    VisualizationGuide,
    VisualizationGuideEntry,
    VisualizationGuideSection,
    VisualizationLayout,
    VisualizationLayoutCell,
    VisualizationMetric,
    VisualizationPayload,
    VisualizationRecord,
    VisualizationScale,
    VisualizationSemantic,
    VisualizationSemanticContent,
    XSemanticTarget,
    XYSemanticTarget,
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


class IncompatibleVisualDomainError(ValueError):
    """The fixed composition cannot share one coordinate system."""


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
                mode = str(getattr(plan, "mode", "records") or "records")
                if mode == "wide_events":
                    if len(records) != 1:
                        raise ValueError(
                            f"wide_events semantic projection requires exactly one selected source "
                            f"record; source {source.ref} produced {len(records)} records. Use records "
                            "mode for an already-long event table."
                        )
                    field_semantics = {
                        "event_role": "event_role",
                        "timestamp": "event_time",
                        "value": "event_value",
                    }
                    event_index = 0
                    for record in records:
                        binding = source_bindings.get(str(record.get("item_id") or "")) or source_bindings.get("")
                        for event in getattr(plan, "events", []) or []:
                            timestamp_path = str(getattr(event, "timestamp_path", "") or "")
                            value_path = str(getattr(event, "value_path", "") or "")
                            try:
                                timestamp = _value_at_path(record, timestamp_path)
                                value = _value_at_path(record, value_path)
                            except ValueError as exc:
                                raise ValueError(
                                    f"semantic event paths '{timestamp_path}'/'{value_path}' are unavailable "
                                    f"within record_path {record_path!r}"
                                ) from exc
                            output = {
                                "event_role": str(getattr(event, "event_role", "") or "event"),
                                "timestamp": timestamp,
                                "value": value,
                            }
                            if binding is not None:
                                binding_id = f"semantic:{getattr(plan, 'view_id', 'view')}:{event_index}"
                                bindings[binding_id] = binding.model_copy(update={"binding_id": binding_id})
                                output["__binding_id"] = binding_id
                            projected.append(output)
                            event_index += 1
                else:
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
                            path_contracts = _projection_path_contracts(source)
                            selected_paths = next(
                                (
                                    item["source_paths"]
                                    for item in path_contracts["record_paths"]
                                    if item["record_path"] == record_path
                                ),
                                path_contracts["default_source_paths"],
                            )
                            raise ValueError(
                                f"semantic source path '{path}' is unavailable in every record within "
                                f"record_path {record_path!r}; source_path is relative to each selected "
                                f"record. Valid source paths for this record_path: {selected_paths}. "
                                f"Executable projection path contracts: {path_contracts}"
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
                    shape=("event_set" if mode == "wide_events" else str(getattr(plan, "grain", "") or "records")),
                    rows=projected,
                    scalar=None,
                    lineage=[source.ref],
                    field_semantics=field_semantics,
                    bindings=bindings,
                )
                refs.append(ref)
                render_contract = _render_contract(self.resolve(ref).value)
                if not render_contract["allowed_layer_types"]:
                    raise ValueError(
                        f"semantic view '{view_id}' has no executable visual consumer; its materialized "
                        f"contract is {render_contract}. Request a grounded source exposing data-layer fields, "
                        "interval boundaries, or scalar/text content suitable for an explicit guide"
                    )
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
                "render_contract": _render_contract(value),
                "lineage": value.lineage,
                "time_range": _row_time_range(value.rows),
                "materialization_complete": materialization_complete,
                "query_context": [
                    context
                    for item in lineage_sources
                    if (context := _query_context(item)) is not None
                ],
                "semantic_contract": _source_semantic_contract(lineage_sources),
            }
            result["data_structure"] = _structure_outline(records)
            # Small grounded sources (for example anomaly points or interval
            # boundaries) carry semantic facts that a structure-only outline
            # cannot express.  Keep the preview strictly bounded; full series
            # remain reference-backed and are resolved only at materialization.
            preview_eligible = any(
                item.kind in {"anomaly", "forecast", "derived_evidence"}
                for item in lineage_sources
            )
            if preview_eligible:
                if value.scalar is not None:
                    result["grounded_preview"] = _bounded_row(value.scalar)
                elif 0 < len(value.rows) <= 12:
                    result["grounded_preview"] = [
                        _bounded_row(row)
                        for row in value.rows
                    ]
            full_projection_root = _projection_root(source)
            result["projection_root"] = {
                "data_structure": _structure_outline([full_projection_root]),
                "record_path_candidates": _record_path_candidates(full_projection_root),
                "executable_path_contracts": _projection_path_contracts(source),
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
                "semantic_contract": {
                    "data_role": "key_insight_claim",
                    "materializes_input_transformation": False,
                    "operation_description": insight.calculation_trace,
                    "supported_visual_uses": ["target_claim", "annotation", "interval_boundaries"],
                },
            }
            full_projection_root = _projection_root(source)
            result["projection_root"] = {
                "data_structure": _structure_outline([full_projection_root]),
                "record_path_candidates": _record_path_candidates(full_projection_root),
                "executable_path_contracts": _projection_path_contracts(source),
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
                "semantic_contract": {
                    "data_role": "key_insight_item",
                    "materializes_input_transformation": False,
                    "operation_description": insight.calculation_trace,
                    "supported_visual_uses": ["target_point", "annotation"],
                },
            }
            full_projection_root = _projection_root(source)
            result["projection_root"] = {
                "data_structure": _structure_outline([full_projection_root]),
                "record_path_candidates": _record_path_candidates(full_projection_root),
                "executable_path_contracts": _projection_path_contracts(source),
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
    """Compile LLM-planned semantic views into a grounded V4 scene."""

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
            raise ValueError(f"visual goal '{goal.purpose}' requires at least one data mark")
        data_views: list[VisualizationDataView] = []
        marks: list[VisualizationDataMark] = []
        semantics: list[VisualizationSemantic] = []
        bindings: dict[str, VisualizationBinding] = {}
        source_refs: list[str] = []
        scales: dict[str, VisualizationScale] = {}
        space_id = "space_0"

        for layer_index, plan in enumerate(goal.layers):
            source = self.catalog.resolve(plan.source_ref)
            view, layer_bindings = self._materialize_view(
                source,
                view_id=f"view_{layer_index}",
                transforms=plan.transform,
            )
            encoding = _encoding_fields(plan.encoding)
            _validate_layer_encoding(plan.mark, encoding, source)
            x_field = _first_encoding_field(encoding.get("x"))
            y_field = _first_encoding_field(encoding.get("y"))
            if not x_field or not y_field:
                raise ValueError(f"data mark '{plan.role}' requires explicit x and y fields")
            x_scale_id = "scale_x_0"
            y_axis_index = _requested_y_axis_index(plan.presentation)
            y_scale_id = f"scale_y_{y_axis_index}"
            self._merge_scale(
                scales,
                _scale_for_field(
                    view,
                    x_field,
                    scale_id=x_scale_id,
                    channel="x",
                    scale_type="time" if _field_type(view, x_field) == "time" else "category",
                ),
            )
            self._merge_scale(
                scales,
                _scale_for_field(
                    view,
                    y_field,
                    scale_id=y_scale_id,
                    channel="y",
                    scale_type=_goal_y_scale_type(goal, y_axis_index),
                ),
            )
            marks.append(VisualizationDataMark(
                mark_id=f"mark_{layer_index}",
                mark=plan.mark,
                role=plan.role,
                source_ref=source.ref,
                data_view_id=view.view_id,
                space_id=space_id,
                encoding={
                    channel: VisualizationChannelBinding(
                        fields=value if isinstance(value, list) else [value],
                        scale_id=(
                            x_scale_id if channel == "x"
                            else y_scale_id if channel in {"y", "lower", "upper"}
                            else None
                        ),
                    )
                    for channel, value in encoding.items()
                },
                transform=[item.model_dump(mode="json") for item in plan.transform],
                presentation=_presentation_options(plan.presentation),
                label=plan.label,
            ))
            data_views.append(view)
            for provenance_ref in [source.ref, *plan.provenance_source_refs]:
                source_refs.extend(self._public_source_refs(provenance_ref))
            for binding in layer_bindings:
                bindings[binding.binding_id] = binding

        semantic_materializations = []
        for semantic_index, plan in enumerate(goal.semantics):
            source = self.catalog.resolve(plan.source_ref)
            view, semantic_bindings = self._materialize_view(
                source,
                view_id=f"semantic_view_{semantic_index}",
            )
            _validate_semantic_plan(plan, view)
            data_views.append(view)
            semantic_materializations.append((plan, view))
            source_refs.extend(self._public_source_refs(source.ref))
            for provenance_ref in plan.provenance_source_refs:
                source_refs.extend(self._public_source_refs(provenance_ref))
            for binding in semantic_bindings:
                bindings[binding.binding_id] = binding

        for plan, view in semantic_materializations:
            if plan.semantic_type == "relation":
                continue
            semantics.extend(self._materialize_semantic_group(
                plan,
                view,
                scales=scales,
                space_id=space_id,
            ))
        for plan, view in semantic_materializations:
            if plan.semantic_type != "relation":
                continue
            semantics.extend(self._materialize_relation_group(
                plan,
                view,
                semantics=semantics,
                bindings=bindings,
            ))

        y_scale_ids = [scale.scale_id for scale in scales.values() if scale.channel == "y"]
        coordinate_spaces = [VisualizationCoordinateSpace(
            space_id=space_id,
            x_scale_id="scale_x_0",
            y_scale_ids=y_scale_ids,
        )]
        guides = _materialize_guides(goal, marks, semantics)
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
            required_roles=list(dict.fromkeys(goal.required_roles)),
            data_views=data_views,
            scales=list(scales.values()),
            coordinate_spaces=coordinate_spaces,
            marks=marks,
            semantics=semantics,
            guides=guides,
            layout=VisualizationLayout(
                mode=_layout_from_chart_plan(goal),
                cells=[VisualizationLayoutCell(cell_id="cell_0", space_ids=[space_id])],
            ),
            bindings=list(bindings.values()),
            presentation=_presentation_options(goal.presentation),
            accessibility=_accessibility_v4(goal, data_views, semantics),
        )
        return payload

    def _materialize_view(
        self,
        source: PresentationSource,
        *,
        view_id: str,
        transforms=(),
    ) -> tuple[VisualizationDataView, list[VisualizationBinding]]:
        rows, scalar = _source_data(source)
        rows = _apply_presentation_transforms(rows, transforms, source)
        if transforms:
            scalar = None
        records, bindings = _records_for_source(source, rows, scalar, view_id=view_id)
        if not records:
            raise ValueError(f"visual source '{source.ref}' produced no grounded records")
        fields = _visualization_fields(source, records)
        allowed_fields = {field.name for field in fields}
        normalized_records = [record.model_copy(update={
            "values": {
                key: value
                for key, value in record.values.items()
                if key in allowed_fields
            },
        }) for record in records]
        return VisualizationDataView(
            view_id=view_id,
            source_ref=source.ref,
            fields=fields,
            records=normalized_records,
        ), bindings

    def _merge_scale(
        self,
        scales: dict[str, VisualizationScale],
        candidate: VisualizationScale,
    ) -> None:
        current = scales.get(candidate.scale_id)
        if current is None:
            scales[candidate.scale_id] = candidate
            return
        current_type = "category" if current.data_type in {"category", "string"} else current.data_type
        candidate_type = "category" if candidate.data_type in {"category", "string"} else candidate.data_type
        if current_type != candidate_type:
            raise IncompatibleVisualDomainError(
                f"scale '{candidate.scale_id}' mixes incompatible data types "
                f"{current.data_type!r} and {candidate.data_type!r}"
            )
        if current.unit and candidate.unit and current.unit != candidate.unit:
            raise IncompatibleVisualDomainError(
                f"scale '{candidate.scale_id}' mixes incompatible units {current.unit!r} and {candidate.unit!r}"
            )

    def _materialize_semantic_group(
        self,
        plan: VisualSemanticPlan,
        view: VisualizationDataView,
        *,
        scales: dict[str, VisualizationScale],
        space_id: str,
    ) -> list[VisualizationSemantic]:
        result = []
        for index, record in enumerate(view.records):
            semantic_id = plan.semantic_id if len(view.records) == 1 else f"{plan.semantic_id}:{record.record_id}"
            target = _semantic_target(
                plan,
                record,
                scales=scales,
                space_id=space_id,
            )
            result.append(VisualizationSemantic(
                semantic_id=semantic_id,
                group_id=plan.semantic_id,
                semantic_type=plan.semantic_type,
                role=plan.role,
                source_ref=plan.source_ref,
                target=target,
                content=_semantic_content(plan, view, record),
                importance=plan.importance,
                line_style=plan.line_style,
                symbol=plan.symbol,
                presentation=_presentation_options(plan.presentation),
                binding_id=record.binding_id,
            ))
        return result

    def _materialize_relation_group(
        self,
        plan: VisualSemanticPlan,
        view: VisualizationDataView,
        *,
        semantics: list[VisualizationSemantic],
        bindings: dict[str, VisualizationBinding],
    ) -> list[VisualizationSemantic]:
        by_group: dict[str, list[VisualizationSemantic]] = {}
        for semantic in semantics:
            by_group.setdefault(semantic.group_id, []).append(semantic)
        members = []
        for related_id in plan.related_semantic_ids:
            candidates = by_group.get(related_id, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"relation '{plan.semantic_id}' requires exactly one grounded member for "
                    f"semantic group '{related_id}', got {len(candidates)}"
                )
            if candidates[0].semantic_type not in {"event", "observation"}:
                raise ValueError("relation members must be grounded event or observation semantics")
            members.append(candidates[0])
        member_item_ids = {
            binding.item_id
            for member in members
            if member.binding_id
            if (binding := bindings.get(member.binding_id)) is not None and binding.item_id
        }
        if len(member_item_ids) != len(members):
            raise ValueError(f"relation '{plan.semantic_id}' members require stable Insight item ids")

        result = []
        for index, record in enumerate(view.records):
            relation_binding = bindings.get(record.binding_id or "")
            related_item_ids = set(relation_binding.related_item_ids) if relation_binding else set()
            if related_item_ids != member_item_ids:
                raise ValueError(
                    f"relation '{plan.semantic_id}' provenance does not match its selected members; "
                    f"expected={sorted(member_item_ids)}, actual={sorted(related_item_ids)}"
                )
            semantic_id = plan.semantic_id if len(view.records) == 1 else f"{plan.semantic_id}:{record.record_id}"
            result.append(VisualizationSemantic(
                semantic_id=semantic_id,
                group_id=plan.semantic_id,
                semantic_type="relation",
                role=plan.role,
                source_ref=plan.source_ref,
                target=RelationSemanticTarget(
                    semantic_ids=[member.semantic_id for member in members],
                ),
                content=_semantic_content(plan, view, record),
                importance=plan.importance,
                line_style=plan.line_style,
                symbol=plan.symbol,
                presentation=_presentation_options(plan.presentation),
                binding_id=record.binding_id,
            ))
        return result

    def _public_source_refs(self, ref: str) -> list[str]:
        source = self.catalog.resolve(ref)
        if source.ref.startswith("semantic:") and source.kind == "view":
            return list(source.value.lineage)
        return [source.ref]


class VisualizationSemanticValidator:
    """Validate grounded invariants independently of the chart-planning LLM."""

    def __init__(self, catalog: PresentationCatalog, *, required_located_roles: set[str] | None = None):
        self.catalog = catalog
        self.required_located_roles = required_located_roles or set()

    def validate(self, goal: VisualGoal, payload: VisualizationPayload) -> None:
        materialized_roles = {
            item.role.strip().casefold()
            for item in [*payload.marks, *payload.semantics]
        }
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
        views = {view.view_id: view for view in payload.data_views}
        for mark in payload.marks:
            source = self.catalog.resolve(mark.source_ref)
            encoding = {
                channel: binding.fields if len(binding.fields) > 1 else binding.fields[0]
                for channel, binding in mark.encoding.items()
            }
            _validate_layer_encoding(mark.mark, encoding, source)
            if mark.mark.strip().casefold() in {"line", "area"} and len(views[mark.data_view_id].records) < 2:
                raise ValueError(
                    f"{mark.mark} mark '{mark.role}' requires at least two grounded records"
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


def _records_for_source(
    source: PresentationSource,
    rows: list[dict],
    scalar: dict | None,
    *,
    view_id: str,
) -> tuple[list[VisualizationRecord], list[VisualizationBinding]]:
    source_bindings = _projection_bindings(source)
    bindings_by_id = {
        binding.binding_id: binding
        for binding in source_bindings.values()
    }
    records = list(rows)
    if scalar is not None and not records:
        records = [dict(scalar)]
    output: list[VisualizationRecord] = []
    used_bindings: dict[str, VisualizationBinding] = {}
    for index, row in enumerate(records):
        item_id = str(row.get("item_id") or "")
        explicit_binding_id = str(row.get("__binding_id") or "")
        binding = (
            bindings_by_id.get(explicit_binding_id)
            or source_bindings.get(item_id)
            or source_bindings.get("")
        )
        if binding is not None:
            used_bindings[binding.binding_id] = binding
        record_id = item_id or f"{view_id}:{index}"
        output.append(VisualizationRecord(
            record_id=record_id,
            values={
                str(key): value
                for key, value in row.items()
                if not str(key).startswith("__")
            },
            binding_id=binding.binding_id if binding else None,
        ))
    return output, list(used_bindings.values())


def _visualization_fields(
    source: PresentationSource,
    records: list[VisualizationRecord],
) -> list[VisualizationField]:
    schema = {
        str(item.get("name")): str(item.get("data_type") or "string")
        for item in _source_schema(source)
        if isinstance(item, dict) and item.get("name")
    }
    semantics = source.value.field_semantics if source.kind == "view" else {}
    supported = {"time", "number", "category", "string", "boolean"}
    present = {key for record in records for key in record.values}
    result = []
    for name, data_type in schema.items():
        if name not in present or data_type not in supported:
            continue
        result.append(VisualizationField(
            name=name,
            data_type=data_type,
            semantic_role=str(semantics.get(name) or name),
            unit=_source_field_unit(source),
        ))
    return result


def _source_field_unit(source: PresentationSource) -> str | None:
    if source.kind == "insight":
        return source.value.unit
    if source.kind == "insight_item":
        return source.value[0].unit
    return None


def _field(view: VisualizationDataView, name: str) -> VisualizationField:
    candidate = next((field for field in view.fields if field.name == name), None)
    if candidate is None:
        raise ValueError(
            f"visual data view '{view.view_id}' does not expose field '{name}'; "
            f"available={sorted(field.name for field in view.fields)}"
        )
    return candidate


def _field_type(view: VisualizationDataView, name: str) -> str:
    return _field(view, name).data_type


def _scale_for_field(
    view: VisualizationDataView,
    field_name: str,
    *,
    scale_id: str,
    channel: str,
    scale_type: str,
) -> VisualizationScale:
    field = _field(view, field_name)
    data_type = field.data_type
    if data_type == "boolean":
        data_type = "category"
    return VisualizationScale(
        scale_id=scale_id,
        channel=channel,
        data_type=data_type,
        semantic_role=field.semantic_role,
        unit=field.unit,
        scale_type=scale_type,
    )


def _requested_y_axis_index(presentation: dict | None) -> int:
    value = (presentation or {}).get("yAxisIndex")
    return value if isinstance(value, int) and value >= 0 else 0


def _goal_y_scale_type(goal: VisualGoal, index: int) -> str:
    y_axis = goal.presentation.get("yAxis") if isinstance(goal.presentation, dict) else None
    entries = y_axis if isinstance(y_axis, list) else [y_axis] if isinstance(y_axis, dict) else []
    entry = entries[index] if index < len(entries) and isinstance(entries[index], dict) else {}
    return "log" if entry.get("type") == "log" else "linear"


def _validate_semantic_plan(plan: VisualSemanticPlan, view: VisualizationDataView) -> None:
    fields = {field.name for field in view.fields}
    referenced = {
        *plan.target_encoding.values(),
        *plan.metric_fields,
        *([plan.description_field] if plan.description_field else []),
    }
    unknown = referenced - fields
    if unknown:
        raise ValueError(
            f"semantic '{plan.semantic_id}' references unavailable fields {sorted(unknown)} "
            f"from '{plan.source_ref}'"
        )
    if plan.semantic_type in {"observation", "reference"}:
        numeric_field = plan.target_encoding["y" if plan.semantic_type == "observation" else "value"]
        if _field_type(view, numeric_field) != "number":
            raise ValueError(f"semantic '{plan.semantic_id}' requires a numeric y target")
    if plan.semantic_type == "event" and _field_type(view, plan.target_encoding["x"]) not in {
        "time", "category", "string", "boolean",
    }:
        raise ValueError(f"event semantic '{plan.semantic_id}' requires a time or category x target")
    if plan.semantic_type == "observation" and _field_type(view, plan.target_encoding["x"]) not in {
        "time", "category", "string", "boolean",
    }:
        raise ValueError(f"observation semantic '{plan.semantic_id}' requires a compatible x target")
    if plan.semantic_type == "interval":
        start_type = _field_type(view, plan.target_encoding["start"])
        end_type = _field_type(view, plan.target_encoding["end"])
        if start_type != end_type or start_type not in {"time", "number", "category", "string"}:
            raise ValueError(f"interval semantic '{plan.semantic_id}' requires compatible boundaries")


def _semantic_target(
    plan: VisualSemanticPlan,
    record: VisualizationRecord,
    *,
    scales: dict[str, VisualizationScale],
    space_id: str,
):
    values = record.values
    x_scale = scales["scale_x_0"]
    y_scale_id = f"scale_y_{_requested_y_axis_index(plan.presentation)}"
    if plan.semantic_type == "fact":
        return ChartSemanticTarget()
    if plan.semantic_type == "event":
        x = values.get(plan.target_encoding["x"])
        _require_value(x, semantic_id=plan.semantic_id, channel="x")
        _require_target_type(plan.semantic_id, _python_visual_type(x), x_scale.data_type, channel="x")
        return XSemanticTarget(
            space_id=space_id,
            scale_id=x_scale.scale_id,
            record_id=record.record_id,
            x=x,
        )
    if plan.semantic_type == "observation":
        x = values.get(plan.target_encoding["x"])
        y = _number(values.get(plan.target_encoding["y"]))
        _require_value(x, semantic_id=plan.semantic_id, channel="x")
        _require_value(y, semantic_id=plan.semantic_id, channel="y")
        y_scale = scales.get(y_scale_id)
        if y_scale is None:
            raise ValueError(f"semantic '{plan.semantic_id}' targets unavailable y scale '{y_scale_id}'")
        _require_target_type(plan.semantic_id, _python_visual_type(x), x_scale.data_type, channel="x")
        return XYSemanticTarget(
            space_id=space_id,
            x_scale_id=x_scale.scale_id,
            y_scale_id=y_scale.scale_id,
            record_id=record.record_id,
            x=x,
            y=y,
        )
    if plan.semantic_type == "interval":
        start = values.get(plan.target_encoding["start"])
        end = values.get(plan.target_encoding["end"])
        _require_value(start, semantic_id=plan.semantic_id, channel="start")
        _require_value(end, semantic_id=plan.semantic_id, channel="end")
        if _ordered_filter_value(start) > _ordered_filter_value(end):
            raise ValueError(f"interval semantic '{plan.semantic_id}' has reversed boundaries")
        _require_target_type(plan.semantic_id, _python_visual_type(start), x_scale.data_type, channel="interval")
        return IntervalSemanticTarget(
            space_id=space_id,
            scale_id=x_scale.scale_id,
            axis="x",
            record_id=record.record_id,
            start=start,
            end=end,
        )
    if plan.semantic_type == "reference":
        value = _number(values.get(plan.target_encoding["value"]))
        _require_value(value, semantic_id=plan.semantic_id, channel="value")
        y_scale = scales.get(y_scale_id)
        if y_scale is None:
            raise ValueError(f"semantic '{plan.semantic_id}' targets unavailable y scale '{y_scale_id}'")
        return ReferenceSemanticTarget(
            space_id=space_id,
            scale_id=y_scale.scale_id,
            axis="y",
            record_id=record.record_id,
            value=value,
        )
    raise ValueError("relation targets are resolved only after their member semantics")


def _require_value(value: Any, *, semantic_id: str, channel: str) -> None:
    if value is None:
        raise ValueError(f"semantic '{semantic_id}' has null target channel '{channel}'")


def _python_visual_type(value: Any) -> str:
    if _time_value(value) is not None:
        return "time"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    return "category"


def _require_target_type(semantic_id: str, actual: str, expected: str, *, channel: str) -> None:
    normalized_expected = "category" if expected in {"category", "string"} else expected
    normalized_actual = "category" if actual in {"category", "string"} else actual
    if normalized_actual != normalized_expected:
        raise IncompatibleVisualDomainError(
            f"semantic '{semantic_id}' {channel} target type '{actual}' is incompatible "
            f"with host scale type '{expected}'"
        )


def _semantic_content(
    plan: VisualSemanticPlan,
    view: VisualizationDataView,
    record: VisualizationRecord,
) -> VisualizationSemanticContent:
    description = None
    if plan.description_field:
        raw_description = record.values.get(plan.description_field)
        description = str(raw_description) if raw_description is not None else None
    metrics = []
    for metric_field in plan.metric_fields:
        field = _field(view, metric_field)
        value = record.values.get(metric_field)
        if value is None:
            continue
        metrics.append(VisualizationMetric(
            label=field.semantic_role,
            value=value,
            data_type=field.data_type,
            unit=field.unit,
        ))
    return VisualizationSemanticContent(
        title=plan.label,
        description=description,
        metrics=metrics,
    )


def _materialize_guides(
    goal: VisualGoal,
    marks: list[VisualizationDataMark],
    semantics: list[VisualizationSemantic],
) -> list[VisualizationGuide]:
    if not goal.guides:
        return []
    marks_by_id = {mark.mark_id: mark for mark in marks}
    semantics_by_group: dict[str, list[VisualizationSemantic]] = {}
    for semantic in semantics:
        semantics_by_group.setdefault(semantic.group_id, []).append(semantic)
    sections = []
    for section in goal.guides:
        entries = []
        for target_id in section.target_ids:
            if section.section_type == "data":
                mark = marks_by_id[target_id]
                entries.append(VisualizationGuideEntry(
                    entry_id=f"guide:{section.section_id}:{target_id}",
                    label=mark.label or mark.role,
                    target_type="mark",
                    target_id=target_id,
                    interaction="toggle",
                    swatch=_mark_swatch(mark.mark),
                ))
                continue
            group = semantics_by_group[target_id]
            binding_ids = {semantic.binding_id for semantic in group if semantic.binding_id}
            entries.append(VisualizationGuideEntry(
                entry_id=f"guide:{section.section_id}:{target_id}",
                label=group[0].content.title,
                target_type="semantic",
                target_id=target_id,
                interaction="select" if len(binding_ids) == 1 else "none",
                swatch=group[0].semantic_type,
                binding_id=next(iter(binding_ids)) if len(binding_ids) == 1 else None,
            ))
        sections.append(VisualizationGuideSection(
            section_id=section.section_id,
            section_type=section.section_type,
            label=section.label,
            entries=entries,
        ))
    return [VisualizationGuide(guide_id="legend_0", sections=sections)]


def _mark_swatch(mark: str) -> str:
    normalized = str(mark or "").strip().casefold()
    return normalized if normalized in {"line", "point", "bar", "area", "band"} else "mark"


def _accessibility_v4(
    goal: VisualGoal,
    data_views: list[VisualizationDataView],
    semantics: list[VisualizationSemantic],
) -> VisualizationAccessibility:
    rows = [
        {"view": view.view_id, **record.values}
        for view in data_views
        for record in view.records[:12]
    ]
    rows.extend({
        "semantic_type": semantic.semantic_type,
        "role": semantic.role,
        "title": semantic.content.title,
        "metrics": [metric.model_dump(mode="json") for metric in semantic.content.metrics],
    } for semantic in semantics[:12])
    return VisualizationAccessibility(
        description=goal.summary or goal.purpose,
        table_columns=_row_fields(rows),
        table_rows=rows[:24],
    )


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
    elif normalized_mark == "annotation":
        if not _first_encoding_field(encoding.get("label")) and not y_field:
            raise ValueError("annotation layer requires an explicit label or value encoding")
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
            x=row.get(x_field) if x_field else None, y=y,
            lower=lower, upper=upper,
            label=str(row.get(label_field) or row.get("label") or row.get("type") or "") or None,
            binding_id=str(row.get("__binding_id") or binding_by_item.get(item_id) or "") or None,
            # Keep an immutable copy of every grounded field so renderer-native
            # encodings can address shapes beyond the normalized x/y pair.
            metadata=dict(row),
        )
        if (
            point.x is not None
            or point.y is not None
            or point.lower is not None
            or point.upper is not None
            or point.label is not None
        ):
            points.append(point)
    if scalar and not points:
        y_key = y_field
        x_key = x_field
        binding_id = bindings[0].binding_id if bindings else None
        point = VisualizationPoint(
            x=scalar.get(x_key) if x_key else None,
            y=_number(scalar.get(y_key)) if y_key else None,
            label=str(scalar.get(label_field) or scalar.get("label") or "") or None,
            binding_id=binding_id,
            metadata=dict(scalar),
        )
        if point.x is not None or point.y is not None or point.label is not None:
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
        related_item_ids=list(item.source_item_ids),
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
    axis_dataset_ids = {
        dataset.dataset_id
        for plan, dataset in zip(goal.layers, datasets, strict=True)
        if plan.mark.strip().casefold() not in {"rule", "annotation"}
    }
    domain_types = {
        dimension.data_type
        for dataset in datasets
        if dataset.dataset_id in axis_dataset_ids
        for dimension in dataset.dimensions
        if dimension.role == "x"
    }
    normalized = {"category" if item in {"category", "string"} else item for item in domain_types}
    if len(normalized) > 1:
        raise IncompatibleVisualDomainError(
            "one shared chart requires compatible x-domain semantics across all layers; "
            "keep time-aligned layers in the primary visual goal and leave scalar summaries in answer text "
            "or an explicitly requested supporting chart"
        )
    annotation_domain_types = {
        dimension.data_type
        for plan, dataset in zip(goal.layers, datasets, strict=True)
        if plan.mark.strip().casefold() == "annotation"
        and _first_encoding_field(_encoding_fields(plan.encoding).get("x"))
        for dimension in dataset.dimensions
        if dimension.role == "x"
    }
    normalized_annotations = {
        "category" if item in {"category", "string"} else item
        for item in annotation_domain_types
    }
    if normalized_annotations and normalized_annotations != normalized:
        raise IncompatibleVisualDomainError(
            "a positioned annotation must use the same x-domain semantics as its host layer; "
            f"host={sorted(normalized)}, annotation={sorted(normalized_annotations)}"
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


def _source_semantic_contract(sources: list[PresentationSource]) -> dict:
    """Describe what an artifact-backed view has actually materialized.

    Lineage alone says where values came from, but not whether an operation was
    applied to the input. Downstream LLMs must not infer that anomaly detections
    are an exclusion-applied dataset or that raw evidence is a derived result.
    """

    by_kind: dict[str, PresentationSource] = {}
    for source in sources:
        by_kind.setdefault(source.kind, source)
    if "insight" in by_kind or "insight_item" in by_kind:
        source = by_kind.get("insight") or by_kind["insight_item"]
        value = source.value[0] if source.kind == "insight_item" else source.value
        return {
            "data_role": "key_insight_claim",
            "materializes_input_transformation": False,
            "operation_description": getattr(value, "calculation_trace", None),
            "supported_visual_uses": ["target_claim", "reference_guide", "annotation"],
            "limitations": [
                "A scalar claim does not materialize the contextual series used to calculate it."
            ],
        }
    if "derived_evidence" in by_kind:
        artifact = by_kind["derived_evidence"].value
        return {
            "data_role": "derived_transformation_result",
            "materializes_input_transformation": True,
            "operation_description": getattr(artifact, "transform_summary", None),
            "supported_visual_uses": ["transformed_context", "derived_series", "derived_markers"],
            "limitations": [],
        }
    if "forecast" in by_kind:
        return {
            "data_role": "forecast_output",
            "materializes_input_transformation": True,
            "operation_description": "Model-produced forecast values or intervals.",
            "supported_visual_uses": ["forecast_series", "forecast_interval"],
            "limitations": ["Does not represent cleaned or exclusion-applied historical observations."],
        }
    if "anomaly" in by_kind:
        return {
            "data_role": "anomaly_detection_output",
            "materializes_input_transformation": False,
            "operation_description": "Detected anomaly points, scores, spans, or detector status.",
            "supported_visual_uses": ["anomaly_markers", "exclusion_markers", "anomaly_scores"],
            "limitations": [
                "Does not contain the retained/cleaned input records after exclusions.",
                "Does not apply anomaly exclusions to the input series.",
            ],
        }
    if "evidence" in by_kind:
        return {
            "data_role": "raw_observations",
            "materializes_input_transformation": False,
            "operation_description": "Database observations materialized from the executed query.",
            "supported_visual_uses": ["complete_context", "raw_series", "comparison_baseline"],
            "limitations": ["Does not materialize downstream analysis transformations."],
        }
    return {
        "data_role": "unclassified_grounded_view",
        "materializes_input_transformation": False,
        "operation_description": None,
        "supported_visual_uses": [],
        "limitations": ["No transformation result is declared by lineage."],
    }


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


def _render_contract(value: DataViewValue) -> dict:
    """Compute executable visual consumers from materialized shape and cardinality."""

    fields_by_type: dict[str, list[str]] = {}
    for field in value.schema_fields:
        if not isinstance(field, dict):
            continue
        fields_by_type.setdefault(str(field.get("data_type") or "unknown"), []).append(
            str(field.get("name") or "")
        )
    point_count = len(value.rows) or int(value.scalar is not None)
    time_fields = [field for field in fields_by_type.get("time", []) if field]
    number_fields = [field for field in fields_by_type.get("number", []) if field]
    category_fields = [
        field
        for data_type in ("category", "string", "boolean")
        for field in fields_by_type.get(data_type, [])
        if field
    ]
    semantic_roles = {
        str(field): str(value.field_semantics.get(field) or "").strip().casefold()
        for field in number_fields
    }
    lower_fields = [
        field
        for field, role in semantic_roles.items()
        if role == "lower_bound" or role.startswith("lower_") or role.endswith("_lower")
    ]
    upper_fields = [
        field
        for field, role in semantic_roles.items()
        if role == "upper_bound" or role.startswith("upper_") or role.endswith("_upper")
    ]
    allowed: list[str] = []
    if point_count >= 1 and time_fields and number_fields:
        allowed.append("event_points")
    if point_count >= 2 and time_fields and number_fields:
        allowed.append("series")
    # Two arbitrary numeric columns do not constitute an uncertainty band.
    # The semantic projection LLM must explicitly identify complementary
    # lower/upper roles before the chart planner may consume them as a band.
    if point_count >= 2 and time_fields and lower_fields and upper_fields:
        allowed.append("band")
    if point_count >= 1 and category_fields and number_fields:
        allowed.append("comparison")
    if point_count >= 1 and len(time_fields) >= 2:
        allowed.append("interval_bounds")
    if point_count >= 1 and number_fields:
        allowed.append("reference_line")
    if point_count >= 1 and any(fields_by_type.values()):
        allowed.append("annotation")
    return {
        "data_shape": value.shape,
        "point_count": point_count,
        "allowed_layer_types": allowed,
        "time_fields": time_fields,
        "number_fields": number_fields,
        "category_fields": category_fields,
        "lower_fields": lower_fields,
        "upper_fields": upper_fields,
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
        "timestamp": item.timestamp, "source_item_ids": list(item.source_item_ids),
        **item.dimensions, **item.locator,
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


def _projection_path_contracts(source: PresentationSource) -> dict:
    """Describe the two executable path namespaces used by semantic projection.

    With ``record_path=null`` the materializer reads the source's normalized
    presentation rows.  With a selected record path it reads records from the
    nested projection document and every source path becomes relative to one
    such record.  Publishing both namespaces prevents an LLM from combining a
    container path from the latter with a field path from the former.
    """

    rows, scalar = _source_data(source)
    default_records = rows or ([dict(scalar)] if scalar else [])
    root = _projection_root(source)
    record_contracts = []
    for record_path in _record_path_candidates(root):
        try:
            selected = _records_at_path(root, record_path)
        except ValueError:
            continue
        record_contracts.append({
            "record_path": record_path,
            "source_paths": _relative_leaf_paths(selected),
        })
    return {
        "default_source_paths": _relative_leaf_paths(default_records),
        "record_paths": record_contracts,
    }


def _relative_leaf_paths(records: list[dict], *, max_depth: int = 6) -> list[str]:
    """Return bounded scalar leaf paths relative to one selected record."""

    paths: list[str] = []

    def child_path(path: str, key: Any) -> str:
        name = str(key)
        if re.fullmatch(r"[^.\[\]]+", name):
            return f"{path}.{name}"
        escaped = name.replace("'", "\\'")
        return f"{path}['{escaped}']"

    def visit(value: Any, path: str, depth: int) -> None:
        if len(paths) >= 128 or depth > max_depth:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, child_path(path, key), depth + 1)
            return
        # Arrays define another record grain.  A field source path selects one
        # scalar value and therefore never crosses an array boundary.
        if isinstance(value, (list, tuple)):
            return
        if path != "$" and path not in paths:
            paths.append(path)

    for record in records[:4]:
        if isinstance(record, dict):
            visit(record, "$", 0)
    return paths


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
