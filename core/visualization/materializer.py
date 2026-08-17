"""Grounded Visualization V3 catalog, layer materializer, and semantic validation."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from schemas.key_insight import KeyInsight, InsightItem
from schemas.output import VisualGoal, VisualLayerPlan
from schemas.state import RequestStateModel
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


SUPPORTED_MARKS = ["line", "point", "bar", "area", "band", "rule", "rect", "text", "boxplot", "table"]


@dataclass(frozen=True)
class DataViewValue:
    rows: list[dict]
    scalar: dict | None
    shape: str
    schema_fields: list[dict]
    lineage: list[str]
    name: str


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
                {"timestamp": item.timestamp, "lower": item.lower, "upper": item.upper}
                for item in forecast.confidence_interval
            ]
            if interval_rows:
                self._register_view(
                    f"view:forecast:{forecast_id}:interval", name="Confidence interval", shape="intervals",
                    rows=interval_rows, scalar=None, lineage=[f"forecast:{forecast_id}"],
                )
        for anomaly_id, anomaly in request_state.anomaly_artifacts.items():
            self._register_artifact(
                "anomaly", anomaly_id, anomaly,
                source_type="anomaly", label=anomaly.detector_name,
            )
            rows = [dict(item) for item in anomaly.anomaly_points if isinstance(item, dict)]
            if rows:
                lineage = [f"anomaly:{anomaly_id}"]
                evidence_ref = _anomaly_evidence_ref(anomaly)
                if evidence_ref:
                    lineage.append(evidence_ref)
                self._register_view(
                    f"view:anomaly:{anomaly_id}:points", name="Anomaly points", shape="records",
                    rows=rows, scalar=None, lineage=lineage,
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
    ) -> None:
        fields = schema_fields or _schema_fields(rows or ([scalar] if scalar else []))
        self._sources[ref] = PresentationSource(
            ref, "view", DataViewValue(rows, scalar, shape, fields, list(dict.fromkeys(lineage)), name)
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

    def canonical_refs(self) -> list[str]:
        return sorted(key for key, source in self._sources.items() if key == source.ref and ":" in key)

    def renderable_refs(self) -> list[str]:
        return sorted(
            ref
            for ref, source in self._sources.items()
            if ref == source.ref and source.kind in {"view", "insight", "insight_item"}
        )

    def expand_preferences(self, refs: list[str]) -> tuple[set[str], set[str]]:
        """Expand stable artifact refs into renderable sources without exposing storage refs."""

        renderable = set(self.renderable_refs())
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
                continue
            expanded.update(self._views_for_lineage_ref(source.ref, renderable))
            if source.kind == "analysis":
                expanded.update(self._analysis_renderable_refs(source.value, renderable))
        return expanded, unknown

    def _views_for_lineage_ref(self, artifact_ref: str, renderable: set[str]) -> set[str]:
        return {
            ref
            for ref in renderable
            if (source := self._sources.get(ref)) is not None
            and source.kind == "view"
            and artifact_ref in source.value.lineage
        }

    def _analysis_renderable_refs(self, analysis, renderable: set[str]) -> set[str]:
        refs: set[str] = set()
        evidence_id = str(getattr(analysis, "input_evidence_id", "") or "").removeprefix("evidence:")
        if evidence_id:
            refs.update(self._views_for_lineage_ref(f"evidence:{evidence_id}", renderable))
        for derived in getattr(analysis, "derived_evidence", []) or []:
            derived_id = str(getattr(derived, "evidence_id", "") or "")
            if derived_id:
                refs.update(self._views_for_lineage_ref(f"derived_evidence:{derived_id}", renderable))
        for insight in getattr(analysis, "produced_insights", []) or []:
            candidate = f"insight:{getattr(insight, 'insight_id', '')}"
            if candidate in renderable:
                refs.add(candidate)
            refs.update(ref for ref in renderable if ref.startswith(f"{candidate}#"))
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

    def planner_inventory(self) -> dict:
        return {
            "schema_version": "3",
            "marks": SUPPORTED_MARKS,
            "sources": [
                self._source_inventory(source)
                for ref, source in sorted(self._sources.items())
                if ref == source.ref and source.kind in {"view", "insight", "insight_item"}
            ],
            "rules": [
                "Plan visual_goals around the user's purpose and declare every required semantic role.",
                "Each layer owns exactly one grounded source_ref; never replace a base series with an annotation collection.",
                "Use typed view:* sources for chart data and insight:*#item sources for semantic decision points.",
                "Artifact refs are lineage only and are intentionally absent; outer artifact preferences have already been expanded into renderable sources.",
                "Use only listed fields in encoding and never invent rows, field names, renderer options, or colors.",
                "Business filtering and calculations must already exist in a Data View; presentation does not recompute them.",
                "materialization_complete only describes whether the executed query result was stored without truncation; it does not prove coverage of the user's analysis interval.",
                "Use time_range, query_context, lineage, and the user request together to judge whether a source is complete enough for visual verification.",
            ],
        }

    def _source_inventory(self, source: PresentationSource) -> dict:
        if source.kind == "view":
            value: DataViewValue = source.value
            preview = value.rows[:4] if value.rows else ([value.scalar] if value.scalar else [])
            lineage_sources = []
            for ref in value.lineage:
                resolved = self._sources.get(ref)
                if resolved is not None:
                    lineage_sources.append(resolved)
            materialization_complete = _full_fidelity_status(lineage_sources)
            capabilities = _render_capabilities(value.schema_fields, scalar=value.scalar is not None)
            return {
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
                "preview": [_bounded_row(item) for item in preview if item],
            }
        if source.kind == "insight":
            insight: KeyInsight = source.value
            locator_row = _insight_locator_row(insight)
            fields = _schema_fields(
                [_insight_item_row(item) for item in insight.items]
                or ([locator_row] if locator_row else [])
            )
            return {
                "source_ref": source.ref, "kind": "insight", "status": insight.status, "insight_type": insight.insight_type,
                "semantic_class": insight.semantic_class, "statement": insight.statement, "value_shape": insight.value_shape,
                "item_refs": [f"{source.ref}#{item.item_id}" for item in insight.items[:12]],
                "locator_fields": list(locator_row) if locator_row else [],
                "locator_preview": _bounded_row(locator_row) if locator_row else None,
                "render_capabilities": _render_capabilities(fields, scalar=not bool(fields)),
            }
        if source.kind == "insight_item":
            insight, item = source.value
            fields = _schema_fields([_insight_item_row(item)])
            return {
                "source_ref": source.ref, "kind": "insight_item", "status": insight.status,
                "label": item.label or insight.name, "timestamp": item.timestamp, "value": item.value,
                "schema_fields": fields,
                "render_capabilities": _render_capabilities(fields, scalar=False),
            }
        value = source.value
        return {
            "source_ref": source.ref, "kind": source.kind,
            "summary": getattr(value, "summary", None) or getattr(value, "analysis_goal", None),
        }


class VisualizationMaterializer:
    """Compile grounded V3 layer plans without guessing cross-source precedence."""

    def __init__(self, request_state: RequestStateModel):
        self.request_state = request_state
        self.catalog = PresentationCatalog(request_state)

    def materialize_all(self, goals: list[VisualGoal]) -> list[VisualizationPayload]:
        output: list[VisualizationPayload] = []
        purposes: set[str] = set()
        for index, goal in enumerate(goals):
            key = goal.purpose.strip().casefold()
            if goal.priority == "primary" and key in purposes:
                raise ValueError(f"multiple primary visualizations cover the same purpose: {goal.purpose}")
            if goal.priority == "primary":
                purposes.add(key)
            try:
                output.append(self.materialize(goal, index=index))
            except ValueError:
                if goal.priority == "supporting":
                    continue
                raise
        return output

    def materialize(self, goal: VisualGoal, *, index: int = 0) -> VisualizationPayload:
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
            source_refs.append(source.ref)
            for binding in layer_bindings:
                bindings[binding.binding_id] = binding
        digest = hashlib.sha1(
            f"{goal.purpose}|{'|'.join(source_refs)}|{'|'.join(layer.role for layer in layers)}".encode("utf-8")
        ).hexdigest()[:12]
        payload = VisualizationPayload(
            visualization_id=f"viz_{digest}_{index}", purpose=goal.purpose, priority=goal.priority,
            title=goal.title, summary=goal.summary, source_refs=list(dict.fromkeys(source_refs)),
            required_roles=list(dict.fromkeys(goal.required_roles)), datasets=datasets, layers=layers,
            bindings=list(bindings.values()), layout=_legible_layout_from_datasets(datasets),
            accessibility=_accessibility(goal, datasets),
        )
        VisualizationSemanticValidator(self.catalog).validate(goal, payload)
        return payload

    def _materialize_layer(
        self, plan: VisualLayerPlan, source: PresentationSource, index: int,
    ) -> tuple[VisualizationDataset, VisualizationLayer, list[VisualizationBinding]]:
        rows, scalar = _source_data(source)
        encoding = _encoding_fields(plan.encoding)
        if plan.mark == "table":
            if not rows and scalar:
                rows = [dict(scalar)]
            if not rows:
                raise ValueError(f"layer role '{plan.role}' requires row-shaped data")
            columns = _table_columns(plan.encoding, rows)
            dataset_id = f"dataset_{index}"
            dataset = VisualizationDataset(
                dataset_id=dataset_id, source_ref=source.ref, rows=rows, columns=columns,
            )
            return dataset, VisualizationLayer(
                layer_id=f"layer_{index}", mark=plan.mark, role=plan.role, source_ref=source.ref,
                encoding=encoding, dataset_id=dataset_id, label=plan.label,
            ), []
        points, bindings, x_field, y_field = _points_for_source(source, rows, scalar, encoding)
        if plan.mark == "text" and scalar:
            dataset_id = f"dataset_{index}"
            metric = dict(scalar)
            metric.setdefault("label", plan.label or plan.role)
            dataset = VisualizationDataset(dataset_id=dataset_id, source_ref=source.ref, metric=metric)
            return dataset, VisualizationLayer(
                layer_id=f"layer_{index}", mark=plan.mark, role=plan.role, source_ref=source.ref,
                encoding=encoding, dataset_id=dataset_id, points=points, label=plan.label,
            ), bindings
        if not points:
            raise ValueError(f"layer role '{plan.role}' produced no renderable points from {source.ref}")
        dataset_id = f"dataset_{index}"
        series_id = f"series_{index}"
        series = VisualizationSeries(series_id=series_id, name=plan.label or plan.role, role=plan.role, points=points)
        dataset = VisualizationDataset(
            dataset_id=dataset_id, source_ref=source.ref,
            dimensions=_dimensions(x_field, y_field, rows, scalar), series=[series],
        )
        layer = VisualizationLayer(
            layer_id=f"layer_{index}", mark=plan.mark, role=plan.role, source_ref=source.ref,
            encoding=encoding, dataset_id=dataset_id, series_id=series_id,
            points=points if plan.mark in {"point", "rule", "text", "rect"} else [], label=plan.label,
        )
        return dataset, layer, bindings


class VisualizationSemanticValidator:
    """Validate that materialized layers satisfy the roles authored by the LLM."""

    def __init__(self, catalog: PresentationCatalog):
        self.catalog = catalog

    def validate(self, goal: VisualGoal, payload: VisualizationPayload) -> None:
        role_layers: dict[str, list[VisualizationLayer]] = {}
        datasets = {dataset.dataset_id: dataset for dataset in payload.datasets}
        for layer in payload.layers:
            role_layers.setdefault(layer.role.strip().casefold(), []).append(layer)
            source = self.catalog.resolve(layer.source_ref)
            if source.kind == "insight" and source.value.status != "verified":
                raise ValueError(f"visual layer '{layer.role}' uses non-verified Insight {source.ref}")
            if source.kind == "insight_item" and source.value[0].status != "verified":
                raise ValueError(f"visual layer '{layer.role}' uses non-verified Insight item {source.ref}")
            if source.kind == "view":
                for lineage_ref in source.value.lineage:
                    self.catalog.resolve(lineage_ref)
            dataset = datasets[layer.dataset_id]
            if layer.mark == "table":
                nonempty = bool(dataset.rows)
            elif layer.mark == "text":
                nonempty = bool(dataset.metric)
            else:
                nonempty = bool(dataset.series and dataset.series[0].points)
            if not nonempty:
                raise ValueError(f"visual layer '{layer.role}' is empty")
            available = {field["name"] for field in _source_schema(source)}
            invalid_fields = sorted({value for value in layer.encoding.values() if value not in available})
            if invalid_fields:
                raise ValueError(
                    f"visual layer '{layer.role}' references unavailable fields {invalid_fields}; available fields: {sorted(available)}"
                )
        missing = [role for role in goal.required_roles if not role_layers.get(role.strip().casefold())]
        if missing:
            raise ValueError(f"visualization semantic validation failed; missing required roles: {missing}")


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


def _encoding_fields(encoding: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for channel, value in (encoding or {}).items():
        if isinstance(value, str):
            if channel != "columns":
                result[channel] = value
            continue
        field = getattr(value, "field", None)
        if isinstance(field, str) and field.strip():
            result[channel] = field.strip()
    return result


def _table_columns(encoding: dict, rows: list[dict]) -> list[str]:
    requested = (encoding or {}).get("columns")
    if isinstance(requested, list):
        columns = [str(item).strip() for item in requested if str(item).strip()]
    else:
        columns = list(_encoding_fields(encoding).values())
    return list(dict.fromkeys(columns)) or _row_fields(rows)


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
    x_field = encoding.get("x") or _first_field(rows, "time") or _first_field(rows, "category")
    y_field = encoding.get("y") or encoding.get("value") or _first_field(rows, "number")
    lower_field = encoding.get("lower") or "lower"
    upper_field = encoding.get("upper") or "upper"
    label_field = encoding.get("label")
    points: list[VisualizationPoint] = []
    binding_by_item = {binding.item_id: binding.binding_id for binding in bindings if binding.item_id}
    for row in rows:
        y = _number(row.get(y_field)) if y_field else None
        lower = _number(row.get(lower_field))
        upper = _number(row.get(upper_field))
        item_id = str(row.get("item_id") or "")
        point = VisualizationPoint(
            x=row.get(x_field) if x_field else row.get("label") or row.get("name"), y=y,
            lower=lower, upper=upper,
            label=str(row.get(label_field) or row.get("label") or row.get("type") or "") or None,
            binding_id=binding_by_item.get(item_id),
            metadata={key: value for key, value in row.items() if key not in {x_field, y_field, lower_field, upper_field}},
        )
        if point.x is not None or point.y is not None or point.lower is not None or point.upper is not None:
            points.append(point)
    if scalar:
        y_key = y_field or next((key for key, value in scalar.items() if _number(value) is not None), None)
        x_key = x_field or next((key for key, value in scalar.items() if _time_value(value) is not None), None)
        binding_id = bindings[0].binding_id if bindings else None
        point = VisualizationPoint(
            x=scalar.get(x_key) if x_key else scalar.get("label"),
            y=_number(scalar.get(y_key)) if y_key else None,
            label=str(scalar.get("label") or "") or None, binding_id=binding_id,
        )
        if point.x is not None or point.y is not None:
            points.append(point)
        x_field = x_key or "label"
        y_field = y_key or "value"
    return points, bindings, x_field or "x", y_field or "value"


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
    for name in _row_fields(rows):
        values = [row.get(name) for row in rows if row.get(name) is not None]
        kind = "string"
        if any(_time_value(value) is not None for value in values[:12]):
            kind = "time"
        elif any(_number(value) is not None for value in values[:12]):
            kind = "number"
        elif any(isinstance(value, bool) for value in values[:12]):
            kind = "boolean"
        elif any(isinstance(value, (dict, list)) for value in values[:12]):
            kind = "object"
        elif len({str(value) for value in values[:40]}) <= 12:
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
        if dataset.rows:
            rows.extend(dataset.rows[:12])
        for series in dataset.series:
            rows.extend(
                {"role": series.role, "x": point.x, "y": point.y, "lower": point.lower, "upper": point.upper}
                for point in series.points[:12]
            )
    columns = _row_fields(rows)
    return VisualizationAccessibility(
        description=goal.summary or goal.purpose, table_columns=columns, table_rows=rows[:24],
    )


def _legible_layout_from_datasets(datasets):
    series = [series for dataset in datasets for series in dataset.series]
    scales = []
    for item in series:
        values = [abs(point.y) for point in item.points if point.y is not None and point.y != 0]
        if values:
            scales.append((min(values), max(values)))
    if len(scales) > 1:
        positive = [low for low, _high in scales if low > 0]
        if positive and max(high for _low, high in scales) / min(positive) >= 1000:
            return "facets"
    return "overlay"


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
    if scalar:
        marks = ["text", "table"]
    elif timestamped_numeric:
        marks = ["line", "point", "area", "band", "bar", "rule", "rect", "text", "table"]
    else:
        marks = ["bar", "point", "text", "table"]
    return {
        "timestamped_numeric": timestamped_numeric,
        "scalar_only": scalar,
        "supported_marks": marks,
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
    coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    result = {
        "evidence_ref": source.ref,
        "result_type": evidence.result_type,
        "summary": evidence.summary,
        "query_language": evidence.query_language,
        "query": str(evidence.query or "")[:1600] or None,
        "columns": list(evidence.columns or [])[:40],
        "query_task_contract": coverage.get("query_task_contract"),
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


def _bounded_row(row: dict) -> dict:
    return {str(key): _bounded_value(value) for key, value in list(row.items())[:12]}


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
