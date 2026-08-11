"""Materialize LLM-selected visual intents from canonical request artifacts.

The planner selects an analytical template and grounded source references.  This
module owns all mechanical concerns: field validation, normalization, sampling,
scale legibility, bindings, and the renderer-independent V2 payload.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from schemas.data_fact import DataFact, FactItem
from schemas.output import VisualIntent
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


TEMPLATE_SOURCE_KINDS: dict[str, set[str]] = {
    "metric.single": {"fact", "analysis", "evidence"},
    "table.detail": {"fact", "analysis", "evidence", "forecast", "anomaly"},
    "ranking.topk": {"fact", "evidence", "analysis"},
    "timeseries.trend": {"evidence", "analysis"},
    "timeseries.highlight": {"evidence", "analysis", "fact"},
    "interval.highlight": {"evidence", "analysis", "fact"},
    "timeseries.forecast": {"forecast", "evidence"},
    "timeseries.anomaly": {"anomaly"},
    "category.comparison": {"evidence", "analysis", "fact"},
    "timeseries.comparison": {"evidence", "analysis"},
    "distribution.histogram": {"evidence", "analysis", "fact"},
    "distribution.boxplot": {"evidence", "analysis", "fact"},
    "relationship.scatter": {"evidence", "analysis"},
}


@dataclass(frozen=True)
class PresentationSource:
    ref: str
    kind: str
    value: Any


class PresentationCatalog:
    """Index canonical artifacts and expose a bounded planner inventory."""

    def __init__(self, request_state: RequestStateModel):
        self.request_state = request_state
        self._sources: dict[str, PresentationSource] = {}
        for evidence_id, evidence in request_state.database_evidence_artifacts.items():
            self._register("evidence", evidence_id, evidence)
        for analysis_id, analysis in request_state.analysis_artifacts.items():
            self._register("analysis", analysis_id, analysis)
        for forecast_id, forecast in request_state.forecast_artifacts.items():
            self._register("forecast", forecast_id, forecast)
        for anomaly_id, anomaly in request_state.anomaly_artifacts.items():
            self._register("anomaly", anomaly_id, anomaly)
        for fact in request_state.fact_set.facts:
            self._register("fact", fact.fact_id, fact)

    def _register(self, kind: str, source_id: str, value: Any) -> None:
        canonical = f"{kind}:{source_id}"
        source = PresentationSource(canonical, kind, value)
        self._sources[canonical] = source
        self._sources.setdefault(source_id, source)

    def resolve(self, ref: str) -> PresentationSource:
        source = self._sources.get(str(ref or "").strip())
        if source is None:
            available = sorted(key for key in self._sources if ":" in key)
            raise ValueError(f"unknown presentation source '{ref}'. Available source_refs: {available}")
        return source

    def canonical_refs(self) -> list[str]:
        return sorted(key for key in self._sources if ":" in key)

    def planner_inventory(self) -> dict:
        return {
            "templates": [
                {"template_id": template_id, "accepted_source_kinds": sorted(kinds)}
                for template_id, kinds in TEMPLATE_SOURCE_KINDS.items()
            ],
            "sources": [
                self._source_inventory(source)
                for ref, source in sorted(self._sources.items())
                if ref == source.ref
            ],
            "rules": [
                "Plan views around user goals, not one chart per Fact.",
                "Use one primary view per analytical purpose; supporting views must add precision or a distinct comparison.",
                "Use ranking.topk when only ranked rows are available. Use timeseries.highlight only when an existing full series is relevant to the question.",
                "Use metric.single for a standalone point value; place it on a time series only when temporal position matters and that series already exists.",
                "Use timeseries.forecast as one view combining available history, forecast boundary, forecast points, and confidence interval.",
                "Never request extra data for presentation and never invent source refs or field names.",
                "Encodings may select existing x, y, series, label, lower, or upper fields; omit them when the source shape is unambiguous.",
            ],
        }

    def _source_inventory(self, source: PresentationSource) -> dict:
        value = source.value
        if source.kind == "evidence":
            rows = _rows_from_evidence(value)
            return {
                "source_ref": source.ref,
                "kind": source.kind,
                "result_type": value.result_type,
                "summary": value.summary,
                "columns": list(dict.fromkeys([*(value.columns or []), *_row_fields(rows)])),
                "row_count": len(rows) or _evidence_point_count(value),
                "candidate_time_fields": _candidate_fields(rows, "time"),
                "candidate_numeric_fields": _candidate_fields(rows, "number"),
                "candidate_dimension_fields": _candidate_fields(rows, "category"),
                "preview": [_bounded_row(row) for row in rows[:4]],
            }
        if source.kind == "fact":
            fact: DataFact = value
            return {
                "source_ref": source.ref,
                "kind": source.kind,
                "fact_type": fact.fact_type,
                "semantic_class": fact.semantic_class,
                "derivation": fact.derivation,
                "value_shape": fact.value_shape,
                "statement": fact.statement,
                "unit": fact.unit,
                "item_count": len(fact.items),
                "item_fields": _row_fields([_fact_item_row(item) for item in fact.items]),
                "item_preview": [_bounded_row(_fact_item_row(item)) for item in fact.items[:4]],
            }
        if source.kind == "analysis":
            result = value.result if isinstance(value.result, dict) else {}
            rows = _rows_from_analysis(result)
            return {
                "source_ref": source.ref,
                "kind": source.kind,
                "goal": value.analysis_goal,
                "summary": value.summary,
                "input_evidence_ref": f"evidence:{value.input_evidence_id}",
                "metrics": _bounded_mapping(result.get("metrics")),
                "detail_fields": _row_fields(rows),
                "row_count": len(rows),
                "candidate_time_fields": _candidate_fields(rows, "time"),
                "candidate_numeric_fields": _candidate_fields(rows, "number"),
                "preview": [_bounded_row(row) for row in rows[:4]],
            }
        if source.kind == "forecast":
            coverage = value.diagnostics.get("coverage") if isinstance(value.diagnostics, dict) else {}
            return {
                "source_ref": source.ref,
                "kind": source.kind,
                "model": value.model_name,
                "horizon": value.horizon,
                "forecast_point_count": len(value.forecast_points),
                "confidence_interval_count": len(value.confidence_interval),
                "input_evidence_refs": list((coverage or {}).get("input_evidence_refs") or []),
                "time_range": _point_time_range(value.forecast_points),
            }
        return {
            "source_ref": source.ref,
            "kind": source.kind,
            "detector": value.detector_name,
            "anomaly_point_count": len(value.anomaly_points),
            "anomaly_span_count": len(value.anomaly_spans),
            "input_evidence_ref": _anomaly_evidence_ref(value),
        }


class VisualizationMaterializer:
    def __init__(self, request_state: RequestStateModel):
        self.request_state = request_state
        self.catalog = PresentationCatalog(request_state)
        self.max_points = max(8, int(request_state.context_budget.get("max_visible_points") or 600))

    def materialize_all(self, intents: list[VisualIntent]) -> list[VisualizationPayload]:
        output: list[VisualizationPayload] = []
        seen: set[tuple[str, tuple[str, ...], str]] = set()
        for index, intent in enumerate(intents):
            key = (intent.template_id, tuple(sorted(intent.source_refs)), intent.purpose.strip().lower())
            if key in seen:
                raise ValueError(f"duplicate visualization intent for purpose '{intent.purpose}'")
            seen.add(key)
            output.append(self.materialize(intent, index=index))
        return output

    def materialize(self, intent: VisualIntent, *, index: int = 0) -> VisualizationPayload:
        sources = [self.catalog.resolve(ref) for ref in intent.source_refs]
        fact_sources = [self.catalog.resolve(ref) for ref in intent.fact_refs]
        if not sources and not fact_sources:
            raise ValueError(f"visual intent '{intent.purpose}' must reference at least one source")
        accepted = TEMPLATE_SOURCE_KINDS[intent.template_id]
        incompatible = sorted({source.kind for source in sources if source.kind not in accepted})
        if incompatible:
            raise ValueError(f"template {intent.template_id} does not accept source kinds {incompatible}")
        if any(source.kind != "fact" for source in fact_sources):
            raise ValueError("fact_refs may only reference Fact sources")
        method_name = "_materialize_" + intent.template_id.replace(".", "_")
        method = getattr(self, method_name)
        dataset, layers, bindings, layout = method(intent, sources, fact_sources)
        source_refs = list(dict.fromkeys(source.ref for source in sources))
        fact_refs = list(dict.fromkeys(source.ref for source in fact_sources))
        accessibility = self._accessibility(intent, dataset, layers)
        digest = hashlib.sha1(
            f"{intent.template_id}|{intent.purpose}|{'|'.join(source_refs)}".encode("utf-8")
        ).hexdigest()[:12]
        return VisualizationPayload(
            visualization_id=f"viz_{digest}_{index}",
            template_id=intent.template_id,
            purpose=intent.purpose,
            priority=intent.priority,
            title=intent.title,
            summary=intent.summary,
            source_refs=source_refs,
            fact_refs=fact_refs,
            dataset=dataset,
            layers=layers,
            bindings=bindings,
            layout=layout,
            accessibility=accessibility,
        )

    def _materialize_metric_single(self, intent, sources, facts):
        source = (facts or sources)[0]
        metric = _metric_from_source(source, intent.encodings)
        dataset = VisualizationDataset(metric=metric, rows=[metric], columns=list(metric))
        return dataset, [], _bindings_for_fact_sources(facts or ([source] if source.kind == "fact" else [])), "overlay"

    def _materialize_table_detail(self, intent, sources, facts):
        rows = _rows_from_sources(_unique_sources([*sources, *facts]))
        if not rows:
            raise ValueError("table.detail requires row-shaped source data")
        columns = list(dict.fromkeys(intent.encodings.values())) if intent.encodings else _row_fields(rows)
        columns = [column for column in columns if any(column in row for row in rows)] or _row_fields(rows)
        dataset = VisualizationDataset(rows=rows[: self.max_points], columns=columns)
        return dataset, [], _bindings_for_fact_sources(facts), "overlay"

    def _materialize_ranking_topk(self, intent, sources, facts):
        rows = _rows_from_sources(_unique_sources([*facts, *sources]))
        if not rows:
            raise ValueError("ranking.topk requires ranked items or rows")
        x_field = intent.encodings.get("x") or _ranking_label_field(rows)
        y_field = intent.encodings.get("y") or _first_field(rows, "number")
        if not x_field or not y_field:
            raise ValueError("ranking.topk requires a category/label field and a numeric field")
        ranked = [row for row in rows if _number(row.get(y_field)) is not None]
        ranked.sort(key=lambda row: (row.get("rank") is None, row.get("rank") or 0, -(_number(row.get(y_field)) or 0)))
        bindings = _bindings_for_fact_sources(facts or [source for source in sources if source.kind == "fact"])
        binding_by_item = {binding.item_id: binding.binding_id for binding in bindings if binding.item_id}
        points = [
            VisualizationPoint(
                x=row.get(x_field), y=_number(row.get(y_field)), label=str(row.get(x_field) or ""),
                binding_id=binding_by_item.get(str(row.get("item_id") or "")),
                metadata={"rank": row.get("rank")},
            )
            for row in ranked[: self.max_points]
        ]
        dataset = _series_dataset(x_field, y_field, [VisualizationSeries(series_id="ranking", name=intent.title, role="ranking", points=points)])
        return dataset, [VisualizationLayer(kind="bar", role="fact", series_id="ranking")], bindings, "overlay"

    def _materialize_timeseries_trend(self, intent, sources, facts):
        return self._time_series_view(intent, sources, facts, add_fact_layers=False)

    def _materialize_timeseries_highlight(self, intent, sources, facts):
        return self._time_series_view(intent, sources, facts, add_fact_layers=True)

    def _materialize_interval_highlight(self, intent, sources, facts):
        return self._time_series_view(intent, sources, facts, add_fact_layers=True, interval=True)

    def _materialize_timeseries_comparison(self, intent, sources, facts):
        return self._time_series_view(intent, sources, facts, add_fact_layers=False)

    def _time_series_view(self, intent, sources, facts, *, add_fact_layers, interval=False):
        series = _series_from_sources(sources, intent.encodings)
        if not series:
            raise ValueError(f"{intent.template_id} requires time-series source data")
        fact_points = _fact_layer_points(facts)
        preserve_x = {point.x for point in fact_points if point.x is not None}
        sampled = [
            item.model_copy(update={"points": _sample_points(item.points, self.max_points, preserve_x)})
            for item in series
        ]
        layers = [VisualizationLayer(kind="line", role="context", series_id=item.series_id) for item in sampled]
        bindings = _bindings_for_fact_sources(facts)
        if add_fact_layers and fact_points:
            kind = "area" if interval else "point"
            layers.append(VisualizationLayer(kind=kind, role="fact", points=fact_points, label="Fact highlight"))
        dataset = _series_dataset(
            intent.encodings.get("x") or "timestamp",
            intent.encodings.get("y") or "value",
            sampled,
        )
        return dataset, layers, bindings, _legible_layout(sampled)

    def _materialize_timeseries_forecast(self, intent, sources, facts):
        forecast_source = _single_kind(sources, "forecast", intent.template_id)
        forecast = forecast_source.value
        evidence = self._forecast_evidence(forecast)
        supplied_evidence_ids = {source.value.evidence_id for source in sources if source.kind == "evidence"}
        if supplied_evidence_ids and supplied_evidence_ids != {evidence.evidence_id}:
            raise ValueError(
                "timeseries.forecast Evidence source must be the Forecast artifact's canonical input Evidence"
            )
        historical = _series_from_evidence(evidence, intent.encodings)
        if not historical:
            raise ValueError("forecast input evidence does not contain a readable time series")
        history = historical[0]
        forecast_bindings: list[VisualizationBinding] = []
        forecast_points: list[VisualizationPoint] = []
        evidence_id = evidence.evidence_id
        interval_by_x = _confidence_by_timestamp(forecast.confidence_interval)
        for index, point in enumerate(forecast.forecast_points):
            binding_id = f"{forecast_source.ref}:point:{index}"
            bounds = interval_by_x.get(str(point.timestamp), {})
            forecast_points.append(VisualizationPoint(
                x=point.timestamp,
                y=float(point.value),
                lower=_number(bounds.get("lower")),
                upper=_number(bounds.get("upper")),
                binding_id=binding_id,
            ))
            forecast_bindings.append(VisualizationBinding(
                binding_id=binding_id,
                source_type="prediction_point",
                source_ref=forecast_source.ref,
                evidence_id=evidence_id,
                locator={"timestamp": point.timestamp, "forecast_index": index},
            ))
        history_budget = max(2, self.max_points - len(forecast_points))
        history_points = _sample_points(history.points, history_budget, set(), prefer_recent=True)
        history_series = history.model_copy(update={"series_id": "historical", "name": history.name or "Historical", "role": "historical", "points": history_points})
        predicted_series = VisualizationSeries(
            series_id="forecast", name="Forecast", role="forecast", unit=history.unit, points=forecast_points
        )
        series = [history_series, predicted_series]
        layers = [
            VisualizationLayer(kind="line", role="context", series_id="historical"),
            VisualizationLayer(kind="line", role="forecast", series_id="forecast"),
        ]
        if any(point.lower is not None and point.upper is not None for point in forecast_points):
            layers.append(VisualizationLayer(kind="band", role="confidence", series_id="forecast", label="Confidence interval"))
        if forecast_points:
            layers.append(VisualizationLayer(
                kind="rule", role="forecast", points=[VisualizationPoint(x=forecast_points[0].x, label="Forecast starts")]
            ))
        dataset = _series_dataset(evidence.data.get("time_field", "timestamp"), history.name or "value", series)
        return dataset, layers, forecast_bindings, _legible_layout(series)

    def _materialize_timeseries_anomaly(self, intent, sources, facts):
        anomaly_source = _single_kind(sources, "anomaly", intent.template_id)
        anomaly = anomaly_source.value
        evidence = self._anomaly_evidence(anomaly)
        series = _series_from_evidence(evidence, intent.encodings)
        if not series:
            raise ValueError("anomaly input evidence does not contain a readable time series")
        anomaly_points: list[VisualizationPoint] = []
        bindings: list[VisualizationBinding] = []
        for index, item in enumerate(anomaly.anomaly_points):
            if not isinstance(item, dict):
                continue
            x = item.get("timestamp") or item.get("time") or item.get("date")
            y = _number(item.get("value") if "value" in item else item.get("y"))
            binding_id = f"{anomaly_source.ref}:point:{index}"
            anomaly_points.append(VisualizationPoint(x=x, y=y, label=str(item.get("label") or "Anomaly"), binding_id=binding_id, metadata=item))
            bindings.append(VisualizationBinding(
                binding_id=binding_id, source_type="anomaly_point", source_ref=anomaly_source.ref,
                evidence_id=evidence.evidence_id, locator={"timestamp": x, "anomaly_index": index},
            ))
        preserve = {point.x for point in anomaly_points}
        sampled = [item.model_copy(update={"points": _sample_points(item.points, self.max_points, preserve)}) for item in series]
        dataset = _series_dataset(intent.encodings.get("x") or "timestamp", intent.encodings.get("y") or "value", sampled)
        layers = [VisualizationLayer(kind="line", role="context", series_id=item.series_id) for item in sampled]
        layers.append(VisualizationLayer(kind="point", role="anomaly", points=anomaly_points, label="Anomaly"))
        return dataset, layers, bindings, _legible_layout(sampled)

    def _materialize_category_comparison(self, intent, sources, facts):
        return self._categorical_series(intent, sources, facts)

    def _categorical_series(self, intent, sources, facts):
        rows = _rows_from_sources([*sources, *facts])
        x_field = intent.encodings.get("x") or _first_field(rows, "category")
        y_field = intent.encodings.get("y") or _first_field(rows, "number")
        series_field = intent.encodings.get("series")
        if not x_field or not y_field:
            raise ValueError("category.comparison requires category and numeric fields")
        grouped: dict[str, list[VisualizationPoint]] = {}
        for row in rows:
            value = _number(row.get(y_field))
            if value is None:
                continue
            name = str(row.get(series_field) or y_field) if series_field else y_field
            grouped.setdefault(name, []).append(VisualizationPoint(x=row.get(x_field), y=value))
        series = [VisualizationSeries(series_id=f"category_{index}", name=name, role="comparison", points=points[: self.max_points]) for index, (name, points) in enumerate(grouped.items())]
        dataset = _series_dataset(x_field, y_field, series)
        layers = [VisualizationLayer(kind="bar", role="comparison", series_id=item.series_id) for item in series]
        return dataset, layers, _bindings_for_fact_sources(facts), _legible_layout(series)

    def _materialize_distribution_histogram(self, intent, sources, facts):
        values, value_field = _numeric_values([*sources, *facts], intent.encodings.get("y") or intent.encodings.get("value"))
        if len(values) < 2:
            raise ValueError("distribution.histogram requires at least two numeric values")
        bins = _histogram(values)
        points = [VisualizationPoint(x=item["label"], y=float(item["count"]), metadata=item) for item in bins]
        series = [VisualizationSeries(series_id="distribution", name=value_field, role="distribution", points=points)]
        dataset = _series_dataset("bin", "count", series)
        return dataset, [VisualizationLayer(kind="bar", role="comparison", series_id="distribution")], [], "overlay"

    def _materialize_distribution_boxplot(self, intent, sources, facts):
        rows = _rows_from_sources([*sources, *facts])
        value_field = intent.encodings.get("y") or intent.encodings.get("value") or _first_field(rows, "number")
        group_field = intent.encodings.get("x") or intent.encodings.get("series") or _first_field(rows, "category")
        if not value_field:
            raise ValueError("distribution.boxplot requires a numeric field")
        grouped: dict[str, list[float]] = {}
        for row in rows:
            value = _number(row.get(value_field))
            if value is not None:
                grouped.setdefault(str(row.get(group_field) if group_field else value_field), []).append(value)
        points = []
        for label, values in grouped.items():
            if values:
                q1, median, q3 = _quartiles(values)
                points.append(VisualizationPoint(x=label, y=median, lower=min(values), upper=max(values), metadata={"q1": q1, "median": median, "q3": q3}))
        if not points:
            raise ValueError("distribution.boxplot has no numeric groups")
        series = [VisualizationSeries(series_id="boxplot", name=value_field, role="distribution", points=points)]
        return _series_dataset(group_field or "group", value_field, series), [VisualizationLayer(kind="boxplot", role="comparison", series_id="boxplot")], [], "overlay"

    def _materialize_relationship_scatter(self, intent, sources, facts):
        rows = _rows_from_sources(sources)
        numeric_fields = _candidate_fields(rows, "number")
        x_field = intent.encodings.get("x") or (numeric_fields[0] if numeric_fields else None)
        y_field = intent.encodings.get("y") or (numeric_fields[1] if len(numeric_fields) > 1 else None)
        if not x_field or not y_field or x_field == y_field:
            raise ValueError("relationship.scatter requires two distinct numeric encodings")
        points = [VisualizationPoint(x=_number(row.get(x_field)), y=_number(row.get(y_field))) for row in rows if _number(row.get(x_field)) is not None and _number(row.get(y_field)) is not None]
        points = _sample_points(points, self.max_points, set())
        series = [VisualizationSeries(series_id="relationship", name=f"{x_field} vs {y_field}", role="relationship", points=points)]
        return _series_dataset(x_field, y_field, series, x_type="number"), [VisualizationLayer(kind="scatter", role="comparison", series_id="relationship")], [], "overlay"

    def _forecast_evidence(self, forecast):
        coverage = forecast.diagnostics.get("coverage") if isinstance(forecast.diagnostics, dict) else {}
        refs = list((coverage or {}).get("input_evidence_refs") or [])
        if not refs:
            ref = forecast.diagnostics.get("resolved_evidence_id") if isinstance(forecast.diagnostics, dict) else None
            refs = [ref] if ref else []
        if not refs:
            raise ValueError("forecast artifact does not identify its input Evidence")
        source = self.catalog.resolve(str(refs[0]))
        if source.kind != "evidence":
            raise ValueError("forecast input reference must resolve to Evidence")
        return source.value

    def _anomaly_evidence(self, anomaly):
        ref = _anomaly_evidence_ref(anomaly)
        if not ref:
            raise ValueError("anomaly artifact does not identify its input Evidence")
        source = self.catalog.resolve(ref)
        if source.kind != "evidence":
            raise ValueError("anomaly input reference must resolve to Evidence")
        return source.value

    def _accessibility(self, intent, dataset, layers):
        rows = dataset.rows[:24]
        columns = dataset.columns
        if not rows and dataset.series:
            rows = []
            for series in dataset.series:
                for point in series.points[:12]:
                    rows.append({"series": series.name, "x": point.x, "y": point.y, "lower": point.lower, "upper": point.upper})
            columns = ["series", "x", "y", "lower", "upper"]
        if dataset.metric:
            rows = [dataset.metric]
            columns = list(dataset.metric)
        return VisualizationAccessibility(
            description=intent.summary or intent.purpose,
            table_columns=columns,
            table_rows=rows[:24],
        )


def _single_kind(sources, kind, template_id):
    matches = [source for source in sources if source.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"{template_id} requires exactly one {kind} source")
    return matches[0]


def _series_dataset(x_field, y_field, series, *, x_type="time"):
    return VisualizationDataset(
        dimensions=[
            VisualizationDimension(name=x_field, data_type=x_type, role="x"),
            VisualizationDimension(name=y_field, data_type="number", role="y"),
        ],
        series=series,
    )


def _rows_from_sources(sources: Iterable[PresentationSource]) -> list[dict]:
    rows: list[dict] = []
    for source in sources:
        if source.kind == "evidence":
            rows.extend(_rows_from_evidence(source.value))
        elif source.kind == "analysis":
            rows.extend(_rows_from_analysis(source.value.result))
        elif source.kind == "fact":
            fact: DataFact = source.value
            if fact.items:
                rows.extend(_fact_item_row(item) for item in fact.items)
            elif isinstance(fact.value, list):
                rows.extend(item if isinstance(item, dict) else {"value": item} for item in fact.value)
            elif isinstance(fact.value, dict):
                rows.append(dict(fact.value))
            else:
                rows.append({"name": fact.name, "value": fact.value, "unit": fact.unit})
        elif source.kind == "forecast":
            rows.extend({"timestamp": point.timestamp, "value": point.value} for point in source.value.forecast_points)
        elif source.kind == "anomaly":
            rows.extend(item for item in source.value.anomaly_points if isinstance(item, dict))
    return rows


def _unique_sources(sources: Iterable[PresentationSource]) -> list[PresentationSource]:
    return list({source.ref: source for source in sources}.values())


def _rows_from_evidence(evidence) -> list[dict]:
    data = evidence.data if isinstance(evidence.data, dict) else {}
    rows = data.get("rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    points = data.get("points")
    if isinstance(points, list):
        return [dict(point) for point in points if isinstance(point, dict)]
    statistics = data.get("statistics")
    if isinstance(statistics, dict):
        return [{"metric": key, "value": value} for key, value in statistics.items()]
    series = data.get("series")
    if isinstance(series, list):
        output: list[dict] = []
        for item in series:
            if not isinstance(item, dict):
                continue
            name = item.get("series_name") or item.get("value_field") or "series"
            for point in item.get("points") or []:
                if isinstance(point, dict):
                    output.append({**point, "series": name})
        return output
    return []


def _rows_from_analysis(result) -> list[dict]:
    if not isinstance(result, dict):
        return []
    details = result.get("details")
    if isinstance(details, dict):
        for key in ("rows", "points", "items", "values", "data"):
            value = details.get(key)
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"value": item} for item in value]
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return [{"metric": key, "value": value} for key, value in metrics.items()]
    return []


def _series_from_sources(sources, encodings):
    output: list[VisualizationSeries] = []
    for source in sources:
        if source.kind == "evidence":
            output.extend(_series_from_evidence(source.value, encodings))
        elif source.kind == "analysis":
            rows = _rows_from_analysis(source.value.result)
            output.extend(_series_from_rows(rows, encodings, prefix=source.value.analysis_id))
    return output


def _series_from_evidence(evidence, encodings):
    data = evidence.data if isinstance(evidence.data, dict) else {}
    payloads = data.get("series")
    if isinstance(payloads, list) and payloads:
        output = []
        for index, payload in enumerate(payloads):
            if not isinstance(payload, dict):
                continue
            points = [
                VisualizationPoint(x=point.get("timestamp"), y=_number(point.get("value")))
                for point in payload.get("points") or []
                if isinstance(point, dict) and _number(point.get("value")) is not None
            ]
            output.append(VisualizationSeries(
                series_id=f"series_{index}",
                name=str(payload.get("series_name") or payload.get("value_field") or f"Series {index + 1}"),
                role="comparison" if len(payloads) > 1 else "historical",
                unit=(payload.get("labels") or {}).get("unit") if isinstance(payload.get("labels"), dict) else None,
                points=points,
            ))
        return output
    return _series_from_rows(_rows_from_evidence(evidence), encodings, prefix=evidence.evidence_id)


def _series_from_rows(rows, encodings, *, prefix):
    if not rows:
        return []
    x_field = encodings.get("x") or _first_field(rows, "time")
    numeric_fields = _candidate_fields(rows, "number")
    requested_y = encodings.get("y")
    y_fields = [requested_y] if requested_y else numeric_fields
    series_field = encodings.get("series")
    if not x_field or not y_fields:
        return []
    grouped: dict[tuple[str, str], list[VisualizationPoint]] = {}
    for row in rows:
        for y_field in y_fields:
            value = _number(row.get(y_field))
            if value is None:
                continue
            group = str(row.get(series_field)) if series_field and row.get(series_field) is not None else y_field
            grouped.setdefault((group, y_field), []).append(VisualizationPoint(x=row.get(x_field), y=value))
    output = []
    for index, ((group, _), points) in enumerate(grouped.items()):
        points.sort(key=lambda point: _x_sort_key(point.x))
        output.append(VisualizationSeries(
            series_id=f"{prefix}_{index}", name=group, role="comparison" if len(grouped) > 1 else "historical", points=points
        ))
    return output


def _fact_layer_points(facts):
    points: list[VisualizationPoint] = []
    for source in facts:
        fact: DataFact = source.value
        if fact.items:
            for item in fact.items:
                points.append(VisualizationPoint(
                    x=item.timestamp or item.locator.get("timestamp"),
                    y=_number(item.value),
                    label=item.label or (f"#{item.rank}" if item.rank is not None else fact.name),
                    binding_id=f"{source.ref}:item:{item.item_id}",
                    metadata={"rank": item.rank} if item.rank is not None else {},
                ))
            continue
        value = fact.value
        if isinstance(value, dict):
            start = value.get("start") or value.get("start_time")
            end = value.get("end") or value.get("end_time")
            if start is not None or end is not None:
                points.extend([
                    VisualizationPoint(x=start, label=fact.name, binding_id=f"{source.ref}:value", metadata={"boundary": "start"}),
                    VisualizationPoint(x=end, label=fact.name, binding_id=f"{source.ref}:value", metadata={"boundary": "end"}),
                ])
                continue
            x = value.get("timestamp") or value.get("time") or value.get("date")
            y = _number(value.get("value") if "value" in value else value.get("y"))
        else:
            x = (fact.time_range or {}).get("end") if fact.time_range else None
            y = _number(value)
        points.append(VisualizationPoint(x=x, y=y, label=fact.name, binding_id=f"{source.ref}:value"))
    return [point for point in points if point.x is not None or point.y is not None]


def _bindings_for_fact_sources(facts):
    bindings: list[VisualizationBinding] = []
    for source in facts:
        fact: DataFact = source.value
        evidence_id = next((ref.source_id for ref in fact.evidence_refs if ref.source_type in {"query", "database_evidence"}), None)
        if fact.items:
            for item in fact.items:
                bindings.append(VisualizationBinding(
                    binding_id=f"{source.ref}:item:{item.item_id}", source_type="fact_item", fact_id=fact.fact_id,
                    item_id=item.item_id, evidence_id=evidence_id, source_ref=source.ref, locator=item.locator,
                ))
        else:
            bindings.append(VisualizationBinding(
                binding_id=f"{source.ref}:value", source_type="fact", fact_id=fact.fact_id,
                evidence_id=evidence_id, source_ref=source.ref,
                locator={"time_range": fact.time_range} if fact.time_range else {},
            ))
    return bindings


def _metric_from_source(source, encodings):
    if source.kind == "fact":
        fact = source.value
        return {"label": fact.name, "value": fact.value, "unit": fact.unit, "statement": fact.statement}
    rows = _rows_from_sources([source])
    value_field = encodings.get("value") or encodings.get("y") or _first_field(rows, "number")
    if not rows or not value_field:
        raise ValueError("metric.single requires a scalar Fact or numeric source field")
    return {"label": value_field, "value": rows[0].get(value_field)}


def _numeric_values(sources, requested_field):
    rows = _rows_from_sources(sources)
    field = requested_field or _first_field(rows, "number")
    if not field:
        return [], "value"
    return [value for row in rows if (value := _number(row.get(field))) is not None], field


def _sample_points(points, limit, preserve_x, *, prefer_recent=False):
    if len(points) <= limit:
        return list(points)
    if prefer_recent:
        selected = list(points[-limit:])
        if points[0].x in preserve_x and all(item.x != points[0].x for item in selected):
            selected[0] = points[0]
        return selected
    keep = {0, len(points) - 1}
    keep.update(index for index, point in enumerate(points) if point.x in preserve_x)
    remaining = max(0, limit - len(keep))
    bucket_count = max(1, remaining // 2)
    bucket_size = len(points) / bucket_count
    for bucket in range(bucket_count):
        start = int(bucket * bucket_size)
        end = min(len(points), int((bucket + 1) * bucket_size))
        candidates = [(index, points[index].y) for index in range(start, end) if points[index].y is not None]
        if candidates:
            keep.add(min(candidates, key=lambda item: item[1])[0])
            keep.add(max(candidates, key=lambda item: item[1])[0])
    if len(keep) < limit:
        stride = (len(points) - 1) / max(limit - 1, 1)
        keep.update(round(index * stride) for index in range(limit))
    return [points[index] for index in sorted(keep)[:limit]]


def _legible_layout(series):
    populated = [[point.y for point in item.points if point.y is not None] for item in series]
    populated = [values for values in populated if values]
    if len(populated) < 2:
        return "overlay"
    all_values = [value for values in populated for value in values]
    domain = max(all_values) - min(all_values)
    if domain <= 0:
        return "overlay"
    chart_height = 280.0
    minimum_readable_span = 12.0
    for values in populated:
        projected_span = ((max(values) - min(values)) / domain) * chart_height
        if projected_span < minimum_readable_span:
            return "facets"
    return "overlay"


def _histogram(values):
    ordered = sorted(values)
    q1, _, q3 = _quartiles(ordered)
    iqr = q3 - q1
    width = (2 * iqr / (len(ordered) ** (1 / 3))) if iqr > 0 else 0
    count = max(1, min(50, math.ceil((ordered[-1] - ordered[0]) / width))) if width > 0 else max(1, round(math.sqrt(len(ordered))))
    minimum, maximum = ordered[0], ordered[-1]
    span = maximum - minimum
    if span == 0:
        return [{"start": minimum, "end": maximum, "label": str(minimum), "count": len(ordered)}]
    bin_width = span / count
    bins = [{"start": minimum + index * bin_width, "end": minimum + (index + 1) * bin_width, "count": 0} for index in range(count)]
    for value in ordered:
        index = min(count - 1, int((value - minimum) / bin_width))
        bins[index]["count"] += 1
    for item in bins:
        item["label"] = f"{item['start']:.4g}–{item['end']:.4g}"
    return bins


def _quartiles(values):
    ordered = sorted(values)
    return (_percentile(ordered, 0.25), _percentile(ordered, 0.5), _percentile(ordered, 0.75))


def _percentile(values, fraction):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _confidence_by_timestamp(intervals):
    output = {}
    for index, item in enumerate(intervals or []):
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp") or item.get("time") or item.get("date") or str(index)
        output[str(timestamp)] = {
            "lower": item.get("lower") if "lower" in item else item.get("lower_bound"),
            "upper": item.get("upper") if "upper" in item else item.get("upper_bound"),
        }
    return output


def _anomaly_evidence_ref(anomaly):
    diagnostics = anomaly.diagnostics if isinstance(anomaly.diagnostics, dict) else {}
    evidence_id = diagnostics.get("resolved_evidence_id") or diagnostics.get("selected_evidence_id")
    if evidence_id:
        return f"evidence:{evidence_id}"
    if anomaly.anomaly_id.startswith("anomaly_"):
        return f"evidence:{anomaly.anomaly_id[len('anomaly_') :]}"
    return None


def _evidence_point_count(evidence):
    data = evidence.data if isinstance(evidence.data, dict) else {}
    if isinstance(data.get("points"), list):
        return len(data["points"])
    if isinstance(data.get("series"), list):
        return sum(len(item.get("points") or []) for item in data["series"] if isinstance(item, dict))
    return 0


def _fact_item_row(item: FactItem):
    return {
        "item_id": item.item_id,
        "label": item.label,
        "rank": item.rank,
        "timestamp": item.timestamp,
        "value": item.value,
        **item.dimensions,
        **item.locator,
    }


def _candidate_fields(rows, kind):
    fields = _row_fields(rows)
    output = []
    for field in fields:
        values = [row.get(field) for row in rows[:64] if row.get(field) is not None]
        if not values:
            continue
        if kind == "number" and sum(_number(value) is not None for value in values) >= max(1, len(values) * 0.8):
            output.append(field)
        elif kind == "time" and sum(_time_value(value) is not None for value in values) >= max(1, len(values) * 0.8):
            output.append(field)
        elif kind == "category" and not all(_number(value) is not None for value in values) and not all(_time_value(value) is not None for value in values):
            output.append(field)
    return output


def _first_field(rows, kind):
    fields = _candidate_fields(rows, kind)
    return fields[0] if fields else None


def _ranking_label_field(rows):
    for field in ("label", "name", "category", "timestamp", "item_id"):
        if any(row.get(field) not in (None, "") for row in rows):
            return field
    return _first_field(rows, "category") or _first_field(rows, "time")


def _row_fields(rows):
    return list(dict.fromkeys(key for row in rows if isinstance(row, dict) for key in row))


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _time_value(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _x_sort_key(value):
    parsed = _time_value(value)
    if parsed is not None:
        return (0, parsed)
    numeric = _number(value)
    if numeric is not None:
        return (1, numeric)
    return (2, str(value))


def _bounded_row(row):
    return {str(key): value for key, value in list(row.items())[:10]}


def _bounded_mapping(value):
    if not isinstance(value, dict):
        return {}
    return {str(key): child for key, child in list(value.items())[:12]}


def _point_time_range(points):
    if not points:
        return None
    return {"start": points[0].timestamp, "end": points[-1].timestamp}
