"""Compile and validate LLM-authored LineChart plans against grounded sources."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from core.visualization.materializer import PresentationCatalog, _projection_bindings, _schema_fields, _source_data
from schemas.linechart_plan import LineChartGoalPlan, LineChartPlan, VisualContentGoal, VisualContentPlan
from schemas.visual_verification import VisualizationVerification
from schemas.visualization import (
    ChartAnnotationTarget,
    IntervalAnnotationTarget,
    LineChartAnnotation,
    LineChartBand,
    LineChartInterval,
    LineChartLegend,
    LineChartLine,
    LineChartPoint,
    LineChartReferenceLine,
    LineChartTooltip,
    LineChartXAxis,
    LineChartYAxis,
    LineChartZoom,
    VisualizationAccessibility,
    VisualizationBinding,
    VisualizationDataView,
    VisualizationField,
    VisualizationPayload,
    VisualizationRecord,
    XAnnotationTarget,
    XYAnnotationTarget,
)


class LineChartCompiler:
    def __init__(self, catalog: PresentationCatalog):
        self.catalog = catalog

    def compile(
        self,
        content: VisualContentPlan,
        plan: LineChartPlan,
    ) -> list[VisualizationPayload]:
        goals = {goal.goal_id: goal for goal in content.goals}
        chart_ids = [chart.goal_id for chart in plan.charts]
        if set(chart_ids) != set(goals) or len(chart_ids) != len(set(chart_ids)):
            raise ValueError("line-chart plan must contain every content goal exactly once")
        verification = VisualizationVerification(
            target_insight_ids=content.target_insight_ids,
            verification_question=str(content.visual_question),
            interpretation=str(content.interpretation),
        )
        output = [self._compile_chart(goals[chart.goal_id], chart, verification) for chart in plan.charts]
        self._validate_supporting_charts(output)
        return output

    def _compile_chart(
        self,
        goal: VisualContentGoal,
        plan: LineChartGoalPlan,
        verification: VisualizationVerification,
    ) -> VisualizationPayload:
        content_by_id = {item.content_id: item for item in goal.content}
        components = _plan_components(plan)
        covered = {component.content_id for component in components}
        missing = set(content_by_id) - covered
        unknown = covered - set(content_by_id)
        if missing or unknown:
            raise ValueError(f"chart content coverage mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        source_refs = list(dict.fromkeys(content_by_id[component.content_id].source_ref for component in components))
        if goal.host_source_ref not in source_refs:
            raise ValueError(f"chart '{goal.goal_id}' does not render its declared host source")
        views, bindings, view_by_source = self._materialize_views(source_refs)
        payload = VisualizationPayload(
            visualization_id=_visualization_id(goal, plan),
            purpose=goal.purpose,
            priority=goal.priority,
            title=goal.title,
            summary=goal.summary,
            verification=verification,
            source_refs=_public_source_refs(self.catalog, source_refs),
            required_roles=list(dict.fromkeys(item.purpose for item in goal.content)),
            data_views=views,
            x_axis=LineChartXAxis(data_type=plan.x_axis_type, label=plan.x_axis_label),
            y_axes=[LineChartYAxis.model_validate(item.model_dump()) for item in plan.y_axes],
            lines=[
                self._line(item, content_by_id, view_by_source, bindings, index)
                for index, item in enumerate([plan.host_line, *plan.lines])
            ],
            points=[self._point(item, content_by_id, view_by_source, bindings, index) for index, item in enumerate(plan.points)],
            bands=[self._band(item, content_by_id, view_by_source, bindings, index) for index, item in enumerate(plan.bands)],
            intervals=[self._interval(item, content_by_id, view_by_source, bindings, index) for index, item in enumerate(plan.intervals)],
            reference_lines=[self._reference(item, content_by_id, view_by_source, bindings, index) for index, item in enumerate(plan.reference_lines)],
            annotations=[self._annotation(item, content_by_id, view_by_source, bindings, index) for index, item in enumerate(plan.annotations)],
            legend=LineChartLegend(
                visible=plan.legend_visible,
                toggle_components="legend_toggle" in goal.required_interactions,
                position=plan.legend_position,
            ),
            tooltip=LineChartTooltip(
                mode=plan.tooltip_mode,
                show_source="evidence_link" in goal.required_interactions,
            ),
            zoom=LineChartZoom(enabled=plan.zoom_enabled, start=plan.zoom_start, end=plan.zoom_end),
            bindings=list(bindings.values()),
            accessibility=_accessibility(goal, views),
        )
        LineChartValidator().validate(payload)
        return payload

    def _materialize_views(self, source_refs: list[str]):
        views: list[VisualizationDataView] = []
        bindings: dict[str, VisualizationBinding] = {}
        view_by_source: dict[str, str] = {}
        for index, source_ref in enumerate(source_refs):
            source = self.catalog.resolve(source_ref)
            rows, scalar = _source_data(source)
            records = [dict(row) for row in rows] or ([dict(scalar)] if scalar else [])
            if not records:
                raise ValueError(f"visual source '{source_ref}' contains no materializable records")
            view_id = f"view_{index}"
            source_bindings = _projection_bindings(source)
            output_records: list[VisualizationRecord] = []
            for row_index, row in enumerate(records):
                source_binding = source_bindings.get(str(row.get("item_id") or "")) or source_bindings.get("")
                binding_id = None
                if source_binding is not None:
                    binding_id = f"{view_id}:record_{row_index}"
                    bindings[binding_id] = source_binding.model_copy(update={"binding_id": binding_id})
                output_records.append(VisualizationRecord(
                    record_id=f"record_{row_index}",
                    values={str(key): value for key, value in row.items() if not str(key).startswith("__")},
                    binding_id=binding_id,
                ))
            schema = _schema_fields([record.values for record in output_records])
            semantics = source.value.field_semantics if source.kind == "view" else {}
            fields = [VisualizationField(
                name=str(item["name"]),
                data_type=_field_type(str(item.get("data_type") or "string")),
                semantic_role=str(semantics.get(str(item["name"])) or item["name"]),
                measure=item.get("measure"),
                unit=item.get("unit"),
            ) for item in schema]
            views.append(VisualizationDataView(
                view_id=view_id,
                source_ref=source.ref,
                fields=fields,
                records=output_records,
            ))
            view_by_source[source_ref] = view_id
        return views, bindings, view_by_source

    @staticmethod
    def _component_base(item, content_by_id, view_by_source, bindings, component_type: str, index: int) -> dict:
        source_ref = content_by_id[item.content_id].source_ref
        view_id = view_by_source[source_ref]
        binding_ids = [key for key in bindings if key.startswith(f"{view_id}:")]
        return {
            "component_id": f"{component_type}_{index}_{item.content_id}",
            "role": item.role,
            "importance": item.importance,
            "source_ref": source_ref,
            "view_id": view_id,
            "label": item.label,
            "binding_ids": binding_ids,
            "presentation": {},
        }

    def _line(self, item, content, views, bindings, index):
        return LineChartLine(**self._component_base(item, content, views, bindings, "line", index), x_field=item.x_field, y_field=item.y_field,
                             y_axis_id=item.y_axis_id, line_style=item.line_style, symbol=item.symbol)

    def _point(self, item, content, views, bindings, index):
        return LineChartPoint(**self._component_base(item, content, views, bindings, "point", index), x_field=item.x_field, y_field=item.y_field,
                              y_axis_id=item.y_axis_id, symbol=item.symbol, size=item.size)

    def _band(self, item, content, views, bindings, index):
        return LineChartBand(**self._component_base(item, content, views, bindings, "band", index), x_field=item.x_field,
                             lower_field=item.lower_field, upper_field=item.upper_field, y_axis_id=item.y_axis_id)

    def _interval(self, item, content, views, bindings, index):
        return LineChartInterval(**self._component_base(item, content, views, bindings, "interval", index), start_field=item.start_field,
                                 end_field=item.end_field)

    def _reference(self, item, content, views, bindings, index):
        return LineChartReferenceLine(**self._component_base(item, content, views, bindings, "reference", index), value_field=item.value_field,
                                      y_axis_id=item.y_axis_id)

    def _annotation(self, item, content, views, bindings, index):
        target = item.target.model_dump(exclude_none=True)
        target_type = target.pop("target_type")
        target_model = {
            "chart": ChartAnnotationTarget, "x": XAnnotationTarget, "xy": XYAnnotationTarget,
            "interval": IntervalAnnotationTarget,
        }[target_type]
        return LineChartAnnotation(**self._component_base(item, content, views, bindings, "annotation", index), content_field=item.content_field,
                                   target=target_model(**target))

    @staticmethod
    def _validate_supporting_charts(charts: list[VisualizationPayload]) -> None:
        primary = next(item for item in charts if item.priority == "primary")
        primary_domains = {(axis.measure.casefold(), (axis.unit or "").casefold()) for axis in primary.y_axes}
        for chart in charts:
            if chart.priority != "supporting":
                continue
            domains = {(axis.measure.casefold(), (axis.unit or "").casefold()) for axis in chart.y_axes}
            if chart.x_axis.data_type == primary.x_axis.data_type and domains <= primary_domains:
                raise ValueError(
                    f"supporting chart '{chart.purpose}' is compatible with the primary chart and must be merged"
                )


class LineChartValidator:
    def validate(self, payload: VisualizationPayload) -> None:
        views = {view.view_id: view for view in payload.data_views}
        axes = {axis.axis_id: axis for axis in payload.y_axes}
        for line in payload.lines:
            self._validate_xy(line, views, axes, payload.x_axis.data_type)
            view = views[line.view_id]
            if sum(record.values.get(line.y_field) is not None for record in view.records) < 2:
                raise ValueError(f"line '{line.component_id}' requires at least two grounded points")
        for point in payload.points:
            self._validate_xy(point, views, axes, payload.x_axis.data_type)
        for band in payload.bands:
            self._require_fields(views[band.view_id], [band.x_field, band.lower_field, band.upper_field])
            self._require_numeric(views[band.view_id], [band.lower_field, band.upper_field])
            for record in views[band.view_id].records:
                lower, upper = record.values.get(band.lower_field), record.values.get(band.upper_field)
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
                    raise ValueError(f"band '{band.component_id}' contains reversed bounds")
        for interval in payload.intervals:
            self._require_fields(views[interval.view_id], [interval.start_field, interval.end_field])
        for reference in payload.reference_lines:
            view = views[reference.view_id]
            self._require_fields(view, [reference.value_field])
            self._require_numeric(view, [reference.value_field])
            values = [record.values.get(reference.value_field) for record in view.records if record.values.get(reference.value_field) is not None]
            if len(values) != 1:
                raise ValueError(f"reference line '{reference.component_id}' requires exactly one grounded scalar")
            field = next(item for item in view.fields if item.name == reference.value_field)
            axis = axes[reference.y_axis_id]
            if field.measure and _norm(field.measure) != _norm(axis.measure):
                raise ValueError(f"reference line '{reference.component_id}' measure conflicts with its y axis")
            if field.unit and _norm(field.unit) != _norm(axis.unit):
                raise ValueError(f"reference line '{reference.component_id}' unit conflicts with its y axis")
        for annotation in payload.annotations:
            view = views[annotation.view_id]
            fields = [annotation.content_field]
            fields.extend(value for key, value in annotation.target.model_dump().items() if key.endswith("_field"))
            self._require_fields(view, fields)

    def _validate_xy(self, component, views, axes, x_axis_type):
        view = views[component.view_id]
        self._require_fields(view, [component.x_field, component.y_field])
        self._require_numeric(view, [component.y_field])
        x_field = next(item for item in view.fields if item.name == component.x_field)
        if x_axis_type == "time" and x_field.data_type != "time":
            raise ValueError(f"component '{component.component_id}' requires a temporal x field")
        y_field = next(item for item in view.fields if item.name == component.y_field)
        axis = axes[component.y_axis_id]
        if y_field.measure and _norm(y_field.measure) != _norm(axis.measure):
            raise ValueError(f"component '{component.component_id}' measure conflicts with its y axis")
        if y_field.unit and _norm(y_field.unit) != _norm(axis.unit):
            raise ValueError(f"component '{component.component_id}' unit conflicts with its y axis")

    @staticmethod
    def _require_fields(view, fields):
        known = {item.name for item in view.fields}
        unknown = set(fields) - known
        if unknown:
            raise ValueError(f"view '{view.view_id}' is missing fields: {sorted(unknown)}")

    @staticmethod
    def _require_numeric(view, fields):
        types = {item.name: item.data_type for item in view.fields}
        invalid = [field for field in fields if types.get(field) != "number"]
        if invalid:
            raise ValueError(f"view '{view.view_id}' fields must be numeric: {invalid}")


def _plan_components(plan: LineChartGoalPlan) -> list:
    return [plan.host_line, *plan.lines, *plan.points, *plan.bands, *plan.intervals, *plan.reference_lines, *plan.annotations]


def _field_type(value: str) -> str:
    return value if value in {"time", "number", "category", "string", "boolean"} else "string"


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _visualization_id(goal: VisualContentGoal, plan: LineChartGoalPlan) -> str:
    raw = json.dumps({"goal": goal.model_dump(mode="json"), "plan": plan.model_dump(mode="json")}, sort_keys=True, default=str)
    return f"viz_{hashlib.sha1(raw.encode()).hexdigest()[:12]}"


def _public_source_refs(catalog: PresentationCatalog, source_refs: list[str]) -> list[str]:
    result: list[str] = []
    for source_ref in source_refs:
        source = catalog.resolve(source_ref)
        refs = source.value.lineage if source.kind == "view" else [source.ref]
        result.extend(str(item) for item in refs)
    return list(dict.fromkeys(result))


def _accessibility(goal: VisualContentGoal, views: list[VisualizationDataView]) -> VisualizationAccessibility:
    rows = []
    for view in views:
        rows.extend({"view": view.view_id, "binding_id": record.binding_id, **record.values} for record in view.records[:12])
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return VisualizationAccessibility(
        description=goal.summary or goal.purpose,
        table_columns=columns,
        table_rows=rows[:24],
    )
