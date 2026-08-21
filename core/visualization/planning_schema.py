"""Build provider-closed LineChart schemas from one grounded content contract."""
from __future__ import annotations

import hashlib
from functools import reduce
from operator import or_
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from schemas.linechart_plan import (
    AnnotationPlan,
    BandPlan,
    ChartAnnotationTargetPlan,
    IntervalAnnotationTargetPlan,
    IntervalPlan,
    LineChartGoalPlan,
    LinePlan,
    PointPlan,
    ReferenceLinePlan,
    StructuredLineChartPlan,
    XAnnotationTargetPlan,
    XYAnnotationTargetPlan,
    VisualContentPlan,
)


class _UnavailableComponent(BaseModel):
    """Closed placeholder used only for component arrays constrained to length zero."""

    model_config = ConfigDict(extra="forbid")


def build_linechart_response_schema(content: VisualContentPlan, inventory: dict):
    """Return a strict Pydantic response type whose fields are source-specific enums."""

    source_contracts = {
        str(item.get("source_ref")): _field_contract(item)
        for item in inventory.get("sources", [])
        if isinstance(item, dict) and item.get("source_ref")
    }
    goal_models = [
        _goal_model(goal, source_contracts, index)
        for index, goal in enumerate(content.goals)
    ]
    chart_type = _union(goal_models)
    suffix = _schema_suffix([goal.goal_id for goal in content.goals])
    return create_model(
        f"GroundedLineChartPlan_{suffix}",
        __base__=StructuredLineChartPlan,
        charts=(list[chart_type], Field(default_factory=list)),
    )


def content_line_capability_errors(content: VisualContentPlan, inventory: dict) -> list[dict]:
    """Check that each goal's declared host can provide a real LineChart line."""

    sources = {
        str(item.get("source_ref")): item
        for item in inventory.get("sources", [])
        if isinstance(item, dict) and item.get("source_ref")
    }
    errors: list[dict] = []
    for goal in content.goals:
        source = sources.get(goal.host_source_ref)
        contract = _field_contract(source or {})
        row_count = _source_row_count(source or {})
        structure = (source or {}).get("data_structure")
        if isinstance(structure, dict):
            row_count = max(row_count, int(structure.get("row_count") or 0))
        if not contract["x"] or not contract["numeric"] or contract["scalar"]:
            errors.append({
                "goal_id": goal.goal_id,
                "host_source_ref": goal.host_source_ref,
                "error": "host_not_line_capable",
                "available_time_or_category_fields": contract["x"],
                "available_numeric_fields": contract["numeric"],
                "row_count": row_count or None,
            })
    return errors


def _goal_model(goal, source_contracts: dict[str, dict[str, Any]], index: int):
    component_models: dict[str, list[type]] = {
        "lines": [], "points": [], "bands": [], "intervals": [],
        "reference_lines": [], "annotations": [],
    }
    for content_item in goal.content:
        fields = source_contracts.get(content_item.source_ref, _field_contract({}))
        tag = _schema_suffix([goal.goal_id, content_item.content_id, str(index)])
        common = {"content_id": (_literal([content_item.content_id]), ...)}
        if fields["time"] and fields["numeric"] and not fields["scalar"]:
            component_models["lines"].append(create_model(
                f"GroundedLine_{tag}", __base__=LinePlan, **common,
                x_field=(_literal(fields["time"]), ...), y_field=(_literal(fields["numeric"]), ...),
            ))
            component_models["bands"].append(create_model(
                f"GroundedBand_{tag}", __base__=BandPlan, **common,
                x_field=(_literal(fields["time"]), ...),
                lower_field=(_literal(fields["numeric"]), ...),
                upper_field=(_literal(fields["numeric"]), ...),
            ))
        if fields["time"] and fields["numeric"]:
            component_models["points"].append(create_model(
                f"GroundedPoint_{tag}", __base__=PointPlan, **common,
                x_field=(_literal(fields["time"]), ...), y_field=(_literal(fields["numeric"]), ...),
            ))
        if fields["time"]:
            component_models["intervals"].append(create_model(
                f"GroundedInterval_{tag}", __base__=IntervalPlan, **common,
                start_field=(_literal(fields["time"]), ...), end_field=(_literal(fields["time"]), ...),
            ))
        if fields["numeric"] and fields["scalar"]:
            component_models["reference_lines"].append(create_model(
                f"GroundedReference_{tag}", __base__=ReferenceLinePlan, **common,
                value_field=(_literal(fields["numeric"]), ...),
            ))
        if fields["all"]:
            target_models: list[type] = [ChartAnnotationTargetPlan]
            if fields["time"]:
                target_models.append(create_model(
                    f"GroundedXTarget_{tag}", __base__=XAnnotationTargetPlan,
                    x_field=(_literal(fields["time"]), ...),
                ))
            if fields["time"] and fields["numeric"]:
                target_models.append(create_model(
                    f"GroundedXYTarget_{tag}", __base__=XYAnnotationTargetPlan,
                    x_field=(_literal(fields["time"]), ...), y_field=(_literal(fields["numeric"]), ...),
                ))
            if fields["time"]:
                target_models.append(create_model(
                    f"GroundedIntervalTarget_{tag}", __base__=IntervalAnnotationTargetPlan,
                    start_field=(_literal(fields["time"]), ...), end_field=(_literal(fields["time"]), ...),
                ))
            component_models["annotations"].append(create_model(
                f"GroundedAnnotation_{tag}", __base__=AnnotationPlan, **common,
                content_field=(_literal(fields["all"]), ...), target=(_union(target_models), ...),
            ))
    host_content_ids = {
        item.content_id for item in goal.content if item.source_ref == goal.host_source_ref
    }
    host_line_models = [
        model for model in component_models["lines"]
        if _line_content_id(model) in host_content_ids
    ]
    if not host_line_models:
        raise ValueError(f"content goal '{goal.goal_id}' has no source with temporal/category and numeric fields")
    overrides = {
        "goal_id": (_literal([goal.goal_id]), ...),
        "host_line": (_union(host_line_models), ...),
        "lines": (list[_union(component_models["lines"])], Field(default_factory=list)),
    }
    for field_name in ("points", "bands", "intervals", "reference_lines", "annotations"):
        candidates = component_models[field_name]
        overrides[field_name] = (
            list[_union(candidates)] if candidates else list[_UnavailableComponent],
            Field(default_factory=list, **({} if candidates else {"max_length": 0})),
        )
    return create_model(
        f"GroundedChartGoal_{_schema_suffix([goal.goal_id, str(index)])}",
        __base__=LineChartGoalPlan,
        **overrides,
    )


def _line_content_id(model: type) -> str:
    return str(model.model_fields["content_id"].annotation.__args__[0])


def _field_contract(source: dict) -> dict[str, Any]:
    fields = [item for item in source.get("schema_fields", []) if isinstance(item, dict) and item.get("name")]
    names = [str(item["name"]) for item in fields]
    numeric = [
        str(item["name"]) for item in fields
        if str(item.get("data_type") or item.get("type") or "").lower() in {"number", "float", "int", "integer"}
    ]
    time = [
        str(item["name"]) for item in fields
        if str(item.get("data_type") or item.get("type") or "").lower() in {"time", "datetime", "date", "timestamp"}
    ]
    capabilities = source.get("render_capabilities") if isinstance(source.get("render_capabilities"), dict) else {}
    row_count = _source_row_count(source)
    return {
        "all": names, "numeric": numeric, "time": time, "x": time,
        "scalar": bool(capabilities.get("scalar_only")) or row_count == 1,
    }


def _source_row_count(source: dict) -> int:
    if source.get("kind") == "insight" and source.get("value") is not None:
        item_count = source.get("item_count")
        if not isinstance(item_count, int) or item_count <= 1:
            return 1
    for value in (source.get("row_count"), source.get("item_count")):
        if isinstance(value, int) and value >= 0:
            return value
    structure = source.get("data_structure")
    if isinstance(structure, dict) and isinstance(structure.get("row_count"), int):
        return int(structure["row_count"])
    return 0


def _literal(values: list[str]):
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise ValueError("cannot construct an empty field enum")
    return Literal.__getitem__(unique)


def _union(models: list[type]):
    if not models:
        raise ValueError("cannot construct an empty model union")
    return models[0] if len(models) == 1 else reduce(or_, models)


def _schema_suffix(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
