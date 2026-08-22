"""Compile a safe native ECharts option by resolving grounded placeholders."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from core.visualization.materializer import PresentationCatalog, _projection_bindings, _schema_fields, _source_data
from schemas.echarts_plan import EChartsChartPlan, EChartsPlan
from schemas.visual_verification import VisualizationVerification
from schemas.visualization import VisualizationAccessibility, VisualizationBinding, VisualizationPayload


MAX_OPTION_BYTES = 128 * 1024
MAX_DEPTH = 20
MAX_DATASETS = 12
MAX_SERIES = 12
MAX_Y_AXES = 2
ALLOWED_SERIES = {"line"}
MARK_KEYS = {"markPoint", "markLine", "markArea"}
FORBIDDEN_KEYS = {"transform", "renderItem", "javascript", "script", "html", "dom"}
URL_PATTERN = re.compile(r"(?:https?|ftp|data|javascript):", re.IGNORECASE)
EXECUTABLE_PATTERN = re.compile(r"(?:\bfunction\s*\(|=>|<\s*script\b|\b(?:window|document)\s*\.)", re.IGNORECASE)


class EChartsValidationError(ValueError):
    def __init__(self, pointer: str, message: str):
        self.pointer = pointer or "/"
        self.message = message
        super().__init__(f"{self.pointer}: {message}")


@dataclass
class _Resolution:
    option: dict[str, Any]
    source_refs: list[str]
    bindings: dict[str, VisualizationBinding]
    target_ids: set[str]
    table_rows: list[dict[str, Any]]


class EChartsCompiler:
    def __init__(self, catalog: PresentationCatalog):
        self.catalog = catalog

    def compile(self, plan: EChartsPlan) -> list[VisualizationPayload]:
        verification = VisualizationVerification(
            target_insight_ids=plan.target_insight_ids,
            verification_question=str(plan.visual_question),
            interpretation=str(plan.interpretation),
        )
        payloads = [self._compile_chart(chart, plan.target_insight_ids, verification) for chart in plan.charts]
        _validate_nonredundant_charts(plan, payloads)
        covered = {
            target
            for payload in payloads
            for target in (payload.verification.target_insight_ids if payload.verification else [])
            if any(_ref_matches_insight(ref, target) for ref in payload.source_refs)
        }
        missing = set(plan.target_insight_ids) - covered
        if missing:
            raise EChartsValidationError("/target_insight_ids", f"target Insights are not grounded by placeholders: {sorted(missing)}")
        return payloads

    def _compile_chart(
        self,
        chart: EChartsChartPlan,
        target_ids: list[str],
        verification: VisualizationVerification,
    ) -> VisualizationPayload:
        option = self._parse(chart.option_json)
        resolution = self._validate_and_resolve(option)
        visualization_id = _visualization_id(chart, option)
        columns = chart.accessibility_table_columns or _columns(resolution.table_rows)
        return VisualizationPayload(
            visualization_id=visualization_id,
            purpose=chart.purpose,
            priority=chart.priority,
            title=chart.title,
            summary=chart.summary,
            verification=verification,
            option=resolution.option,
            source_refs=resolution.source_refs,
            bindings=list(resolution.bindings.values()),
            accessibility=VisualizationAccessibility(
                description=chart.accessibility_description,
                table_columns=columns,
                table_rows=resolution.table_rows[:24],
            ),
        )

    @staticmethod
    def _parse(option_json: str) -> dict[str, Any]:
        if len(option_json.encode("utf-8")) > MAX_OPTION_BYTES:
            raise EChartsValidationError("/option_json", f"option exceeds {MAX_OPTION_BYTES} bytes")
        try:
            option = json.loads(option_json)
        except json.JSONDecodeError as exc:
            raise EChartsValidationError("/option_json", f"invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
        if not isinstance(option, dict):
            raise EChartsValidationError("/", "ECharts option must be an object")
        _validate_tree(option, "", 0)
        return option

    def _validate_and_resolve(self, option: dict[str, Any]) -> _Resolution:
        datasets = _as_list(option.get("dataset"))
        series = _as_list(option.get("series"))
        x_axes = _as_list(option.get("xAxis"))
        y_axes = _as_list(option.get("yAxis"))
        if not datasets or len(datasets) > MAX_DATASETS:
            raise EChartsValidationError("/dataset", f"option requires 1-{MAX_DATASETS} datasets")
        if not series or len(series) > MAX_SERIES:
            raise EChartsValidationError("/series", f"option requires 1-{MAX_SERIES} series")
        if not x_axes:
            raise EChartsValidationError("/xAxis", "at least one x axis is required")
        if not y_axes or len(y_axes) > MAX_Y_AXES:
            raise EChartsValidationError("/yAxis", f"option requires 1-{MAX_Y_AXES} y axes")
        for index, axis in enumerate(x_axes):
            if not isinstance(axis, dict):
                raise EChartsValidationError(f"/xAxis/{index}", "x axis must be an object")
        for index, axis in enumerate(y_axes):
            if not isinstance(axis, dict):
                raise EChartsValidationError(f"/yAxis/{index}", "y axis must be an object")

        refs: list[str] = []
        bindings: dict[str, VisualizationBinding] = {}
        target_ids: set[str] = set()
        table_rows: list[dict[str, Any]] = []
        resolved_datasets: list[dict[str, Any]] = []
        dataset_fields: list[set[str]] = []
        dataset_field_types: list[dict[str, str]] = []
        dataset_row_counts: list[int] = []
        dataset_ids: dict[str, int] = {}
        for index, dataset in enumerate(datasets):
            path = f"/dataset/{index}"
            if not isinstance(dataset, dict):
                raise EChartsValidationError(path, "dataset must be an object")
            if "transform" in dataset:
                raise EChartsValidationError(path + "/transform", "dataset transforms are forbidden")
            placeholder = dataset.get("source")
            if not _is_dataset_placeholder(placeholder):
                raise EChartsValidationError(path + "/source", "dataset.source must be exactly a $dataset placeholder")
            source_ref = str(placeholder["$dataset"])
            source = self._resolve_source(source_ref, path + "/source/$dataset")
            rows, scalar = _source_data(source)
            materialized = [dict(row) for row in rows] or ([dict(scalar)] if scalar else [])
            if not materialized:
                raise EChartsValidationError(path + "/source", f"source '{source_ref}' contains no records")
            output_rows = []
            source_bindings = _projection_bindings(source)
            for row_index, row in enumerate(materialized):
                binding_id = f"dataset_{index}:row_{row_index}"
                source_binding = source_bindings.get(str(row.get("item_id") or "")) or source_bindings.get("")
                binding = source_binding.model_copy(update={"binding_id": binding_id}) if source_binding else _generic_binding(
                    binding_id, source.ref, source.kind, row_index,
                )
                bindings[binding_id] = binding
                output = {str(key): value for key, value in row.items() if not str(key).startswith("__")}
                output["bindingId"] = binding_id
                output_rows.append(output)
            schema = _schema_fields(output_rows)
            fields = {str(item["name"]) for item in schema} | {"bindingId"}
            dataset_fields.append(fields)
            dataset_field_types.append({str(item["name"]): str(item.get("data_type") or "string") for item in schema})
            dataset_row_counts.append(len(output_rows))
            dataset_id = dataset.get("id")
            if dataset_id is not None:
                if not isinstance(dataset_id, str) or not dataset_id:
                    raise EChartsValidationError(path + "/id", "dataset id must be a non-empty string")
                if dataset_id in dataset_ids:
                    raise EChartsValidationError(path + "/id", f"duplicate dataset id '{dataset_id}'")
                dataset_ids[dataset_id] = index
            resolved_datasets.append({**dataset, "source": output_rows})
            refs.append(source.ref)
            target_ids.update(_source_insight_ids(source))
            table_rows.extend({"source": source.ref, **row} for row in output_rows[:12])

        geometries: set[str] = set()
        has_continuous_series = False
        resolved_series = []
        for index, item in enumerate(series):
            path = f"/series/{index}"
            if not isinstance(item, dict):
                raise EChartsValidationError(path, "series must be an object")
            series_type = item.get("type")
            if series_type not in ALLOWED_SERIES:
                raise EChartsValidationError(path + "/type", f"series type must be one of {sorted(ALLOWED_SERIES)}")
            if "data" in item:
                raise EChartsValidationError(path + "/data", "inline series.data is forbidden; bind a dataset")
            dataset_index = _series_dataset_index(item, dataset_ids, len(datasets), path)
            _axis_index(item, "xAxisIndex", len(x_axes), path)
            _axis_index(item, "yAxisIndex", len(y_axes), path)
            encode = item.get("encode")
            if not isinstance(encode, dict) or not encode:
                raise EChartsValidationError(path + "/encode", "dataset-backed series requires encode")
            _validate_encode(
                encode,
                dataset_fields[dataset_index],
                dataset_field_types[dataset_index],
                path + "/encode",
                x_axis_type=str(x_axes[item.get("xAxisIndex", 0)].get("type") or "category"),
            )
            if dataset_row_counts[dataset_index] >= 2:
                has_continuous_series = True
            signature = json.dumps(
                [series_type, dataset_index, encode, item.get("xAxisIndex", 0), item.get("yAxisIndex", 0)],
                sort_keys=True,
            )
            if signature in geometries:
                raise EChartsValidationError(path, "duplicate geometry uses the same dataset, encode, type, and axes")
            geometries.add(signature)
            resolved_item = dict(item)
            for mark_key in MARK_KEYS:
                if mark_key in resolved_item:
                    resolved_item[mark_key], mark_refs, mark_bindings, mark_targets = self._resolve_mark(
                        resolved_item[mark_key], path + f"/{mark_key}", mark_key,
                    )
                    refs.extend(mark_refs)
                    bindings.update(mark_bindings)
                    target_ids.update(mark_targets)
            resolved_series.append(resolved_item)

        if not has_continuous_series:
            raise EChartsValidationError("/series", "at least one line series requires two grounded records")

        _validate_legend(option.get("legend"), resolved_series)
        _validate_visual_scales(
            resolved_series,
            resolved_datasets,
            dataset_ids,
            x_axes,
            y_axes,
            y_axes_are_list=isinstance(option.get("yAxis"), list),
        )

        resolved = dict(option)
        resolved["dataset"] = resolved_datasets if isinstance(option.get("dataset"), list) else resolved_datasets[0]
        resolved["series"] = resolved_series if isinstance(option.get("series"), list) else resolved_series[0]
        return _Resolution(resolved, list(dict.fromkeys(refs)), bindings, target_ids, table_rows)

    def _resolve_mark(self, value: Any, path: str, mark_key: str):
        if not isinstance(value, dict):
            raise EChartsValidationError(path, f"{mark_key} must be an object")
        data = value.get("data")
        if not isinstance(data, list):
            raise EChartsValidationError(path + "/data", f"{mark_key}.data must be an array")
        refs: list[str] = []
        bindings: dict[str, VisualizationBinding] = {}
        targets: set[str] = set()
        resolved_data = []
        for index, item in enumerate(data):
            _validate_mark_data_grounding(item, f"{path}/data/{index}")
            resolved, item_refs, item_bindings, item_targets = self._resolve_values(item, f"{path}/data/{index}")
            if item_bindings and isinstance(resolved, dict):
                binding_ids = list(item_bindings)
                if len(binding_ids) == 1:
                    resolved["bindingId"] = binding_ids[0]
            resolved_data.append(resolved)
            refs.extend(item_refs)
            bindings.update(item_bindings)
            targets.update(item_targets)
        return {**value, "data": resolved_data}, refs, bindings, targets

    def _resolve_values(self, value: Any, path: str):
        if _is_value_placeholder(value):
            spec = value["$value"]
            source_ref = str(spec.get("source_ref") or "")
            source = self._resolve_source(source_ref, path + "/$value/source_ref")
            field = str(spec.get("field") or "")
            if field.startswith("/"):
                if spec.get("item_id") is not None or spec.get("record_id") is not None:
                    raise EChartsValidationError(
                        path + "/$value/field",
                        "a JSON Pointer field addresses the unique grounding document and cannot use a record selector",
                    )
                resolved_value = _json_pointer_value(
                    _value_grounding_document(source), field, path + "/$value/field",
                )
                if isinstance(resolved_value, (dict, list)):
                    raise EChartsValidationError(
                        path + "/$value/field",
                        f"JSON Pointer '{field}' must resolve to one scalar value",
                    )
                binding_id = "value:" + hashlib.sha256(f"{source.ref}:grounding_document".encode()).hexdigest()[:16]
                binding = _generic_binding(binding_id, source.ref, source.kind, 0)
                return resolved_value, [source.ref], {binding_id: binding}, _source_insight_ids(source)
            rows, scalar = _source_data(source)
            records = [dict(row) for row in rows] or ([dict(scalar)] if scalar else [])
            item_selector = spec.get("item_id")
            record_selector = spec.get("record_id")
            if record_selector == "scalar":
                records = _scalar_records(source)
            elif item_selector is not None:
                records = [row for row in records if str(row.get("item_id") or "") == str(item_selector)]
            elif record_selector is not None:
                records = [
                    row for row in records
                    if str(row.get("item_id") or row.get("record_id") or "") == str(record_selector)
                ]
            if len(records) != 1:
                raise EChartsValidationError(path, f"$value source must select exactly one record, found {len(records)}")
            if field not in records[0]:
                raise EChartsValidationError(path + "/$value/field", f"unknown field '{field}' in source '{source.ref}'")
            source_bindings = _projection_bindings(source)
            original = source_bindings.get(str(records[0].get("item_id") or "")) or source_bindings.get("")
            selector = item_selector or record_selector
            binding_id = "value:" + hashlib.sha256(f"{source.ref}:{selector}".encode()).hexdigest()[:16]
            binding = original.model_copy(update={"binding_id": binding_id}) if original else _generic_binding(
                binding_id, source.ref, source.kind, 0,
            )
            return records[0][field], [source.ref], {binding_id: binding}, _source_insight_ids(source)
        if isinstance(value, dict):
            output, refs, bindings, targets = {}, [], {}, set()
            for key, child in value.items():
                resolved, child_refs, child_bindings, child_targets = self._resolve_values(child, path + "/" + _escape(key))
                output[key] = resolved
                refs.extend(child_refs)
                bindings.update(child_bindings)
                targets.update(child_targets)
            return output, refs, bindings, targets
        if isinstance(value, list):
            output, refs, bindings, targets = [], [], {}, set()
            for index, child in enumerate(value):
                resolved, child_refs, child_bindings, child_targets = self._resolve_values(child, f"{path}/{index}")
                output.append(resolved)
                refs.extend(child_refs)
                bindings.update(child_bindings)
                targets.update(child_targets)
            return output, refs, bindings, targets
        return value, [], {}, set()

    def _resolve_source(self, ref: str, pointer: str):
        try:
            source = self.catalog.resolve(ref)
        except ValueError as exc:
            raise EChartsValidationError(pointer, str(exc)) from exc
        if source.ref not in set(self.catalog.projection_refs()):
            raise EChartsValidationError(pointer, f"source '{ref}' is not exposed by the Grounded Source Inventory")
        return source


def _validate_nonredundant_charts(plan: EChartsPlan, payloads: list[VisualizationPayload]) -> None:
    if len(payloads) < 2:
        return
    primary_index = next(index for index, chart in enumerate(plan.charts) if chart.priority == "primary")
    primary = payloads[primary_index]
    primary_geometry = _chart_geometry(primary.option)
    primary_refs = set(primary.source_refs)
    for index, payload in enumerate(payloads):
        if index == primary_index:
            continue
        if (
            _chart_geometry(payload.option) <= primary_geometry
            and set(payload.source_refs) <= primary_refs
            and not _chart_has_marks(payload.option)
        ):
            raise EChartsValidationError(
                f"/charts/{index}",
                "supporting chart is visually redundant with the primary chart; remove it or use distinct grounded data",
            )


def _chart_geometry(option: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in _as_list(option.get("series")):
        if not isinstance(item, dict):
            continue
        result.add(json.dumps(
            [item.get("type"), item.get("datasetId"), item.get("datasetIndex", 0), item.get("encode")],
            sort_keys=True,
        ))
    return result


def _chart_has_marks(option: dict[str, Any]) -> bool:
    return any(
        any(key in item for key in MARK_KEYS)
        for item in _as_list(option.get("series"))
        if isinstance(item, dict)
    )


def _validate_tree(value: Any, path: str, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise EChartsValidationError(path, f"option nesting exceeds {MAX_DEPTH}")
    if isinstance(value, dict):
        if "$dataset" in value and not _is_dataset_placeholder(value):
            raise EChartsValidationError(path, "malformed $dataset placeholder")
        if "$value" in value and not _is_value_placeholder(value):
            raise EChartsValidationError(path, "malformed $value placeholder")
        for key, child in value.items():
            pointer = path + "/" + _escape(key)
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise EChartsValidationError(pointer, f"'{key}' is forbidden")
            _validate_tree(child, pointer, depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_tree(child, f"{path}/{index}", depth + 1)
    elif isinstance(value, str) and (URL_PATTERN.search(value) or EXECUTABLE_PATTERN.search(value)):
        raise EChartsValidationError(path, "URLs and executable content are forbidden")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise EChartsValidationError(path, f"unsupported JSON value type '{type(value).__name__}'")


def _is_dataset_placeholder(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"$dataset"} and isinstance(value["$dataset"], str)


def _is_value_placeholder(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"$value"} or not isinstance(value["$value"], dict):
        return False
    spec = value["$value"]
    return set(spec) <= {"source_ref", "field", "item_id", "record_id"} and {"source_ref", "field"} <= set(spec)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _series_dataset_index(item: dict, ids: dict[str, int], count: int, path: str) -> int:
    if "datasetId" in item:
        dataset_id = item["datasetId"]
        if dataset_id not in ids:
            raise EChartsValidationError(path + "/datasetId", f"unknown dataset id '{dataset_id}'")
        resolved = ids[dataset_id]
        if "datasetIndex" in item:
            index = item["datasetIndex"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= count:
                raise EChartsValidationError(path + "/datasetIndex", f"dataset index must be between 0 and {count - 1}")
            if index != resolved:
                raise EChartsValidationError(path + "/datasetIndex", "datasetId and datasetIndex resolve to different datasets")
        return resolved
    index = item.get("datasetIndex", 0)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= count:
        raise EChartsValidationError(path + "/datasetIndex", f"dataset index must be between 0 and {count - 1}")
    return index


def _axis_index(item: dict, key: str, count: int, path: str) -> None:
    index = item.get(key, 0)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= count:
        raise EChartsValidationError(path + f"/{key}", f"axis index must be between 0 and {count - 1}")


def _validate_encode(
    encode: dict,
    fields: set[str],
    field_types: dict[str, str],
    path: str,
    *,
    x_axis_type: str,
) -> None:
    selected = []
    for channel, value in encode.items():
        values = value if isinstance(value, list) else [value]
        for index, field in enumerate(values):
            if isinstance(field, str):
                if field not in fields:
                    suffix = f"/{_escape(channel)}/{index}" if isinstance(value, list) else f"/{_escape(channel)}"
                    raise EChartsValidationError(path + suffix, f"unknown encoded field '{field}'")
                selected.append(field)
                if channel == "y" and field_types.get(field) != "number":
                    raise EChartsValidationError(path + f"/{_escape(channel)}", f"y field '{field}' must be numeric")
                if channel == "x" and x_axis_type == "time" and field_types.get(field) not in {"time", "number"}:
                    raise EChartsValidationError(
                        path + f"/{_escape(channel)}",
                        f"x field '{field}' must be temporal or numeric for a time axis",
                    )
            else:
                raise EChartsValidationError(path + f"/{_escape(channel)}", "encode values must be grounded field names")
    if not selected:
        raise EChartsValidationError(path, "encode must select grounded fields")


def _scalar_records(source) -> list[dict[str, Any]]:
    if source.kind == "insight":
        value = source.value.value
        return [dict(value)] if isinstance(value, dict) else [{"label": source.value.name, "value": value}]
    if source.kind == "view" and source.value.scalar is not None:
        return [dict(source.value.scalar)]
    return []


def _value_grounding_document(source) -> dict[str, Any]:
    """Expose one stable, read-only document for nested grounded Insight values."""

    if source.kind == "insight":
        insight = source.value
        locator = {}
        rows, _scalar = _source_data(source)
        if len(rows) == 1 and not insight.items:
            locator = dict(rows[0])
        return {
            "source_ref": source.ref,
            "statement": insight.statement,
            "value": insight.value,
            "unit": insight.unit,
            "time_range": insight.time_range,
            "dimensions": insight.dimensions,
            "selection": insight.selection,
            "calculation_trace": insight.calculation_trace,
            "items": [_insight_item_document(item) for item in insight.items],
            "locator": locator or None,
        }
    if source.kind == "insight_item":
        insight, item = source.value
        return {
            "source_ref": source.ref,
            "value": item.value,
            "unit": insight.unit,
            "calculation_trace": insight.calculation_trace,
            "item": _insight_item_document(item),
        }
    rows, scalar = _source_data(source)
    return {
        "source_ref": source.ref,
        "rows": rows,
        "scalar": scalar,
    }


def _insight_item_document(item) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "item_id": item.item_id,
            "value": item.value,
            "label": item.label,
            "rank": item.rank,
            "timestamp": item.timestamp,
            "source_item_ids": list(item.source_item_ids),
            **item.dimensions,
            **item.locator,
        }.items()
        if value is not None
    }


def _json_pointer_value(document: Any, pointer: str, error_path: str) -> Any:
    current = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise EChartsValidationError(error_path, f"JSON Pointer '{pointer}' has invalid list index '{token}'") from exc
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise EChartsValidationError(error_path, f"JSON Pointer '{pointer}' does not exist in source grounding document")
    return current


def _validate_legend(legend: Any, series: list[dict[str, Any]]) -> None:
    names = {str(item.get("name")) for item in series if item.get("name")}
    for legend_index, item in enumerate(_as_list(legend)):
        if not isinstance(item, dict) or "data" not in item:
            continue
        data = item.get("data")
        if not isinstance(data, list):
            continue
        for index, entry in enumerate(data):
            name = entry if isinstance(entry, str) else entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str) and name not in names:
                prefix = f"/legend/{legend_index}" if isinstance(legend, list) else "/legend"
                raise EChartsValidationError(prefix + f"/data/{index}", f"legend entry '{name}' has no matching series")


def _validate_visual_scales(
    series: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    dataset_ids: dict[str, int],
    x_axes: list[dict[str, Any]],
    y_axes: list[dict[str, Any]],
    *,
    y_axes_are_list: bool,
) -> None:
    by_axis: dict[int, list[tuple[int, float]]] = {}
    for series_index, item in enumerate(series):
        encode = item.get("encode") if isinstance(item.get("encode"), dict) else {}
        y_fields = encode.get("y")
        y_fields = y_fields if isinstance(y_fields, list) else [y_fields]
        fields = [field for field in y_fields if isinstance(field, str)]
        if not fields:
            continue
        dataset_index = dataset_ids.get(item.get("datasetId"), item.get("datasetIndex", 0))
        if not isinstance(dataset_index, int) or dataset_index >= len(datasets):
            continue
        rows = datasets[dataset_index].get("source")
        values = []
        if isinstance(rows, list):
            values = [
                float(row[field])
                for row in rows if isinstance(row, dict)
                for field in fields
                if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
            ]
        if values:
            if any(not math.isfinite(value) for value in values):
                raise EChartsValidationError(f"/series/{series_index}/encode/y", "series contains a non-finite y value")
            series_median = abs(statistics.median(values))
            nonzero = [abs(value) for value in values if value]
            robust_center = statistics.median(nonzero) if nonzero else 0.0
            if len(nonzero) >= 3 and robust_center > 0 and max(nonzero) / robust_center > 100:
                raise EChartsValidationError(
                    f"/series/{series_index}/datasetIndex",
                    "series contains extreme values that make its main range unreadable; use the relevant filtered or cleaned source",
                )
            y_axis_index = int(item.get("yAxisIndex", 0))
            if (
                item.get("type") == "line"
                and min(values) > 0
                and max(values) / min(values) < 10
                and y_axes[y_axis_index].get("scale") is not True
            ):
                prefix = f"/yAxis/{y_axis_index}" if y_axes_are_list else "/yAxis"
                raise EChartsValidationError(
                    prefix + "/scale",
                    "positive line data with a narrow range requires scale=true to avoid an unreadable zero baseline",
                )
            by_axis.setdefault(int(item.get("yAxisIndex", 0)), []).append((series_index, series_median))
            mark_line = item.get("markLine") if isinstance(item.get("markLine"), dict) else {}
            for mark_index, mark in enumerate(mark_line.get("data") or []):
                if not isinstance(mark, dict) or not isinstance(mark.get("yAxis"), (int, float)):
                    continue
                mark_value = abs(float(mark["yAxis"]))
                low, high = sorted((series_median, mark_value))
                if low > 0 and high / low > 4:
                    raise EChartsValidationError(
                        f"/series/{series_index}/markLine/data/{mark_index}/yAxis",
                        "reference value is not visually compatible with the series y scale",
                    )
            _validate_mark_coordinates(item, series_index, series_median, x_axes)
    for items in by_axis.values():
        positive = [(index, median) for index, median in items if median > 0]
        if len(positive) < 2:
            continue
        low_index, low = min(positive, key=lambda value: value[1])
        _high_index, high = max(positive, key=lambda value: value[1])
        if high / low > 4:
            raise EChartsValidationError(
                f"/series/{low_index}/yAxisIndex",
                "series sharing one y axis have incompatible visual scales; remove the unrelated series or use a justified second axis",
            )


def _validate_mark_coordinates(
    series: dict[str, Any],
    series_index: int,
    series_median: float,
    x_axes: list[dict[str, Any]],
) -> None:
    mark_point = series.get("markPoint") if isinstance(series.get("markPoint"), dict) else {}
    x_axis_index = int(series.get("xAxisIndex", 0))
    x_axis_type = str(x_axes[x_axis_index].get("type") or "category")
    for mark_index, mark in enumerate(mark_point.get("data") or []):
        if not isinstance(mark, dict) or "coord" not in mark:
            continue
        symbol = mark.get("symbol", mark_point.get("symbol"))
        symbol_size = mark.get("symbolSize", mark_point.get("symbolSize"))
        label = mark.get("label", mark_point.get("label"))
        style_pointer = f"/series/{series_index}/markPoint/data/{mark_index}"
        if symbol != "circle":
            raise EChartsValidationError(style_pointer + "/symbol", "markPoint symbol must be 'circle'")
        if not isinstance(symbol_size, (int, float)) or isinstance(symbol_size, bool) or not 8 <= symbol_size <= 16:
            raise EChartsValidationError(style_pointer + "/symbolSize", "markPoint symbolSize must be between 8 and 16")
        if not isinstance(label, dict) or label.get("show") is not False:
            raise EChartsValidationError(style_pointer + "/label/show", "markPoint labels must be hidden to prevent overlap")
        coord = mark.get("coord")
        pointer = f"/series/{series_index}/markPoint/data/{mark_index}/coord"
        if not isinstance(coord, list) or len(coord) != 2:
            raise EChartsValidationError(pointer, "markPoint coord must contain exactly [x, y]")
        if x_axis_type == "time" and not _is_time_coordinate(coord[0]):
            raise EChartsValidationError(pointer + "/0", "markPoint x coordinate is not valid for the time axis")
        if not isinstance(coord[1], (int, float)) or isinstance(coord[1], bool) or not math.isfinite(float(coord[1])):
            raise EChartsValidationError(pointer + "/1", "markPoint y coordinate must be a finite number")
        mark_value = abs(float(coord[1]))
        low, high = sorted((series_median, mark_value))
        if low > 0 and high / low > 4:
            raise EChartsValidationError(pointer + "/1", "markPoint y coordinate is not visually compatible with the series y scale")


def _is_time_coordinate(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _validate_mark_data_grounding(value: Any, path: str) -> None:
    if not _contains_value_placeholder(value):
        raise EChartsValidationError(path, "mark data must contain at least one grounded $value placeholder")
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + "/" + _escape(key)
            if key in {"xAxis", "yAxis", "value"} and not _contains_value_placeholder(child):
                raise EChartsValidationError(child_path, f"mark {key} data must use a $value placeholder")
            if key == "coord":
                if not isinstance(child, list) or not child or any(not _contains_value_placeholder(item) for item in child):
                    raise EChartsValidationError(child_path, "every mark coordinate must use a $value placeholder")
            _validate_nested_mark_coordinates(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_mark_coordinates(child, f"{path}/{index}")


def _validate_nested_mark_coordinates(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + "/" + _escape(key)
            if key in {"xAxis", "yAxis", "value"} and not _contains_value_placeholder(child):
                raise EChartsValidationError(child_path, f"mark {key} data must use a $value placeholder")
            if key == "coord" and (
                not isinstance(child, list) or not child or any(not _contains_value_placeholder(item) for item in child)
            ):
                raise EChartsValidationError(child_path, "every mark coordinate must use a $value placeholder")
            _validate_nested_mark_coordinates(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_mark_coordinates(child, f"{path}/{index}")


def _contains_value_placeholder(value: Any) -> bool:
    if _is_value_placeholder(value):
        return True
    if isinstance(value, dict):
        return any(_contains_value_placeholder(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_value_placeholder(child) for child in value)
    return False


def _generic_binding(binding_id: str, source_ref: str, source_type: str, row_index: int) -> VisualizationBinding:
    return VisualizationBinding(
        binding_id=binding_id,
        source_type=source_type,
        source_ref=source_ref,
        evidence_id=_evidence_id(source_ref),
        locator={"row_index": row_index},
    )


def _source_insight_ids(source) -> set[str]:
    if source.kind == "insight":
        return {str(source.value.insight_id)}
    if source.kind == "insight_item":
        return {str(source.value[0].insight_id)}
    return set()


def _ref_matches_insight(ref: str, insight_id: str) -> bool:
    return ref == f"insight:{insight_id}" or ref.startswith(f"insight:{insight_id}#")


def _evidence_id(ref: str) -> str | None:
    if ref.startswith("view:evidence:"):
        return ref.removeprefix("view:evidence:").rsplit(":", 1)[0]
    if ref.startswith("evidence:"):
        return ref.split(":", 1)[1]
    return None


def _visualization_id(chart: EChartsChartPlan, option: dict[str, Any]) -> str:
    stable = json.dumps({"chart": chart.model_dump(exclude={"option_json"}), "option": option}, sort_keys=True, default=str)
    return "viz_" + hashlib.sha256(stable.encode()).hexdigest()[:20]


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    result = []
    for row in rows[:24]:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def _escape(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")
