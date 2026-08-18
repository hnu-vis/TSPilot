"""Canonical, read-only data sources shared by analysis and presentation."""
from __future__ import annotations

from typing import Any

from schemas.database import DatabaseEvidence


def resolve_artifact_sources(request_state, refs: list[str]) -> list[dict]:
    """Resolve stable artifact refs into complete, typed datasets.

    Business interpretation remains with the consuming LLM. This resolver only
    exposes authoritative values, structure, and lineage.
    """

    resolved: list[dict] = []
    seen: set[str] = set()
    for raw_ref in refs:
        ref = _canonical_ref(str(raw_ref or "").strip(), request_state)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        kind, source_id = ref.split(":", 1)
        if kind == "evidence":
            artifact = request_state.database_evidence_artifacts.get(source_id)
            if artifact is None and getattr(request_state.latest_database_evidence, "evidence_id", None) == source_id:
                artifact = request_state.latest_database_evidence
            if artifact is None:
                raise ValueError(f"unknown artifact source '{ref}'")
            rows = _database_rows(artifact)
            resolved.append(_source(ref, kind, getattr(artifact, "summary", None), [ref], [
                _dataset("records", rows=rows, shape=_row_shape(rows)),
            ]))
            continue
        if kind == "forecast":
            artifact = request_state.forecast_artifacts.get(source_id)
            if artifact is None:
                raise ValueError(f"unknown artifact source '{ref}'")
            points = [_model_dict(item) for item in artifact.forecast_points]
            intervals = [_model_dict(item) for item in artifact.confidence_interval]
            lineage = [ref, *_forecast_evidence_refs(artifact)]
            datasets = [_dataset("forecast_points", rows=points, shape="timeseries")]
            if intervals:
                datasets.append(_dataset("confidence_intervals", rows=intervals, shape="intervals"))
            quality = artifact.diagnostics.get("input_quality") if isinstance(artifact.diagnostics, dict) else None
            if isinstance(quality, dict) and quality:
                datasets.append(_dataset("forecast_quality", scalar=quality, shape="scalar"))
            resolved.append(_source(ref, kind, artifact.model_name, lineage, datasets))
            continue
        if kind == "anomaly":
            artifact = request_state.anomaly_artifacts.get(source_id)
            if artifact is None:
                raise ValueError(f"unknown artifact source '{ref}'")
            points = [_model_dict(item) for item in artifact.anomaly_points]
            spans = [_model_dict(item) for item in artifact.anomaly_spans]
            scores = [_model_dict(item) for item in artifact.scores]
            evidence_ref = _anomaly_evidence_ref(artifact)
            datasets = [
                _dataset("anomaly_points", rows=points, shape="records"),
                _dataset("anomaly_status", scalar={"detected_count": len(points)}, shape="scalar"),
            ]
            if spans:
                datasets.append(_dataset("anomaly_spans", rows=spans, shape="intervals"))
            if scores:
                datasets.append(_dataset("anomaly_scores", rows=scores, shape=_row_shape(scores)))
            resolved.append(_source(
                ref, kind, artifact.detector_name,
                [ref, *([evidence_ref] if evidence_ref else [])], datasets,
            ))
            continue
        if kind == "derived_evidence":
            artifact = request_state.derived_evidence_artifacts.get(source_id)
            if artifact is None:
                raise ValueError(f"unknown artifact source '{ref}'")
            resolved.append(_source(ref, kind, artifact.name, [ref, *artifact.lineage], [
                _dataset(artifact.name, rows=list(artifact.rows), scalar=artifact.scalar, shape=artifact.shape),
            ]))
            continue
        if kind == "insight":
            artifact = next(
                (
                    item for item in request_state.insight_set.insights
                    if item.insight_id == source_id or item.insight_key == source_id
                ),
                None,
            )
            if artifact is None or artifact.status not in {"verified", "unavailable"}:
                raise ValueError(f"unknown or unverified artifact source '{ref}'")
            rows = [_insight_item_row(item) for item in artifact.items]
            scalar = None if rows else {
                "value": artifact.value,
                "statement": artifact.statement,
                "status": artifact.status,
                "unavailable_reason": artifact.unavailable_reason,
            }
            evidence_refs = [f"{item.source_type}:{item.source_id}" for item in artifact.evidence_refs]
            resolved.append(_source(ref, kind, artifact.name, [ref, *evidence_refs], [
                _dataset("insight", rows=rows, scalar=scalar, shape="records" if rows else "scalar"),
            ]))
            continue
        raise ValueError(f"unsupported artifact source '{ref}'")
    return resolved


def database_evidence_for_sources(request_state, sources: list[dict]) -> DatabaseEvidence | None:
    """Return the database ancestor used for legacy canonical variables."""

    for source in sources:
        for ref in source.get("lineage", []):
            if not str(ref).startswith("evidence:"):
                continue
            evidence_id = str(ref).split(":", 1)[1]
            artifact = request_state.database_evidence_artifacts.get(evidence_id)
            if artifact is not None:
                return artifact
    return request_state.latest_database_evidence


def source_prompt_manifest(sources: list[dict], *, preview_rows: int = 5) -> list[dict]:
    """Bounded source description for LLM code generation."""

    return [
        {
            "source_ref": source["source_ref"],
            "source_type": source["source_type"],
            "label": source.get("label"),
            "lineage": source.get("lineage", []),
            "datasets": [
                {
                    "name": dataset["name"],
                    "shape": dataset["shape"],
                    "row_count": dataset["row_count"],
                    "schema_fields": dataset["schema_fields"],
                    "preview": dataset.get("rows", [])[:preview_rows]
                    or ([dataset["scalar"]] if dataset.get("scalar") is not None else []),
                }
                for dataset in source.get("datasets", [])
            ],
        }
        for source in sources
    ]


def primary_analysis_input(sources: list[dict]) -> dict | None:
    """Bind the first explicitly referenced artifact to canonical analysis rows.

    Source order is caller intent.  The first non-empty dataset of that source is
    therefore the primary computation input; lineage evidence remains available
    through ``source_by_ref`` but must not silently replace the requested artifact.
    """

    for source in sources:
        for dataset in source.get("datasets", []):
            rows = [dict(item) for item in dataset.get("rows", []) if isinstance(item, dict)]
            scalar = dataset.get("scalar")
            if not rows and isinstance(scalar, dict):
                rows = [dict(scalar)]
            if not rows:
                continue
            columns = list(dict.fromkeys(key for row in rows for key in row))
            points = rows if dataset.get("shape") == "timeseries" else []
            return {
                "source_ref": source["source_ref"],
                "dataset_name": dataset.get("name"),
                "shape": dataset.get("shape"),
                "rows": rows,
                "points": points,
                "columns": columns,
            }
    return None


def _source(ref: str, kind: str, label: str | None, lineage: list[str], datasets: list[dict]) -> dict:
    source = {
        "source_ref": ref,
        "source_type": kind,
        "label": str(label or ref),
        "lineage": list(dict.fromkeys(item for item in lineage if item)),
        "datasets": datasets,
    }
    # Expose every dataset by its authoritative name as a generic convenience
    # view.  Names are artifact-owned rather than enumerated here, so new tool
    # datasets automatically become accessible without consumer changes.
    for dataset in datasets:
        name = str(dataset.get("name") or "").strip()
        if not name or name in source:
            continue
        source[name] = (
            dataset.get("rows")
            if dataset.get("rows")
            else dataset.get("scalar")
        )
    return source


def _dataset(name: str, *, rows: list[dict] | None = None, scalar: dict | None = None, shape: str) -> dict:
    items = [dict(item) for item in rows or [] if isinstance(item, dict)]
    return {
        "name": name,
        "shape": shape,
        "rows": items,
        "scalar": dict(scalar) if isinstance(scalar, dict) else None,
        "row_count": len(items) or int(isinstance(scalar, dict)),
        "schema_fields": _schema_fields(items or ([scalar] if isinstance(scalar, dict) else [])),
    }


def _canonical_ref(ref: str, request_state) -> str:
    if not ref:
        return ""
    if ":" in ref:
        return ref
    for kind, artifacts in (
        ("evidence", request_state.database_evidence_artifacts),
        ("forecast", request_state.forecast_artifacts),
        ("anomaly", request_state.anomaly_artifacts),
        ("derived_evidence", request_state.derived_evidence_artifacts),
    ):
        if ref in artifacts:
            return f"{kind}:{ref}"
    return ref


def _database_rows(evidence) -> list[dict]:
    data = evidence.data if isinstance(evidence.data, dict) else {}
    for key in ("rows", "points"):
        if isinstance(data.get(key), list):
            return [dict(item) for item in data[key] if isinstance(item, dict)]
    rows: list[dict] = []
    for series in data.get("series", []) if isinstance(data.get("series"), list) else []:
        if not isinstance(series, dict):
            continue
        identity = series.get("series_name") or series.get("value_field")
        rows.extend({**item, **({"series": identity} if identity else {})} for item in series.get("points", []) if isinstance(item, dict))
    return rows


def _forecast_evidence_refs(forecast) -> list[str]:
    diagnostics = forecast.diagnostics if isinstance(forecast.diagnostics, dict) else {}
    coverage = diagnostics.get("coverage") if isinstance(diagnostics.get("coverage"), dict) else {}
    refs = coverage.get("input_evidence_refs") if isinstance(coverage.get("input_evidence_refs"), list) else []
    return [item if str(item).startswith("evidence:") else f"evidence:{item}" for item in refs if item]


def _anomaly_evidence_ref(anomaly) -> str | None:
    diagnostics = anomaly.diagnostics if isinstance(anomaly.diagnostics, dict) else {}
    for key in ("resolved_evidence_id", "input_evidence_id", "selected_evidence_id"):
        value = str(diagnostics.get(key) or "").strip()
        if value:
            return value if value.startswith("evidence:") else f"evidence:{value}"
    anomaly_id = str(getattr(anomaly, "anomaly_id", "") or "")
    return f"evidence:{anomaly_id.removeprefix('anomaly_')}" if anomaly_id.startswith("anomaly_") else None


def _model_dict(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {"value": value}


def _insight_item_row(item) -> dict:
    row = item.model_dump(mode="json", exclude_none=True)
    dimensions = row.pop("dimensions", {})
    locator = row.pop("locator", {})
    return {**row, **dimensions, **locator}


def _schema_fields(rows: list[dict]) -> list[dict]:
    fields: list[dict] = []
    for row in rows[:40]:
        for key, value in row.items():
            if any(item["name"] == str(key) for item in fields):
                continue
            fields.append({"name": str(key), "type": type(value).__name__})
    return fields


def _row_shape(rows: list[dict]) -> str:
    fields = {str(key).lower() for row in rows for key in row}
    if {"lower", "upper"}.issubset(fields):
        return "intervals"
    if fields & {"timestamp", "time", "_time", "date", "datetime"}:
        return "timeseries"
    return "records"
