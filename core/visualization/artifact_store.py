"""Durable storage for full-fidelity visualization payloads."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading

from schemas.visualization import VisualizationPayload


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class VisualizationArtifactStore:
    """Persist complete visualizations while returning lightweight descriptors."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.RLock()

    def put(self, visualization: VisualizationPayload) -> VisualizationPayload:
        artifact_id = self._validate_id(visualization.visualization_id)
        data_ref = f"/api/v1/visualizations/{artifact_id}/data"
        complete_datasets = []
        for dataset in visualization.datasets:
            row_count = sum(len(series.points) for series in dataset.series)
            complete_datasets.append(dataset.model_copy(update={
                "data_ref": data_ref,
                "row_count": row_count,
                "time_range": _dataset_time_range(dataset),
            }, deep=True))
        complete = visualization.model_copy(update={
            "data_ref": data_ref,
            "datasets": complete_datasets,
        }, deep=True)
        payload = complete.model_dump(mode="json")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / f"{artifact_id}.json"
            temporary = self.root / f".{artifact_id}.tmp"
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
        return self.descriptor(complete)

    def get(self, artifact_id: str) -> VisualizationPayload | None:
        safe_id = self._validate_id(artifact_id)
        target = self.root / f"{safe_id}.json"
        if not target.is_file():
            return None
        with self._lock:
            return VisualizationPayload.model_validate_json(target.read_text(encoding="utf-8"))

    def descriptor(self, visualization: VisualizationPayload) -> VisualizationPayload:
        datasets = []
        for dataset in visualization.datasets:
            row_count = sum(len(series.points) for series in dataset.series)
            datasets.append(dataset.model_copy(update={
                "data_ref": visualization.data_ref,
                "row_count": row_count,
                "time_range": _dataset_time_range(dataset),
                "series": [],
            }, deep=True))
        layers = [layer.model_copy(update={"points": []}, deep=True) for layer in visualization.layers]
        return visualization.model_copy(update={
            "datasets": datasets,
            "layers": layers,
            "accessibility": visualization.accessibility.model_copy(update={"table_rows": []}),
        }, deep=True)

    def _validate_id(self, artifact_id: str) -> str:
        value = str(artifact_id or "").strip()
        if not value or not _SAFE_ID.fullmatch(value):
            raise ValueError("invalid visualization artifact id")
        return value


def _dataset_time_range(dataset) -> dict | None:
    values = []
    for series in dataset.series:
        values.extend(point.x for point in series.points if point.x not in (None, ""))
    if not values:
        return None
    ordered = sorted(values, key=str)
    return {"start": ordered[0], "end": ordered[-1]}
