"""Durable storage for full-fidelity visualization payloads."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading

from pydantic import ValidationError

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
        complete_views = []
        for view in visualization.data_views:
            complete_views.append(view.model_copy(update={
                "data_ref": data_ref,
                "row_count": len(view.records),
                "time_range": _view_time_range(view),
            }, deep=True))
        complete = visualization.model_copy(update={
            "data_ref": data_ref,
            "data_views": complete_views,
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
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if not isinstance(payload, dict) or payload.get("schema_version") != "4":
                return None
            try:
                return VisualizationPayload.model_validate(payload)
            except ValidationError:
                # Old V3 and pre-LineChart V4 artifacts remain on disk but are
                # intentionally not migrated or interpreted by the new runtime.
                return None

    def descriptor(self, visualization: VisualizationPayload) -> VisualizationPayload:
        views = []
        for view in visualization.data_views:
            views.append(view.model_copy(update={
                "data_ref": visualization.data_ref,
                "row_count": len(view.records),
                "time_range": _view_time_range(view),
                "records": [],
            }, deep=True))
        return visualization.model_copy(update={
            "data_views": views,
            "accessibility": visualization.accessibility.model_copy(update={"table_rows": []}),
        }, deep=True)

    def _validate_id(self, artifact_id: str) -> str:
        value = str(artifact_id or "").strip()
        if not value or not _SAFE_ID.fullmatch(value):
            raise ValueError("invalid visualization artifact id")
        return value


def _view_time_range(view) -> dict | None:
    time_fields = {field.name for field in view.fields if field.data_type == "time"}
    values = [
        record.values.get(field)
        for record in view.records
        for field in time_fields
        if record.values.get(field) not in (None, "")
    ]
    if not values:
        return None
    ordered = sorted(values, key=str)
    return {"start": ordered[0], "end": ordered[-1]}
