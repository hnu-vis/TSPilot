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
        complete = visualization.model_copy(update={
            "data_ref": data_ref,
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
            if not isinstance(payload, dict) or payload.get("schema_version") != "5":
                return None
            try:
                return VisualizationPayload.model_validate(payload)
            except ValidationError:
                # Older artifacts remain on disk but are intentionally not
                # migrated or interpreted by the V5 runtime.
                return None

    def descriptor(self, visualization: VisualizationPayload) -> VisualizationPayload:
        return visualization.model_copy(update={
            "option": _without_dataset_sources(visualization.option),
            "accessibility": visualization.accessibility.model_copy(update={"table_rows": []}),
        }, deep=True)

    def _validate_id(self, artifact_id: str) -> str:
        value = str(artifact_id or "").strip()
        if not value or not _SAFE_ID.fullmatch(value):
            raise ValueError("invalid visualization artifact id")
        return value

def _without_dataset_sources(option: dict) -> dict:
    """Preserve native option structure while removing only full record arrays."""
    clone = json.loads(json.dumps(option, ensure_ascii=False))
    datasets = clone.get("dataset")
    for dataset in datasets if isinstance(datasets, list) else ([datasets] if isinstance(datasets, dict) else []):
        if isinstance(dataset, dict):
            dataset["source"] = []
    return clone
