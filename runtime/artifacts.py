"""Artifact persistence helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def persist_json_artifact(
    *,
    artifact_id: str,
    artifact_kind: str,
    payload: dict,
    subdir: str | None = None,
    directory: str | Path | None = None,
) -> dict:
    """Persist one artifact payload to cache_data and return its reference."""

    if directory is not None:
        root = Path(directory)
    elif subdir is not None:
        root = Path(__file__).resolve().parents[1] / "cache_data" / subdir
    else:
        raise ValueError("persist_json_artifact requires either subdir or directory.")
    root.mkdir(parents=True, exist_ok=True)
    filename = _artifact_filename(artifact_id)
    path = root / f"{filename}.json"
    envelope = {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "uri": str(path),
    }


def _artifact_filename(artifact_id: str) -> str:
    """Build a filesystem-safe artifact filename with a stable hash suffix."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(artifact_id)).strip("._-")
    prefix = normalized[:80] or "artifact"
    digest = hashlib.sha1(str(artifact_id).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
