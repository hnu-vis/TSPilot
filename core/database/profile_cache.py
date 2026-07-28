"""Persistent database profile cache helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_TTL_SECONDS = 15 * 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_profile_id(database_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(database_id)).strip("._") or "database"


def profile_path(profile_dir: Path, database_id: str) -> Path:
    return profile_dir / f"{safe_profile_id(database_id)}.json"


def read_profile(profile_dir: Path, database_id: str) -> dict[str, Any] | None:
    path = profile_path(profile_dir, database_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_profile(profile_dir: Path, database_id: str, payload: dict[str, Any]) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_path(profile_dir, database_id)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temp_path.chmod(0o600)
    temp_path.replace(path)
    return path


def profile_is_fresh(payload: dict[str, Any], *, ttl_seconds: int) -> bool:
    generated_at = str(payload.get("generated_at") or "")
    if not generated_at:
        return False
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    return age_seconds <= max(1, ttl_seconds)
