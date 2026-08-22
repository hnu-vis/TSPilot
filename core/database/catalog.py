"""Shared product database catalog consumed by backend and frontend."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "database_catalog.json"


@lru_cache(maxsize=1)
def database_catalog() -> tuple[dict[str, Any], ...]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("database catalog must be a non-empty list")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"database catalog entry {index} must be an object")
        db_type = str(raw.get("type") or "").strip().lower()
        if not db_type or db_type in seen:
            raise ValueError(f"database catalog contains invalid or duplicate type '{db_type}'")
        if not str(raw.get("label") or "").strip():
            raise ValueError(f"database catalog type '{db_type}' requires a label")
        seen.add(db_type)
        entries.append({**raw, "type": db_type})
    return tuple(entries)


def supported_database_types() -> tuple[str, ...]:
    return tuple(entry["type"] for entry in database_catalog())


def database_catalog_entry(db_type: str) -> dict[str, Any] | None:
    normalized = str(db_type or "").strip().lower()
    return next((entry for entry in database_catalog() if entry["type"] == normalized), None)


def public_extra_config(db_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return only catalog-declared, non-secret connector options."""
    entry = database_catalog_entry(db_type)
    if entry is None:
        return {}
    result: dict[str, Any] = {}
    for field in entry.get("extraFields") or []:
        if not isinstance(field, dict) or field.get("secret"):
            continue
        key = str(field.get("key") or "")
        if key and key in config:
            result[key] = config[key]
    return result


def missing_required_config_fields(config: dict[str, Any]) -> list[str]:
    """Return missing common or catalog-declared required connection fields."""
    missing = [key for key in ("host", "port") if config.get(key) in (None, "")]
    entry = database_catalog_entry(str(config.get("type") or config.get("db_type") or ""))
    if entry is None:
        return missing
    defaults = entry.get("defaults") if isinstance(entry.get("defaults"), dict) else {}
    default_extra = defaults.get("extra") if isinstance(defaults.get("extra"), dict) else {}
    for field in entry.get("extraFields") or []:
        if not isinstance(field, dict) or not field.get("required"):
            continue
        key = str(field.get("key") or "")
        if key and config.get(key, default_extra.get(key)) in (None, ""):
            missing.append(key)
    return missing
