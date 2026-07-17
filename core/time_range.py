"""Time range normalization helpers."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any


def parse_time_to_utc(value: Any) -> datetime:
    """Parse a user/backend timestamp and return an aware UTC datetime."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Timestamp value cannot be empty.")
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc_rfc3339(value: datetime) -> str:
    """Format an aware datetime as an RFC3339 UTC timestamp with Z."""

    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def normalize_time_value(value: Any) -> str:
    return format_utc_rfc3339(parse_time_to_utc(value))


def normalize_time_range(time_range: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize start/end/stop fields while preserving unrelated metadata."""

    if not time_range:
        return time_range
    normalized = dict(time_range)
    if normalized.get("start"):
        normalized["start"] = normalize_time_value(normalized["start"])
    raw_end = normalized.get("end", normalized.get("stop"))
    if raw_end:
        normalized["end"] = normalize_time_value(raw_end)
        normalized.pop("stop", None)
    if normalized.get("start") and normalized.get("end"):
        start = parse_time_to_utc(normalized["start"])
        end = parse_time_to_utc(normalized["end"])
        if end < start:
            raise ValueError("time_range end must be greater than or equal to start.")
    normalized.setdefault("timezone", "UTC")
    return normalized
