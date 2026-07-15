"""Trace event models."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TraceEventModel(BaseModel):
    """One runtime trace event."""

    event_type: str
    payload: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

