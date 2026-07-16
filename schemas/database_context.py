"""Database context model."""
from __future__ import annotations

from pydantic import BaseModel


class DatabaseContext(BaseModel):
    """Normalized database selection object."""

    database_id: str
    database_type: str
    display_name: str | None = None
    connection_hint: str | None = None
    schema_hint: dict | None = None
    selected_at: str | None = None
