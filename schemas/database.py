"""Database evidence models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.data_fact import DataFact, FactCoverage


class DatabaseEvidence(BaseModel):
    """Normalized database evidence."""

    evidence_id: str
    result_type: Literal["schema", "metric_list", "statistics", "table", "timeseries"]
    database: str
    query_language: str | None = None
    query: str | None = None
    summary: str
    data: dict = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    produced_facts: list[DataFact] = Field(default_factory=list)
    rejected_facts: list[DataFact] = Field(default_factory=list)
    fact_coverage: FactCoverage | None = None
