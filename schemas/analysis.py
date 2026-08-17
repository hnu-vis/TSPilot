"""Code-interpreter computation and derived-evidence models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from schemas.key_insight import KeyInsight, InsightCoverage


class ComputedInsight(BaseModel):
    """A semantic-keyed value calculated by Python, before LLM binding."""

    insight_key: str
    value: Any = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    calculation_trace: str | dict[str, Any] | list[Any]
    derived_evidence_ids: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_value(self):
        if self.value is None and not self.items and not str(self.unavailable_reason or "").strip():
            raise ValueError("computed insight requires value, items, or unavailable_reason")
        trace = self.calculation_trace
        if (isinstance(trace, str) and not trace.strip()) or (not isinstance(trace, str) and not trace):
            raise ValueError("computed insight requires a non-empty calculation_trace")
        return self


class DerivedEvidence(BaseModel):
    """A reusable row/scalar artifact produced by a computation."""

    evidence_id: str
    name: str
    shape: Literal["timeseries", "records", "scalar", "intervals"]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    scalar: dict[str, Any] | None = None
    lineage: list[str] = Field(min_length=1)
    transform_summary: str

    @model_validator(mode="after")
    def validate_content(self):
        if self.shape == "scalar":
            if not isinstance(self.scalar, dict) or not self.scalar:
                raise ValueError("scalar derived evidence requires a non-empty scalar object")
        elif not self.rows:
            raise ValueError(f"{self.shape} derived evidence requires non-empty rows")
        return self


class AnalysisResult(BaseModel):
    """Stable analysis envelope; semantic Insights are bound after computation."""

    analysis_id: str
    analysis_goal: str
    code_type: Literal["code_interpreter_v2"] = "code_interpreter_v2"
    code_hash: str
    input_evidence_id: str
    input_row_count: int
    status: Literal["succeeded", "failed"]
    summary: str
    computed_insights: list[ComputedInsight] = Field(default_factory=list)
    derived_evidence: list[DerivedEvidence] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
    produced_insights: list[KeyInsight] = Field(default_factory=list)
    rejected_insights: list[KeyInsight] = Field(default_factory=list)
    insight_coverage: InsightCoverage | None = None

    @model_validator(mode="after")
    def validate_succeeded_result_contract(self):
        if self.status == "succeeded":
            if not self.summary.strip():
                raise ValueError("succeeded analysis requires a non-empty summary")
            if not self.computed_insights:
                raise ValueError("succeeded analysis requires at least one computed insight")
        return self
