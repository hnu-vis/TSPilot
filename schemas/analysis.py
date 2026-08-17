"""Generated-code analysis result models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from schemas.key_insight import KeyInsight, InsightCoverage


class DataViewField(BaseModel):
    name: str
    data_type: Literal["time", "number", "category", "string", "boolean", "object"]


class AnalysisDataView(BaseModel):
    """A named, lineage-preserving dataset produced by an analysis."""

    view_id: str
    name: str
    shape: Literal["timeseries", "records", "scalar", "intervals"]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    scalar: dict[str, Any] | None = None
    schema_fields: list[DataViewField] = Field(default_factory=list)
    lineage: list[str] = Field(min_length=1)
    transform_summary: str | None = None

    @model_validator(mode="after")
    def validate_content(self):
        if self.shape == "scalar":
            if not isinstance(self.scalar, dict) or not self.scalar:
                raise ValueError("scalar data view requires a non-empty scalar object")
        elif not self.rows:
            raise ValueError(f"{self.shape} data view requires non-empty rows")
        return self


class AnalysisResult(BaseModel):
    analysis_id: str
    analysis_goal: str
    code_type: Literal["python_rows_v1", "python_sandbox_v1", "code_interpreter_v1", "analysis_request_v1"] = "python_rows_v1"
    code_hash: str
    input_evidence_id: str
    input_row_count: int
    status: Literal["succeeded", "failed"]
    summary: str
    result: dict = Field(default_factory=dict)
    data_views: list[AnalysisDataView] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
    produced_insights: list[KeyInsight] = Field(default_factory=list)
    rejected_insights: list[KeyInsight] = Field(default_factory=list)
    insight_coverage: InsightCoverage | None = None

    @model_validator(mode="after")
    def validate_succeeded_result_contract(self):
        if self.status != "succeeded":
            return self
        result_summary = self.result.get("summary") if isinstance(self.result, dict) else None
        if not isinstance(result_summary, str) or not result_summary.strip():
            raise ValueError("succeeded analysis result must include result.summary.")
        if not isinstance(self.result.get("metrics"), dict):
            raise ValueError("succeeded analysis result must include result.metrics object.")
        if not isinstance(self.result.get("details"), dict):
            raise ValueError("succeeded analysis result must include result.details object.")
        self.result = {
            **self.result,
            "summary": result_summary.strip(),
            "metrics": dict(self.result["metrics"]),
            "details": dict(self.result["details"]),
        }
        raw_views = self.result.get("data_views")
        if raw_views is not None:
            if not isinstance(raw_views, list):
                raise ValueError("result.data_views must be a list")
            self.data_views = [AnalysisDataView.model_validate(item) for item in raw_views]
        self.summary = result_summary.strip()
        return self
