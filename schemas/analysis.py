"""Generated-code analysis result models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schemas.data_fact import DataFact, FactCoverage


class AnalysisResult(BaseModel):
    analysis_id: str
    analysis_goal: str
    code_type: Literal["python_rows_v1", "python_sandbox_v1", "code_interpreter_v1"] = "python_rows_v1"
    code_hash: str
    input_evidence_id: str
    input_row_count: int
    status: Literal["succeeded", "failed"]
    summary: str
    result: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    produced_facts: list[DataFact] = Field(default_factory=list)
    rejected_facts: list[DataFact] = Field(default_factory=list)
    fact_coverage: FactCoverage | None = None

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
        self.summary = result_summary.strip()
        return self
