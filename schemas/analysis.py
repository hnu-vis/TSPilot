"""Generated-code analysis result models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
