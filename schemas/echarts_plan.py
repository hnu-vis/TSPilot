"""Closed structured-output envelope for native ECharts planning."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.key_insight import KeyInsightRequest


class VisualizationEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_action: Literal["sql_query", "anomaly", "forecast", "code_interpreter"]
    purpose: str = Field(min_length=1)
    message: str | None = None
    required_shape: str = Field(min_length=1)
    required_fields: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    input_evidence: str | None = None
    input_source_refs: list[str] = Field(default_factory=list)
    insight_requests: list[KeyInsightRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_owner(self):
        if self.input_evidence and not self.input_source_refs:
            self.input_source_refs.append(self.input_evidence)
        if self.required_action == "code_interpreter" and not self.insight_requests:
            raise ValueError("code_interpreter dependency requires insight_requests")
        if self.required_action in {"anomaly", "forecast"}:
            self.insight_requests = []
        return self


class EChartsChartPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chart_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    purpose: str = Field(min_length=1)
    priority: Literal["primary", "supporting"]
    title: str = Field(min_length=1)
    summary: str | None = None
    accessibility_description: str = Field(min_length=1)
    accessibility_table_columns: list[str] = Field(default_factory=list)
    option_json: str = Field(min_length=2, max_length=131072)


class EChartsPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visual_question: str | None = None
    interpretation: str | None = None
    target_insight_ids: list[str] = Field(default_factory=list)
    charts: list[EChartsChartPlan] = Field(default_factory=list, max_length=4)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def require_charts_or_dependency(self):
        if bool(self.charts) == bool(self.required_data_request):
            raise ValueError("ECharts planning must return charts or one data request")
        if self.charts:
            if not self.visual_question or not self.interpretation:
                raise ValueError("charts require visual_question and interpretation")
            if len([chart for chart in self.charts if chart.priority == "primary"]) != 1:
                raise ValueError("ECharts planning requires exactly one primary chart")
            ids = [chart.chart_id for chart in self.charts]
            if len(ids) != len(set(ids)):
                raise ValueError("chart ids must be unique")
        return self


class StructuredInsightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    insight_type: str = Field(min_length=1)
    insight_key: str | None = None

    def to_runtime(self) -> KeyInsightRequest:
        return KeyInsightRequest(name=self.name, insight_type=self.insight_type, insight_key=self.insight_key)


class StructuredVisualizationEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_action: Literal["sql_query", "anomaly", "forecast", "code_interpreter"]
    purpose: str = Field(min_length=1)
    message: str | None = None
    required_shape: str = Field(min_length=1)
    required_fields: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    input_evidence: str | None = None
    input_source_refs: list[str] = Field(default_factory=list)
    insight_requests: list[StructuredInsightRequest] = Field(default_factory=list)

    def to_runtime(self) -> VisualizationEvidenceRequest:
        return VisualizationEvidenceRequest(
            **self.model_dump(exclude={"insight_requests"}),
            insight_requests=[item.to_runtime() for item in self.insight_requests],
        )


class StructuredEChartsPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visual_question: str | None = None
    interpretation: str | None = None
    target_insight_ids: list[str] = Field(default_factory=list)
    charts: list[EChartsChartPlan] = Field(default_factory=list, max_length=4)
    required_data_request: StructuredVisualizationEvidenceRequest | None = None

    def to_runtime(self) -> EChartsPlan:
        return EChartsPlan(
            visual_question=self.visual_question,
            interpretation=self.interpretation,
            target_insight_ids=self.target_insight_ids,
            charts=self.charts,
            required_data_request=(self.required_data_request.to_runtime() if self.required_data_request else None),
        )
