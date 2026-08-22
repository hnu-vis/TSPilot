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


class EChartsGroundedTimeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_ref: str = Field(min_length=1)
    value_id: str = Field(min_length=1, pattern=r"^time_[1-9][0-9]*$")


class EChartsGroundedNumberRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_ref: str = Field(min_length=1)
    value_id: str = Field(min_length=1, pattern=r"^number_[1-9][0-9]*$")


class EChartsLineSeriesPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)


class EChartsPointAnnotationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    time: EChartsGroundedTimeRef
    value: EChartsGroundedNumberRef


class EChartsIntervalAnnotationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    start: EChartsGroundedTimeRef
    end: EChartsGroundedTimeRef


class EChartsReferenceLinePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: EChartsGroundedNumberRef


class EChartsChartPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chart_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    purpose: str = Field(min_length=1)
    priority: Literal["primary", "supporting"]
    title: str = Field(min_length=1)
    summary: str | None = None
    accessibility_description: str = Field(min_length=1)
    accessibility_table_columns: list[str] = Field(default_factory=list)
    series: list[EChartsLineSeriesPlan] = Field(default_factory=list, max_length=2)
    point_annotations: list[EChartsPointAnnotationPlan] = Field(default_factory=list, max_length=12)
    interval_annotations: list[EChartsIntervalAnnotationPlan] = Field(default_factory=list, max_length=6)
    reference_lines: list[EChartsReferenceLinePlan] = Field(default_factory=list, max_length=6)
    y_axis_name: str | None = None
    option_json: str | None = Field(default=None, min_length=2, max_length=131072, exclude=True)

    @model_validator(mode="after")
    def require_typed_or_legacy_option(self):
        if self.option_json is not None and self.series:
            raise ValueError("chart cannot contain both typed series and legacy option_json")
        if self.option_json is None and not self.series:
            raise ValueError("chart requires typed series")
        return self


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


class StructuredEChartsChartPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chart_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    purpose: str = Field(min_length=1)
    priority: Literal["primary", "supporting"]
    title: str = Field(min_length=1)
    summary: str | None = None
    accessibility_description: str = Field(min_length=1)
    accessibility_table_columns: list[str] = Field(default_factory=list)
    series: list[EChartsLineSeriesPlan] = Field(min_length=1, max_length=2)
    point_annotations: list[EChartsPointAnnotationPlan] = Field(default_factory=list, max_length=12)
    interval_annotations: list[EChartsIntervalAnnotationPlan] = Field(default_factory=list, max_length=6)
    reference_lines: list[EChartsReferenceLinePlan] = Field(default_factory=list, max_length=6)
    y_axis_name: str | None = None

    def to_runtime(self) -> EChartsChartPlan:
        return EChartsChartPlan(**self.model_dump())


class StructuredEChartsPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visual_question: str | None = None
    interpretation: str | None = None
    target_insight_ids: list[str] = Field(default_factory=list)
    charts: list[StructuredEChartsChartPlan] = Field(default_factory=list, max_length=1)
    required_data_request: StructuredVisualizationEvidenceRequest | None = None

    def to_runtime(self) -> EChartsPlan:
        return EChartsPlan(
            visual_question=self.visual_question,
            interpretation=self.interpretation,
            target_insight_ids=self.target_insight_ids,
            charts=[chart.to_runtime() for chart in self.charts],
            required_data_request=(self.required_data_request.to_runtime() if self.required_data_request else None),
        )


class StructuredEChartsChartPlanWithoutTimeAnnotations(StructuredEChartsChartPlan):
    """Provider schema used when no selected Insight exposes a grounded time value."""

    point_annotations: list[EChartsPointAnnotationPlan] = Field(default_factory=list, max_length=0)
    interval_annotations: list[EChartsIntervalAnnotationPlan] = Field(default_factory=list, max_length=0)


class StructuredEChartsPlanWithoutTimeAnnotations(StructuredEChartsPlan):
    charts: list[StructuredEChartsChartPlanWithoutTimeAnnotations] = Field(default_factory=list, max_length=1)
