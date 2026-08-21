"""Closed LLM planning contracts for LineChart-first visualization."""
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


class VisualContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    source_ref: str = Field(min_length=1)
    insight_ids: list[str] = Field(default_factory=list)
    purpose: str = Field(min_length=1)
    importance: Literal["primary", "highlight", "support"]


class VisualContentGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    purpose: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str | None = None
    priority: Literal["primary", "supporting"]
    host_source_ref: str = Field(min_length=1)
    content: list[VisualContentItem] = Field(min_length=1)
    required_interactions: list[Literal["tooltip", "legend_toggle", "zoom", "evidence_link"]] = Field(default_factory=list)


class VisualContentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visual_question: str | None = None
    interpretation: str | None = None
    target_insight_ids: list[str] = Field(default_factory=list)
    goals: list[VisualContentGoal] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def require_goals_or_dependency(self):
        if bool(self.goals) == bool(self.required_data_request):
            raise ValueError("content planning must return goals or one data request")
        if self.goals:
            if not self.visual_question or not self.interpretation:
                raise ValueError("visual content requires a question and interpretation")
            primary = [goal for goal in self.goals if goal.priority == "primary"]
            if len(primary) != 1:
                raise ValueError("visual content requires exactly one primary chart goal")
            ids = [item.content_id for goal in self.goals for item in goal.content]
            if len(ids) != len(set(ids)):
                raise ValueError("visual content ids must be unique")
        return self


class LineChartYAxisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    axis_id: str = Field(min_length=1)
    label: str | None = None
    measure: str = Field(min_length=1)
    unit: str | None = None
    scale: Literal["linear", "log"] = "linear"


class _ComponentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    role: str = Field(min_length=1)
    importance: Literal["primary", "highlight", "support"]
    label: str | None = None


class LinePlan(_ComponentPlan):
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)
    line_style: Literal["solid", "dashed", "dotted"] = "solid"
    symbol: Literal["none", "circle", "diamond", "triangle"] = "none"


class PointPlan(_ComponentPlan):
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)
    symbol: Literal["circle", "diamond", "triangle", "pin"] = "circle"
    size: Literal["small", "medium", "large"] = "medium"


class BandPlan(_ComponentPlan):
    x_field: str = Field(min_length=1)
    lower_field: str = Field(min_length=1)
    upper_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)


class IntervalPlan(_ComponentPlan):
    start_field: str = Field(min_length=1)
    end_field: str = Field(min_length=1)


class ReferenceLinePlan(_ComponentPlan):
    value_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)


class ChartAnnotationTargetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["chart"]


class XAnnotationTargetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["x"]
    x_field: str = Field(min_length=1)


class XYAnnotationTargetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["xy"]
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)


class IntervalAnnotationTargetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["interval"]
    start_field: str = Field(min_length=1)
    end_field: str = Field(min_length=1)


AnnotationTargetPlan = (
    ChartAnnotationTargetPlan
    | XAnnotationTargetPlan
    | XYAnnotationTargetPlan
    | IntervalAnnotationTargetPlan
)


class AnnotationPlan(_ComponentPlan):
    content_field: str = Field(min_length=1)
    target: AnnotationTargetPlan


class LineChartGoalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal_id: str = Field(min_length=1)
    x_axis_type: Literal["time", "category"]
    x_axis_label: str | None = None
    y_axes: list[LineChartYAxisPlan] = Field(min_length=1, max_length=2)
    host_line: LinePlan
    lines: list[LinePlan] = Field(default_factory=list)
    points: list[PointPlan] = Field(default_factory=list)
    bands: list[BandPlan] = Field(default_factory=list)
    intervals: list[IntervalPlan] = Field(default_factory=list)
    reference_lines: list[ReferenceLinePlan] = Field(default_factory=list)
    annotations: list[AnnotationPlan] = Field(default_factory=list)
    legend_visible: bool = True
    legend_position: Literal["top", "bottom"] = "top"
    tooltip_mode: Literal["axis", "item", "none"] = "axis"
    zoom_enabled: bool = False
    zoom_start: str | int | float | None = None
    zoom_end: str | int | float | None = None

    @model_validator(mode="after")
    def validate_components(self):
        if (self.zoom_start is None) != (self.zoom_end is None):
            raise ValueError("chart zoom requires both start and end")
        return self


class LineChartPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    charts: list[LineChartGoalPlan] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def require_charts_or_dependency(self):
        if bool(self.charts) == bool(self.required_data_request):
            raise ValueError("line-chart planning must return charts or one data request")
        return self


class StructuredInsightRequest(BaseModel):
    """Provider-safe projection of a runtime KeyInsightRequest.

    Arbitrary analytical details stay in the evidence request's purpose/message. The
    provider response schema must not expose KeyInsightRequest's open JSON objects.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    insight_type: str = Field(min_length=1)
    insight_key: str | None = None

    def to_runtime(self) -> KeyInsightRequest:
        return KeyInsightRequest(
            name=self.name,
            insight_type=self.insight_type,
            insight_key=self.insight_key,
        )


class StructuredVisualizationEvidenceRequest(BaseModel):
    """Closed response-format contract converted to the richer runtime request."""

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

    @model_validator(mode="after")
    def validate_owner(self):
        if self.required_action == "code_interpreter" and not self.insight_requests:
            raise ValueError("code_interpreter dependency requires insight_requests")
        if self.required_action in {"anomaly", "forecast"} and self.insight_requests:
            raise ValueError(f"{self.required_action} dependency must not contain insight_requests")
        return self

    def to_runtime(self) -> VisualizationEvidenceRequest:
        return VisualizationEvidenceRequest(
            required_action=self.required_action,
            purpose=self.purpose,
            message=self.message,
            required_shape=self.required_shape,
            required_fields=self.required_fields,
            required_properties=self.required_properties,
            input_evidence=self.input_evidence,
            input_source_refs=self.input_source_refs,
            insight_requests=[item.to_runtime() for item in self.insight_requests],
        )


class StructuredVisualContentPlan(VisualContentPlan):
    """Strict provider response for stage one."""

    required_data_request: StructuredVisualizationEvidenceRequest | None = None

    def to_runtime(self) -> VisualContentPlan:
        return VisualContentPlan(
            visual_question=self.visual_question,
            interpretation=self.interpretation,
            target_insight_ids=self.target_insight_ids,
            goals=self.goals,
            required_data_request=(
                self.required_data_request.to_runtime() if self.required_data_request else None
            ),
        )


class StructuredLineChartPlan(LineChartPlan):
    """Strict provider response for stage two."""

    required_data_request: StructuredVisualizationEvidenceRequest | None = None

    def to_runtime(self) -> LineChartPlan:
        return LineChartPlan(
            charts=self.charts,
            required_data_request=(
                self.required_data_request.to_runtime() if self.required_data_request else None
            ),
        )
