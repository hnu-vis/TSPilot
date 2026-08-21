"""LLM-planned, full-fidelity visualization artifact tool."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator

from core.visualization import (
    IncompatibleVisualDomainError,
    PresentationCatalog,
    VisualizationArtifactStore,
    VisualizationMaterializer,
    VisualizationSemanticValidator,
)
from runtime.llm_trace import llm_trace_span
from runtime.prompt_locale import prompt_locale_instruction
from runtime.token_usage import record_llm_token_usage
from runtime.timeout_policy import load_timeout_policy
from schemas.output import (
    AnnotationVisualLayerIR,
    BandVisualLayerIR,
    ComparisonVisualLayerIR,
    EventPointsVisualLayerIR,
    IntervalLiteralVisualLayerIR,
    IntervalSourceVisualLayerIR,
    ReferenceLineVisualLayerIR,
    SeriesVisualLayerIR,
    VisualEncodingIR,
    VisualFilterTransform,
    VisualGoal,
    VisualGoalIR,
    VisualLayerIR,
    VisualLayerPlan,
)
from schemas.key_insight import KeyInsightRequest
from schemas.state import RequestStateModel
from schemas.visual_verification import (
    VisualProofObligation,
    VisualizationVerification,
)
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool, StructuredToolError


class VisualizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    source_refs: list[str] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)


_VISUALIZATION_CONTROL_CONSTRAINT_KEYS = {
    "mode",
    "repair_contract",
    "validation_failure",
}


class VisualizationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_goals: list[VisualGoal] = Field(default_factory=list)
    required_data_request: "VisualizationEvidenceRequest | None" = None

    @model_validator(mode="after")
    def require_goal_or_data_request(self):
        if bool(self.visual_goals) == bool(self.required_data_request):
            raise ValueError("chart planning must produce either visual_goals or required_data_request, never both")
        return self


class VisualVerificationDecision(BaseModel):
    """Question-first decision about what the chart can genuinely verify."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["visualize", "needs_sources", "not_visualizable"]
    target_insight_ids: list[str] = Field(default_factory=list)
    verification_question: str | None = Field(default=None, max_length=1200)
    interpretation: str | None = Field(default=None, max_length=1200)
    visual_relation: str | None = Field(default=None, max_length=1200)
    proof_obligations: list[VisualProofObligation] = Field(default_factory=list, max_length=8)
    required_context: list[str] = Field(default_factory=list, max_length=8)
    non_visual_insight_ids: list[str] = Field(default_factory=list)
    required_data_request: "VisualizationEvidenceRequest | None" = None

    @model_validator(mode="after")
    def validate_decision(self):
        if self.decision == "visualize":
            if not str(self.verification_question or "").strip():
                raise ValueError("a visual verification decision requires verification_question")
            if not str(self.interpretation or "").strip():
                raise ValueError("a visual verification decision requires interpretation")
            if self.required_data_request is not None:
                raise ValueError("a visual verification decision cannot also request data")
        elif self.decision == "needs_sources":
            if self.required_data_request is None:
                raise ValueError("needs_sources requires required_data_request")
        elif self.required_data_request is not None:
            raise ValueError("not_visualizable cannot request data")
        return self

    def public_contract(self) -> VisualizationVerification:
        if self.decision != "visualize":
            raise ValueError("only a visualizable decision has a public verification contract")
        return VisualizationVerification(
            target_insight_ids=self.target_insight_ids,
            verification_question=str(self.verification_question),
            interpretation=str(self.interpretation),
            proof_obligations=self.proof_obligations,
        )


class VisualizationCandidateAudit(BaseModel):
    """Independent semantic review of a fully materialized candidate."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "revise", "needs_sources", "unavailable"]
    issues: list[str] = Field(default_factory=list)
    required_data_request: "VisualizationEvidenceRequest | None" = None

    @model_validator(mode="after")
    def validate_resolution(self):
        if self.decision == "approve" and (self.issues or self.required_data_request is not None):
            raise ValueError("an approved audit cannot contain issues or a dependency")
        if self.decision == "revise" and not self.issues:
            raise ValueError("a revision audit requires actionable issues")
        if self.decision == "needs_sources" and self.required_data_request is None:
            raise ValueError("a source audit requires required_data_request")
        if self.decision != "needs_sources" and self.required_data_request is not None:
            raise ValueError("only needs_sources may include required_data_request")
        if self.decision == "unavailable" and not self.issues:
            raise ValueError("an unavailable audit requires a reason")
        return self


class SemanticFieldPlan(BaseModel):
    """One LLM-authored semantic column backed by an existing source value."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$",
    )
    semantic_role: str = Field(min_length=1, max_length=256)
    source_path: str = Field(min_length=1, max_length=512)


class SemanticEventPlan(BaseModel):
    """Expand one source record into one located event row."""

    model_config = ConfigDict(extra="forbid")

    event_role: str = Field(min_length=1, max_length=256)
    timestamp_path: str = Field(min_length=1, max_length=512)
    value_path: str = Field(min_length=1, max_length=512)


class _SemanticViewPlanBase(BaseModel):
    """Common, source-grounded identity for a semantic view."""

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=300)
    purpose: str = Field(min_length=1, max_length=800)
    grain: str = Field(min_length=1, max_length=300)
    source_ref: str = Field(min_length=1, max_length=512)
    record_path: str | None


class RecordsSemanticViewPlan(_SemanticViewPlanBase):
    """A regular record table, with one semantic column per source path."""

    mode: Literal["records"]
    fields: list[SemanticFieldPlan] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_semantic_columns(self):
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("semantic view field names must be unique")
        return self


class WideEventsSemanticViewPlan(_SemanticViewPlanBase):
    """One wide source record expanded into its separately located events."""

    mode: Literal["wide_events"]
    events: list[SemanticEventPlan] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_semantic_events(self):
        event_roles = [item.event_role for item in self.events]
        if len(event_roles) != len(set(event_roles)):
            raise ValueError("semantic event roles must be unique")
        event_paths = [(item.timestamp_path, item.value_path) for item in self.events]
        if len(event_paths) != len(set(event_paths)):
            raise ValueError("wide semantic events must use distinct timestamp/value path pairs")
        return self


# A literal-tagged union makes the two source shapes mutually exclusive in the
# structured-output contract itself.  Avoid Pydantic's explicit discriminator:
# it emits JSON Schema ``oneOf``, which is rejected by supported OpenAI-style
# strict-output providers; the plain union emits provider-compatible ``anyOf``.
SemanticViewPlan = RecordsSemanticViewPlan | WideEventsSemanticViewPlan


class SemanticProjectionPlan(BaseModel):
    """First-stage LLM decision: semantic views or an upstream evidence request."""

    model_config = ConfigDict(extra="forbid")

    semantic_views: list[SemanticViewPlan] = Field(default_factory=list)
    required_data_request: "VisualizationEvidenceRequest | None" = None

    @model_validator(mode="after")
    def require_views_or_data_request(self):
        if bool(self.semantic_views) == bool(self.required_data_request):
            raise ValueError("semantic projection must produce either semantic_views or required_data_request, never both")
        return self


class VisualizationEvidenceRequest(BaseModel):
    """A planner-selected request for the tool that owns a missing visual source."""

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
    def validate_action_contract(self):
        if self.input_evidence and not self.input_source_refs:
            self.input_source_refs.append(self.input_evidence)
        if self.required_action == "code_interpreter" and not self.insight_requests:
            raise ValueError("code_interpreter visualization dependency requires insight_requests")
        if self.required_action in {"anomaly", "forecast"}:
            self.insight_requests = []
        return self


class _StructuredInsightRequest(BaseModel):
    """Closed insight dependency emitted only at the visualization LLM boundary."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    insight_type: str = Field(min_length=1)
    insight_key: str | None

    def to_runtime(self) -> KeyInsightRequest:
        return KeyInsightRequest(
            name=self.name,
            insight_type=self.insight_type,
            insight_key=self.insight_key,
        )


class _StructuredEvidenceRequest(BaseModel):
    """Strict structured-output form of VisualizationEvidenceRequest."""

    model_config = ConfigDict(extra="forbid")

    required_action: Literal["sql_query", "anomaly", "forecast", "code_interpreter"]
    purpose: str = Field(min_length=1)
    message: str | None
    required_shape: str = Field(min_length=1)
    required_fields: list[str]
    required_properties: list[str]
    input_evidence: str | None
    input_source_refs: list[str]
    insight_requests: list[_StructuredInsightRequest]

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


class VisualEvidenceUse(BaseModel):
    """One upstream source selected to fulfill the fixed presentation goal."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1, max_length=512)
    purpose: str = Field(min_length=1, max_length=800)


class VisualEvidenceConsumption(BaseModel):
    """LLM-authored contract between a visual goal and upstream lineage."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ready", "needs_sources", "unavailable"]
    rationale: str = Field(min_length=1, max_length=1600)
    source_uses: list[VisualEvidenceUse] = Field(default_factory=list, max_length=12)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def validate_decision(self):
        refs = [item.source_ref for item in self.source_uses]
        if len(refs) != len(set(refs)):
            raise ValueError("visual evidence consumption source refs must be unique")
        if self.decision == "ready":
            if not self.source_uses:
                raise ValueError("ready visual evidence consumption requires source_uses")
            if self.required_data_request is not None:
                raise ValueError("ready visual evidence consumption cannot request data")
        elif self.decision == "needs_sources":
            if self.required_data_request is None:
                raise ValueError("needs_sources visual evidence consumption requires a data request")
        elif self.required_data_request is not None:
            raise ValueError("unavailable visual evidence consumption cannot request data")
        return self


class VisualCompositionLayer(BaseModel):
    """One semantic layer decision made before field encoding."""

    model_config = ConfigDict(extra="forbid")

    layer_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    family: Literal["primary", "highlight", "support"]
    layer_type: Literal[
        "series",
        "event_points",
        "band",
        "interval_overlay",
        "comparison",
        "reference_line",
        "annotation",
    ]
    role: str = Field(min_length=1, max_length=300)
    purpose: str = Field(min_length=1, max_length=800)
    source_ref: str = Field(min_length=1, max_length=512)
    interval_source_ref: str | None = Field(max_length=512)
    label: str | None = Field(max_length=300)

    @model_validator(mode="after")
    def validate_interval_source(self):
        if self.layer_type != "interval_overlay" and self.interval_source_ref is not None:
            raise ValueError("only an interval_overlay composition layer may select interval_source_ref")
        return self


class VisualCompositionGoal(BaseModel):
    """Chart-level composition without renderer field bindings."""

    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(min_length=1, max_length=800)
    title: str = Field(min_length=1, max_length=300)
    priority: Literal["primary", "supporting"]
    summary: str | None = Field(max_length=1200)
    show_legend: bool
    tooltip: Literal["axis", "item", "none"]
    enable_zoom: bool
    viewport_start: str | int | float | None
    viewport_end: str | int | float | None
    y_scale: Literal["linear", "log"]
    layers: list[VisualCompositionLayer] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_goal(self):
        ids = [layer.layer_id for layer in self.layers]
        if len(ids) != len(set(ids)):
            raise ValueError("visual composition layer_id values must be unique")
        host_layers = [
            layer
            for layer in self.layers
            if layer.layer_type in {"series", "event_points", "band", "comparison"}
        ]
        if not any(layer.family == "primary" for layer in host_layers):
            raise ValueError(
                "a visual composition requires at least one primary axis-bearing host layer; "
                "reference lines, interval overlays, and annotations must attach to a host"
            )
        axis_domains = {
            "categorical" if layer.layer_type == "comparison" else "temporal"
            for layer in host_layers
        }
        if len(axis_domains) > 1:
            raise ValueError(
                "one visual goal cannot mix temporal and categorical axis domains; "
                "split them into separate goals or attach scalar evidence as a reference_line/annotation"
            )
        if (self.viewport_start is None) != (self.viewport_end is None):
            raise ValueError("visual composition viewport requires both start and end")
        return self


class VisualCompositionPlan(BaseModel):
    """LLM-authored display composition, prior to exact field encoding."""

    model_config = ConfigDict(extra="forbid")

    visual_goals: list[VisualCompositionGoal] = Field(default_factory=list, max_length=4)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def require_goal_or_data_request(self):
        if bool(self.visual_goals) == bool(self.required_data_request):
            raise ValueError("visual composition must produce either visual_goals or required_data_request")
        layer_ids = [layer.layer_id for goal in self.visual_goals for layer in goal.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("visual composition layer_id values must be globally unique")
        return self


class _StructuredEvidenceUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1, max_length=512)
    purpose: str = Field(min_length=1, max_length=800)


class _StructuredEvidenceConsumption(BaseModel):
    """Provider-strict evidence selection after the presentation goal is fixed."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ready", "needs_sources", "unavailable"]
    rationale: str = Field(min_length=1, max_length=1600)
    source_uses: list[_StructuredEvidenceUse] = Field(max_length=12)
    required_data_request: _StructuredEvidenceRequest | None

    def to_runtime(self) -> VisualEvidenceConsumption:
        return VisualEvidenceConsumption(
            decision=self.decision,
            rationale=self.rationale,
            source_uses=[
                VisualEvidenceUse(source_ref=item.source_ref, purpose=item.purpose)
                for item in self.source_uses
            ],
            required_data_request=(
                self.required_data_request.to_runtime()
                if self.required_data_request is not None
                else None
            ),
        )


class _StructuredVisualCompositionPlan(BaseModel):
    """Provider-strict display composition before exact field encoding."""

    model_config = ConfigDict(extra="forbid")

    visual_goals: list[VisualCompositionGoal]
    required_data_request: _StructuredEvidenceRequest | None

    @model_validator(mode="after")
    def require_goal_or_data_request(self):
        if bool(self.visual_goals) == bool(self.required_data_request):
            raise ValueError("visual composition must produce either visual_goals or required_data_request")
        return self

    def to_runtime(self) -> VisualCompositionPlan:
        return VisualCompositionPlan(
            visual_goals=self.visual_goals,
            required_data_request=(
                self.required_data_request.to_runtime()
                if self.required_data_request is not None
                else None
            ),
        )


class _EncodedLayerBase(BaseModel):
    """Presentation and field bindings for one already-fixed composition layer."""

    model_config = ConfigDict(extra="forbid")

    layer_id: str = Field(min_length=1, max_length=64)
    emphasis: Literal["normal", "subtle", "strong"]
    line_style: Literal["solid", "dashed", "dotted"]
    symbol: Literal["none", "circle", "diamond", "triangle", "pin"]
    axis: Literal["primary", "secondary"]


class _EncodedXYLayer(_EncodedLayerBase):
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    series_field: str | None
    label_field: str | None


class _EncodedBandLayer(_EncodedLayerBase):
    x_field: str = Field(min_length=1)
    lower_field: str = Field(min_length=1)
    upper_field: str = Field(min_length=1)
    series_field: str | None
    label_field: str | None


class _EncodedReferenceLineLayer(_EncodedLayerBase):
    value_field: str = Field(min_length=1)
    label_field: str | None


class _EncodedAnnotationLayer(_EncodedLayerBase):
    content_field: str = Field(min_length=1)
    value_field: str | None
    x_field: str | None


class _EncodedIntervalSourceLayer(_EncodedXYLayer):
    interval_start_field: str = Field(min_length=1)
    interval_end_field: str = Field(min_length=1)


class _EncodedIntervalLiteralLayer(_EncodedXYLayer):
    interval_start_value: str | int | float
    interval_end_value: str | int | float


class _StructuredEncodedVisualPlan(BaseModel):
    """Base shape specialized at runtime to the fixed composition vocabulary."""

    model_config = ConfigDict(extra="forbid")

    layers: list[_EncodedLayerBase]
    required_data_request: _StructuredEvidenceRequest | None

    @model_validator(mode="after")
    def require_layers_or_data_request(self):
        if bool(self.layers) == bool(self.required_data_request):
            raise ValueError("visual encoding must produce either layers or required_data_request")
        return self


class _StructuredCandidateAudit(BaseModel):
    """Provider-strict form of the post-materialization semantic audit."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "revise", "needs_sources", "unavailable"]
    issues: list[str]
    required_data_request: _StructuredEvidenceRequest | None

    @model_validator(mode="after")
    def validate_resolution(self):
        if self.decision == "approve" and (self.issues or self.required_data_request is not None):
            raise ValueError("an approved audit cannot contain issues or a dependency")
        if self.decision == "revise" and not self.issues:
            raise ValueError("a revision audit requires actionable issues")
        if self.decision == "needs_sources" and self.required_data_request is None:
            raise ValueError("a source audit requires required_data_request")
        if self.decision != "needs_sources" and self.required_data_request is not None:
            raise ValueError("only needs_sources may include required_data_request")
        if self.decision == "unavailable" and not self.issues:
            raise ValueError("an unavailable audit requires a reason")
        return self

    def to_runtime(self) -> VisualizationCandidateAudit:
        return VisualizationCandidateAudit(
            decision=self.decision,
            issues=self.issues,
            required_data_request=(
                self.required_data_request.to_runtime()
                if self.required_data_request is not None
                else None
            ),
        )


class _StructuredProjectionProofAssessment(BaseModel):
    """One explicit coverage judgment for one required proof obligation."""

    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1, max_length=32)
    status: Literal["directly_materialized", "repairable", "missing", "not_visualizable"]
    missing_evidence_kind: Literal[
        "existing_database_observations",
        "calculation_from_available_evidence",
        "anomaly_detector_output",
        "forecast_model_output",
    ] | None
    view_ids: list[str]
    rationale: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_missing_kind(self):
        if self.status == "missing" and self.missing_evidence_kind is None:
            raise ValueError("a missing proof obligation requires missing_evidence_kind")
        return self


class _StructuredProofAuditBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[_StructuredProjectionProofAssessment]


class _StructuredProofApproveAudit(_StructuredProofAuditBranch):
    decision: Literal["approve"]
    issues: list[str]
    required_data_request: None

    @model_validator(mode="after")
    def require_clean_approval(self):
        if self.issues:
            raise ValueError("an approved proof audit cannot contain issues")
        return self


class _StructuredProofReviseAudit(_StructuredProofAuditBranch):
    decision: Literal["revise"]
    issues: list[str] = Field(min_length=1)
    required_data_request: None


class _StructuredProofNeedsSourcesAudit(_StructuredProofAuditBranch):
    decision: Literal["needs_sources"]
    issues: list[str]
    required_data_request: _StructuredEvidenceRequest


class _StructuredProofUnavailableAudit(_StructuredProofAuditBranch):
    decision: Literal["unavailable"]
    issues: list[str] = Field(min_length=1)
    required_data_request: None


class _StructuredProjectionAudit(BaseModel):
    """Provider-enforced exclusive, obligation-by-obligation proof audit."""

    model_config = ConfigDict(extra="forbid")

    outcome: (
        _StructuredProofApproveAudit
        | _StructuredProofReviseAudit
        | _StructuredProofNeedsSourcesAudit
        | _StructuredProofUnavailableAudit
    )

    @property
    def assessments(self) -> list[_StructuredProjectionProofAssessment]:
        return self.outcome.assessments

    @property
    def decision(self) -> Literal["approve", "revise", "needs_sources", "unavailable"]:
        return self.outcome.decision

    @property
    def required_data_request(self) -> _StructuredEvidenceRequest | None:
        return self.outcome.required_data_request

    def to_runtime(self) -> VisualizationCandidateAudit:
        return VisualizationCandidateAudit(
            decision=self.decision,
            issues=self.outcome.issues,
            required_data_request=(
                self.required_data_request.to_runtime()
                if self.required_data_request is not None
                else None
            ),
        )


def _structured_proof_audit_schema(
    verification: VisualVerificationDecision,
    *,
    available_view_ids: set[str],
    allowed_input_source_refs: set[str],
) -> type[_StructuredProjectionAudit]:
    """Close identity vocabularies while leaving semantic judgments to the LLM."""

    obligation_ids = tuple(sorted({item.obligation_id for item in verification.proof_obligations}))
    view_ids = tuple(sorted(available_view_ids))
    source_refs = tuple(sorted(allowed_input_source_refs))
    if not obligation_ids or not view_ids or not source_refs:
        raise ValueError(
            "proof audit schema requires non-empty obligation, view, and input-source vocabularies"
        )

    obligation_id_type = Literal.__getitem__(obligation_ids)
    view_id_type = Literal.__getitem__(view_ids)
    source_ref_type = Literal.__getitem__(source_refs)
    assessment_type = create_model(
        "FixedProofAssessment",
        __base__=_StructuredProjectionProofAssessment,
        obligation_id=(obligation_id_type, ...),
        view_ids=(list[view_id_type], ...),
    )
    request_type = create_model(
        "FixedProofEvidenceRequest",
        __base__=_StructuredEvidenceRequest,
        input_evidence=(source_ref_type | None, ...),
        input_source_refs=(list[source_ref_type], ...),
    )
    approve_type = create_model(
        "FixedProofApproveAudit",
        __base__=_StructuredProofApproveAudit,
        assessments=(list[assessment_type], ...),
    )
    revise_type = create_model(
        "FixedProofReviseAudit",
        __base__=_StructuredProofReviseAudit,
        assessments=(list[assessment_type], ...),
    )
    needs_sources_type = create_model(
        "FixedProofNeedsSourcesAudit",
        __base__=_StructuredProofNeedsSourcesAudit,
        assessments=(list[assessment_type], ...),
        required_data_request=(request_type, ...),
    )
    unavailable_type = create_model(
        "FixedProofUnavailableAudit",
        __base__=_StructuredProofUnavailableAudit,
        assessments=(list[assessment_type], ...),
    )
    return create_model(
        "FixedProofAudit",
        __base__=_StructuredProjectionAudit,
        outcome=(
            approve_type | revise_type | needs_sources_type | unavailable_type,
            ...,
        ),
    )


def _validate_proof_audit(
    audit: _StructuredProjectionAudit,
    verification: VisualVerificationDecision,
    *,
    available_view_ids: set[str],
    scope: str,
) -> None:
    """Validate proof coverage without deciding or repairing the LLM's semantics."""

    expected_ids = [item.obligation_id for item in verification.proof_obligations]
    actual_ids = [item.obligation_id for item in audit.assessments]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError(
            f"{scope} audit must assess every proof obligation exactly once; "
            f"expected={expected_ids}, actual={actual_ids}"
        )

    unknown_view_ids = sorted({
        view_id
        for item in audit.assessments
        for view_id in item.view_ids
        if view_id not in available_view_ids
    })
    if unknown_view_ids:
        raise ValueError(
            f"{scope} audit references unknown or non-visible semantic view ids: {unknown_view_ids}"
        )

    uncovered_direct = [
        item.obligation_id
        for item in audit.assessments
        if item.status == "directly_materialized" and not item.view_ids
    ]
    if uncovered_direct:
        raise ValueError(
            f"{scope} audit calls obligations directly materialized without visible views: "
            f"{uncovered_direct}"
        )

    if audit.decision == "approve":
        invalid = [
            item.obligation_id
            for item in audit.assessments
            if item.status != "directly_materialized" or not item.view_ids
        ]
        if invalid:
            raise ValueError(f"approved {scope} audit has uncovered obligations: {invalid}")

    missing = [item for item in audit.assessments if item.status == "missing"]
    if missing and audit.decision != "needs_sources":
        raise ValueError(
            f"{scope} audit classified missing proof but did not request its source owner: "
            f"{[item.obligation_id for item in missing]}"
        )
    if audit.decision == "needs_sources":
        if not missing or audit.required_data_request is None:
            raise ValueError(
                f"{scope} source request requires at least one explicitly classified missing obligation"
            )
        owner_by_kind = {
            "existing_database_observations": "sql_query",
            "calculation_from_available_evidence": "code_interpreter",
            "anomaly_detector_output": "anomaly",
            "forecast_model_output": "forecast",
        }
        allowed_owners = {
            owner_by_kind[item.missing_evidence_kind]
            for item in missing
            if item.missing_evidence_kind is not None
        }
        if audit.required_data_request.required_action not in allowed_owners:
            raise ValueError(
                f"{scope} audit selected a source owner inconsistent with its missing evidence classification; "
                f"allowed={sorted(allowed_owners)}, actual={audit.required_data_request.required_action}"
            )


def _proof_audit_input_source_refs(inventory: dict) -> set[str]:
    refs: set[str] = set()
    for collection in ("sources", "views", "proof_bundle"):
        for item in inventory.get(collection, []) if isinstance(inventory.get(collection), list) else []:
            if not isinstance(item, dict):
                continue
            ref = str(item.get("source_ref") or "").strip()
            if ref:
                refs.add(ref)
    return refs


class _StructuredVerificationBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_insight_ids: list[str]
    verification_question: str | None
    interpretation: str | None
    visual_relation: str | None
    proof_obligations: list[VisualProofObligation] = Field(max_length=8)
    required_context: list[str] = Field(max_length=8)
    non_visual_insight_ids: list[str]


class _StructuredVisualizeDecision(_StructuredVerificationBranch):
    decision: Literal["visualize"]
    required_data_request: None


class _StructuredNeedsSourcesDecision(_StructuredVerificationBranch):
    decision: Literal["needs_sources"]
    required_data_request: _StructuredEvidenceRequest


class _StructuredNotVisualizableDecision(_StructuredVerificationBranch):
    decision: Literal["not_visualizable"]
    required_data_request: None


class _StructuredVerificationDecision(BaseModel):
    """Provider-enforced exclusive visual-verification decision branch."""

    model_config = ConfigDict(extra="forbid")

    outcome: (
        _StructuredVisualizeDecision
        | _StructuredNeedsSourcesDecision
        | _StructuredNotVisualizableDecision
    )

    def to_runtime(self) -> VisualVerificationDecision:
        decision = self.outcome
        requirement = decision.required_data_request
        return VisualVerificationDecision(
            decision=decision.decision,
            target_insight_ids=decision.target_insight_ids,
            verification_question=decision.verification_question,
            interpretation=decision.interpretation,
            visual_relation=decision.visual_relation,
            proof_obligations=decision.proof_obligations,
            required_context=decision.required_context,
            non_visual_insight_ids=decision.non_visual_insight_ids,
            required_data_request=(
                requirement.to_runtime()
                if requirement is not None
                else None
            ),
        )


class _StructuredSemanticProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_views: list[SemanticViewPlan]
    required_data_request: _StructuredEvidenceRequest | None

    @model_validator(mode="after")
    def require_views_or_data_request(self):
        if bool(self.semantic_views) == bool(self.required_data_request):
            raise ValueError("semantic projection must produce either semantic_views or required_data_request")
        return self

    def to_runtime(self) -> SemanticProjectionPlan:
        return SemanticProjectionPlan(
            semantic_views=self.semantic_views,
            required_data_request=(
                self.required_data_request.to_runtime()
                if self.required_data_request is not None
                else None
            ),
        )


class _StructuredVisualizationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_goals: list[VisualGoalIR]
    required_data_request: _StructuredEvidenceRequest | None

    @model_validator(mode="after")
    def require_goal_or_data_request(self):
        if bool(self.visual_goals) == bool(self.required_data_request):
            raise ValueError("chart planning must produce either visual_goals or required_data_request")
        return self


VisualizationPlan.model_rebuild()
SemanticProjectionPlan.model_rebuild()
VisualVerificationDecision.model_rebuild()
VisualizationCandidateAudit.model_rebuild()
VisualCompositionPlan.model_rebuild()


class VisualizationResult(BaseModel):
    status: Literal["created", "needs_sources", "unavailable"] = "created"
    summary: str
    visualization_ids: list[str] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None
    unavailable_reason: str | None = None


class VisualizationTool(BaseTool):
    """Plan semantic layers with an LLM and persist complete renderer data."""

    def __init__(
        self,
        *,
        llm,
        artifact_store: VisualizationArtifactStore,
        render_auditor=None,
        llm_timeout_seconds: float | None = None,
    ):
        self._llm = llm
        self._artifact_store = artifact_store
        self._render_auditor = render_auditor
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else load_timeout_policy().tool("visualization").stage_seconds("llm_call_seconds")
        )

    async def close(self) -> None:
        if self._render_auditor is not None and hasattr(self._render_auditor, "close"):
            await self._render_auditor.close()

    async def execute(
        self,
        validated_input: VisualizationInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        # Runtime repair metadata is control-plane state, not part of the
        # user's visual semantics.  Feeding a rejected candidate back as a
        # constraint causes prompt amplification across outer ReAct retries.
        validated_input = validated_input.model_copy(update={
            "constraints": _semantic_visualization_constraints(validated_input.constraints),
        })
        catalog = PresentationCatalog(request_state)
        requested_refs = _resolve_visualization_lineage_refs(validated_input.source_refs, request_state)
        source_preferences, unknown = catalog.expand_preferences(requested_refs)
        inventory = catalog.planner_inventory(source_preferences)
        if unknown:
            raise _semantic_error(ValueError(f"unknown requested source refs: {sorted(unknown)}"), inventory)

        verification = await self._select_verification(
            validated_input,
            inventory,
            request_state,
            source_preferences=source_preferences,
            catalog=catalog,
        )
        if verification.decision == "needs_sources":
            return _dependency_result(verification.required_data_request, request_state)
        if verification.decision == "not_visualizable":
            return _unavailable_result(
                verification.interpretation
                or "The requested conclusion has no grounded visual relationship that the available evidence can verify."
            )
        primary_host_refs: set[str] = set()
        if verification.target_insight_ids and verification.proof_obligations:
            host_resolution = await self._resolve_primary_visual_host(
                validated_input,
                verification,
                inventory,
                request_state,
                catalog=catalog,
            )
            if host_resolution.decision == "needs_sources":
                return _dependency_result(host_resolution.required_data_request, request_state)
            if host_resolution.decision == "unavailable":
                return _unavailable_result(host_resolution.rationale)
            primary_host_refs = {
                item.source_ref for item in host_resolution.source_uses
            }
        evidence_consumption: VisualEvidenceConsumption | None = None
        if verification.target_insight_ids:
            source_preferences, unknown_targets = catalog.expand_preferences([
                f"insight:{insight_id}" for insight_id in verification.target_insight_ids
            ])
            if unknown_targets:
                raise _semantic_error(
                    ValueError(f"verification selected unknown Insight targets: {sorted(unknown_targets)}"),
                    inventory,
                )
            inventory = catalog.targeted_planner_inventory(source_preferences)
            evidence_consumption = await self._consume_key_insight_evidence(
                validated_input,
                verification,
                inventory,
                request_state,
                catalog=catalog,
            )
            if evidence_consumption.decision == "needs_sources":
                return _dependency_result(
                    evidence_consumption.required_data_request,
                    request_state,
                )
            if evidence_consumption.decision == "unavailable":
                return _unavailable_result(evidence_consumption.rationale)
            source_preferences = primary_host_refs | {
                item.source_ref for item in evidence_consumption.source_uses
            }
            inventory = catalog.targeted_planner_inventory(source_preferences)
        elif primary_host_refs:
            source_preferences = primary_host_refs
            inventory = catalog.targeted_planner_inventory(source_preferences)

        projection = None
        projection_error = ""
        semantic_refs: list[str] = []
        proof_bundle: list[dict] = []
        for projection_attempt in range(3):
            projection = await self._project(
                validated_input,
                inventory,
                request_state,
                verification=verification,
                source_preferences=source_preferences,
                evidence_consumption=evidence_consumption,
                repair_context=(
                    None
                    if projection_attempt == 0
                    else _planning_diagnostic(
                        ValueError(projection_error),
                        stage="semantic_projection_execution",
                        allowed_values=_default_source_paths(inventory),
                    )
                ),
            )
            if projection.required_data_request:
                requirement = projection.required_data_request
                try:
                    requirement = _normalize_requirement_input(requirement, catalog)
                except ValueError as exc:
                    projection_error = str(exc)
                    if projection_attempt >= 2:
                        raise _semantic_error(exc, inventory, scope="semantic_projection") from exc
                    continue
                return _dependency_result(requirement, request_state)
            try:
                _validate_projection_consumes_target_claims(
                    projection,
                    evidence_consumption=evidence_consumption,
                    inventory=inventory,
                    target_insight_ids=verification.target_insight_ids,
                )
                if verification.proof_obligations:
                    projection_audit = await self._audit_projection(
                        validated_input,
                        verification,
                        projection,
                        inventory,
                        request_state,
                        catalog=catalog,
                    )
                    if projection_audit.decision == "needs_sources":
                        return _dependency_result(
                            projection_audit.required_data_request,
                            request_state,
                        )
                    if projection_audit.decision == "unavailable":
                        return _unavailable_result("; ".join(projection_audit.issues))
                    if projection_audit.decision != "approve":
                        raise ValueError(
                            "semantic projection does not directly materialize every visual proof obligation: "
                            + "; ".join(projection_audit.issues)
                        )
                proof_bundle = _resolve_semantic_evidence_bundle(
                    verification,
                    projection,
                    [f"semantic:{view.view_id}" for view in projection.semantic_views],
                )
                semantic_refs = catalog.materialize_semantic_views(projection.semantic_views)
                break
            except ValueError as exc:
                projection_error = str(exc)
                if projection_attempt >= 2:
                    raise _semantic_error(exc, inventory, scope="semantic_projection") from exc
        if not semantic_refs:
            raise _semantic_error(
                ValueError("visualization semantic projection produced no executable views"),
                inventory,
                scope="semantic_projection",
            )

        semantic_inventory = catalog.semantic_inventory(semantic_refs)
        semantic_inventory["proof_bundle"] = proof_bundle
        complete = None
        composition_repair = None
        for composition_attempt in range(3):
            composition = await self._compose(
                validated_input,
                semantic_inventory,
                request_state,
                verification=verification,
                repair_context=composition_repair,
            )
            if composition.required_data_request:
                try:
                    requirement = _normalize_requirement_input(composition.required_data_request, catalog)
                except ValueError as exc:
                    if composition_attempt >= 2:
                        raise _semantic_error(
                            exc,
                            semantic_inventory,
                            scope="visual_composition_requirement",
                        ) from exc
                    composition_repair = _planning_diagnostic(
                        exc,
                        stage="visual_composition_requirement",
                        allowed_values=_closed_visual_field_contract(semantic_inventory),
                    )
                    continue
                return _dependency_result(requirement, request_state)

            plan = None
            encoding_repair = None
            for encoding_attempt in range(3):
                try:
                    plan = await self._encode(
                        validated_input,
                        semantic_inventory,
                        request_state,
                        verification=verification,
                        catalog=catalog,
                        composition=composition,
                        repair_context=encoding_repair,
                    )
                    if plan.required_data_request:
                        requirement = _normalize_requirement_input(plan.required_data_request, catalog)
                        return _dependency_result(requirement, request_state)
                    complete = VisualizationMaterializer(
                        request_state,
                        catalog=catalog,
                        visual_constraints=validated_input.constraints,
                    ).materialize_all(
                        plan.visual_goals,
                        verification=verification.public_contract(),
                    )
                    validator = VisualizationSemanticValidator(catalog)
                    for goal, payload in zip(plan.visual_goals, complete, strict=True):
                        validator.validate(goal, payload)
                    break
                except IncompatibleVisualDomainError as exc:
                    complete = None
                    composition_repair = _planning_diagnostic(
                        exc,
                        stage="visual_composition_domain",
                        allowed_values=_visual_composition_inventory(semantic_inventory),
                    )
                    break
                except ValueError as exc:
                    complete = None
                    if encoding_attempt >= 2:
                        return _unavailable_result(
                            "Visual field encoding could not be materialized without semantic ambiguity: "
                            + _compact_validation_message(exc)
                        )
                    encoding_repair = _planning_diagnostic(
                        exc,
                        stage=(
                            "requirement_grounding"
                            if "input" in str(exc).casefold() and "source" in str(exc).casefold()
                            else "visual_field_materialization"
                        ),
                        allowed_values=_closed_visual_field_contract(semantic_inventory),
                    )
            if complete is None or plan is None:
                continue
            audit = await self._audit_candidate(
                validated_input,
                verification,
                plan,
                complete,
                semantic_inventory,
                request_state,
                catalog=catalog,
            )
            if audit.decision == "approve":
                break
            if audit.decision == "needs_sources":
                return _dependency_result(audit.required_data_request, request_state)
            if audit.decision == "unavailable":
                return _unavailable_result("; ".join(audit.issues))
            complete = None
            if composition_attempt >= 2:
                return _unavailable_result(
                    "Visual verification remained semantically inconsistent after LLM repair: "
                    + "; ".join(audit.issues)
                )
            composition_repair = {
                "stage": "semantic_candidate_audit",
                "error_code": "SEMANTIC_AUDIT_REJECTED",
                "issues": audit.issues,
            }

        if complete is None:
            return _unavailable_result("Visualization did not materialize a verified candidate.")
        descriptors = [self._artifact_store.put(item) for item in complete]
        source_refs = list(dict.fromkeys(ref for item in descriptors for ref in item.source_refs))
        return VisualizationResult(
            summary=f"Created {len(descriptors)} grounded visualization artifact(s).",
            visualization_ids=[item.visualization_id for item in descriptors],
            visualizations=descriptors,
            source_refs=source_refs,
        ).model_dump(mode="json")

    async def _resolve_primary_visual_host(
        self,
        request: VisualizationInput,
        verification: VisualVerificationDecision,
        inventory: dict,
        request_state: RequestStateModel,
        *,
        catalog: PresentationCatalog,
    ) -> VisualEvidenceConsumption:
        """Resolve the axis-bearing relationship before guides or chart composition."""

        allowed_refs = {
            str(item.get("source_ref") or "").strip()
            for item in inventory.get("sources", [])
            if isinstance(item, dict) and str(item.get("source_ref") or "").strip()
        }
        fact_packet = _visual_evidence_fact_packet(
            request_state,
            verification.target_insight_ids,
            inventory,
        )
        repair_error: Exception | None = None
        for _attempt in range(3):
            prompt = prompt_locale_instruction(request_state.response_language) + (
                "You resolve only the primary axis-bearing data relationship for a fixed visualization goal. Do not design a chart, "
                "select guides, or calculate values. Read visual_relation and required proof obligations, then ask whether one supplied "
                "record source already materializes that relationship at the requested grain. A source is a valid primary host only when "
                "its records directly contain the x/category and measure needed for that relationship. Raw calculation inputs, background "
                "context, scalar aggregates, thresholds, conclusions, and annotations are supporting evidence; they do not become the host "
                "for a different derived relationship. Selection-only access cannot group, aggregate, subtract, join, resample, or transform. "
                "Return decision=ready only with the minimal exact host source_ref values. If the relationship must be calculated from supplied "
                "evidence, return needs_sources for code_interpreter with non-empty insight_requests and exact input source refs. Missing raw "
                "database records belong to sql_query, anomaly detector results to anomaly, and forecasts to forecast. Return unavailable only "
                "when the goal has no defensible axis-bearing relationship. Do not invent or substitute a host. input_evidence must be null or "
                "one exact Allowed source_ref, and every input_source_refs item must be copied exactly from that vocabulary.\n"
                f"Fixed visualization goal: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
                f"Grounded source facts: {json.dumps(fact_packet, ensure_ascii=False)}\n"
                f"Allowed source refs: {json.dumps(sorted(allowed_refs), ensure_ascii=False)}\n"
                f"Repair context: {str(repair_error)[:1000] if repair_error else 'none'}"
            )
            messages = [("system", prompt), ("user", request.message)]
            started_at = time.perf_counter()
            response, content, parsed, parse_error = await _invoke_structured(
                self._llm,
                _StructuredEvidenceConsumption,
                messages,
                timeout_seconds=self._llm_timeout_seconds,
                trace_title="Primary Visual Host Resolution",
                trace_summary="确认主数据关系是否已由上游物化",
            )
            record_llm_token_usage(
                request_state,
                source="visualization.primary_host_resolution",
                response=response,
                messages=messages,
                output_text=content,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            try:
                if parse_error is not None:
                    raise parse_error
                resolution = parsed.to_runtime()
                selected = {item.source_ref for item in resolution.source_uses}
                unknown = selected - allowed_refs
                if unknown:
                    raise ValueError(
                        f"primary visual host selected unknown source refs: {sorted(unknown)}"
                    )
                if resolution.decision == "needs_sources":
                    resolution = resolution.model_copy(update={
                        "required_data_request": _normalize_requirement_input(
                            resolution.required_data_request,
                            catalog,
                        ),
                    })
                return resolution
            except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
                repair_error = exc

        raise _semantic_error(
            ValueError(f"invalid primary visual host resolution after LLM repair: {repair_error}"),
            inventory,
            scope="primary_visual_host",
        ) from repair_error

    async def _select_verification(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        *,
        source_preferences: set[str],
        catalog: PresentationCatalog,
    ) -> VisualVerificationDecision:
        if self._llm is None:
            raise RuntimeError("visual verification planning requires an LLM")
        insight_inventory = _verified_insight_inventory(request_state)
        allowed = {item["insight_id"] for item in insight_inventory}
        repair_error: Exception | None = None
        for attempt in range(3):
            repair_context = "none" if repair_error is None else (
                "The preceding candidate was rejected by runtime validation. Re-plan the complete decision; "
                f"do not preserve invalid values. Validation feedback: {repair_error}. "
                f"The only valid values for target_insight_ids and non_visual_insight_ids are: {sorted(allowed)}."
            )
            prompt = prompt_locale_instruction(request_state.response_language) + (
                "You define the presentation goal for grounded visualization inside a strict outer ReAct loop. Decide what conclusion or "
                "observable relationship the user wants to inspect before looking at data fields, source schemas, or chart types. Return one "
                "schema-valid discriminated outcome: "
                "visualize/not_visualizable require required_data_request=null; needs_sources requires exactly one complete request. "
                "Select only supplied verified insight_id values. target_insight_ids and non_visual_insight_ids are IDs, never source refs: "
                "do not add an 'insight:' prefix. A target Insight owns the conclusion; a later evidence-consumption stage will follow its "
                "evidence_refs and decide whether the related artifacts can execute the goal. Do not perform source selection or lineage "
                "consistency adjudication in this stage. Target a located Insight only when one item co-locates its timestamp and numeric value; do not reconstruct "
                "a point from unrelated scalar Insights. A raw descriptive chart may have no target Insight, but must state an observable relation. "
                "For visualize, define proof_obligations as short stable machine ids (at most 32 characters; examples: complete_context, "
                "located_claim) plus descriptions of the independently inspectable evidence roles the chart needs. An id is only an opaque "
                "cross-stage key: never put dates, field names, source refs, descriptions, or repeated phrases into it. Do not use chart types as obligations. "
                "When the user asks to inspect anomaly exclusions or anomaly results, the default presentation goal is a complete raw context series "
                "with detected/excluded anomaly points visibly marked; this does not imply a separate cleaned line. Require a cleaned-only or retained-only "
                "series only when the user explicitly asks to see that transformed trajectory. "
                "Use needs_sources when the conclusion or required context is absent; do not calculate or invent a fallback. code_interpreter "
                "requests require non-empty insight_requests; SQL may request SQL-owned atomic Insights. Use not_visualizable only when a chart "
                "adds no inspectable evidence (for example, a causal claim unsupported by observational data).\n"
                f"User request: {request.message}\n"
                f"Original analytical request: {request_state.message}\n"
                f"Analytical task goal: {getattr(request_state.task_contract, 'goal', None) or request_state.focus}\n"
                f"Task visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
                f"Verified Key Insights: {json.dumps(insight_inventory, ensure_ascii=False)}\n"
                f"Preferred source refs: {json.dumps(sorted(source_preferences), ensure_ascii=False)}\n"
                f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
                f"Repair context: {repair_context}"
            )
            messages = [("system", prompt), ("user", request.message)]
            started_at = time.perf_counter()
            response, content, parsed, parse_error = await _invoke_structured(
                self._llm,
                _StructuredVerificationDecision,
                messages,
                timeout_seconds=self._llm_timeout_seconds,
                trace_title="Visual Verification Planning",
                trace_summary="选择需要通过图表验证的数据发现",
            )
            record_llm_token_usage(
                request_state,
                source="visualization.verification_plan",
                response=response,
                messages=messages,
                output_text=content,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            try:
                if parse_error is not None:
                    raise parse_error
                decision = parsed.to_runtime()
                referenced = set(decision.target_insight_ids) | set(decision.non_visual_insight_ids)
                unknown = referenced - allowed
                if unknown:
                    raise ValueError(
                        "visual verification plan references unknown or unverified insights: "
                        f"{sorted(unknown)}"
                    )
                if decision.decision == "needs_sources":
                    decision = decision.model_copy(update={
                        "required_data_request": _normalize_requirement_input(
                            decision.required_data_request,
                            catalog,
                        ),
                    })
                return decision
            except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
                repair_error = exc

        raise _semantic_error(
            ValueError(f"invalid visual verification decision after LLM repair: {repair_error}"),
            inventory,
            scope="verification_selection",
        ) from repair_error

    async def _consume_key_insight_evidence(
        self,
        request: VisualizationInput,
        verification: VisualVerificationDecision,
        inventory: dict,
        request_state: RequestStateModel,
        *,
        catalog: PresentationCatalog,
    ) -> VisualEvidenceConsumption:
        """Bind a fixed presentation goal to executable Key Insight lineage."""

        allowed_refs = {
            str(item.get("source_ref") or "").strip()
            for item in inventory.get("sources", [])
            if isinstance(item, dict) and str(item.get("source_ref") or "").strip()
        }
        target_refs = {
            f"insight:{insight_id}" for insight_id in verification.target_insight_ids
        }
        fact_packet = _visual_evidence_fact_packet(
            request_state,
            verification.target_insight_ids,
            inventory,
        )
        repair_error: Exception | None = None
        for _attempt in range(3):
            prompt = prompt_locale_instruction(request_state.response_language) + (
                "You bind a fixed visualization goal to upstream Key Insight lineage. The presentation goal is already decided: "
                "do not redesign the chart, choose chart types, or invent fields. Decide only whether the target Key Insights and their "
                "related source refs can supply every required visual evidence role. For ready, select the minimal exact source_ref set and "
                "state each source's purpose. Every target Insight source_ref must remain selected because it owns the claim; evidence refs "
                "own the context used to inspect it. Select refs only from Allowed lineage source refs. Treat a verified target Key Insight as the "
                "authoritative upstream conclusion; visualization must display it and must not recompute, overturn, or semantically re-audit it. Bind "
                "sources according to semantic_contract.supported_visual_uses. First identify the primary data relationship and grain named by "
                "visual_relation and the required proof obligations. decision=ready requires one selected record source that already materializes that "
                "primary relationship at that grain. A calculation input, background/context series, scalar conclusion, threshold, or annotation is not "
                "the primary relationship. It may still be selected for its supporting role, but it cannot make the decision ready. If no source directly "
                "materializes the primary relationship, decision=needs_sources even when all inputs needed to calculate it are present. "
                "A complete raw observation view can supply the background line, while "
                "an anomaly detection output can supply anomaly/exclusion markers on that same line. This combination is sufficient to inspect anomaly "
                "results and does not require a cleaned series. If target locations overlap anomaly markers, retain both sources so the overlap is visible. "
                "Ready means every required inspectable relationship is directly materialized by the selected source contracts, not merely calculable "
                "from them. Raw observations plus a scalar aggregate do not materialize a missing grouped, differenced, rate, interval, or transformed "
                "series. Request code_interpreter whenever the fixed presentation goal requires such a derived relationship from already available "
                "evidence and no derived result source exists. anomaly owns detection points/scores/status; sql_query owns raw database context; "
                "forecast owns predictions. Use unavailable only when the fixed goal has no inspectable visual relation. "
                "When the original analytical request or the target calculation explicitly uses anomaly exclusion and grounded anomaly points exist, select "
                "the anomaly point view for visible exclusion markers. Do not select a score series or scalar detector status unless the fixed goal asks for it. "
                "Return exactly one strict object: decision=ready with non-empty source_uses and required_data_request=null; decision=needs_sources "
                "with one complete request; or decision=unavailable with no request. Do not use a deterministic or synthetic substitute.\n"
                f"Fixed presentation goal: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
                f"Key Insight lineage fact packet: {json.dumps(fact_packet, ensure_ascii=False)}\n"
                f"Allowed lineage source refs: {json.dumps(sorted(allowed_refs), ensure_ascii=False)}\n"
                f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
                f"Repair context: {str(repair_error)[:800] if repair_error else 'none'}"
            )
            messages = [("system", prompt), ("user", request.message)]
            started_at = time.perf_counter()
            response, content, parsed, parse_error = await _invoke_structured(
                self._llm,
                _StructuredEvidenceConsumption,
                messages,
                timeout_seconds=self._llm_timeout_seconds,
                trace_title="Visual Evidence Consumption",
                trace_summary="沿 Key Insight 引用选择可执行可视化证据",
            )
            record_llm_token_usage(
                request_state,
                source="visualization.evidence_consumption",
                response=response,
                messages=messages,
                output_text=content,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            try:
                if parse_error is not None:
                    raise parse_error
                consumption = parsed.to_runtime()
                selected = {item.source_ref for item in consumption.source_uses}
                unknown = selected - allowed_refs
                if unknown:
                    raise ValueError(
                        f"visual evidence consumption selected refs outside target lineage: {sorted(unknown)}"
                    )
                if consumption.decision == "ready":
                    missing_targets = target_refs - selected
                    if missing_targets:
                        raise ValueError(
                            "ready visual evidence consumption omitted target Key Insight refs: "
                            f"{sorted(missing_targets)}"
                        )
                if consumption.decision == "needs_sources":
                    consumption = consumption.model_copy(update={
                        "required_data_request": _normalize_requirement_input(
                            consumption.required_data_request,
                            catalog,
                        ),
                    })
                return consumption
            except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
                repair_error = exc

        raise _semantic_error(
            ValueError(f"invalid Key Insight evidence consumption after LLM repair: {repair_error}"),
            inventory,
            scope="evidence_consumption",
        ) from repair_error

    async def _audit_candidate(
        self,
        request: VisualizationInput,
        verification: VisualVerificationDecision,
        plan: VisualizationPlan,
        visualizations: list[VisualizationPayload],
        inventory: dict,
        request_state: RequestStateModel,
        *,
        catalog: PresentationCatalog,
        repair_context: dict | None = None,
        contract_repair_attempt: int = 0,
    ) -> VisualizationCandidateAudit:
        target_insights = _target_insight_inventory(request_state, verification.target_insight_ids)
        proof_audit = bool(verification.proof_obligations)
        visible_view_ids = {
            layer.source_ref.removeprefix("semantic:")
            for visualization in visualizations
            for layer in visualization.layers
            if layer.source_ref.startswith("semantic:")
        }
        audit_schema = (
            _structured_proof_audit_schema(
                verification,
                available_view_ids=visible_view_ids,
                allowed_input_source_refs=_proof_audit_input_source_refs(inventory),
            )
            if proof_audit
            else _StructuredCandidateAudit
        )
        response_contract = (
            "Return exactly one JSON object matching {\"outcome\":{\"assessments\":[{\"obligation_id\":str,"
            "\"status\":\"directly_materialized\"|\"repairable\"|\"missing\"|\"not_visualizable\","
            "\"missing_evidence_kind\":\"existing_database_observations\"|\"calculation_from_available_evidence\"|"
            "\"anomaly_detector_output\"|\"forecast_model_output\"|null,\"view_ids\":[str],\"rationale\":str}],"
            "\"decision\":\"approve\"|\"revise\"|\"needs_sources\"|\"unavailable\","
            "\"issues\":[str],\"required_data_request\":object|null}}. Return exactly one assessment for every "
            "required proof obligation, copying its short obligation_id verbatim. A directly_materialized assessment must name "
            "the semantic view_id values that visibly carry the relationship in the materialized candidate. A missing relationship "
            "must be status=missing and decision=needs_sources, not revise. Use status=repairable only when an already supplied semantic "
            "view directly materializes the relationship and the current chart can expose it through composition or encoding repair. "
            "Classify missing evidence before choosing its owner: "
            "existing_database_observations means records that can be queried without deriving the requested relationship and belongs to sql_query; "
            "calculation_from_available_evidence includes grouping, aggregation, differences, rates, resampling, and transformations and belongs to "
            "code_interpreter; anomaly_detector_output belongs to anomaly; forecast_model_output belongs to forecast. "
            if proof_audit
            else (
                "Return exactly one JSON object matching {\"decision\":\"approve\"|\"revise\"|\"needs_sources\"|"
                "\"unavailable\",\"issues\":[str],\"required_data_request\":object|null}. "
            )
        )
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You independently audit a fully materialized visual-verification candidate. You did not author the plan. "
            + response_contract
            + "Approve only when the chart lets a human inspect the "
            "verification question and the claimed meaning is entailed by the grounded field semantics and lineage. Check every "
            "target Insight, required contextual baseline, complete comparison set, time/grain/unit compatibility, interval semantics, "
            "role coverage, title, legend meaning, and interpretation. For a localized interval in a longer series, require the exact highlighted "
            "interval to remain inside a broader initial dataZoom observation window while the complete series remains scrollable; values outside "
            "that viewport must not distort its visible y scale. Exact values must remain owned by their artifacts. Do not accept "
            "a causal claim from merely correlated series. Treat verified target Key Insights as authoritative upstream conclusions and do not "
            "recompute or overturn them. A complete raw series plus anomaly detection points is a valid way to inspect anomaly exclusions: the line "
            "must be labeled as complete/raw context and anomaly records must be visibly marked as detected/excluded points. If a target interval "
            "overlaps those markers, preserve and expose the overlap rather than rejecting it. Do not call the raw line cleaned or exclusion-applied. "
            "Audit actual materialized cardinality, not just layer labels: event_points renders every source row and cannot be called a single start, "
            "end, or anomaly marker when its dataset contains the full context series. Reject redundant layers that redraw the same complete source "
            "with identical x/y fields under a narrower label. When the request asks for surrounding context, the initial viewport must extend beyond "
            "the target interval wherever grounded observations exist; a viewport equal to the two target boundaries shows no surrounding context. "
            "A source-backed interval_overlay is intentionally a filtered highlight over the primary context, not a redundant redraw, when its "
            "provenance contains the authoritative target boundary source. Its exact filter boundaries are sufficient to expose start/end; do not "
            "require duplicate endpoint layers unless the fixed composition calls for additional endpoint information. A null viewport with enabled "
            "zoom validly shows the complete surrounding context. "
            "Treat scalar guides explicitly: a reference_line is valid only when its numeric measure, unit, population, and time scope match the host "
            "y meaning; an annotation may state grounded scalar/text evidence but must not be counted as visual verification of a missing relationship. "
            "If the verification question requires a derived series that is not materialized, return needs_sources for code_interpreter rather than "
            "approving an annotation attached to a semantically different raw series. Before any other audit decision, compare every guide layer's "
            "encoding_semantics against the data host's encoding_semantics in the same candidate. Equal primitive data types or equal units do not imply "
            "equal measures: price level, price difference/range, rate, count, score, and threshold are distinct y meanings. If a rule guide is not the "
            "same measure as its host y, decision=revise and require chart annotation instead, unless the verification question actually needs that "
            "measure as an inspectable series; then decision=needs_sources with its source owner. Never approve first and mention this only as prose. "
            "Require a transformed source only when the fixed goal explicitly asks for a cleaned-only or retained-only trajectory. Use revise for "
            "presentation or plan errors that existing semantic views can "
            "repair, needs_sources for genuinely missing evidence, and unavailable when the requested conclusion has no defensible "
            "visual verification. Never approve a decorative scalar chart or invent a fallback.\n"
            "Treat source materialization_complete and query_execution coverage as authoritative. In sampled time series, the last "
            "observed timestamp may validly precede an exclusive query stop boundary by one sampling interval; do not call a complete "
            "source incomplete merely because its final observed timestamp is not identical to the requested stop.\n"
            f"User request: {request.message}\n"
            f"Verification decision: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(target_insights, ensure_ascii=False)}\n"
            f"Chart plan: {json.dumps(plan.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Materialized candidates: {json.dumps(_candidate_audit_view(visualizations, inventory), ensure_ascii=False)}\n"
            f"Allowed dependency input source refs (exact closed vocabulary; never write prose here): "
            f"{json.dumps(sorted(_proof_audit_input_source_refs(inventory)), ensure_ascii=False)}\n"
            f"Semantic inventory: {json.dumps(inventory, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            audit_schema,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title="Visualization Audit",
            trace_summary="检查图表候选是否忠实表达证据",
        )
        record_llm_token_usage(
            request_state,
            source="visualization.semantic_audit",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
        try:
            if parse_error is not None:
                raise parse_error
            if proof_audit:
                _validate_proof_audit(
                    parsed,
                    verification,
                    available_view_ids=visible_view_ids,
                    scope="materialized candidate",
                )
            audit = parsed.to_runtime()
            if audit.decision == "needs_sources":
                audit = audit.model_copy(update={
                    "required_data_request": _normalize_requirement_input(
                        audit.required_data_request,
                        catalog,
                    ),
                })
            return audit
        except (json.JSONDecodeError, ValueError) as exc:
            if contract_repair_attempt < 2:
                return await self._audit_candidate(
                    request,
                    verification,
                    plan,
                    visualizations,
                    inventory,
                    request_state,
                    catalog=catalog,
                    repair_context={
                        **(repair_context or {}),
                        "contract_error": str(exc),
                        "rejected_response": content,
                        "schema_instruction": (
                            "Return decision=revise with required_data_request=null for a plan-only repair, or "
                            "decision=needs_sources with a complete VisualizationEvidenceRequest for genuinely missing evidence."
                        ),
                    },
                    contract_repair_attempt=contract_repair_attempt + 1,
                )
            return VisualizationCandidateAudit(
                decision="unavailable",
                issues=[f"Semantic visual audit did not return a valid decision: {exc}"],
            )

    async def _project(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        verification: VisualVerificationDecision,
        source_preferences: set[str],
        evidence_consumption: VisualEvidenceConsumption | None = None,
        repair_context: dict | None = None,
        schema_repair_attempt: int = 0,
    ) -> SemanticProjectionPlan:
        if self._llm is None:
            raise RuntimeError("visualization planning requires an LLM")
        target_insights = _target_insight_inventory(request_state, verification.target_insight_ids)
        source_contracts = _projection_source_inventory(inventory)
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You are the semantic projection stage for grounded visualization. Project the target Insight and the complete context needed "
            "to inspect it into executable, same-grain semantic views. Source records are resolved only during materialization: use the "
            "supplied schema, projection_root, and lineage; never calculate, infer, or manufacture values. Keep a located point in one "
            "record that contains both timestamp and numeric value. Preserve the complete baseline/comparison context rather than replacing "
            "it with a summary Insight. Return exactly one JSON object matching: "
            "{\"semantic_views\":[{\"view_id\":str,\"name\":str,\"purpose\":str,\"grain\":str,"
            "\"source_ref\":str,\"record_path\":str|null,\"mode\":\"records\","
            "\"fields\":[{\"name\":str,\"semantic_role\":str,\"source_path\":str}]}|"
            "{\"view_id\":str,\"name\":str,\"purpose\":str,\"grain\":str,\"source_ref\":str,"
            "\"record_path\":str|null,\"mode\":\"wide_events\",\"events\":[{\"event_role\":str,"
            "\"timestamp_path\":str,\"value_path\":str}]}],"
            "\"required_data_request\":{\"required_action\":\"sql_query\"|\"anomaly\"|\"forecast\"|\"code_interpreter\","
            "\"purpose\":str,\"message\":str|null,\"required_shape\":str,\"required_fields\":[str],"
            "\"required_properties\":[str],\"input_evidence\":str|null,\"input_source_refs\":[str],\"insight_requests\":[{"
            "\"name\":str,\"insight_type\":str,\"insight_key\":str|null}]}|null}. "
            "Use one exact source_ref per view. record_path is optional, not a preferred default. Follow projection_root.executable_path_contracts "
            "as a closed path vocabulary: with record_path=null choose source_path only from default_source_paths; with a non-null record_path choose "
            "it only from record_path_candidates and choose source_path only from that record path's source_paths. source_path is relative to each "
            "selected record, so never repeat the record_path container as a source_path; wildcards belong only in record_path. For each view choose exactly one representation: records includes a non-empty "
            "fields property and MUST NOT include events; wide_events includes a non-empty events property and MUST NOT include fields. wide_events is only for one wide record containing "
            "multiple distinct timestamp/value pairs; it emits event_role, timestamp, value. Use a separate view for another grain. "
                "Semantic field name is a stable ASCII machine identifier (for example timestamp, start_time, observed_value); put localized or descriptive "
                "meaning only in semantic_role. Never translate field identifiers. A semantic view can only select values within its one source_ref: it cannot join, anti-join, subtract, filter one "
                "source by another, or apply a business transformation. If a proof obligation requires a cleaned/transformed series and no single "
                "grounded source already contains it, request code_interpreter instead of projecting raw and exclusion sources separately. "
                "Use stable graphical semantic roles: interval boundaries end in _start/_end, and genuine uncertainty bounds end in _lower/_upper. "
                "Never label an ordinary value, score, or target endpoint as an uncertainty bound merely to make a band chart available. "
                "A proof role that asks to inspect anomaly exclusions or anomaly results is satisfied by projecting the complete raw context and the "
                "anomaly detection points as separate marker views; it does not require a cleaned series. Preserve target/anomaly overlap for display. "
                "Request a transformed source only when the fixed presentation goal explicitly requires the transformed trajectory itself. Forecast "
                "visuals require historical actuals when available; request sql_query when that baseline is absent. "
                "Evidence consumption is an executable hand-off, not advisory prose. When a selected target Key Insight exposes graphical location "
                "or interval fields, emit a semantic view from that exact Insight source_ref so it remains the authoritative target/annotation/boundary "
                "source. A raw or anomaly source with coincident values cannot substitute for the target claim. Use the related raw and anomaly sources "
                "for their separate context and marker roles. "
                "Every semantic view must have an executable consumer. Time+number and category+number views can own data layers; number-only scalar "
                "views can own a reference_line or annotation; text/status scalar views can own an annotation. A scalar guide never becomes a standalone "
                "chart or an artificial category axis: it must attach to a host data layer whose meaning it truthfully explains. "
            "Create the minimal set of semantic views whose purposes collectively satisfy every required proof obligation in the visual verification "
            "contract. Do not copy obligation ids into view ids, names, purposes, or any other field; obligation ids are local to verification and are "
            "not cross-stage foreign keys. Every emitted semantic view is a required proof source and must be used by the final chart. If no grounded "
            "view set can satisfy an obligation, request its owner. "
            "Do not invent uncertainty unless requested. Return either non-empty semantic_views with null required_data_request, or an empty "
            "view list with one owner request. Owners: sql_query for raw context, anomaly only for detecting anomaly points/scores/status, forecast "
            "for predictions, and code_interpreter for derived calculations with non-empty insight_requests. Applying anomaly decisions to a source "
            "to produce retained/excluded flags or a cleaned/transformed series is a derived calculation owned by code_interpreter, never anomaly. "
            "input_evidence is an exact source_ref or null. "
            "On repair, fix the reported schema, source, or path violation in the complete plan; never repeat it or create a fallback view.\n"
            f"Visualization request: {request.message}\n"
            f"Visual verification contract: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(target_insights, ensure_ascii=False)}\n"
            f"Evidence consumption contract: {json.dumps(evidence_consumption.model_dump(mode='json'), ensure_ascii=False) if evidence_consumption else 'none'}\n"
            f"Authoritative visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
            f"Preferred source refs: {json.dumps(sorted(source_preferences), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Allowed paths when record_path is null: {json.dumps(_default_source_paths(inventory), ensure_ascii=False)}\n"
            f"Grounded source contracts: {json.dumps(source_contracts, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            _StructuredSemanticProjection,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title="Semantic Projection",
            trace_summary="将证据映射为可视化语义字段",
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.semantic_projection",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            if parse_error is not None:
                raise parse_error
            return parsed.to_runtime()
        except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
            if schema_repair_attempt < 2:
                return await self._project(
                    request,
                    inventory,
                    request_state,
                    verification,
                    source_preferences,
                    evidence_consumption,
                    repair_context=_planning_diagnostic(
                        exc,
                        stage="semantic_projection_schema",
                    ),
                    schema_repair_attempt=schema_repair_attempt + 1,
                )
            raise _semantic_error(
                ValueError(f"invalid semantic projection plan: {exc}"), inventory, scope="semantic_projection",
            ) from exc

    async def _audit_projection(
        self,
        request: VisualizationInput,
        verification: VisualVerificationDecision,
        projection: SemanticProjectionPlan,
        inventory: dict,
        request_state: RequestStateModel,
        *,
        catalog: PresentationCatalog,
        repair_context: str | None = None,
        contract_repair_attempt: int = 0,
    ) -> VisualizationCandidateAudit:
        """Independently verify that projected views directly cover the proof goal."""

        available_view_ids = {view.view_id for view in projection.semantic_views}
        audit_schema = _structured_proof_audit_schema(
            verification,
            available_view_ids=available_view_ids,
            allowed_input_source_refs=_proof_audit_input_source_refs(inventory),
        )
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You independently audit a semantic projection before visual composition. You did not author the projection. Return exactly one JSON "
            "object matching {\"outcome\":{\"assessments\":[{\"obligation_id\":str,\"status\":\"directly_materialized\"|\"missing\"|"
            "\"repairable\"|\"not_visualizable\",\"missing_evidence_kind\":\"existing_database_observations\"|\"calculation_from_available_evidence\"|\"anomaly_detector_output\"|"
            "\"forecast_model_output\"|null,\"view_ids\":[str],\"rationale\":str}],\"decision\":\"approve\"|\"revise\"|\"needs_sources\"|"
            "\"unavailable\",\"issues\":[str],\"required_data_request\":object|null}}. Return exactly one assessment for every supplied required "
            "proof obligation, copying its short obligation_id verbatim. For every required proof obligation, identify a projected view whose selected fields directly "
            "materialize the inspectable relationship at the required grain. Projection is selection-only: it cannot calculate, aggregate, resample, "
            "subtract, or infer a missing relationship. Raw price observations do not materialize daily high-low differences; a scalar average does not "
            "materialize the per-period series from which it was calculated; an annotation does not replace a missing relationship. Scalar/text target "
            "views are valid guide candidates only after a genuine data host covers the inspectable relationship. Use status=repairable only when one "
            "of the supplied grounded source contracts already directly materializes the relationship and a different projection can select it. "
            "If no supplied source contract materializes the relationship, status=missing and decision=needs_sources are mandatory. Equal units do not make price level, "
            "price range, rate, count, score, or threshold the same measure. Approve only if every required obligation is directly covered. Use revise "
            "when existing supplied sources can be projected differently. Use needs_sources with the correct owner when a required derived/raw/forecast/"
            "anomaly relationship is absent; code_interpreter requests require non-empty insight_requests. Classify each missing obligation before "
            "choosing its owner: existing_database_observations means directly queryable source records without deriving the requested relation and belongs "
            "to sql_query; calculation_from_available_evidence includes grouping, differences, rates, aggregation, resampling, or transformed series and "
            "belongs to code_interpreter; anomaly_detector_output belongs to anomaly; forecast_model_output belongs to forecast. SQL is not the owner of a "
            "calculation merely because a database originally supplied its input. "
            "Use unavailable only when no defensible source "
            "path exists. decision=approve requires every assessment status=directly_materialized with at least one valid view_id. Do not invent a fallback.\n"
            f"Visualization request: {request.message}\n"
            f"Visual verification contract: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(_target_insight_inventory(request_state, verification.target_insight_ids), ensure_ascii=False)}\n"
            f"Candidate semantic projection: {json.dumps(projection.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Allowed dependency input source refs (exact closed vocabulary; never write prose here): "
            f"{json.dumps(sorted(_proof_audit_input_source_refs(inventory)), ensure_ascii=False)}\n"
            f"Grounded source contracts: {json.dumps(_projection_source_inventory(inventory), ensure_ascii=False)}\n"
            f"Repair context: {repair_context or 'none'}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            audit_schema,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title="Semantic Projection Audit",
            trace_summary="检查语义视图是否直接覆盖视觉证明目标",
        )
        record_llm_token_usage(
            request_state,
            source="visualization.semantic_projection_audit",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
        try:
            if parse_error is not None:
                raise parse_error
            _validate_proof_audit(
                parsed,
                verification,
                available_view_ids=available_view_ids,
                scope="semantic projection",
            )
            audit = parsed.to_runtime()
            if audit.decision == "needs_sources":
                audit = audit.model_copy(update={
                    "required_data_request": _normalize_requirement_input(
                        audit.required_data_request,
                        catalog,
                    ),
                })
            return audit
        except (json.JSONDecodeError, ValueError) as exc:
            if contract_repair_attempt < 2:
                return await self._audit_projection(
                    request,
                    verification,
                    projection,
                    inventory,
                    request_state,
                    catalog=catalog,
                    repair_context=str(exc),
                    contract_repair_attempt=contract_repair_attempt + 1,
                )
            return VisualizationCandidateAudit(
                decision="unavailable",
                issues=[f"Semantic projection audit did not return a valid decision: {exc}"],
            )

    async def _compose(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        verification: VisualVerificationDecision,
        repair_context: dict | None = None,
        schema_repair_attempt: int = 0,
    ) -> VisualCompositionPlan:
        if self._llm is None:
            raise RuntimeError("visual composition requires an LLM")
        target_insights = _target_insight_inventory(request_state, verification.target_insight_ids)
        proof_bundle = inventory.get("proof_bundle") if isinstance(inventory.get("proof_bundle"), list) else []
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You design the semantic composition of a grounded visualization before any fields are encoded. Decide what the human must see, "
            "not how columns map to renderer channels. Separate every layer into family=primary (the complete contextual visual), highlight "
            "(the target Key Insight location/interval), or support (anomaly markers, thresholds, forecast bounds, or comparison evidence that "
            "helps interpret the target). Return one branch: non-empty visual_goals with required_data_request=null, or an empty goal list with "
            "one complete owner request. Use short stable ASCII layer_id values. Choose source_ref and optional interval_source_ref only from the "
            "semantic composition inventory. Do not output field names, encodings, transforms, renderer options, datasets, or series. layer_id is the "
            "unique execution identity. role is a semantic responsibility and may repeat across distinct nonredundant layers that jointly fulfill it; "
            "never rename a role merely to force uniqueness. "
            "For a localized target in a longer series, keep the complete series as primary and use interval_overlay over that context with the "
            "target boundary view as interval_source_ref when available. A target view owns its highlight; coincident raw/anomaly values cannot "
            "substitute for it. Anomaly views are support markers, never cleaned lines. event_points renders every row, so use it only when the "
            "entire selected source grain consists of intended events; never relabel a complete context series as one start/end point. "
            "Use band only when allowed consumers explicitly admit a true lower/upper band. A scalar is not a comparison series: use reference_line for "
            "a grounded numeric guide only when its measure, unit, population, and time scope match the host y meaning; use annotation for grounded text "
            "or a scalar callout that must not own an axis. An annotation may retain a grounded event/observation x position from its source; a global "
            "scalar annotation omits it. Neither guide owns an x domain, and both must attach to a goal containing a primary data host. "
            "If the requested inspectable relationship needs a derived series that no source materializes, request code_interpreter instead of attaching "
            "the scalar conclusion to a semantically different raw y measure. Each visual goal is one shared coordinate domain: keep temporal layers "
            "together, split categorical comparisons into separate goals, and never mix a time x axis with a category x axis. Use every proof evidence "
            "source at least once, either as a layer source_ref or as interval_source_ref, without redundant redraws; reuse a source only when distinct "
            "fields express genuinely different guide roles. An interval_overlay is an intentional filtered highlight over "
            "the primary context and is not redundant when its boundaries come from the authoritative target source. Do not add separate start/end "
            "layers unless they provide information the interval overlay cannot express. If surrounding context is requested, set enable_zoom=true "
            "and either choose a viewport strictly wider than the target wherever grounded observations exist, or set both viewport values to null to "
            "show the complete context. Never set the initial viewport exactly equal to the target boundaries. When multiple visible roles need "
            "identification, enable the legend or make their labels and marks unambiguous. "
            "Choose one cohesive shared plot for time-aligned layers. Owners: sql_query for raw context, anomaly for anomaly outputs, forecast for "
            "predictions, and code_interpreter for missing derived calculations with non-empty insight_requests. Do not invent a fallback.\n"
            f"Visualization request: {request.message}\n"
            f"Original analytical request: {request_state.message}\n"
            f"Visual verification contract: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(target_insights, ensure_ascii=False)}\n"
            f"Authoritative visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Proof evidence bundle: {json.dumps(proof_bundle, ensure_ascii=False)}\n"
            f"Semantic composition inventory: {json.dumps(_visual_composition_inventory(inventory), ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            _StructuredVisualCompositionPlan,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title="Visual Composition",
            trace_summary="设计主视觉、高亮与辅助证据层",
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.composition",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            if parse_error is not None:
                raise parse_error
            composition = parsed.to_runtime()
            _validate_visual_composition(composition, inventory, proof_bundle=proof_bundle)
            return composition
        except (json.JSONDecodeError, ValueError) as exc:
            if schema_repair_attempt < 2:
                return await self._compose(
                    request,
                    inventory,
                    request_state,
                    verification,
                    repair_context=_planning_diagnostic(
                        exc,
                        stage="visual_composition",
                        allowed_values=_visual_composition_inventory(inventory),
                    ),
                    schema_repair_attempt=schema_repair_attempt + 1,
                )
            raise _semantic_error(
                ValueError(f"invalid visual composition: {_compact_validation_message(exc)}"),
                inventory,
                scope="visual_composition",
            ) from exc

    async def _encode(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        verification: VisualVerificationDecision,
        catalog: PresentationCatalog,
        composition: VisualCompositionPlan,
        repair_context: dict | None = None,
        schema_repair_attempt: int = 0,
    ) -> VisualizationPlan:
        if self._llm is None:
            raise RuntimeError("visual encoding requires an LLM")
        field_contract = _closed_visual_field_contract(inventory)
        proof_bundle = inventory.get("proof_bundle") if isinstance(inventory.get("proof_bundle"), list) else []
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You bind exact semantic fields to an already-fixed visual composition. You may not add, remove, duplicate, rename, or change a "
            "layer's family, layer_type, role, source_ref, interval_source_ref, goal, viewport, or evidence purpose. Return exactly one encoding "
            "for every supplied layer_id, or one genuine owner request. Choose fields only from the closed field contract for that layer's fixed "
            "source. Fill the schema's explicit slots: series/event_points/comparison/interval_overlay require x_field and y_field; band requires "
            "x_field, lower_field, and upper_field; reference_line requires value_field and optional label_field; annotation requires content_field, "
            "optional value_field, and optional x_field. Bind annotation.x_field when the source grounds a concrete event/observation position on "
            "the host domain; keep it null for a global or scalar metric. In particular, preserve a real timestamp/category emitted by "
            "code_interpreter for an important located result instead of reducing it to an unpositioned callout. An annotation x binding borrows "
            "the host axis and does not create a new domain. A reference_line never receives an x_field. Use series_field or label_field only when their grounded "
            "meanings help the fixed role; otherwise null. For a source-backed interval "
            "choose its exact start/end time fields from the fixed interval_source_ref. For a literal interval, copy grounded boundary values from "
            "the target Insight; do not calculate new business values. event_points consumes every source row. Do not use an encoding to simulate "
            "selection, filtering, joins, anomaly exclusion, or another transformation. Presentation choices here are limited to emphasis, line style, "
            "symbol, and axis. On repair, rebuild only rejected field bindings and do not redesign the composition.\n"
            f"Visualization request: {request.message}\n"
            f"Visual verification contract: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(_target_insight_inventory(request_state, verification.target_insight_ids), ensure_ascii=False)}\n"
            f"Fixed visual composition: {json.dumps(composition.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Closed encoding field contract: {json.dumps(field_contract, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        schema = _structured_visual_encoding_schema(composition, field_contract)
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            schema,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title="Visual Field Encoding",
            trace_summary="为固定图层绑定精确语义字段",
        )
        record_llm_token_usage(
            request_state,
            source="visualization.field_encoding",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
        try:
            if parse_error is not None:
                raise parse_error
            return _compile_visual_encoding_plan(
                parsed,
                composition,
                catalog,
                proof_bundle=proof_bundle,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            if schema_repair_attempt < 2:
                return await self._encode(
                    request,
                    inventory,
                    request_state,
                    verification,
                    catalog,
                    composition,
                    repair_context=_planning_diagnostic(
                        exc,
                        stage="visual_field_encoding",
                        allowed_values=field_contract,
                    ),
                    schema_repair_attempt=schema_repair_attempt + 1,
                )
            raise _semantic_error(
                ValueError(f"invalid visual field encoding: {_compact_validation_message(exc)}"),
                inventory,
                scope="visual_field_encoding",
            ) from exc


def _visual_composition_inventory(inventory: dict) -> dict:
    """Expose visual roles and capabilities while withholding encoding details."""

    sources = []
    for view in inventory.get("views", []) if isinstance(inventory.get("views"), list) else []:
        if not isinstance(view, dict):
            continue
        render_contract = view.get("render_contract") if isinstance(view.get("render_contract"), dict) else {}
        semantics = view.get("field_semantics") if isinstance(view.get("field_semantics"), dict) else {}
        sources.append({
            "source_ref": view.get("source_ref"),
            "name": view.get("name"),
            "purpose": view.get("purpose"),
            "grain": view.get("grain"),
            "point_count": int(render_contract.get("point_count") or 0),
            "allowed_consumers": list(render_contract.get("allowed_layer_types") or []),
            "semantic_roles": list(dict.fromkeys(str(value) for value in semantics.values() if str(value).strip())),
            "semantic_contract": view.get("semantic_contract"),
            "lineage": view.get("lineage"),
            "time_range": view.get("time_range"),
        })
    return {
        "sources": sources,
        "rules": [
            "primary supplies the complete contextual visual",
            "highlight is owned by the target Key Insight location or interval",
            "support exposes relevant anomaly, threshold, uncertainty, forecast, or comparison evidence",
            "reference_line and annotation attach grounded evidence without owning an x domain; annotation may borrow a grounded host x position",
            "one visual goal contains one shared temporal or categorical coordinate domain",
            "field selection belongs to the later encoding stage",
        ],
    }


def _validate_visual_composition(
    composition: VisualCompositionPlan,
    inventory: dict,
    *,
    proof_bundle: list[dict],
) -> None:
    if composition.required_data_request is not None:
        return
    contract = _closed_visual_field_contract(inventory)
    for goal in composition.visual_goals:
        for layer in goal.layers:
            source = contract.get(layer.source_ref)
            if source is None:
                raise ValueError(
                    f"visual composition layer {layer.layer_id!r} selected unknown semantic source {layer.source_ref!r}"
                )
            allowed = set(source.get("allowed_consumers") or [])
            if layer.layer_type == "interval_overlay":
                if "series" not in allowed:
                    raise ValueError(
                        f"interval overlay {layer.layer_id!r} requires a series-capable context source; "
                        f"{layer.source_ref!r} allows {sorted(allowed)}"
                    )
                if layer.interval_source_ref is not None:
                    boundary = contract.get(layer.interval_source_ref)
                    if boundary is None:
                        raise ValueError(
                            f"interval overlay {layer.layer_id!r} selected unknown boundary source "
                            f"{layer.interval_source_ref!r}"
                        )
                    if "interval_bounds" not in set(boundary.get("allowed_consumers") or []):
                        raise ValueError(
                            f"interval source {layer.interval_source_ref!r} does not expose interval_bounds"
                        )
            elif layer.layer_type not in allowed:
                raise ValueError(
                    f"composition layer {layer.layer_id!r} cannot use {layer.layer_type!r} with {layer.source_ref!r}; "
                    f"allowed consumers are {sorted(allowed)}"
                )
    _validate_visual_proof_bundle(composition.visual_goals, proof_bundle)


def _structured_visual_encoding_schema(
    composition: VisualCompositionPlan,
    source_contract: dict[str, dict[str, Any]],
) -> type[BaseModel]:
    """Build a small closed schema for the already-fixed layer set."""

    def literal(values: list[str]):
        return Literal.__getitem__(tuple(values))

    def nullable_literal(values: list[str]):
        return literal(values) | None if values else type(None)

    host_domain_by_layer_id = {
        layer.layer_id: (
            "categorical"
            if any(item.layer_type == "comparison" for item in goal.layers)
            else "temporal"
        )
        for goal in composition.visual_goals
        for layer in goal.layers
    }
    layer_models: list[type[BaseModel]] = []
    for index, layer in enumerate(
        item for goal in composition.visual_goals for item in goal.layers
    ):
        source = source_contract.get(layer.source_ref) or {}
        fields = [
            str(item.get("name") or "").strip()
            for item in source.get("fields", [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        if not fields:
            raise ValueError(f"composition source {layer.source_ref!r} exposes no encodable fields")
        number_fields = list(source.get("number_fields", []))
        category_fields = list(source.get("category_fields", []))
        model_fields: dict[str, Any] = {"layer_id": (literal([layer.layer_id]), ...)}
        base: type[BaseModel]
        if layer.layer_type == "reference_line":
            if not number_fields:
                raise ValueError(
                    f"reference guide source {layer.source_ref!r} exposes no numeric value field"
                )
            base = _EncodedReferenceLineLayer
            model_fields["value_field"] = (literal(number_fields), ...)
            model_fields["label_field"] = (nullable_literal(fields), ...)
        elif layer.layer_type == "annotation":
            base = _EncodedAnnotationLayer
            model_fields["content_field"] = (literal(fields), ...)
            model_fields["value_field"] = (nullable_literal(number_fields), ...)
            annotation_x_fields = list(dict.fromkeys(
                source.get("category_fields", [])
                if host_domain_by_layer_id[layer.layer_id] == "categorical"
                else source.get("time_fields", [])
            ))
            model_fields["x_field"] = (nullable_literal(annotation_x_fields), ...)
        else:
            x_fields = list(dict.fromkeys([
                *source.get("time_fields", []),
                *source.get("category_fields", []),
            ]))
            if not x_fields or not number_fields:
                raise ValueError(
                    f"composition source {layer.source_ref!r} does not expose compatible x and numeric fields"
                )
            model_fields.update({
                "x_field": (literal(x_fields), ...),
                "series_field": (nullable_literal(category_fields), ...),
                "label_field": (nullable_literal(fields), ...),
            })
            base = _EncodedXYLayer
        if layer.layer_type == "band":
            lower_fields = list(source.get("lower_fields", []))
            upper_fields = list(source.get("upper_fields", []))
            if not lower_fields or not upper_fields:
                raise ValueError(f"band source {layer.source_ref!r} lacks semantic lower/upper fields")
            base = _EncodedBandLayer
            model_fields["lower_field"] = (literal(lower_fields), ...)
            model_fields["upper_field"] = (literal(upper_fields), ...)
        elif layer.layer_type not in {"reference_line", "annotation"}:
            model_fields["y_field"] = (literal(number_fields), ...)
        if layer.layer_type == "interval_overlay" and layer.interval_source_ref is not None:
            base = _EncodedIntervalSourceLayer
            boundary = source_contract.get(layer.interval_source_ref) or {}
            time_fields = [
                str(item.get("name") or "").strip()
                for item in boundary.get("fields", [])
                if isinstance(item, dict)
                and item.get("data_type") == "time"
                and str(item.get("name") or "").strip()
            ]
            if not time_fields:
                raise ValueError(
                    f"interval source {layer.interval_source_ref!r} exposes no time boundary fields"
                )
            model_fields["interval_start_field"] = (literal(time_fields), ...)
            model_fields["interval_end_field"] = (literal(time_fields), ...)
        elif layer.layer_type == "interval_overlay":
            base = _EncodedIntervalLiteralLayer
        layer_models.append(create_model(
            f"FixedEncodedLayer{index}",
            __base__=base,
            **model_fields,
        ))

    layer_union = layer_models[0]
    for model in layer_models[1:]:
        layer_union = layer_union | model
    return create_model(
        "FixedCompositionEncodingPlan",
        __base__=_StructuredEncodedVisualPlan,
        layers=(list[layer_union], ...),
    )


def _compile_visual_encoding_plan(
    encoded: _StructuredEncodedVisualPlan,
    composition: VisualCompositionPlan,
    catalog: PresentationCatalog,
    *,
    proof_bundle: list[dict],
) -> VisualizationPlan:
    if encoded.required_data_request is not None:
        return VisualizationPlan(
            visual_goals=[],
            required_data_request=encoded.required_data_request.to_runtime(),
        )
    expected = {
        layer.layer_id: layer
        for goal in composition.visual_goals
        for layer in goal.layers
    }
    decisions = {layer.layer_id: layer for layer in encoded.layers}
    if len(decisions) != len(encoded.layers):
        raise ValueError("visual encoding layer_id values must be unique")
    if set(decisions) != set(expected):
        raise ValueError(
            "visual encoding must cover the fixed composition exactly; "
            f"missing={sorted(set(expected) - set(decisions))}, extra={sorted(set(decisions) - set(expected))}"
        )

    goals: list[VisualGoalIR] = []
    for goal in composition.visual_goals:
        layers: list[VisualLayerIR] = []
        for layer in goal.layers:
            decision = decisions[layer.layer_id]
            if layer.layer_type == "reference_line":
                encodings = [
                    VisualEncodingIR(channel="value", field=getattr(decision, "value_field")),
                ]
                if getattr(decision, "label_field") is not None:
                    encodings.append(
                        VisualEncodingIR(channel="label", field=getattr(decision, "label_field"))
                    )
            elif layer.layer_type == "annotation":
                encodings = [
                    VisualEncodingIR(channel="label", field=getattr(decision, "content_field")),
                ]
                if getattr(decision, "value_field") is not None:
                    encodings.append(
                        VisualEncodingIR(channel="value", field=getattr(decision, "value_field"))
                    )
                if getattr(decision, "x_field") is not None:
                    encodings.append(
                        VisualEncodingIR(channel="x", field=getattr(decision, "x_field"))
                    )
            else:
                encodings = [VisualEncodingIR(channel="x", field=getattr(decision, "x_field"))]
            if layer.layer_type == "band":
                encodings.extend([
                    VisualEncodingIR(channel="lower", field=getattr(decision, "lower_field")),
                    VisualEncodingIR(channel="upper", field=getattr(decision, "upper_field")),
                ])
            elif layer.layer_type not in {"reference_line", "annotation"}:
                encodings.append(VisualEncodingIR(channel="y", field=getattr(decision, "y_field")))
            if getattr(decision, "series_field", None) is not None:
                encodings.append(VisualEncodingIR(channel="series", field=decision.series_field))
            if layer.layer_type not in {"reference_line", "annotation"} and getattr(decision, "label_field", None) is not None:
                encodings.append(VisualEncodingIR(channel="label", field=decision.label_field))
            common = {
                "layer_type": layer.layer_type,
                "role": layer.role,
                "source_ref": layer.source_ref,
                "encodings": encodings,
                "emphasis": decision.emphasis,
                "line_style": decision.line_style,
                "symbol": decision.symbol,
                "axis": decision.axis,
                "label": layer.label,
            }
            if layer.layer_type == "interval_overlay" and layer.interval_source_ref is not None:
                visual_layer = IntervalSourceVisualLayerIR(
                    **common,
                    interval_source_ref=layer.interval_source_ref,
                    interval_start_field=getattr(decision, "interval_start_field"),
                    interval_end_field=getattr(decision, "interval_end_field"),
                )
            elif layer.layer_type == "interval_overlay":
                visual_layer = IntervalLiteralVisualLayerIR(
                    **common,
                    interval_start_value=getattr(decision, "interval_start_value"),
                    interval_end_value=getattr(decision, "interval_end_value"),
                )
            else:
                visual_layer = {
                    "series": SeriesVisualLayerIR,
                    "event_points": EventPointsVisualLayerIR,
                    "band": BandVisualLayerIR,
                    "comparison": ComparisonVisualLayerIR,
                    "reference_line": ReferenceLineVisualLayerIR,
                    "annotation": AnnotationVisualLayerIR,
                }[layer.layer_type](**common)
            layers.append(visual_layer)
        goals.append(VisualGoalIR(
            purpose=goal.purpose,
            title=goal.title,
            priority=goal.priority,
            summary=goal.summary,
            required_roles=[layer.role for layer in goal.layers],
            show_legend=goal.show_legend,
            tooltip=goal.tooltip,
            enable_zoom=goal.enable_zoom,
            viewport_start=goal.viewport_start,
            viewport_end=goal.viewport_end,
            y_scale=goal.y_scale,
            layers=layers,
        ))
    plan = _StructuredVisualizationPlan(visual_goals=goals, required_data_request=None)
    return _compile_visualization_plan(plan, catalog, proof_bundle=proof_bundle)


def _compile_visualization_plan(
    plan: _StructuredVisualizationPlan,
    catalog: PresentationCatalog,
    *,
    proof_bundle: list[dict] | None = None,
) -> VisualizationPlan:
    if plan.required_data_request is not None:
        return VisualizationPlan(
            visual_goals=[],
            required_data_request=plan.required_data_request.to_runtime(),
        )
    _validate_nonredundant_visual_layers(plan.visual_goals)
    _validate_visual_proof_bundle(plan.visual_goals, proof_bundle or [])
    return VisualizationPlan(
        visual_goals=[_compile_visual_goal(goal, catalog) for goal in plan.visual_goals],
        required_data_request=None,
    )


def _validate_nonredundant_visual_layers(goals: list[VisualGoalIR]) -> None:
    """Reject relabeled redraws of the same rows and encoding.

    An interval overlay is intentionally a filtered redraw. Other layers with
    the same source and channel bindings consume exactly the same records, so a
    narrower role label cannot turn a complete series into selected events.
    The LLM must re-plan against a source whose grain actually matches the role.
    """

    for goal in goals:
        seen: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
        for layer in goal.layers:
            if layer.layer_type == "interval_overlay":
                continue
            fingerprint = (
                layer.source_ref,
                tuple(sorted((item.channel, item.field) for item in layer.encodings)),
            )
            previous = seen.get(fingerprint)
            if previous is not None:
                raise ValueError(
                    f"duplicate visual layer consumption: roles {previous!r} and {layer.role!r} redraw "
                    f"the same source and fields {fingerprint[0]!r}. Keep one truthful layer or use a "
                    "semantic source whose row grain matches the narrower role."
                )
            seen[fingerprint] = layer.role


def _validate_visual_proof_bundle(
    goals: list[VisualGoalIR],
    proof_bundle: list[dict],
) -> None:
    """Require every selected proof source to remain visible in the final IR."""

    required_refs = {
        str(item.get("source_ref") or "").strip()
        for item in proof_bundle
        if isinstance(item, dict) and str(item.get("source_ref") or "").strip()
    }
    if not required_refs:
        return
    used_refs = {
        ref
        for goal in goals
        for layer in goal.layers
        for ref in (layer.source_ref, getattr(layer, "interval_source_ref", None))
        if ref
    }
    missing = sorted(required_refs - used_refs)
    if missing:
        raise ValueError(
            f"visual IR does not expose required proof evidence sources: {missing}"
        )


def _compile_visual_goal(goal: VisualGoalIR, catalog: PresentationCatalog) -> VisualGoal:
    presentation: dict[str, Any] = {
        "legend": {"show": goal.show_legend},
        "tooltip": (
            {"show": False}
            if goal.tooltip == "none"
            else {"trigger": goal.tooltip}
        ),
        "grid": {"containLabel": True},
        "yAxis": [{"type": "log" if goal.y_scale == "log" else "value"}],
    }
    if any(
        layer.axis == "secondary" and layer.layer_type != "annotation"
        for layer in goal.layers
    ):
        presentation["yAxis"].append({"type": "value"})
    if goal.enable_zoom:
        zoom_window = {}
        if goal.viewport_start is not None and goal.viewport_end is not None:
            zoom_window = {
                "startValue": goal.viewport_start,
                "endValue": goal.viewport_end,
            }
        presentation["dataZoom"] = [
            {"type": "inside", "filterMode": "filter", **zoom_window},
            {"type": "slider", "filterMode": "filter", **zoom_window},
        ]
    return VisualGoal(
        purpose=goal.purpose,
        title=goal.title,
        priority=goal.priority,
        summary=goal.summary,
        required_roles=goal.required_roles,
        presentation=presentation,
        layers=[_compile_visual_layer(layer, catalog) for layer in goal.layers],
    )


def _compile_visual_layer(
    layer: VisualLayerIR,
    catalog: PresentationCatalog,
) -> VisualLayerPlan:
    _validate_visual_ir_source(layer, catalog)
    mark = {
        "series": "line",
        "event_points": "point",
        "band": "band",
        "interval_overlay": "line",
        "comparison": "bar",
        "reference_line": "rule",
        "annotation": "annotation",
    }[layer.layer_type]
    encoding = {item.channel: item.field for item in layer.encodings}
    transforms: list[VisualFilterTransform] = []
    provenance_source_refs: list[str] = []
    if layer.layer_type == "interval_overlay":
        interval_source_ref = getattr(layer, "interval_source_ref", None)
        if interval_source_ref is not None:
            provenance_source_refs.append(interval_source_ref)
            start, end = _interval_boundaries(
                catalog,
                interval_source_ref,
                str(getattr(layer, "interval_start_field")),
                str(getattr(layer, "interval_end_field")),
            )
        else:
            start = getattr(layer, "interval_start_value")
            end = getattr(layer, "interval_end_value")
        transforms.append(VisualFilterTransform(
            field=encoding["x"],
            operator="between",
            value=[start, end],
        ))

    line_width = {"subtle": 1.5, "normal": 2, "strong": 3}[layer.emphasis]
    opacity = {"subtle": 0.65, "normal": 0.9, "strong": 1.0}[layer.emphasis]
    presentation: dict[str, Any] = {
        "opacity": opacity,
        "emphasis": {"focus": "series"},
    }
    if mark in {"line", "area", "rule"}:
        presentation["lineStyle"] = {
            "width": line_width,
            "type": layer.line_style,
        }
        presentation["showSymbol"] = layer.symbol != "none"
    if layer.symbol != "none":
        presentation["symbol"] = layer.symbol
        presentation["symbolSize"] = 12 if layer.emphasis == "strong" else 9
    if layer.axis == "secondary" and layer.layer_type != "annotation":
        presentation["yAxisIndex"] = 1
    return VisualLayerPlan(
        role=layer.role,
        source_ref=layer.source_ref,
        mark=mark,
        encoding=encoding,
        transform=transforms,
        presentation=presentation,
        provenance_source_refs=provenance_source_refs,
        label=layer.label,
    )


def _validate_visual_ir_source(
    layer: VisualLayerIR,
    catalog: PresentationCatalog,
) -> None:
    """Validate renderer-independent intent against an executable data contract."""

    source = catalog.resolve(layer.source_ref)
    if source.kind != "view":
        # Legacy catalog sources remain materializable, but only typed data views
        # expose the closed render contract used by the new planning boundary.
        return

    fields = {
        str(item.get("name")): str(item.get("data_type") or "unknown")
        for item in source.value.schema_fields
        if isinstance(item, dict) and item.get("name")
    }
    encodings = {item.channel: item.field for item in layer.encodings}
    unknown = sorted(set(encodings.values()) - set(fields))
    if unknown:
        raise ValueError(
            f"visual IR layer '{layer.role}' references unavailable fields {unknown} "
            f"from {source.ref}; available fields: {sorted(fields)}"
        )

    rows = list(source.value.rows or ([source.value.scalar] if source.value.scalar else []))
    point_count = len(rows)
    minimum_points = 2 if layer.layer_type in {"series", "band", "interval_overlay"} else 1
    if point_count < minimum_points:
        raise ValueError(
            f"visual IR {layer.layer_type} layer '{layer.role}' requires at least "
            f"{minimum_points} grounded points; source {source.ref} has {point_count}"
        )

    numeric_channels = {"y", "value", "lower", "upper"}
    invalid_numeric = sorted(
        f"{channel}={field}"
        for channel, field in encodings.items()
        if channel in numeric_channels and fields.get(field) != "number"
    )
    if invalid_numeric:
        raise ValueError(
            f"visual IR layer '{layer.role}' has incompatible numeric encodings "
            f"{invalid_numeric}"
        )

    x_field = encodings.get("x")
    x_type = fields.get(x_field or "")
    expected_x_types = (
        {"category", "string", "boolean"}
        if layer.layer_type == "comparison"
        else {"time", "category", "string", "boolean"}
        if layer.layer_type == "annotation"
        else {"time"}
    )
    if x_field and x_type not in expected_x_types:
        raise ValueError(
            f"visual IR {layer.layer_type} layer '{layer.role}' has incompatible "
            f"x encoding {x_field}={x_type}; expected {sorted(expected_x_types)}"
        )


def _interval_boundaries(
    catalog: PresentationCatalog,
    source_ref: str,
    start_field: str,
    end_field: str,
) -> tuple[Any, Any]:
    source = catalog.resolve(source_ref)
    if source.kind != "view":
        raise ValueError(f"interval boundary source '{source_ref}' must be a semantic view")
    records = list(source.value.rows or ([source.value.scalar] if source.value.scalar else []))
    pairs = {
        (record.get(start_field), record.get(end_field))
        for record in records
        if record.get(start_field) is not None and record.get(end_field) is not None
    }
    if len(pairs) != 1:
        raise ValueError(
            f"interval boundary source '{source_ref}' must expose exactly one distinct "
            f"{start_field}/{end_field} pair"
        )
    return next(iter(pairs))


def _planning_diagnostic(
    error: Exception,
    *,
    stage: str,
    allowed_values: Any = None,
) -> dict:
    raw_message = str(error)
    message = _compact_validation_message(error)
    lowered = raw_message.casefold()
    if "requires at least two grounded points" in lowered:
        code = "INSUFFICIENT_SERIES_POINTS"
    elif "proof obligation" in lowered or "proof mapping" in lowered:
        code = "INVALID_PROOF_MAPPING"
    elif "semantic view" in lowered and (
        "extra inputs" in lowered
        or "fields" in lowered
        or "events" in lowered
    ):
        code = "INVALID_SEMANTIC_VIEW_SHAPE"
    elif "unavailable field" in lowered or "unavailable in every record" in lowered:
        code = "UNKNOWN_FIELD"
    elif "literal_error" in lowered or "input should be" in lowered:
        code = "CLOSED_VOCABULARY_VIOLATION"
    elif "duplicate visual layer consumption" in lowered:
        code = "REDUNDANT_LAYER_CONSUMPTION"
    elif "unknown presentation source" in lowered:
        code = "UNKNOWN_SOURCE"
    elif "incompatible" in lowered:
        code = "INCOMPATIBLE_VISUAL_DOMAIN"
    elif "interval" in lowered and "bound" in lowered:
        code = "INVALID_INTERVAL_BOUNDARY"
    else:
        code = "VISUAL_PLAN_EXECUTION_FAILED"
    diagnostic = {
        "stage": stage,
        "error_code": code,
        "message": message[:400],
    }
    repair_instructions = {
        "UNKNOWN_FIELD": (
            "Delete or completely rebuild every rejected layer. Copy exact field identifiers from allowed_values for "
            "that same source_ref; never repeat conventional timestamp/value names that are absent. If the source is "
            "already consumed as interval_source_ref, remove any redundant invalid graphical layer."
        ),
        "CLOSED_VOCABULARY_VIOLATION": (
            "Delete or completely rebuild every rejected layer. Choose only source refs, field identifiers, and layer "
            "consumers admitted by allowed_values; do not preserve an out-of-contract string from the rejected candidate."
        ),
        "INVALID_PROOF_MAPPING": (
            "Use every missing semantic proof source once as either a valid graphical layer or an interval_source_ref; "
            "do not create an additional layer after the source is already consumed."
        ),
        "INCOMPATIBLE_VISUAL_DOMAIN": (
            "Recompose the complete visual goal. Keep one shared temporal or categorical host domain per goal; attach compatible "
            "scalar evidence as reference_line/annotation guides that own no x domain; a positioned annotation may only borrow a "
            "grounded x coordinate compatible with that host. Split genuinely independent domains into "
            "separate goals. Do not ask field encoding to repair a composition decision."
        ),
        "REDUNDANT_LAYER_CONSUMPTION": (
            "Remove relabeled redraws of the same source and fields. Keep the complete context layer, then bind located markers to "
            "their target/anomaly semantic source or use one interval_overlay for the selected segment."
        ),
    }
    if code in repair_instructions:
        diagnostic["repair_instruction"] = repair_instructions[code]
    if allowed_values is not None:
        diagnostic["allowed_values"] = allowed_values
    return diagnostic


def _compact_validation_message(error: Exception, *, max_items: int = 6) -> str:
    """Summarize union-schema failures without reinjecting candidate payloads.

    Pydantic reports every rejected union branch and includes the full invalid
    input in its text form. That output is useful to a Python developer but is
    actively harmful as LLM repair context: the meaningful branch error is
    buried under dozens of literal mismatches and large source documents can be
    repeated verbatim. Keep actionable non-literal errors first and never copy
    ``input`` values into the control plane.
    """

    if not isinstance(error, ValidationError):
        return str(error)
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    actionable = [item for item in errors if item.get("type") != "literal_error"]
    selected = actionable[:max_items] or errors[:max_items]
    parts: list[str] = []
    for item in selected:
        location = ".".join(str(part) for part in item.get("loc", ()))
        detail = str(item.get("msg") or item.get("type") or "invalid value")
        parts.append(f"{location}: {detail}" if location else detail)
    suffix = f"; {len(errors) - len(selected)} additional branch error(s) omitted" if len(errors) > len(selected) else ""
    return f"{len(errors)} validation error(s): " + "; ".join(parts) + suffix


async def _invoke_structured(
    llm,
    schema,
    messages,
    *,
    timeout_seconds: float,
    trace_title: str,
    trace_summary: str | None = None,
):
    response = None
    content = ""
    try:
        async with llm_trace_span(
            trace_title,
            summary=trace_summary,
            messages=messages,
        ) as trace_span:
            if hasattr(llm, "with_structured_output"):
                runnable = llm.with_structured_output(
                    schema,
                    method="json_schema",
                    include_raw=True,
                    strict=True,
                )
                bundle = await asyncio.wait_for(
                    runnable.ainvoke(messages),
                    timeout=timeout_seconds,
                )
                if isinstance(bundle, dict):
                    response = bundle.get("raw")
                    content = _llm_content(response)
                    if trace_span is not None:
                        trace_span.attach_response(
                            response,
                            messages=messages,
                            output_text=content,
                        )
                    parsed = bundle.get("parsed")
                    if parsed is None:
                        error = bundle.get("parsing_error") or ValueError(
                            "structured output was not parsed"
                        )
                        raise ValueError(str(error))
                    parsed = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
                    return response, content, parsed, None
                parsed = bundle if isinstance(bundle, schema) else schema.model_validate(bundle)
                content = _llm_content(bundle)
                if trace_span is not None:
                    trace_span.attach_response(
                        bundle,
                        messages=messages,
                        output_text=content,
                    )
                return bundle, content, parsed, None
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout_seconds,
            )
            content = _llm_content(response)
            if trace_span is not None:
                trace_span.attach_response(
                    response,
                    messages=messages,
                    output_text=content,
                )
            parsed = schema.model_validate(json.loads(_json_object(content)))
            return response, content, parsed, None
    except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
        return response, content, None, exc


def _llm_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def _expand_source_preferences(
    refs: list[str], catalog: PresentationCatalog,
) -> tuple[set[str], set[str]]:
    """Compatibility wrapper around the catalog-owned artifact-to-view resolver."""
    return catalog.expand_preferences(refs)


def _semantic_visualization_constraints(constraints: dict | None) -> dict:
    """Return only user/domain constraints safe to expose to visual planners."""

    if not isinstance(constraints, dict):
        return {}
    return {
        str(key): value
        for key, value in constraints.items()
        if str(key) not in _VISUALIZATION_CONTROL_CONSTRAINT_KEYS
        and not str(key).startswith("_")
    }


def _resolve_visualization_lineage_refs(
    refs: list[str], request_state: RequestStateModel,
) -> list[str]:
    """Dereference an existing presentation artifact to its grounded data lineage."""
    visualizations = {
        str(getattr(visualization, "visualization_id", "") or ""): visualization
        for visualization in request_state.visualizations
    }
    resolved: list[str] = []
    for raw_ref in refs:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        visualization_id = ref.removeprefix("visualization:")
        visualization = visualizations.get(visualization_id)
        if visualization is None:
            resolved.append(ref)
            continue
        resolved.extend(
            str(source_ref).strip()
            for source_ref in getattr(visualization, "source_refs", []) or []
            if str(source_ref).strip()
        )
    return list(dict.fromkeys(resolved))


def _verified_insight_inventory(request_state: RequestStateModel) -> list[dict]:
    # Verification selects semantic Insight identifiers, whereas later planning
    # needs source refs to materialize their lineage. Keep those namespaces
    # separate at the decision boundary so the model cannot mistake a
    # presentation/source ref (``insight:<id>``) for an ``insight_id`` value.
    inventory = []
    for insight in request_state.insight_set.insights:
        if insight.status != "verified":
            continue
        item = _insight_prompt_view(insight)
        item.pop("source_ref", None)
        inventory.append(item)
    return inventory


def _data_source_inventory(inventory: dict) -> dict:
    """Keep only reference-backed data contracts when Insight semantics are supplied separately."""

    return {
        **inventory,
        "sources": [
            source
            for source in inventory.get("sources", [])
            if source.get("kind") not in {"insight", "insight_item"}
        ],
    }


def _projection_source_inventory(inventory: dict) -> dict:
    """Expose Insight access paths without duplicating target Insight content."""

    semantic_keys = {
        "statement", "value", "items", "item", "locator", "timestamp", "label",
    }
    sources = []
    for source in inventory.get("sources", []):
        if source.get("kind") in {"insight", "insight_item"}:
            source = {
                key: value
                for key, value in source.items()
                if key not in semantic_keys
            }
        sources.append(source)
    return {**inventory, "sources": sources}


def _validate_projection_consumes_target_claims(
    projection: SemanticProjectionPlan,
    *,
    evidence_consumption: VisualEvidenceConsumption | None,
    inventory: dict,
    target_insight_ids: list[str],
) -> None:
    """Keep a graphically located target claim authoritative across stages.

    Evidence binding deliberately keeps the target Insight separate from the
    artifacts used to inspect it.  Without enforcing that hand-off, a later LLM
    can accidentally use coincident raw/anomaly values as if they owned the
    target interval.  This validator requests LLM re-planning; it never creates
    or substitutes a deterministic view.
    """

    if evidence_consumption is None or evidence_consumption.decision != "ready":
        return
    selected = {item.source_ref for item in evidence_consumption.source_uses}
    target_refs = {f"insight:{insight_id}" for insight_id in target_insight_ids}
    source_by_ref = {
        str(source.get("source_ref") or "").strip(): source
        for source in inventory.get("sources", [])
        if isinstance(source, dict) and str(source.get("source_ref") or "").strip()
    }

    required: set[str] = set()
    for source_ref in selected & target_refs:
        source = source_by_ref.get(source_ref)
        if source is None or source.get("kind") != "insight":
            continue
        # A scalar conclusion may remain in the textual answer while its
        # evidence supplies the chart context. Only claims with their own
        # grounded locator or located item collection must become a visual
        # boundary/event source.
        if not source.get("locator_fields") and not int(source.get("item_count") or 0):
            continue
        field_types = [
            str(field.get("data_type") or "")
            for field in source.get("schema_fields", [])
            if isinstance(field, dict)
        ]
        has_number = "number" in field_types
        time_count = field_types.count("time")
        has_category = any(item in {"category", "string"} for item in field_types)
        if (time_count and has_number) or (has_category and has_number) or time_count >= 2:
            required.add(source_ref)

    projected = {view.source_ref for view in projection.semantic_views}
    missing = sorted(required - projected)
    if missing:
        raise ValueError(
            "semantic projection omitted graphically located target Key Insight source(s) selected by "
            f"evidence consumption: {missing}. Project each exact target source for its claim, annotation, "
            "or interval-boundary role; coincident context/anomaly values do not own the target claim."
        )


def _visual_evidence_fact_packet(
    request_state: RequestStateModel,
    insight_ids: list[str],
    inventory: dict,
) -> dict:
    """Build a compact, value-bearing packet for Key Insight source consumption.

    Projection schemas and query text are intentionally excluded: this stage
    decides whether the upstream meaning is executable, not how to address its
    fields. Exact small artifact facts and calculation traces remain visible so
    an LLM can detect contradictions before any chart is authored.
    """

    wanted = set(insight_ids)
    target_insights = []
    evidence_refs: list[tuple[str, str]] = []
    for insight in request_state.insight_set.insights:
        if insight.status != "verified" or insight.insight_id not in wanted:
            continue
        item = _insight_prompt_view(insight)
        item["method"] = insight.method
        item["calculation_trace"] = _bounded_semantic_value(insight.calculation_trace)
        target_insights.append(item)
        evidence_refs.extend(
            (ref.source_type, ref.source_id)
            for ref in insight.evidence_refs
        )

    compact_sources = []
    source_keys = {
        "source_ref", "kind", "name", "shape", "row_count", "schema_fields",
        "field_semantics", "lineage", "time_range", "materialization_complete",
        "grounded_preview", "status", "insight_type", "insight_key",
        "derived_from", "evidence_refs", "item_count", "render_contract",
        "semantic_contract",
    }
    for source in inventory.get("sources", []):
        if not isinstance(source, dict):
            continue
        compact_sources.append({
            key: _bounded_semantic_value(value)
            for key, value in source.items()
            if key in source_keys and value is not None
        })

    artifacts = []
    seen_artifacts: set[tuple[str, str]] = set()
    for source_type, source_id in evidence_refs:
        key = (source_type, source_id)
        if key in seen_artifacts:
            continue
        seen_artifacts.add(key)
        if source_type == "analysis":
            artifact = request_state.analysis_artifacts.get(source_id)
            if artifact is not None:
                artifacts.append({
                    "source_ref": f"analysis:{source_id}",
                    "kind": "analysis",
                    "analysis_goal": artifact.analysis_goal,
                    "status": artifact.status,
                    "input_row_count": artifact.input_row_count,
                    "input_source_refs": artifact.input_source_refs,
                    "summary": artifact.summary,
                    "computed_insights": [
                        _bounded_semantic_value(item.model_dump(mode="json"))
                        for item in artifact.computed_insights[:12]
                    ],
                    "derived_evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "name": item.name,
                            "shape": item.shape,
                            "row_count": len(item.rows),
                            "scalar": _bounded_semantic_value(item.scalar),
                            "lineage": item.lineage,
                            "transform_summary": item.transform_summary,
                        }
                        for item in artifact.derived_evidence[:12]
                    ],
                    "diagnostics": _bounded_semantic_value(artifact.diagnostics),
                })
        elif source_type == "anomaly":
            artifact = request_state.anomaly_artifacts.get(source_id)
            if artifact is not None:
                artifacts.append({
                    "source_ref": f"anomaly:{source_id}",
                    "kind": "anomaly",
                    "detector_name": artifact.detector_name,
                    "anomaly_count": len(artifact.anomaly_points),
                    "anomaly_points": [
                        _bounded_semantic_value(item)
                        for item in artifact.anomaly_points[:20]
                    ],
                    "anomaly_span_count": len(artifact.anomaly_spans),
                    "anomaly_spans": [
                        _bounded_semantic_value(item)
                        for item in artifact.anomaly_spans[:12]
                    ],
                    "score_count": len(artifact.scores),
                    "diagnostics": _bounded_semantic_value(artifact.diagnostics),
                })
        elif source_type in {"query", "evidence"}:
            artifact = request_state.database_evidence_artifacts.get(source_id)
            if artifact is not None:
                rows = artifact.data.get("rows", []) if isinstance(artifact.data, dict) else []
                artifacts.append({
                    "source_ref": f"evidence:{source_id}",
                    "kind": "database_evidence",
                    "result_type": artifact.result_type,
                    "summary": artifact.summary,
                    "columns": artifact.columns,
                    "row_count": len(rows),
                    "materialization_complete": bool(
                        (artifact.diagnostics or {}).get("is_full_fidelity", True)
                    ),
                    "query_execution": _bounded_semantic_value(
                        (artifact.metadata or {}).get("query_execution")
                    ),
                })
    return {
        "target_insights": target_insights,
        "related_sources": compact_sources,
        "referenced_artifacts": artifacts,
    }


def _target_insight_inventory(request_state: RequestStateModel, insight_ids: list[str]) -> list[dict]:
    wanted = set(insight_ids)
    return [
        _insight_prompt_view(insight)
        for insight in request_state.insight_set.insights
        if insight.status == "verified" and insight.insight_id in wanted
    ]


def _insight_prompt_view(insight) -> dict:
    items = list(insight.items)
    visible_items = items if len(items) <= 12 else [*items[:6], *items[-6:]]
    return {
        "source_ref": f"insight:{insight.insight_id}",
        "insight_id": insight.insight_id,
        "insight_key": insight.insight_key,
        "name": insight.name,
        "insight_type": insight.insight_type,
        "statement": insight.statement,
        "value": _bounded_semantic_value(insight.value),
        "value_shape": insight.value_shape,
        "semantic_class": insight.semantic_class,
        "derivation": insight.derivation,
        "unit": insight.unit,
        "subject": insight.subject,
        "dimensions": _bounded_semantic_value(insight.dimensions),
        "time_range": _bounded_semantic_value(insight.time_range),
        "evidence_refs": [item.model_dump(mode="json") for item in insight.evidence_refs[:8]],
        "items": [
            _bounded_semantic_value(item.model_dump(mode="json", exclude_none=True))
            for item in visible_items
        ],
        "item_count": len(items),
        "derived_from": insight.derived_from,
        "quality_flags": insight.quality_flags,
    }


def _bounded_semantic_value(value, *, depth: int = 0):
    if depth >= 6:
        return "[nested content available through source_ref]"
    if isinstance(value, dict):
        return {
            str(key): _bounded_semantic_value(item, depth=depth + 1)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        visible = items if len(items) <= 12 else [*items[:6], *items[-6:]]
        return [_bounded_semantic_value(item, depth=depth + 1) for item in visible]
    if isinstance(value, str):
        return value[:1000]
    return value


def _candidate_audit_view(
    visualizations: list[VisualizationPayload],
    inventory: dict | None = None,
) -> list[dict]:
    view_contracts = {
        str(view.get("source_ref") or ""): view
        for view in (inventory or {}).get("views", [])
        if isinstance(view, dict) and str(view.get("source_ref") or "").strip()
    }
    result: list[dict] = []
    for visualization in visualizations:
        payload = visualization.model_dump(
            mode="json",
            exclude={"datasets", "layers", "bindings", "accessibility"},
        )
        payload["accessibility"] = {
            "description": visualization.accessibility.description,
            "table_columns": visualization.accessibility.table_columns,
        }
        payload["datasets"] = [
            {
                "dataset_id": dataset.dataset_id,
                "source_ref": dataset.source_ref,
                "dimensions": [item.model_dump(mode="json") for item in dataset.dimensions],
                "series": [
                    {
                        "series_id": series.series_id,
                        "name": series.name,
                        "role": series.role,
                        "point_count": len(series.points),
                    }
                    for series in dataset.series
                ],
            }
            for dataset in visualization.datasets
        ]
        payload["layers"] = []
        for layer in visualization.layers:
            item = layer.model_dump(mode="json", exclude={"points"})
            contract = view_contracts.get(layer.source_ref, {})
            semantics = (
                contract.get("field_semantics")
                if isinstance(contract.get("field_semantics"), dict)
                else {}
            )
            schema = {
                str(field.get("name")): str(field.get("data_type") or "unknown")
                for field in contract.get("schema_fields", [])
                if isinstance(field, dict) and str(field.get("name") or "").strip()
            }
            item["visual_family"] = (
                "guide" if layer.mark in {"rule", "annotation", "rect"} else "data"
            )
            item["encoding_semantics"] = {
                channel: [
                    {
                        "field": field,
                        "semantic_role": semantics.get(field),
                        "data_type": schema.get(field),
                    }
                    for field in (value if isinstance(value, list) else [value])
                ]
                for channel, value in layer.encoding.items()
            }
            item["source_semantic_contract"] = contract.get("semantic_contract")
            payload["layers"].append(item)
        payload["binding_count"] = len(visualization.bindings)
        result.append(payload)
    return result


def _dependency_signature(requirement: VisualizationEvidenceRequest | dict) -> str:
    payload = (
        requirement.model_dump(mode="json", exclude_none=True)
        if isinstance(requirement, VisualizationEvidenceRequest)
        else requirement
    )
    stable = {
        "required_action": payload.get("required_action"),
        "purpose": payload.get("purpose"),
        "required_shape": payload.get("required_shape"),
        "required_fields": payload.get("required_fields") or [],
        "required_properties": payload.get("required_properties") or [],
        "input_source_refs": payload.get("input_source_refs") or [],
        "insight_keys": sorted(
            str(item.get("insight_key") or item.get("name") or "")
            for item in payload.get("insight_requests") or []
            if isinstance(item, dict)
        ),
    }
    return json.dumps(stable, ensure_ascii=True, sort_keys=True, default=str)


def _dependency_was_attempted(
    requirement: VisualizationEvidenceRequest,
    request_state: RequestStateModel,
) -> bool:
    signature = _dependency_signature(requirement)
    observations = list(request_state.observations or [])
    for index in range(len(observations) - 1, -1, -1):
        observation = observations[index]
        if observation.tool_name != "visualization" or not observation.success:
            continue
        payload = observation.payload if isinstance(observation.payload, dict) else {}
        previous = payload.get("required_data_request")
        if not isinstance(previous, dict) or _dependency_signature(previous) != signature:
            continue
        return any(
            later.success and later.tool_name == requirement.required_action
            for later in observations[index + 1:]
        )
    return False


def _json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    return text[start:end + 1]


def _visual_contract(request_state: RequestStateModel) -> list[dict]:
    """Expose authoritative user-visible visual deliverables without storage details."""
    contract = request_state.task_contract
    if contract is None:
        return []
    visual_outputs = []
    for output in contract.required_outputs:
        if not output.required:
            continue
        if output.output_type != "visualization" and output.evidence_kind != "visualization":
            continue
        visual_outputs.append({
            "id": output.id,
            "description": output.description,
            "success_criteria": output.success_criteria,
            "measures": output.measures,
            "dimensions": output.dimensions,
            "time_scope": output.time_scope,
        })
    return visual_outputs


def _default_source_paths(inventory: dict) -> dict[str, list[str]]:
    """Expose exact executable top-level paths as repair feedback to the LLM."""

    result: dict[str, list[str]] = {}
    for source in inventory.get("sources", []) if isinstance(inventory.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        ref = str(source.get("source_ref") or "").strip()
        fields = source.get("schema_fields") if isinstance(source.get("schema_fields"), list) else []
        paths = [
            f"$.{field.get('name')}"
            for field in fields
            if isinstance(field, dict) and str(field.get("name") or "").strip()
        ]
        if ref and paths:
            result[ref] = paths
    return result


def _closed_visual_field_contract(inventory: dict) -> dict[str, dict[str, Any]]:
    """Return the closed executable source contract for every semantic view.

    Semantic view field names are runtime identifiers, while Insight statements
    and labels are natural language. Keeping this compact contract adjacent to
    the IR request prevents a planner from treating descriptive text as a
    column name without choosing a chart on the tool's behalf.
    """

    contract: dict[str, dict[str, Any]] = {}
    for view in inventory.get("views", []) if isinstance(inventory.get("views"), list) else []:
        if not isinstance(view, dict):
            continue
        ref = str(view.get("source_ref") or "").strip()
        fields = view.get("schema_fields") if isinstance(view.get("schema_fields"), list) else []
        semantics = view.get("field_semantics") if isinstance(view.get("field_semantics"), dict) else {}
        entries = [
            {
                "name": str(field["name"]),
                "data_type": str(field.get("data_type") or "unknown"),
                "semantic_role": str(semantics.get(str(field["name"])) or ""),
            }
            for field in fields
            if isinstance(field, dict) and str(field.get("name") or "").strip()
        ]
        render_contract = (
            view.get("render_contract")
            if isinstance(view.get("render_contract"), dict)
            else {}
        )
        if ref and entries:
            contract[ref] = {
                "fields": entries,
                "point_count": int(render_contract.get("point_count") or 0),
                "allowed_consumers": list(render_contract.get("allowed_layer_types") or []),
                "time_fields": list(render_contract.get("time_fields") or []),
                "number_fields": list(render_contract.get("number_fields") or []),
                "category_fields": list(render_contract.get("category_fields") or []),
                "lower_fields": list(render_contract.get("lower_fields") or []),
                "upper_fields": list(render_contract.get("upper_fields") or []),
            }
    return contract


def _resolve_semantic_evidence_bundle(
    verification: VisualVerificationDecision,
    projection: SemanticProjectionPlan,
    semantic_refs: list[str],
) -> list[dict]:
    """Expose every grounded projection view as required chart proof.

    Proof-obligation identifiers are local to verification.  Treating an
    LLM-authored free string as a foreign key in another LLM response caused
    repeated namespace drift.  Projection instead emits the minimal collective
    proof view set; the chart compiler already enforces that every source in
    this bundle remains visible in the final IR.
    """

    if not verification.proof_obligations:
        return []
    refs_by_view_id = {
        ref.removeprefix("semantic:"): ref
        for ref in semantic_refs
        if ref.startswith("semantic:")
    }
    if len(refs_by_view_id) != len(projection.semantic_views):
        raise ValueError("semantic projection proof views are not uniquely addressable")
    return [
        {
            "view_id": view.view_id,
            "purpose": view.purpose,
            "source_ref": refs_by_view_id[view.view_id],
        }
        for view in projection.semantic_views
    ]


def _missing_evidence_required(requirement: VisualizationEvidenceRequest, inventory: dict) -> StructuredToolError:
    action = requirement.required_action
    modes = {
        "sql_query": "full_timeseries_required",
        "anomaly": "anomaly_evidence_required",
        "forecast": "forecast_evidence_required",
        "code_interpreter": "derived_evidence_required",
    }
    error_types = {
        "sql_query": "visualization_data_incomplete",
        "anomaly": "visualization_anomaly_missing",
        "forecast": "visualization_forecast_missing",
        "code_interpreter": "visualization_analysis_missing",
    }
    messages = {
        "sql_query": "Visualization requires additional full-fidelity database evidence.",
        "anomaly": "Visualization requires an authoritative anomaly artifact.",
        "forecast": "Visualization requires an authoritative forecast artifact.",
        "code_interpreter": "Visualization requires calculated semantic evidence.",
    }
    message = messages[action]
    contract = requirement.model_dump(mode="json", exclude_none=True)
    contract["mode"] = modes[action]
    if action == "sql_query":
        contract["constraints"] = {"evidence_shape": "raw_timeseries", "full_fidelity": True}
    return StructuredToolError(
        message,
        error_type=error_types[action],
        retryable=True,
        recommended_next_action=action,
        diagnostics={"required_data_request": contract},
        validation_failure={
            "scope": "visualization_input_data",
            "capability": "visualization",
            "tool": "visualization",
            "error_code": error_types[action],
            "message": message,
            "repair_contract": contract,
            "retry_policy": {
                "required_action": action,
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )


def _normalize_requirement_input(
    requirement: VisualizationEvidenceRequest,
    catalog: PresentationCatalog,
) -> VisualizationEvidenceRequest:
    refs = list(requirement.input_source_refs)
    if requirement.input_evidence:
        input_evidence_ref = requirement.input_evidence
        if ":" not in input_evidence_ref:
            input_evidence_ref = f"evidence:{input_evidence_ref}"
        if input_evidence_ref not in refs:
            refs.insert(0, input_evidence_ref)
    if not refs:
        return requirement
    source_refs = catalog.analysis_input_source_refs(refs)
    return requirement.model_copy(update={
        "input_source_refs": source_refs,
        "input_evidence": next(
            (ref.split(":", 1)[1] for ref in source_refs if ref.startswith("evidence:")),
            None,
        ),
    })


def _needs_sources_result(requirement: VisualizationEvidenceRequest) -> dict:
    return VisualizationResult(
        status="needs_sources",
        summary="Visualization planning identified additional semantic sources required before materialization.",
        required_data_request=requirement,
    ).model_dump(mode="json")


def _dependency_result(
    requirement: VisualizationEvidenceRequest,
    request_state: RequestStateModel,
) -> dict:
    if _dependency_was_attempted(requirement, request_state):
        return _unavailable_result(
            "The requested visual-verification evidence remained unavailable after its owning "
            f"{requirement.required_action} action completed."
        )
    return _needs_sources_result(requirement)


def _unavailable_result(reason: str) -> dict:
    message = str(reason or "Visual verification is unavailable.").strip()
    return VisualizationResult(
        status="unavailable",
        summary="No visualization was published because visual verification did not pass.",
        unavailable_reason=message,
    ).model_dump(mode="json")


def _semantic_error(
    exc: ValueError,
    _inventory: dict,
    *,
    scope: str = "chart_plan",
) -> StructuredToolError:
    diagnostic = _planning_diagnostic(exc, stage=scope)
    message = (
        "Visualization planning failed "
        f"[{diagnostic['error_code']} at {scope}]: {diagnostic['message']}"
    )
    contract = {
        "mode": "visualization_llm_repair",
        "instruction": (
            "Retry visualization from the current grounded state; the responsible LLM stage must produce a new complete candidate."
        ),
        "error_code": diagnostic["error_code"],
        "failed_stage": scope,
    }
    return StructuredToolError(
        message,
        error_type="visualization_planning_failed",
        retryable=True,
        recommended_next_action="visualization",
        diagnostics=diagnostic,
        validation_failure={
            "scope": scope,
            "capability": "visualization",
            "tool": "visualization",
            "error_code": "visualization_planning_failed",
            "message": message,
            "repair_contract": contract,
            "retry_policy": {
                "required_action": "visualization",
                "max_equivalent_retries": 1,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )
