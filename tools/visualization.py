"""LLM-planned, full-fidelity visualization artifact tool."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.visualization import (
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
    VisualFilterTransform,
    VisualGoal,
    VisualGoalIR,
    VisualLayerIR,
    VisualLayerPlan,
)
from schemas.key_insight import KeyInsightRequest
from schemas.state import RequestStateModel
from schemas.visual_verification import VisualizationVerification
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool, StructuredToolError


class VisualizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    source_refs: list[str] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)


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
    verification_question: str | None = None
    interpretation: str | None = None
    visual_relation: str | None = None
    required_context: list[str] = Field(default_factory=list)
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

    name: str = Field(min_length=1)
    semantic_role: str = Field(min_length=1)
    source_path: str = Field(min_length=1)


class SemanticEventPlan(BaseModel):
    """Expand one source record into one located event row."""

    model_config = ConfigDict(extra="forbid")

    event_role: str = Field(min_length=1)
    timestamp_path: str = Field(min_length=1)
    value_path: str = Field(min_length=1)


class SemanticViewPlan(BaseModel):
    """A grounded semantic view prepared for independent chart planning."""

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    grain: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    record_path: str | None
    mode: Literal["records", "wide_events"]
    fields: list[SemanticFieldPlan]
    events: list[SemanticEventPlan]

    @model_validator(mode="after")
    def unique_semantic_columns(self):
        if self.mode == "records" and not self.fields:
            raise ValueError("records semantic view requires fields")
        if self.mode == "wide_events" and not self.events:
            raise ValueError("wide_events semantic view requires event mappings")
        if self.mode == "records" and self.events:
            raise ValueError("records semantic view cannot include event mappings")
        if self.mode == "wide_events" and self.fields:
            raise ValueError("wide_events semantic view cannot include record fields")
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("semantic view field names must be unique")
        event_roles = [item.event_role for item in self.events]
        if len(event_roles) != len(set(event_roles)):
            raise ValueError("semantic event roles must be unique")
        event_paths = [(item.timestamp_path, item.value_path) for item in self.events]
        if len(event_paths) != len(set(event_paths)):
            raise ValueError("wide semantic events must use distinct timestamp/value path pairs")
        return self


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


class _StructuredVerificationBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_insight_ids: list[str]
    verification_question: str | None
    interpretation: str | None
    visual_relation: str | None
    required_context: list[str]
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
        )
        if verification.decision == "needs_sources":
            requirement = _normalize_requirement_input(verification.required_data_request, catalog)
            return _dependency_result(requirement, request_state)
        if verification.decision == "not_visualizable":
            return _unavailable_result(
                verification.interpretation
                or "The requested conclusion has no grounded visual relationship that the available evidence can verify."
            )
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

        projection = None
        projection_error = ""
        semantic_refs: list[str] = []
        for projection_attempt in range(3):
            projection = await self._project(
                validated_input,
                inventory,
                request_state,
                verification=verification,
                source_preferences=source_preferences,
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
        complete = None
        repair_context = None
        for planning_attempt in range(3):
            try:
                plan = await self._plan(
                    validated_input,
                    semantic_inventory,
                    request_state,
                    verification=verification,
                    catalog=catalog,
                    repair_context=repair_context,
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
            except ValueError as exc:
                if planning_attempt >= 2:
                    return _unavailable_result(
                        f"Visual verification could not be materialized without semantic ambiguity: {exc}"
                    )
                repair_context = _planning_diagnostic(
                    exc,
                    stage=(
                        "requirement_grounding"
                        if "input" in str(exc).casefold() and "source" in str(exc).casefold()
                        else "materialization"
                    ),
                )
                continue
            # Publish the first grounded candidate that satisfies executable
            # materialization invariants. Candidate semantic and screenshot
            # audits are intentionally disabled in the publication path.
            break

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

    async def _select_verification(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        *,
        source_preferences: set[str],
    ) -> VisualVerificationDecision:
        if self._llm is None:
            raise RuntimeError("visual verification planning requires an LLM")
        insight_inventory = _verified_insight_inventory(request_state)
        data_source_inventory = _data_source_inventory(inventory)
        allowed = {item["insight_id"] for item in insight_inventory}
        repair_error: Exception | None = None
        for attempt in range(3):
            repair_context = "none" if repair_error is None else (
                "The preceding candidate was rejected by runtime validation. Re-plan the complete decision; "
                f"do not preserve invalid values. Validation feedback: {repair_error}. "
                f"The only valid values for target_insight_ids and non_visual_insight_ids are: {sorted(allowed)}."
            )
            prompt = prompt_locale_instruction(request_state.response_language) + (
                "You are the visual verification planner inside a strict outer ReAct loop. Decide whether the user's conclusion has "
                "an inspectable visual relationship, before choosing chart types or fields. Return one schema-valid discriminated outcome: "
                "visualize/not_visualizable require required_data_request=null; needs_sources requires exactly one complete request. "
                "Select only supplied verified insight_id values. target_insight_ids and non_visual_insight_ids are IDs, never source refs: "
                "do not add an 'insight:' prefix. Insights state the conclusion; evidence, derived evidence, forecast, and anomaly artifacts "
                "supply its context. Target a located Insight only when one item co-locates its timestamp and numeric value; do not reconstruct "
                "a point from unrelated scalar Insights. A raw descriptive chart may have no target Insight, but must state an observable relation. "
                "Use needs_sources when the conclusion or required context is absent; do not calculate or invent a fallback. code_interpreter "
                "requests require non-empty insight_requests; SQL may request SQL-owned atomic Insights. Use not_visualizable only when a chart "
                "adds no inspectable evidence (for example, a causal claim unsupported by observational data).\n"
                f"User request: {request.message}\n"
                f"Task visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
                f"Verified Key Insights: {json.dumps(insight_inventory, ensure_ascii=False)}\n"
                f"Preferred source refs: {json.dumps(sorted(source_preferences), ensure_ascii=False)}\n"
                f"Referenced data contracts: {json.dumps(data_source_inventory, ensure_ascii=False)}\n"
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
                return decision
            except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
                repair_error = exc

        raise _semantic_error(
            ValueError(f"invalid visual verification decision after LLM repair: {repair_error}"),
            inventory,
            scope="verification_selection",
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
        repair_context: dict | None = None,
        contract_repair_attempt: int = 0,
    ) -> VisualizationCandidateAudit:
        target_insights = _target_insight_inventory(request_state, verification.target_insight_ids)
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You independently audit a fully materialized visual-verification candidate. You did not author the plan. "
            "Return exactly one JSON object matching {\"decision\":\"approve\"|\"revise\"|\"needs_sources\"|\"unavailable\","
            "\"issues\":[str],\"required_data_request\":object|null}. Approve only when the chart lets a human inspect the "
            "verification question and the claimed meaning is entailed by the grounded field semantics and lineage. Check every "
            "target Insight, required contextual baseline, complete comparison set, time/grain/unit compatibility, interval semantics, "
            "role coverage, title, legend meaning, and interpretation. For a localized interval in a longer series, require the exact highlighted "
            "interval to remain inside a broader initial dataZoom observation window while the complete series remains scrollable; values outside "
            "that viewport must not distort its visible y scale. Exact values must remain owned by their artifacts. Do not accept "
            "a causal claim from merely correlated series. Use revise for presentation or plan errors that existing semantic views can "
            "repair, needs_sources for genuinely missing evidence, and unavailable when the requested conclusion has no defensible "
            "visual verification. Never approve a decorative scalar chart or invent a fallback.\n"
            "Treat source materialization_complete and query_execution coverage as authoritative. In sampled time series, the last "
            "observed timestamp may validly precede an exclusive query stop boundary by one sampling interval; do not call a complete "
            "source incomplete merely because its final observed timestamp is not identical to the requested stop.\n"
            f"User request: {request.message}\n"
            f"Verification decision: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(target_insights, ensure_ascii=False)}\n"
            f"Chart plan: {json.dumps(plan.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Materialized candidates: {json.dumps(_candidate_audit_view(visualizations), ensure_ascii=False)}\n"
            f"Semantic inventory: {json.dumps(inventory, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            VisualizationCandidateAudit,
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
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            if contract_repair_attempt < 2:
                return await self._audit_candidate(
                    request,
                    verification,
                    plan,
                    visualizations,
                    inventory,
                    request_state,
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
            "\"source_ref\":str,\"record_path\":str|null,\"mode\":\"records\"|\"wide_events\","
            "\"fields\":[{\"name\":str,\"semantic_role\":str,\"source_path\":str}],"
            "\"events\":[{\"event_role\":str,\"timestamp_path\":str,\"value_path\":str}]}],"
            "\"required_data_request\":{\"required_action\":\"sql_query\"|\"anomaly\"|\"forecast\"|\"code_interpreter\","
            "\"purpose\":str,\"message\":str|null,\"required_shape\":str,\"required_fields\":[str],"
            "\"required_properties\":[str],\"input_evidence\":str|null,\"input_source_refs\":[str],\"insight_requests\":[{"
            "\"name\":str,\"insight_type\":str,\"insight_key\":str|null}]}|null}. "
            "Use one exact source_ref per view. Select record_path only from projection_root.record_path_candidates; source_path is relative "
            "to that record, and wildcards belong only in record_path. For each view choose exactly one representation: records has non-empty "
            "fields and events=[], while wide_events has fields=[] and non-empty events. wide_events is only for one wide record containing "
            "multiple distinct timestamp/value pairs; it emits event_role, timestamp, value. Use a separate view "
            "for another grain. Forecast visuals require historical actuals when available; request sql_query when that baseline is absent. "
            "Do not invent uncertainty unless requested. Return either non-empty semantic_views with null required_data_request, or an empty "
            "view list with one owner request. Owners: sql_query for raw context, anomaly for anomaly detection, forecast for predictions, "
            "code_interpreter only for derived calculations and with non-empty insight_requests. input_evidence is an exact source_ref or null. "
            "On repair, fix the reported schema, source, or path violation in the complete plan; never repeat it or create a fallback view.\n"
            f"Visualization request: {request.message}\n"
            f"Visual verification contract: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(target_insights, ensure_ascii=False)}\n"
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
                    repair_context=_planning_diagnostic(
                        exc,
                        stage="semantic_projection_schema",
                    ),
                    schema_repair_attempt=schema_repair_attempt + 1,
                )
            raise _semantic_error(
                ValueError(f"invalid semantic projection plan: {exc}"), inventory, scope="semantic_projection",
            ) from exc

    async def _plan(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        verification: VisualVerificationDecision,
        catalog: PresentationCatalog,
        repair_context: dict | None = None,
    ) -> VisualizationPlan:
        if self._llm is None:
            raise RuntimeError("visualization planning requires an LLM")
        target_insights = _target_insight_inventory(request_state, verification.target_insight_ids)
        field_contract = _closed_visual_field_contract(inventory)
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You produce a closed, renderer-independent Visual IR for grounded visualization. Semantic projection has already "
            "organized existing values into views. Return one branch: non-empty visual_goals with required_data_request=null, or an empty "
            "goal list with one complete owner request. Do not write renderer options, transforms, datasets, or series; select only semantic "
            "layer intent and fields. Every schema field is required; use null, false, normal, solid, none, primary, or [] when inapplicable. "
            "Allowed layer_type values are series, event_points, band, interval_overlay, comparison. series/event_points/comparison/interval_overlay "
            "need x and y; band needs x/lower/upper. A line-like layer needs at least two points; event_points needs one. required_roles need "
            "same-role layers. interval_overlay uses a multi-point context source and either one interval_source_ref with exact start/end fields, "
            "or literal start/end values, never both; all interval fields are null for other types. "
            "Use one cohesive shared plot for related time-aligned information. Preserve the full series for a localized interval, forecast, anomaly, "
            "or trend claim; forecast additionally needs historical actuals when available. Do not turn scalar summaries or text labels into artificial "
            "graphic layers, or claim semantics not present in field_semantics. If required meaning is absent, request its owning source instead. "
            "Owners: sql_query for raw context, anomaly for anomaly outputs, forecast for predictions, and code_interpreter only for derived calculations "
            "with non-empty insight_requests. input_evidence is an exact source_ref or null. Text cards and tables are unsupported. "
            "The closed encoding field contract below is the complete executable vocabulary for every layer. Copy a field name "
            "verbatim from the entry for its source_ref into encodings and interval boundary fields. Field names are identifiers, "
            "not prose: never translate, paraphrase, or substitute an Insight name, statement, or label for a field identifier. "
            "When Repair context reports a rejected field, choose a different field from that same contract or return a genuine "
            "required_data_request; never repeat the rejected identifier. Do not weaken the requested purpose and do not invent a fallback chart.\n"
            f"Visualization request: {request.message}\n"
            f"Visual verification contract: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Target Key Insights: {json.dumps(target_insights, ensure_ascii=False)}\n"
            f"Authoritative visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Closed encoding field contract: {json.dumps(field_contract, ensure_ascii=False)}\n"
            f"Semantic view inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm,
            _StructuredVisualizationPlan,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title="Chart Planning",
            trace_summary="选择图表结构与视觉编码",
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.chart_plan",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            if parse_error is not None:
                raise parse_error
        except (json.JSONDecodeError, ValueError) as exc:
            raise _semantic_error(ValueError(f"invalid chart plan: {exc}"), inventory) from exc
        return _compile_visualization_plan(parsed, catalog)


def _compile_visualization_plan(
    plan: _StructuredVisualizationPlan,
    catalog: PresentationCatalog,
) -> VisualizationPlan:
    if plan.required_data_request is not None:
        return VisualizationPlan(
            visual_goals=[],
            required_data_request=plan.required_data_request.to_runtime(),
        )
    return VisualizationPlan(
        visual_goals=[_compile_visual_goal(goal, catalog) for goal in plan.visual_goals],
        required_data_request=None,
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
    if any(layer.axis == "secondary" for layer in goal.layers):
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
    interval_fields = (
        layer.interval_source_ref,
        layer.interval_start_field,
        layer.interval_end_field,
    )
    interval_values = (layer.interval_start_value, layer.interval_end_value)
    if layer.layer_type == "interval_overlay":
        uses_source = all(interval_fields)
        uses_values = all(value is not None for value in interval_values)
        if uses_source == uses_values:
            raise ValueError(
                "interval_overlay requires exactly one boundary source or one literal boundary pair"
            )
    elif any(value is not None for value in (*interval_fields, *interval_values)):
        raise ValueError("interval boundary fields are valid only for interval_overlay")

    _validate_visual_ir_source(layer, catalog)
    mark = {
        "series": "line",
        "event_points": "point",
        "band": "band",
        "interval_overlay": "line",
        "comparison": "bar",
    }[layer.layer_type]
    encoding = {item.channel: item.field for item in layer.encodings}
    transforms: list[VisualFilterTransform] = []
    if layer.layer_type == "interval_overlay":
        if layer.interval_source_ref is not None:
            start, end = _interval_boundaries(
                catalog,
                layer.interval_source_ref,
                str(layer.interval_start_field),
                str(layer.interval_end_field),
            )
        else:
            start, end = layer.interval_start_value, layer.interval_end_value
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
    if mark in {"line", "area"}:
        presentation["lineStyle"] = {
            "width": line_width,
            "type": layer.line_style,
        }
        presentation["showSymbol"] = layer.symbol != "none"
    if layer.symbol != "none":
        presentation["symbol"] = layer.symbol
        presentation["symbolSize"] = 12 if layer.emphasis == "strong" else 9
    if layer.axis == "secondary":
        presentation["yAxisIndex"] = 1
    return VisualLayerPlan(
        role=layer.role,
        source_ref=layer.source_ref,
        mark=mark,
        encoding=encoding,
        transform=transforms,
        presentation=presentation,
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
    message = str(error)
    lowered = message.casefold()
    if "requires at least two grounded points" in lowered:
        code = "INSUFFICIENT_SERIES_POINTS"
    elif "unavailable field" in lowered or "unavailable in every record" in lowered:
        code = "UNKNOWN_FIELD"
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
        "message": message[:1200],
    }
    if allowed_values is not None:
        diagnostic["allowed_values"] = allowed_values
    return diagnostic


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


def _candidate_audit_view(visualizations: list[VisualizationPayload]) -> list[dict]:
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
        payload["layers"] = [layer.model_dump(mode="json", exclude={"points"}) for layer in visualization.layers]
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


def _closed_visual_field_contract(inventory: dict) -> dict[str, list[dict[str, str]]]:
    """Return the executable chart-field vocabulary for each semantic view.

    Semantic view field names are runtime identifiers, while Insight statements
    and labels are natural language. Keeping this compact contract adjacent to
    the IR request prevents a planner from treating descriptive text as a
    column name without choosing a chart on the tool's behalf.
    """

    contract: dict[str, list[dict[str, str]]] = {}
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
        if ref and entries:
            contract[ref] = entries
    return contract


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
    if requirement.input_evidence and requirement.input_evidence not in refs:
        refs.insert(0, requirement.input_evidence)
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
    message = f"Visualization planning failed: {exc}"
    contract = {
        "mode": "visualization_llm_repair",
        "instruction": (
            "Retry visualization and let the responsible LLM stage re-plan from the execution feedback below."
        ),
        "execution_error": str(exc),
        "failed_stage": scope,
    }
    return StructuredToolError(
        message,
        error_type="visualization_planning_failed",
        retryable=True,
        recommended_next_action="visualization",
        diagnostics={"execution_error": str(exc), "failed_stage": scope},
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
