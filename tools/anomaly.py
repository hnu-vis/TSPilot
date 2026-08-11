"""Anomaly tool placeholder."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from core.timeseries.anomaly_registry import default_anomaly_detector_name, get_anomaly_detector
from core.timeseries.evidence_resolution import resolve_database_evidence
from core.timeseries.normalization import normalize_timeseries_evidence
from schemas.database import DatabaseEvidence
from schemas.data_fact import DataFactRequest
from schemas.timeseries import AnomalyResult
from tools.base import BaseTool, StructuredToolError


class AnomalyInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    detector_name: str | None = None
    series_name: str | None = None
    constraints: dict | None = Field(default_factory=dict)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_fact_contracts(self):
        if self.fact_requests:
            raise ValueError("anomaly returns an analysis artifact; Data Fact requests must target sql_query or code_interpreter")
        return self


class AnomalyTool(BaseTool):
    async def execute(self, validated_input: AnomalyInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = resolve_database_evidence(database_evidence, request_state, tool_label="Anomaly")
        if database_evidence is None:
            raise ValueError("Anomaly detection requires database_evidence or a latest_database_evidence in request state.")
        constraints = validated_input.constraints or {}
        database_evidence, input_policy_diagnostics = _resolve_anomaly_input_policy(
            database_evidence,
            request_state,
            constraints,
        )
        preferred_series = validated_input.series_name or constraints.get("series_name")
        try:
            series = normalize_timeseries_evidence(
                database_evidence,
                series_name=preferred_series,
                value_field=preferred_series,
            )
        except ValueError as exc:
            raise _insufficient_timeseries_evidence_error(
                message=str(exc),
                tool_name="anomaly",
                evidence=database_evidence,
            ) from exc
        detector_name = validated_input.detector_name or constraints.get("detector_name") or default_anomaly_detector_name()
        detector = get_anomaly_detector(detector_name)
        detector_output = detector.detect(series, params=constraints)
        anomaly_points = detector_output.anomaly_points
        return AnomalyResult(
            anomaly_id=f"anomaly_{database_evidence.evidence_id}",
            detector_name=detector.name,
            anomaly_points=anomaly_points,
            anomaly_spans=detector_output.anomaly_spans,
            scores=detector_output.scores,
            diagnostics={
                **detector_output.diagnostics,
                **input_policy_diagnostics,
                "series_name": series.series_name,
                "detector_registry_name": detector.name,
            },
        ).model_dump(mode="json")


def _resolve_anomaly_input_policy(
    database_evidence: DatabaseEvidence,
    request_state,
    constraints: dict,
) -> tuple[DatabaseEvidence, dict]:
    policy = str(constraints.get("input_policy") or "raw_when_available").strip().lower()
    diagnostics = {
        "input_policy": policy,
        "selected_evidence_id": database_evidence.evidence_id,
        "resolved_evidence_id": database_evidence.evidence_id,
    }
    if policy in {"selected", "current", "as_provided"} or request_state is None:
        return database_evidence, diagnostics
    if not _evidence_looks_outlier_filtered(database_evidence):
        return database_evidence, diagnostics
    raw_candidate = _latest_raw_timeseries_candidate(database_evidence, request_state)
    if raw_candidate is None:
        diagnostics["input_policy_note"] = "selected evidence appears filtered, but no raw candidate was available"
        return database_evidence, diagnostics
    diagnostics["resolved_evidence_id"] = raw_candidate.evidence_id
    diagnostics["input_policy_note"] = "selected evidence appears outlier-filtered; anomaly detection used prior raw time-series evidence"
    return raw_candidate, diagnostics


def _latest_raw_timeseries_candidate(selected: DatabaseEvidence, request_state) -> DatabaseEvidence | None:
    artifacts = getattr(request_state, "database_evidence_artifacts", {}) or {}
    candidates = []
    for evidence in artifacts.values():
        if not isinstance(evidence, DatabaseEvidence):
            try:
                evidence = DatabaseEvidence.model_validate(evidence)
            except Exception:
                continue
        if evidence.evidence_id == selected.evidence_id:
            continue
        if evidence.result_type != "timeseries" or evidence.database != selected.database:
            continue
        if _evidence_looks_outlier_filtered(evidence):
            continue
        candidates.append(evidence)
    return candidates[-1] if candidates else None


def _insufficient_timeseries_evidence_error(
    *,
    message: str,
    tool_name: str,
    evidence: DatabaseEvidence,
) -> StructuredToolError:
    repair_contract = {
        "mode": "timeseries_input_repair",
        "failed_tool": tool_name,
        "input_evidence": evidence.evidence_id,
        "required_evidence_shape": "raw_timeseries",
        "required_min_points": 2,
        "constraints": {
            "evidence_shape": "raw_timeseries",
            "dialect_complexity_policy": "simple_raw_evidence",
        },
    }
    return StructuredToolError(
        message,
        error_type="insufficient_timeseries_evidence",
        retryable=True,
        diagnostics={
            "input_evidence_id": evidence.evidence_id,
            "result_type": evidence.result_type,
            "required_min_points": 2,
            "required_evidence_shape": "raw_timeseries",
        },
        recommended_next_action="sql_query",
        validation_failure={
            "tool": "sql_query",
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "sql_query",
                "max_equivalent_retries": 2,
                "terminal_after_exhausted": False,
            },
        },
    )


def _evidence_looks_outlier_filtered(evidence: DatabaseEvidence) -> bool:
    diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
    for key in (
        "excluded_rows",
        "outlier_rule",
        "adjusted_metrics",
        "outlier_treatment",
        "cleaning_policy",
        "rows_excluded",
    ):
        value = diagnostics.get(key)
        if value:
            return True

    metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
    text_parts = [
        evidence.summary or "",
        str(metadata.get("purpose") or ""),
        str(metadata.get("transformation") or ""),
        str(metadata.get("lineage") or ""),
    ]
    coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    text_parts.extend(
        str(item)
        for item in [
            coverage.get("next_action_hint"),
        ]
        if item
    )
    text = " ".join(text_parts).lower()
    filtered_patterns = (
        r"\boutlier[-_\s]*(filtered|removed|excluded|adjusted|treated)\b",
        r"\banomal(y|ies)[-_\s]*(filtered|removed|excluded|adjusted|treated)\b",
        r"\b(filtered|removed|excluded|adjusted|treated)\s+(outlier|anomal)",
        r"\bclean(ed)?\s+(series|data|dataset|evidence|input)\b",
        r"\b(winsorized|winsoris[ea]d)\b",
        r"(剔除|排除|过滤|清洗).{0,12}(异常|离群)",
        r"(异常|离群).{0,12}(剔除|排除|过滤|清洗)",
        r"清洗后",
        r"过滤后",
    )
    return any(re.search(pattern, text) for pattern in filtered_patterns)
