"""Anomaly tool placeholder."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.timeseries.anomaly_adapter import detect_zscore_anomalies
from core.timeseries.normalization import normalize_timeseries_evidence
from schemas.database import DatabaseEvidence
from schemas.timeseries import AnomalyResult
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool


class AnomalyInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    series_name: str | None = None
    constraints: dict | None = Field(default_factory=dict)


class AnomalyTool(BaseTool):
    async def execute(self, validated_input: AnomalyInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = _resolve_database_evidence(database_evidence, request_state)
        if database_evidence is None:
            raise ValueError("Anomaly detection requires database_evidence or a latest_database_evidence in request state.")
        preferred_series = validated_input.series_name or validated_input.constraints.get("series_name")
        series = normalize_timeseries_evidence(
            database_evidence,
            series_name=preferred_series,
            value_field=preferred_series,
        )
        threshold = float(validated_input.constraints.get("zscore_threshold", 2.5))
        anomaly_points, scores = detect_zscore_anomalies(series, threshold=threshold)
        visualization = VisualizationPayload(
            visualization_id=f"viz_anomaly_{database_evidence.evidence_id}",
            visualization_type="chart",
            visualization_kind="line",
            renderer="linechart",
            title=f"{series.value_field} anomalies",
            summary=f"{series.value_field} 的异常点检测结果。",
            chart={
                "x_axis_data": [point.timestamp for point in series.points],
                "series_data": [
                    {"name": series.value_field, "data": [point.value for point in series.points]},
                ],
            },
            annotations=anomaly_points,
            binding_evidence_ids=[database_evidence.evidence_id],
            requested_fact_types=["anomaly"],
            time_column=series.time_field,
            primary_measure=series.value_field,
            display_priority=2,
        )
        return AnomalyResult(
            anomaly_id=f"anomaly_{database_evidence.evidence_id}",
            detector_name="zscore",
            anomaly_points=anomaly_points,
            anomaly_spans=[],
            scores=scores,
            diagnostics={"threshold": threshold},
            visualizations=[visualization],
        ).model_dump(mode="json")


def _resolve_database_evidence(database_evidence, request_state):
    if database_evidence is None:
        latest = request_state.latest_database_evidence
        if latest is None:
            return None
        return request_state.database_evidence_artifacts.get(latest.evidence_id, latest)
    if isinstance(database_evidence, dict):
        evidence_id = database_evidence.get("evidence_id")
        if evidence_id:
            return request_state.database_evidence_artifacts.get(evidence_id) or DatabaseEvidence.model_validate(database_evidence)
        return DatabaseEvidence.model_validate(database_evidence)
    if isinstance(database_evidence, str):
        return request_state.database_evidence_artifacts.get(database_evidence)
    return request_state.database_evidence_artifacts.get(database_evidence.evidence_id, database_evidence)
