"""Anomaly tool placeholder."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.timeseries.anomaly_registry import default_anomaly_detector_name, get_anomaly_detector
from core.timeseries.evidence_resolution import resolve_database_evidence
from core.timeseries.normalization import normalize_timeseries_evidence
from schemas.database import DatabaseEvidence
from schemas.timeseries import AnomalyResult
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool


class AnomalyInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    detector_name: str | None = None
    series_name: str | None = None
    constraints: dict | None = Field(default_factory=dict)


class AnomalyTool(BaseTool):
    async def execute(self, validated_input: AnomalyInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = resolve_database_evidence(database_evidence, request_state, tool_label="Anomaly")
        if database_evidence is None:
            raise ValueError("Anomaly detection requires database_evidence or a latest_database_evidence in request state.")
        constraints = validated_input.constraints or {}
        preferred_series = validated_input.series_name or constraints.get("series_name")
        series = normalize_timeseries_evidence(
            database_evidence,
            series_name=preferred_series,
            value_field=preferred_series,
        )
        detector_name = validated_input.detector_name or constraints.get("detector_name") or default_anomaly_detector_name()
        detector = get_anomaly_detector(detector_name)
        detector_output = detector.detect(series, params=constraints)
        anomaly_points = detector_output.anomaly_points
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
            requested_capabilities=["anomaly"],
            time_column=series.time_field,
            primary_measure=series.value_field,
            display_priority=2,
        )
        return AnomalyResult(
            anomaly_id=f"anomaly_{database_evidence.evidence_id}",
            detector_name=detector.name,
            anomaly_points=anomaly_points,
            anomaly_spans=detector_output.anomaly_spans,
            scores=detector_output.scores,
            diagnostics={
                **detector_output.diagnostics,
                "series_name": series.series_name,
                "detector_registry_name": detector.name,
            },
            visualizations=[visualization],
        ).model_dump(mode="json")
