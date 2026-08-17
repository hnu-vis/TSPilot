"""Packaged workflow tool."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.skill import execute_skill, list_skills
from tools.base import BaseTool, StructuredToolError


class SkillInput(BaseModel):
    skill_name: str
    task_context: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)


class SkillTool(BaseTool):
    async def execute(self, validated_input: SkillInput, *, request_state=None, **kwargs) -> dict:
        available = set(list_skills())
        if validated_input.skill_name not in available:
            raise StructuredToolError(
                f"Unknown skill '{validated_input.skill_name}'.",
                error_type="unknown_skill",
                retryable=False,
                diagnostics={"available_skills": sorted(available)},
                recommended_next_action="terminate",
            )
        task_context = dict(validated_input.task_context)
        if request_state is not None and not task_context:
            task_context = {
                "latest_database_evidence": (
                    request_state.latest_database_evidence.model_dump(mode="json")
                    if request_state.latest_database_evidence
                    else None
                ),
                "latest_forecast": (
                    {
                        "forecast_id": request_state.latest_forecast.forecast_id,
                        "model_name": request_state.latest_forecast.model_name,
                        "horizon": request_state.latest_forecast.horizon,
                        "status": request_state.latest_forecast.status,
                    }
                    if request_state.latest_forecast
                    else None
                ),
                "latest_anomaly": (
                    {
                        "anomaly_id": request_state.latest_anomaly.anomaly_id,
                        "detector_name": request_state.latest_anomaly.detector_name,
                        "anomaly_point_count": len(request_state.latest_anomaly.anomaly_points),
                    }
                    if request_state.latest_anomaly
                    else None
                ),
                "visualizations": [
                    visualization.model_dump(mode="json")
                    for visualization in request_state.visualizations
                ],
            }
        result = execute_skill(validated_input.skill_name, task_context, validated_input.parameters)
        return {
            "summary": result.get("summary", f"Skill '{validated_input.skill_name}' executed."),
            "skill_name": validated_input.skill_name,
            "results": result.get("results", []),
        }
