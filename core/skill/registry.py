"""Packaged workflow registry."""
from __future__ import annotations

from collections.abc import Callable


def list_skills() -> list[str]:
    return sorted(_SKILLS)


def execute_skill(skill_name: str, task_context: dict, parameters: dict) -> dict:
    handler = _SKILLS.get(skill_name)
    if handler is None:
        return {
            "summary": f"Unknown skill '{skill_name}'.",
            "results": [{"available_skills": list_skills()}],
        }
    return handler(task_context, parameters)


def _reference_extractor(task_context: dict, parameters: dict) -> dict:
    references = []
    for key in ("latest_database_evidence", "latest_insight", "latest_forecast", "latest_anomaly"):
        payload = task_context.get(key)
        if isinstance(payload, dict):
            reference_id = (
                payload.get("evidence_id")
                or payload.get("insight_id")
                or payload.get("forecast_id")
                or payload.get("anomaly_id")
            )
            if reference_id:
                references.append({"slot": key, "source_id": reference_id})
    return {"summary": f"Extracted {len(references)} structured references.", "results": references}


def _next_step_advisor(task_context: dict, parameters: dict) -> dict:
    advice = []
    if not task_context.get("latest_database_evidence"):
        advice.append("query_database")
    elif not task_context.get("latest_insight"):
        advice.append("insight")
    elif parameters.get("need_forecast"):
        advice.append("forecast")
    elif parameters.get("need_anomaly"):
        advice.append("anomaly")
    else:
        advice.append("format_answer")
    return {"summary": f"Suggested next step: {advice[0]}.", "results": [{"recommended_actions": advice}]}


def _visualization_inventory(task_context: dict, parameters: dict) -> dict:
    visualizations = task_context.get("visualizations") or []
    results = [
        {
            "visualization_id": visualization.get("visualization_id"),
            "renderer": visualization.get("renderer"),
            "title": visualization.get("title"),
        }
        for visualization in visualizations
        if isinstance(visualization, dict)
    ]
    return {
        "summary": f"Collected {len(results)} visualization descriptors.",
        "results": results,
    }


_SKILLS: dict[str, Callable[[dict, dict], dict]] = {
    "reference_extractor": _reference_extractor,
    "next_step_advisor": _next_step_advisor,
    "visualization_inventory": _visualization_inventory,
}
