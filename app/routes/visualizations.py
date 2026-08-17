"""Full-fidelity visualization artifact reads."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.deps import get_visualization_artifact_store


router = APIRouter(prefix="/api/v1/visualizations", tags=["visualizations"])


@router.get("/{visualization_id}/data")
async def visualization_data(visualization_id: str):
    try:
        artifact = get_visualization_artifact_store().get(visualization_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if artifact is None:
        raise HTTPException(status_code=404, detail="Visualization artifact not found.")
    return artifact.model_dump(mode="json")
