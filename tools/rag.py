"""Local knowledge retrieval tool."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.settings import get_settings
from core.rag import retrieve_local_knowledge
from tools.base import BaseTool


class RagInput(BaseModel):
    query: str
    database_context: dict | None = None
    database_evidence: dict | None = None
    filters: dict = Field(default_factory=dict)


class RagTool(BaseTool):
    async def execute(self, validated_input: RagInput, **kwargs) -> dict:
        settings = get_settings()
        root = settings.resolved_knowledge_base_dir
        results = retrieve_local_knowledge(root, validated_input.query, validated_input.filters)
        for result in results:
            result["score"] = int(result["score"])
        summary = (
            f"Retrieved {len(results)} knowledge passages for '{validated_input.query}'."
            if results
            else f"No local knowledge passages matched '{validated_input.query}'."
        )
        return {"summary": summary, "results": results}
