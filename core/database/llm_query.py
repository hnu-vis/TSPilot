"""LLM-driven read-only query generation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator


class LLMGeneratedQuery(BaseModel):
    """Structured query proposal returned by the query-generation LLM."""

    query: str
    query_language: str | None = None
    purpose: str | None = None
    expected_result_type: str | None = None
    selected_fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float | None = None

    @model_validator(mode="after")
    def require_query(self):
        if not self.query.strip():
            raise ValueError("LLM query generation returned an empty query.")
        return self


@dataclass
class LLMQueryGenerationResult:
    """Query generation result plus prompt/repair diagnostics."""

    generated_query: LLMGeneratedQuery
    raw_response: str
    repaired_from_query: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class LLMQueryGenerator:
    """Ask an LLM to generate a datasource-grounded read-only query."""

    def __init__(self, llm):
        self._llm = llm

    async def generate(
        self,
        *,
        database_id: str,
        database_type: str,
        message: str,
        schema_preview: dict,
        time_range: dict | None,
        constraints: dict,
        history: list[dict],
        previous_query: str | None = None,
        error: Exception | str | None = None,
    ) -> LLMQueryGenerationResult:
        if self._llm is None:
            raise RuntimeError("LLM query generation requires an llm instance.")

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(
            database_id=database_id,
            database_type=database_type,
            message=message,
            schema_preview=schema_preview,
            time_range=time_range,
            constraints=constraints,
            history=history,
            previous_query=previous_query,
            error=error,
        )
        raw_response = await self._invoke_model([("system", system_prompt), ("user", user_prompt)])
        generated_query = self._parse_response(raw_response)
        return LLMQueryGenerationResult(
            generated_query=generated_query,
            raw_response=raw_response,
            repaired_from_query=previous_query,
            diagnostics={"repair": bool(previous_query or error)},
        )

    async def _invoke_model(self, messages) -> str:
        response = await self._llm.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    def _parse_response(self, content: str) -> LLMGeneratedQuery:
        payload = self._decode_json(content)
        try:
            return LLMGeneratedQuery.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"LLM query generation returned invalid JSON payload: {exc}") from exc

    def _decode_json(self, content: str) -> dict:
        stripped = content.strip()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise ValueError(f"LLM query generation did not return JSON: {content}")
            decoded = json.loads(match.group(0))
        if not isinstance(decoded, dict):
            raise ValueError("LLM query generation response must be a JSON object.")
        return decoded

    def _system_prompt(self) -> str:
        return (
            "You generate database queries for TSPilot.\n"
            "Return exactly one JSON object and no markdown.\n"
            "The query must be read-only and grounded only in the provided schema.\n"
            "For SQL databases, use SELECT or WITH only.\n"
            "For InfluxDB, generate Flux and do not use to(), experimental.to(), or write functions.\n"
            "For Prometheus, generate PromQL.\n"
            "Treat schema_preview.schema_linking as the first-stage grounding result.\n"
            "Preserve every schema_linking.required_filters item in the generated query unless it is impossible in the dialect.\n"
            "Use schema_linking.candidate_filters and labels_or_tags as auxiliary value-domain evidence. "
            "Map user-mentioned entities, abbreviations, ticker symbols, and multilingual aliases to those domain values before writing filters.\n"
            "If a value-domain candidate is the only plausible match for a requested entity, include that filter even when it was not in required_filters.\n"
            "Prefer raw time-series rows when the user asks for trends, seasonality, anomalies, or forecasting.\n"
            "Use aggregation only when the user asks for an aggregate, ranking, count, summary statistic, bucket, or comparison.\n"
            "JSON schema: {"
            "\"query\": string, "
            "\"query_language\": string, "
            "\"purpose\": string, "
            "\"expected_result_type\": \"timeseries\"|\"table\"|\"statistics\"|\"schema\"|\"metric_list\", "
            "\"selected_fields\": string[], "
            "\"assumptions\": string[], "
            "\"confidence\": number"
            "}."
        )

    def _user_prompt(
        self,
        *,
        database_id: str,
        database_type: str,
        message: str,
        schema_preview: dict,
        time_range: dict | None,
        constraints: dict,
        history: list[dict],
        previous_query: str | None,
        error: Exception | str | None,
    ) -> str:
        payload = {
            "database_id": database_id,
            "database_type": database_type,
            "message": message,
            "schema_preview": schema_preview,
            "time_range": time_range,
            "constraints": constraints,
            "history": history[-6:],
            "previous_query": previous_query,
            "error": str(error) if error is not None else None,
        }
        mode = "repair" if previous_query or error else "generate"
        return "LLM SQL Query Generation JSON:\n" + json.dumps(
            {"mode": mode, "request": payload},
            ensure_ascii=False,
            default=str,
        )
