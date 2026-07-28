"""LLM-driven read-only query generation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from runtime.language import detect_response_language
from runtime.token_usage import record_llm_token_usage


class LLMGeneratedQuery(BaseModel):
    """Structured query proposal returned by the query-generation LLM."""

    query: str
    query_language: str | None = None
    purpose: str | None = None
    expected_result_type: str | None = None
    selected_fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    task_coverage: dict[str, Any] = Field(default_factory=dict)
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
        request_state=None,
    ) -> LLMQueryGenerationResult:
        if self._llm is None:
            raise RuntimeError("LLM query generation requires an llm instance.")

        system_prompt = self._system_prompt()
        response_language = getattr(request_state, "response_language", None) or detect_response_language(message)
        user_prompt = self._user_prompt(
            database_id=database_id,
            database_type=database_type,
            message=message,
            response_language=response_language,
            schema_preview=schema_preview,
            time_range=time_range,
            constraints=constraints,
            history=history,
            previous_query=previous_query,
            error=error,
        )
        raw_response = await self._invoke_model(
            [("system", system_prompt), ("user", user_prompt)],
            request_state=request_state,
            source="sql_query.generation_repair" if previous_query or error else "sql_query.generation",
        )
        generated_query = self._parse_response(raw_response)
        return LLMQueryGenerationResult(
            generated_query=generated_query,
            raw_response=raw_response,
            repaired_from_query=previous_query,
            diagnostics={"repair": bool(previous_query or error)},
        )

    async def _invoke_model(self, messages, *, request_state=None, source: str = "sql_query.generation") -> str:
        response = await self._llm.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content)
        record_llm_token_usage(
            request_state,
            source=source,
            response=response,
            messages=messages,
            output_text=content,
            tool_name="sql_query",
        )
        return content

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
            "If schema_preview.metadata.data_profile is present, treat it as grounded time-coverage evidence for measurement/field/tag series. "
            "Use it to distinguish a valid series with no data in the requested time range from an invalid field or filter. "
            "If the requested time range is outside every matching data_profile source, say so in task_coverage.missing/next_action_hint and prefer a minimal existence/count query over a broad raw-data query. "
            "Do not infer missing data from structural schema alone; structural schema proves fields and tags exist, not that rows exist for every time range.\n"
            "If a value-domain candidate is the only plausible match for a requested entity, include that filter even when it was not in required_filters.\n"
            "For trends, seasonality, anomalies, or forecasting, preserve coverage of the requested time range. "
            "If constraints.max_points limits output size, use time-window aggregation or representative downsampling across the full range; "
            "do not use LIMIT/top/head as a substitute for sampling because it truncates the time range.\n"
            "Use aggregation for explicit aggregate/ranking/count/summary/bucket/comparison requests, and for full-range downsampling needed to keep time-series evidence within constraints.\n"
            "For derived or multi-field statistics, plan from the requested output contract rather than a fixed query template. "
            "Each requested measure, dimension, grouping, comparison, time boundary, or derived quantity must either be represented by an explicit returned column/result shape, or listed in task_coverage.missing. "
            "If one compact query would require fragile dialect-specific state logic, prefer a simpler reliable evidence query and mark the still-missing contract fields instead of claiming coverage.\n"
            "Before returning, self-check whether this single query covers the whole user request. "
            "Use request.response_language for all natural-language JSON values you generate, including purpose, assumptions, "
            "task_coverage.satisfied, task_coverage.missing, and task_coverage.next_action_hint. "
            "Use Simplified Chinese for \"zh\" and English for \"en\". Keep query code, identifiers, and data values unchanged.\n"
            "Use task_coverage.satisfied for request constraints/facts this query is designed to satisfy, "
            "task_coverage.missing for requested facts not directly computed by this query, and "
            "task_coverage.next_action_hint for the next sql_query needed when coverage is incomplete. "
            "Do not claim coverage for extrema, count, grouping, or ordering unless the query explicitly computes it.\n"
            "JSON schema: {"
            "\"query\": string, "
            "\"query_language\": string, "
            "\"purpose\": string, "
            "\"expected_result_type\": \"timeseries\"|\"table\"|\"statistics\"|\"schema\"|\"metric_list\", "
            "\"selected_fields\": string[], "
            "\"assumptions\": string[], "
            "\"task_coverage\": {\"satisfied\": string[], \"missing\": string[], \"next_action_hint\": string|null}, "
            "\"confidence\": number"
            "}."
        )

    def _user_prompt(
        self,
        *,
        database_id: str,
        database_type: str,
        message: str,
        response_language: str,
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
            "response_language": response_language,
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
