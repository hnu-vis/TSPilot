"""LLM-driven read-only query generation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from core.database.dialects import dialect_for_database
from runtime.language import detect_response_language
from runtime.token_usage import record_llm_token_usage


def _looks_like_single_object(value: dict) -> bool:
    object_markers = {
        "id",
        "name",
        "description",
        "output_type",
        "column",
        "source",
        "operator",
        "value",
        "measure",
        "dimension",
        "field",
    }
    return bool(set(value) & object_markers)


class QueryTaskContract(BaseModel):
    """Database-independent contract for one generated evidence query."""

    intent_type: str | None = None
    required_measures: list[str] = Field(default_factory=list)
    required_dimensions: list[dict[str, Any]] = Field(default_factory=list)
    required_filters: list[dict[str, Any]] = Field(default_factory=list)
    required_outputs: list[Any] = Field(default_factory=list)
    downstream_action: str | None = None
    preferred_evidence_shape: str | None = None
    dialect_complexity_policy: str | None = None
    coverage: dict[str, Any] = Field(default_factory=dict)

    @field_validator("required_measures", mode="before")
    @classmethod
    def normalize_string_list(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, dict):
            values = []
            for key, item in value.items():
                if item is True and str(key).strip():
                    values.append(str(key).strip())
                elif isinstance(item, str) and item.strip():
                    values.append(item.strip())
                elif isinstance(item, dict):
                    label = item.get("name") or item.get("id") or item.get("logical_measure")
                    if label:
                        values.append(str(label).strip())
                elif isinstance(item, (int, float, bool)):
                    values.append(str(item))
            return values
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @field_validator("required_dimensions", "required_filters", "required_outputs", mode="before")
    @classmethod
    def normalize_object_list(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            if _looks_like_single_object(value):
                return [value]
            return [
                {str(key): item} if not isinstance(item, dict) else {"id": str(key), **item}
                for key, item in value.items()
            ]
        return [value]


class QueryExecutionContract(BaseModel):
    """Datasource execution semantics that are not encoded in query text."""

    mode: str = "instant"
    start: str | None = None
    end: str | None = None
    lookback_seconds: int | None = Field(default=None, gt=0)
    step_seconds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_range_contract(self):
        self.mode = str(self.mode or "instant").strip().lower()
        if self.mode not in {"instant", "range"}:
            raise ValueError("query_execution.mode must be instant or range.")
        if self.mode == "range" and not ((self.start and self.end) or self.lookback_seconds):
            raise ValueError("Range query execution requires start/end or lookback_seconds.")
        return self


class LLMGeneratedQuery(BaseModel):
    """Structured query proposal returned by the query-generation LLM."""

    query: str
    query_language: str | None = None
    purpose: str | None = None
    expected_result_type: str | None = None
    selected_fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    task_coverage: dict[str, Any] = Field(default_factory=dict)
    query_task_contract: QueryTaskContract | None = None
    query_execution: QueryExecutionContract = Field(default_factory=QueryExecutionContract)
    confidence: float | None = None

    @model_validator(mode="after")
    def require_query(self):
        if not self.query.strip():
            raise ValueError("LLM query generation returned an empty query.")
        return self


class LLMSchemaLinkingContract(BaseModel):
    """LLM-selected schema/query grounding contract."""

    sources: list[dict[str, Any]] = Field(default_factory=list)
    measures: list[dict[str, Any]] = Field(default_factory=list)
    value_columns: list[dict[str, Any]] = Field(default_factory=list)
    aggregate_targets: list[dict[str, Any]] = Field(default_factory=list)
    dimension_columns: list[dict[str, Any]] = Field(default_factory=list)
    required_filters: list[dict[str, Any]] = Field(default_factory=list)
    candidate_filters: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    confidence: str = "low"
    evidence: list[str] = Field(default_factory=list)

    @field_validator(
        "sources",
        "measures",
        "value_columns",
        "aggregate_targets",
        "dimension_columns",
        "required_filters",
        "candidate_filters",
        mode="before",
    )
    @classmethod
    def normalize_object_lists(cls, value, info):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            return []
        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
                continue
            if isinstance(item, str) and item.strip():
                key = "name"
                if info.field_name in {"required_filters", "candidate_filters"}:
                    key = "column"
                elif info.field_name in {"measures", "aggregate_targets"}:
                    key = "logical_measure"
                normalized.append({key: item.strip()})
        return normalized

    @field_validator("required_filters", mode="after")
    @classmethod
    def keep_only_schema_value_filters(cls, value):
        return [item for item in value if not _is_time_boundary_filter(item)]


def _is_time_boundary_filter(item: dict[str, Any]) -> bool:
    column = str(item.get("column") or item.get("name") or "").strip().lower()
    return column in {"time", "timestamp", "datetime", "date", "_time", "_start", "_stop"}


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

    async def generate_schema_linking(
        self,
        *,
        database_id: str,
        database_type: str,
        message: str,
        schema_preview: dict,
        time_range: dict | None,
        constraints: dict,
        history: list[dict],
        request_state=None,
    ) -> dict[str, Any]:
        """Ask the LLM to select the grounded schema contract used by query generation and validation."""
        if self._llm is None:
            raise RuntimeError("LLM schema linking requires an llm instance.")
        response_language = getattr(request_state, "response_language", None) or detect_response_language(message)
        payload = {
            "database_id": database_id,
            "database_type": database_type,
            "message": message,
            "response_language": response_language,
            "schema_preview": schema_preview,
            "time_range": time_range,
            "constraints": constraints,
            "history": history[-4:],
        }
        messages = [
            ("system", self._schema_linking_system_prompt(database_type=database_type)),
            ("user", "LLM Schema Linking JSON:\n" + json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        raw_response = await self._invoke_model(
            messages,
            request_state=request_state,
            source="sql_query.schema_linking",
        )
        contract = LLMSchemaLinkingContract.model_validate(self._decode_json(raw_response))
        return {
            **contract.model_dump(mode="json"),
            "contract_version": "llm_schema_linking.v1",
        }

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
        fact_requests: list[Any] | None = None,
        previous_query: str | None = None,
        error: Exception | str | None = None,
        request_state=None,
    ) -> LLMQueryGenerationResult:
        if self._llm is None:
            raise RuntimeError("LLM query generation requires an llm instance.")

        system_prompt = self._system_prompt(database_type=database_type)
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
            fact_requests=fact_requests or [],
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

    def _system_prompt(self, *, database_type: str) -> str:
        dialect = dialect_for_database(database_type)
        return (
            "You generate database queries for TSPilot.\n"
            "Return exactly one JSON object and no markdown.\n"
            "The query must be read-only and grounded only in the provided schema.\n"
            f"Dialect/query-language rules for this datasource ({database_type}, {dialect.query_language}): {dialect.generation_rules}\n"
            "Treat schema_preview.schema_linking as the first-stage grounding result.\n"
            "If schema_preview.schema_linking.measures or aggregate_targets is present, treat it as authoritative: "
            "use selector_column/selector_value to filter logical measures and aggregate/read physical_value_column or aggregate_column. "
            "Never aggregate a logical_measure name directly when a separate physical_value_column is provided.\n"
            "If schema_preview.physical_model is present, follow its query_generation_constraints before all generic column-name heuristics.\n"
            "Preserve every schema_linking.required_filters item in the generated query unless it is impossible in the dialect.\n"
            "Use schema_linking.candidate_filters and labels_or_tags as auxiliary value-domain evidence. "
            "Map user-mentioned entities, abbreviations, ticker symbols, and multilingual aliases to those domain values before writing filters.\n"
            "If schema_preview.metadata.data_profile is present, treat it as grounded time-coverage evidence for measurement/field/tag series. "
            "Use it to distinguish a valid series with no data in the requested time range from an invalid field or filter. "
            "For relative recency requests such as latest/recent/current window, anchor the range to the latest covered timestamp in data_profile when coverage evidence is available; "
            "do not anchor relative ranges to wall-clock now unless the data_profile also shows current data coverage. "
            "If the requested time range is outside every matching data_profile source, say so in task_coverage.missing/next_action_hint and prefer a minimal existence/count query over a broad raw-data query. "
            "Do not infer missing data from structural schema alone; structural schema proves fields and tags exist, not that rows exist for every time range.\n"
            "If a value-domain candidate is the only plausible match for a requested entity, include that filter even when it was not in required_filters.\n"
            "For trends, seasonality, anomalies, or forecasting, preserve coverage of the requested time range. "
            "If constraints.max_points limits output size, use time-window aggregation or representative downsampling across the full range; "
            "do not use LIMIT/top/head as a substitute for sampling because it truncates the time range.\n"
            "If request.constraints.evidence_shape is raw_timeseries, treat that as an authoritative execution contract: "
            "return native timestamp/value rows for the grounded series, set query_task_contract.preferred_evidence_shape=\"raw_series\", "
            "and do not aggregate, rank, group, join, pivot, or compute derived outputs in the query. "
            "Put every non-raw requested output in task_coverage.missing with next_action_hint naming the downstream analysis tool.\n"
            "Use aggregation for explicit aggregate/ranking/count/summary/bucket/comparison requests, and for full-range downsampling needed to keep time-series evidence within constraints.\n"
            "For derived or multi-field statistics, plan from the requested output contract rather than a fixed query template. "
            "Each requested measure, dimension, grouping, comparison, time boundary, or derived quantity must either be represented by an explicit returned column/result shape, or listed in task_coverage.missing. "
            "If one compact query would require fragile dialect-specific state logic, prefer a simpler reliable evidence query and mark the still-missing contract fields instead of claiming coverage.\n"
            "For PromQL, metric sources must be queried by metric name and must never be repeated as source/name/__name__ label matchers. "
            "Return exactly one syntactically complete PromQL expression; never concatenate independent expressions with newlines or semicolons. "
            "Use only grounded labels inside braces. Use native PromQL range functions for rate/increase/delta requests. "
            "A datasource-native transformation in schema_linking.aggregate_targets is mandatory query semantics: do not defer it "
            "to code_interpreter, and do not claim it in task_coverage unless the generated query actually applies it. "
            "For example, a requested rate must return rate(metric[window]); downstream analysis may calculate statistics over those rate values, not reconstruct rate from raw counters. "
            "When the user requests both a range series and summary statistics, query the requested series expression and assign the statistics to downstream analysis instead of concatenating separate aggregate expressions. "
            "For multiple metric sources, preserve metric identity with label_replace into a metric_name label before an or union when labels may overlap.\n"
            "For Prometheus instant/latest values set query_execution.mode=instant. For a requested time interval or recent window, "
            "set query_execution.mode=range and provide either absolute start/end or an LLM-resolved lookback_seconds plus step_seconds. "
            "The PromQL expression for range evaluation must be an instant-vector expression such as rate(metric[5m]), never a bare metric[15m] range vector.\n"
            "Always create query_task_contract before choosing query shape. "
            "If the user asks for derived arithmetic over database values, such as difference, ratio, change, percentage, return, spread, or any custom calculation, "
            "set query_task_contract.downstream_action=\"code_interpreter\", set preferred_evidence_shape to raw_series or simple aggregate_table, and generate the simplest reliable evidence query. "
            "This downstream rule applies only after all explicitly requested datasource-native operations have been performed in the query. "
            "When query_task_contract.downstream_action is code_interpreter and preferred_evidence_shape is raw_series, the query must return raw evidence only; "
            "do not aggregate, rank, pivot, join, or reshape away the native value/time columns. "
            "For Flux, do not invent helper functions such as summarize() and do not use custom record-property syntax for dynamic aggregates; when multi-aggregate Flux is uncertain, return raw _time/_value evidence and put the derived output in task_coverage.missing with next_action_hint=\"code_interpreter\".\n"
            "Before returning, self-check whether this single query covers the whole user request. "
            "If request.fact_requests is non-empty, treat it as current-tool fact output guidance. "
            "When requested facts can be produced directly by this query, return columns or rows that make those facts verifiable from current evidence. "
            "For point_value or time_boundary facts with requirements.time_position=start|end, the boundary means the first or last available observation inside the requested range. "
            "Query the complete applicable range and preserve native timestamp/value rows; do not require an observation to equal the range boundary timestamp exactly. "
            "When a requested fact requires downstream calculation, preserve the necessary raw/current evidence and list that fact in task_coverage.missing with the downstream action. "
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
            "\"query_task_contract\": {\"intent_type\": string|null, \"required_measures\": array of strings, \"required_dimensions\": array of objects, \"required_filters\": array of objects, \"required_outputs\": array of objects or strings, \"downstream_action\": string|null, \"preferred_evidence_shape\": string|null, \"dialect_complexity_policy\": string|null, \"coverage\": object}, "
            "\"query_execution\": {\"mode\": \"instant\"|\"range\", \"start\": string|null, \"end\": string|null, \"lookback_seconds\": integer|null, \"step_seconds\": integer|null}, "
            "\"confidence\": number"
            "}."
        )

    def _schema_linking_system_prompt(self, *, database_type: str) -> str:
        dialect = dialect_for_database(database_type)
        return (
            "You perform schema linking for TSPilot text-to-query.\n"
            "Return exactly one JSON object and no markdown.\n"
            "Use only schema_preview as schema evidence.\n"
            "Select the minimal physical sources, value columns, dimension columns, and filters needed for the user request.\n"
            f"Dialect-specific grounding rules for this datasource ({database_type}): {dialect.schema_linking_rules}\n"
            "If schema_preview.physical_model is present, use it as authoritative physical-column evidence. "
            "For each requested measure, preserve logical_measure, selector_column/selector_value, physical_value_column, and aggregate_column in measures/aggregate_targets. "
            "Do not put a logical field value into value_columns as if it were a physical aggregate column unless physical_model proves that column exists after pivoting. "
            "Treat schema_preview value-domain candidates as filter evidence. "
            "When the user mentions an entity, unit, ticker, symbol, code, label, or multilingual alias that matches a candidate value, include it in required_filters. "
            "required_filters must contain every source/dimension value that is necessary for the query to answer the specific user request, not optional filters.\n"
            "Do not put time-window boundaries in required_filters; request.time_range is the authoritative temporal contract consumed by query generation.\n"
            "For Prometheus, never emit a metric source as selector_column/selector_value or required_filter. "
            "A Prometheus measure uses physical_value_column=value; actual labels are dimensions or filters. "
            "Do not put rate windows or relative time phrases in unresolved_terms because they are query semantics, not schema objects.\n"
            "If a term cannot be grounded, put it in unresolved_terms and lower confidence. Do not invent schema objects or values.\n"
            "For each filter use {\"source\": string|null, \"column\": string, \"operator\": \"=\", \"value\": string|number|boolean}.\n"
            "JSON schema: {"
            "\"sources\": object[], "
            "\"measures\": object[], "
            "\"value_columns\": object[], "
            "\"aggregate_targets\": object[], "
            "\"dimension_columns\": object[], "
            "\"required_filters\": object[], "
            "\"candidate_filters\": object[], "
            "\"unresolved_terms\": string[], "
            "\"confidence\": \"high\"|\"medium\"|\"low\", "
            "\"evidence\": string[]"
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
        fact_requests: list[Any],
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
            "fact_requests": fact_requests,
            "history": history[-6:],
            "previous_query": previous_query,
            "error": str(error) if error is not None else None,
        }
        mode = "repair" if previous_query or error else "generate"
        if mode == "repair":
            payload["repair_requirements"] = self._repair_requirements(
                schema_preview=schema_preview,
                previous_query=previous_query,
                error=error,
            )
        return "LLM SQL Query Generation JSON:\n" + json.dumps(
            {"mode": mode, "request": payload},
            ensure_ascii=False,
            default=str,
        )

    def _repair_requirements(
        self,
        *,
        schema_preview: dict,
        previous_query: str | None,
        error: Exception | str | None,
    ) -> dict[str, Any]:
        schema_linking = schema_preview.get("schema_linking") if isinstance(schema_preview, dict) else None
        required_filters: list[Any] = []
        candidate_filters: list[Any] = []
        sources: list[Any] = []
        measures: list[Any] = []
        aggregate_targets: list[Any] = []
        if isinstance(schema_linking, dict):
            required_filters = schema_linking.get("required_filters") if isinstance(schema_linking.get("required_filters"), list) else []
            candidate_filters = schema_linking.get("candidate_filters") if isinstance(schema_linking.get("candidate_filters"), list) else []
            sources = schema_linking.get("sources") if isinstance(schema_linking.get("sources"), list) else []
            measures = schema_linking.get("measures") if isinstance(schema_linking.get("measures"), list) else []
            aggregate_targets = schema_linking.get("aggregate_targets") if isinstance(schema_linking.get("aggregate_targets"), list) else []
        physical_model = schema_preview.get("physical_model") if isinstance(schema_preview.get("physical_model"), dict) else None
        return {
            "goal": (
                "Return a corrected read-only query that satisfies the user request and addresses the validation/execution error. "
                "Preserve every required filter that is applicable to the selected source. "
                "Do not remove an entity, unit, time, grouping, aggregate, or ranking constraint unless it is explicitly impossible; "
                "if impossible, explain it in task_coverage.missing. "
                "If the error says the query must be raw_series/raw_timeseries, return only native timestamp/value rows and move all aggregates or derived outputs to task_coverage.missing."
            ),
            "previous_query": previous_query,
            "error": str(error) if error is not None else None,
            "required_filters": required_filters,
            "candidate_filters": candidate_filters,
            "grounded_sources": sources,
            "measures": measures,
            "aggregate_targets": aggregate_targets,
            "physical_model": physical_model,
        }
