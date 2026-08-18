"""LLM semantic binding for computation-only Code Interpreter outputs."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.analysis.python_runner import AnalysisCodeError
from schemas.analysis import ComputedInsight
from schemas.key_insight import InsightEvidenceRef, InsightItem, KeyInsight, KeyInsightRequest
from runtime.prompt_locale import prompt_locale_instruction


class InsightBindingError(AnalysisCodeError):
    """Raised when computed values cannot be semantically bound to their requests."""


class _ItemAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    label: str = Field(min_length=1)


class _Binding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insight_key: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    item_annotations: list[_ItemAnnotation] = Field(default_factory=list)


class _BindingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bindings: list[_Binding] = Field(min_length=1)


class LLMInsightBinder:
    """Bind immutable computed values to requested Insight semantics."""

    def __init__(self, llm):
        self._llm = llm

    async def bind(
        self,
        *,
        requests: list[KeyInsightRequest],
        computed: list[ComputedInsight],
        analysis_id: str,
        analysis_goal: str,
        input_evidence_id: str,
        input_source_refs: list[str] | None = None,
        response_language: str,
    ) -> list[KeyInsight]:
        if self._llm is None:
            raise InsightBindingError("code_interpreter requires an LLM Insight Binder")
        requests_by_key = {request.insight_key: request for request in requests}
        computed_by_key = {item.insight_key: item for item in computed}
        if set(computed_by_key) != set(requests_by_key):
            raise InsightBindingError(
                "computed_insights keys must exactly match requested insight keys; "
                f"requested={sorted(requests_by_key)}, computed={sorted(computed_by_key)}"
            )
        payload = {
            "analysis_goal": analysis_goal,
            "requests": [request.model_dump(mode="json", exclude_none=True) for request in requests],
            "computed_insights": [item.model_dump(mode="json") for item in computed],
        }
        bindings = await self._invoke(payload, response_language=response_language)
        bindings_by_key = {item.insight_key.strip(): item for item in bindings}
        if set(bindings_by_key) != set(requests_by_key):
            raise InsightBindingError("Insight Binder must return exactly one binding for every requested insight")

        result: list[KeyInsight] = []
        for key, request in requests_by_key.items():
            candidate = computed_by_key[key]
            binding = bindings_by_key[key]
            statement = binding.statement.strip()
            items = _bind_items(candidate.items, [item.model_dump(mode="json") for item in binding.item_annotations])
            evidence_refs = [
                InsightEvidenceRef(source_type="analysis", source_id=analysis_id, label=analysis_goal),
                *[
                    _source_evidence_ref(ref)
                    for ref in (input_source_refs or [f"evidence:{input_evidence_id}"])
                ],
                *[
                    InsightEvidenceRef(source_type="derived_evidence", source_id=evidence_id)
                    for evidence_id in candidate.derived_evidence_ids
                ],
            ]
            result.append(
                KeyInsight(
                    insight_id=_insight_id(analysis_id, key, candidate.value, candidate.items),
                    insight_key=key,
                    name=request.name,
                    insight_type=request.insight_type,
                    statement=statement,
                    value=candidate.value,
                    value_shape=request.result_shape or ("collection" if items else None),
                    items=items,
                    semantic_class=request.semantic_class,
                    derivation=request.derivation,
                    selection=request.selection,
                    subject=request.subject,
                    dimensions=request.dimensions,
                    time_range=request.time_range,
                    method="code_interpreter",
                    evidence_refs=evidence_refs,
                    calculation_trace=candidate.calculation_trace,
                    derived_from=request.derived_from,
                    status="unavailable" if candidate.unavailable_reason else "verified",
                    unavailable_reason=candidate.unavailable_reason,
                )
            )
        return result
    async def _invoke(self, payload: dict, *, response_language: str) -> list[_Binding]:
        system = prompt_locale_instruction(response_language) + (
            "You bind immutable Python computation outputs to Key Insight semantics. "
            "Return exactly one JSON object with a bindings list. Each binding must contain only "
            "insight_key, a concise evidence-grounded statement, and optional item_annotations. "
            "Do not calculate, alter, round, replace, or infer any value. Do not add insights. "
            "item_annotations may identify an item by zero-based index and add label only; preserve item order."
        )
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        last_error: Exception | None = None
        for attempt in range(2):
            raw_content = ""
            try:
                if hasattr(self._llm, "with_structured_output"):
                    runnable = self._llm.with_structured_output(
                        _BindingResponse, method="json_schema", include_raw=True,
                    )
                    bundle = await asyncio.wait_for(runnable.ainvoke(messages), timeout=30)
                    if isinstance(bundle, dict):
                        raw_content = _llm_content(bundle.get("raw"))
                        parsed = bundle.get("parsed")
                        if parsed is None:
                            raise ValueError(bundle.get("parsing_error") or "structured output was not parsed")
                        parsed = parsed if isinstance(parsed, _BindingResponse) else _BindingResponse.model_validate(parsed)
                    else:
                        parsed = bundle if isinstance(bundle, _BindingResponse) else _BindingResponse.model_validate(bundle)
                else:
                    response = await asyncio.wait_for(self._llm.ainvoke(messages), timeout=30)
                    raw_content = _llm_content(response)
                    parsed = _BindingResponse.model_validate_json(raw_content)
                return parsed.bindings
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    messages.extend([
                        ("assistant", raw_content),
                        ("human", f"Your binding output violated the required schema: {exc}. Return one corrected schema-valid object only."),
                    ])
        raise InsightBindingError(f"Insight Binder violated its output contract: {last_error}") from last_error


def _source_evidence_ref(ref: str) -> InsightEvidenceRef:
    source_type, separator, source_id = str(ref or "").partition(":")
    if not separator:
        source_type, source_id = "query", source_type
    elif source_type == "evidence":
        source_type = "query"
    return InsightEvidenceRef(source_type=source_type, source_id=source_id)


def _bind_items(raw_items: list[dict[str, Any]], annotations: Any) -> list[InsightItem]:
    annotations_by_index = {
        item.get("index"): item
        for item in annotations or []
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    result: list[InsightItem] = []
    for index, raw in enumerate(raw_items):
        payload = dict(raw)
        annotation = annotations_by_index.get(index, {})
        item_id = str(payload.pop("item_id", "")).strip() or _item_id(index, raw)
        known = {
            key: payload.pop(key)
            for key in ("value", "rank", "timestamp", "source_item_ids", "locator")
            if key in payload
        }
        result.append(
            InsightItem(
                item_id=item_id,
                label=str(annotation.get("label") or payload.pop("label", "")).strip() or None,
                dimensions=payload,
                **known,
            )
        )
    return result


def _item_id(index: int, value: Any) -> str:
    digest = hashlib.sha1(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"item_{index}_{digest}"


def _insight_id(analysis_id: str, key: str, value: Any, items: Any) -> str:
    digest = hashlib.sha1(
        json.dumps({"value": value, "items": items}, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return f"ins_{analysis_id}_{key}_{digest}"


def _llm_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()
