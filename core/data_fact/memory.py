"""Long-term memory for fact definitions and recipes.

This module persists how to understand and generate fact types. It does not
persist concrete numeric fact instances as reusable answers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Iterable

from schemas.data_fact import DataFact, DataFactRequest, FactDefinition, FactMemory, FactRecipe


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fact_memory_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "cache_data" / "database" / "fact_memory"


def fact_memory_path(database_id: str | None = None) -> Path:
    name = "global" if not database_id else _safe_id(database_id)
    return fact_memory_dir() / f"{name}.json"


def read_fact_memory(database_id: str | None = None) -> FactMemory:
    path = fact_memory_path(database_id)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        payload = {}
    memory = FactMemory.model_validate(
        {
            "definitions": [*default_fact_definitions(), *(payload.get("definitions") or [])],
            "recipes": [*default_fact_recipes(), *(payload.get("recipes") or [])],
            "storage_path": str(path),
            "updated_at": payload.get("updated_at"),
        }
    )
    return _dedupe_memory(memory)


def write_fact_memory(memory: FactMemory, database_id: str | None = None) -> Path:
    path = fact_memory_path(database_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dedupe_memory(memory).model_dump(mode="json")
    payload["updated_at"] = utc_now_iso()
    payload["storage_path"] = str(path)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.chmod(0o600)
    temp_path.replace(path)
    return path


def prompt_fact_memory_view(database_id: str | None = None) -> dict:
    global_memory = read_fact_memory(None)
    scoped_memory = read_fact_memory(database_id) if database_id else FactMemory()
    memory = _dedupe_memory(
        FactMemory(
            definitions=[*global_memory.definitions, *scoped_memory.definitions],
            recipes=[*global_memory.recipes, *scoped_memory.recipes],
            storage_path=scoped_memory.storage_path or global_memory.storage_path,
            updated_at=scoped_memory.updated_at or global_memory.updated_at,
        )
    )
    definitions = memory.definitions[:12]
    recipes = memory.recipes[:8]
    return {
        "summary": {
            "definition_count": len(memory.definitions),
            "recipe_count": len(memory.recipes),
            "available_fact_types": [item.fact_type for item in definitions],
            "available_recipes": [item.name for item in recipes],
        },
        "definitions": [
            {
                "fact_type": item.fact_type,
                "preferred_tool": item.preferred_tool,
                "scope": item.scope,
            }
            for item in definitions
        ],
        "recipes": [
            {
                "fact_type": item.fact_type,
                "name": item.name,
                "preferred_tool": item.preferred_tool,
                "scope": item.scope,
            }
            for item in recipes
        ],
        "updated_at": memory.updated_at,
    }


def observe_fact_usage(
    *,
    database_id: str | None,
    tool_name: str,
    requests: Iterable[DataFactRequest],
    facts: Iterable[DataFact],
) -> FactMemory:
    """Persist reusable fact definitions/recipes inferred from usage."""

    memory = read_fact_memory(database_id)
    now = utc_now_iso()
    definitions = {item.fact_type: item for item in memory.definitions}
    recipes = {item.recipe_id: item for item in memory.recipes}

    for request in requests:
        fact_type = request.fact_type or "custom"
        definitions.setdefault(
            fact_type,
            FactDefinition(
                fact_type=fact_type,
                description=f"Observed fact type '{fact_type}' from fact request '{request.name}'.",
                required_evidence=_required_evidence_for_tool(tool_name),
                preferred_tool=tool_name,
                output_schema=_output_schema_from_request(request),
                verification_requirements=_default_verification_requirements(),
                scope=database_id or "global",
                source="observed_fact_request",
                updated_at=now,
            ),
        )
        recipe_id = _recipe_id(fact_type, request.name, tool_name)
        recipes.setdefault(
            recipe_id,
            FactRecipe(
                recipe_id=recipe_id,
                fact_type=fact_type,
                name=request.name,
                preferred_tool=tool_name,
                fact_request_template=request.model_dump(mode="json", exclude_none=True),
                expected_result_schema={"facts": [{"name": request.name, "fact_type": fact_type}]},
                verification_notes=_default_verification_requirements(),
                scope=database_id or "global",
                source="observed_fact_request",
                updated_at=now,
            ),
        )

    for fact in facts:
        fact_type = fact.fact_type or "custom"
        definitions.setdefault(
            fact_type,
            FactDefinition(
                fact_type=fact_type,
                description=f"Observed fact type '{fact_type}' produced by {fact.method}.",
                required_evidence=_required_evidence_for_tool(fact.method),
                preferred_tool=fact.method,
                output_schema=_output_schema_from_fact(fact),
                verification_requirements=_default_verification_requirements(),
                scope=database_id or "global",
                source="observed_data_fact",
                updated_at=now,
            ),
        )
    next_memory = FactMemory(
        definitions=list(definitions.values()),
        recipes=list(recipes.values()),
        storage_path=str(fact_memory_path(database_id)),
        updated_at=now,
    )
    write_fact_memory(next_memory, database_id)
    return next_memory


def default_fact_definitions() -> list[dict]:
    now = None
    common_verification = _default_verification_requirements()
    return [
        {
            "fact_type": "point_value",
            "description": "A value at a specific boundary, timestamp, category, or row.",
            "required_evidence": ["database_evidence"],
            "preferred_tool": "sql_query",
            "output_schema": {"value": "number|string", "timestamp": "string|null"},
            "verification_requirements": common_verification,
            "report_guidance": "Use for start, end, latest, earliest, or current values.",
            "updated_at": now,
        },
        {
            "fact_type": "extreme",
            "description": "A minimum or maximum value over grounded evidence.",
            "required_evidence": ["database_evidence"],
            "preferred_tool": "sql_query",
            "output_schema": {"value": "number", "operator": "min|max", "row": "object"},
            "verification_requirements": common_verification,
            "report_guidance": "Report the value and, when available, its timestamp or group.",
            "updated_at": now,
        },
        {
            "fact_type": "change",
            "description": "A delta, percentage change, return, or rate between comparable values.",
            "required_evidence": ["database_evidence", "analysis_result"],
            "preferred_tool": "code_interpreter",
            "output_schema": {"start_value": "number", "end_value": "number", "absolute_change": "number", "percentage_change": "number"},
            "verification_requirements": common_verification,
            "report_guidance": "Use code for arithmetic and cite input evidence plus calculation trace.",
            "updated_at": now,
        },
        {
            "fact_type": "seasonality",
            "description": "Recurring temporal pattern or periodic behavior in a time series.",
            "required_evidence": ["time_series", "analysis_result"],
            "preferred_tool": "code_interpreter",
            "output_schema": {"period": "string|null", "strength": "number|null", "method": "string", "limitations": "list"},
            "verification_requirements": common_verification,
            "report_guidance": "Include method, confidence, and limitations; avoid claiming periodicity from sparse data.",
            "updated_at": now,
        },
        {
            "fact_type": "custom",
            "description": "A user- or domain-defined fact not covered by built-in fact types.",
            "required_evidence": ["database_evidence|analysis_result"],
            "preferred_tool": "code_interpreter",
            "output_schema": {"value": "object|string|number|null", "method": "string", "limitations": "list"},
            "verification_requirements": common_verification,
            "report_guidance": "Define the meaning before generating the fact; final claims still require current evidence.",
            "updated_at": now,
        },
    ]


def default_fact_recipes() -> list[dict]:
    return [
        {
            "recipe_id": "recipe_range_change",
            "fact_type": "change",
            "name": "range_change",
            "preferred_tool": "code_interpreter",
            "fact_request_template": {"fact_type": "change", "requirements": {"needs_start_value": True, "needs_end_value": True}},
            "expected_result_schema": {
                "facts": [{"name": "percentage_change", "fact_type": "change"}],
                "metrics": {"start_value": "number", "end_value": "number", "absolute_change": "number", "percentage_change": "number"},
            },
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        }
    ]


def _dedupe_memory(memory: FactMemory) -> FactMemory:
    definitions = {item.fact_type: item for item in memory.definitions}
    recipes = {item.recipe_id: item for item in memory.recipes}
    return memory.model_copy(
        update={
            "definitions": list(definitions.values()),
            "recipes": list(recipes.values()),
        }
    )


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "global"


def _recipe_id(fact_type: str, name: str, tool_name: str) -> str:
    return _safe_id(f"recipe_{tool_name}_{fact_type}_{name}")


def _required_evidence_for_tool(tool_name: str) -> list[str]:
    if tool_name in {"sql_query", "query_database"}:
        return ["database_evidence"]
    if tool_name == "code_interpreter":
        return ["database_evidence", "analysis_result"]
    if tool_name == "forecast":
        return ["time_series", "forecast_result"]
    if tool_name == "anomaly":
        return ["time_series", "anomaly_result"]
    return ["tool_observation"]


def _default_verification_requirements() -> list[str]:
    return [
        "must reference current request evidence",
        "must include producing method",
        "must include calculation trace or output schema for computed values",
        "must not reuse old numeric values as final evidence",
    ]


def _output_schema_from_request(request: DataFactRequest) -> dict:
    return {
        "name": request.name,
        "fact_type": request.fact_type,
        "value": "unknown",
        "requirements": request.requirements,
    }


def _output_schema_from_fact(fact: DataFact) -> dict:
    return {
        "name": fact.name,
        "fact_type": fact.fact_type,
        "value_kind": type(fact.value).__name__ if fact.value is not None else "null",
        "quality_flags": "list[str]",
    }
