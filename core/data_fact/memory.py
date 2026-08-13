"""Long-term memory for fact definitions and recipes.

This module persists how to understand and generate fact types. It does not
persist concrete numeric fact instances as reusable answers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import re
from pathlib import Path
from typing import Iterable

from schemas.data_fact import (
    DataFact,
    DataFactRequest,
    FactDefinition,
    FactMemory,
    FactRecipe,
    MemoryCard,
    MemoryDetail,
)
from core.data_fact.contracts import fact_request_contract_error


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fact_memory_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "cache_data" / "database" / "fact_memory"


def fact_memory_path(database_id: str | None = None) -> Path:
    name = "global" if not database_id else _safe_id(database_id)
    return fact_memory_dir() / f"{name}.json"


def read_persisted_fact_memory(database_id: str) -> FactMemory:
    """Read only learned entries physically stored for one database."""

    path = fact_memory_path(database_id)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        payload = {}
    memory = FactMemory.model_validate(
        {
            "definitions": payload.get("definitions") or [],
            "recipes": payload.get("recipes") or [],
            "cards": payload.get("cards") or [],
            "details": payload.get("details") or [],
            "storage_path": str(path),
            "updated_at": payload.get("updated_at"),
        }
    )
    return _materialize_card_memory(_dedupe_memory(memory))


def system_fact_memory() -> FactMemory:
    """Return code-owned defaults without database-learned entries."""

    return _materialize_card_memory(FactMemory(
        definitions=[FactDefinition.model_validate(item) for item in default_fact_definitions()],
        recipes=[FactRecipe.model_validate(item) for item in default_fact_recipes()],
        storage_path=None,
        updated_at=None,
    ))


def read_fact_memory(database_id: str | None = None) -> FactMemory:
    """Return runtime-effective Memory: system defaults plus database learning."""

    system = system_fact_memory()
    if not database_id:
        return system
    scoped = read_persisted_fact_memory(database_id)
    return _materialize_card_memory(_dedupe_memory(FactMemory(
        definitions=[*system.definitions, *scoped.definitions],
        recipes=[*system.recipes, *scoped.recipes],
        cards=[*system.cards, *scoped.cards],
        details=[*system.details, *scoped.details],
        storage_path=scoped.storage_path,
        updated_at=scoped.updated_at,
    )))


def write_fact_memory(memory: FactMemory, database_id: str | None = None) -> Path:
    if not database_id:
        raise ValueError("Fact Memory persistence requires a database scope; system defaults are code-owned.")
    path = fact_memory_path(database_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    persistable = FactMemory(
        definitions=[item for item in memory.definitions if item.source != "system"],
        recipes=[item for item in memory.recipes if item.source != "system"],
        storage_path=str(path),
        updated_at=memory.updated_at,
    )
    payload = _dedupe_memory(persistable).model_dump(mode="json")
    payload["updated_at"] = utc_now_iso()
    payload["storage_path"] = str(path)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.chmod(0o600)
    temp_path.replace(path)
    return path


def prompt_fact_memory_view(database_id: str | None = None) -> dict:
    return memory_cards_view(database_id)


def database_fact_memory_summary(database_id: str) -> dict:
    """Return aggregate metadata for learned memory in one database scope."""

    memory = read_persisted_fact_memory(database_id)
    return {
        "definition_count": len(memory.definitions),
        "recipe_count": len(memory.recipes),
        "card_count": len(memory.definitions) + len(memory.recipes),
        "updated_at": memory.updated_at,
    }


def memory_cards_view(
    database_id: str | None = None,
    *,
    max_cards: int | None = 24,
    include_system: bool = True,
) -> dict:
    memory = (
        read_fact_memory(database_id)
        if include_system or not database_id
        else read_persisted_fact_memory(database_id)
    )
    cards = memory.cards if max_cards is None else memory.cards[:max_cards]
    definition_cards = [card for card in memory.cards if card.kind == "fact_definition"]
    recipe_cards = [card for card in memory.cards if card.kind == "fact_recipe"]
    return {
        "summary": {
            "definition_count": len(definition_cards),
            "recipe_count": len(recipe_cards),
            "card_count": len(memory.cards),
            "available_titles": [item.title for item in cards[:12]],
        },
        "cards": [card.model_dump(mode="json") for card in cards],
        "updated_at": memory.updated_at,
    }


def memory_detail(
    database_id: str | None,
    memory_id: str,
    *,
    include_system: bool = True,
) -> MemoryDetail | None:
    normalized_id = _safe_id(memory_id)
    memory = (
        read_fact_memory(database_id)
        if include_system or not database_id
        else read_persisted_fact_memory(database_id)
    )
    for detail in memory.details:
        if _safe_id(detail.id) == normalized_id:
            return detail
    return None


def memory_details(database_id: str | None, memory_ids: Iterable[str]) -> list[MemoryDetail]:
    memory = read_fact_memory(database_id)
    by_id = {_safe_id(detail.id): detail for detail in memory.details}
    result: list[MemoryDetail] = []
    seen: set[str] = set()
    for memory_id in memory_ids:
        normalized_id = _safe_id(memory_id)
        detail = by_id.get(normalized_id)
        if detail is None or detail.id in seen:
            continue
        result.append(detail)
        seen.add(detail.id)
    return result


def observe_fact_usage(
    *,
    database_id: str | None,
    tool_name: str,
    requests: Iterable[DataFactRequest],
    facts: Iterable[DataFact],
) -> FactMemory:
    """Persist recipes only when a requested fact was produced and verified."""

    if not database_id:
        raise ValueError("Observed Fact usage may only be persisted to a database scope.")
    memory = read_persisted_fact_memory(database_id)
    now = utc_now_iso()
    definitions = {item.fact_type: item for item in memory.definitions}
    recipes = {item.recipe_id: item for item in memory.recipes}
    verified_facts = [fact for fact in facts if fact.status == "verified"]
    verified_by_key = {fact.fact_key: fact for fact in verified_facts if fact.fact_key}

    for request in requests:
        if fact_request_contract_error(request, tool_name):
            continue
        produced_fact = verified_by_key.get(request.fact_key)
        if produced_fact is None:
            continue
        fact_type = request.fact_type or "custom"
        existing_definition = definitions.get(fact_type)
        if existing_definition is None or existing_definition.source in {"observed_fact_request", "observed_data_fact", "verified_data_fact"}:
            definitions[fact_type] = FactDefinition(
                fact_type=fact_type,
                description=f"Verified fact type '{fact_type}' from fact request '{request.name}'.",
                required_evidence=_required_evidence_for_tool(tool_name),
                preferred_tool=tool_name,
                output_schema=_output_schema_from_fact(produced_fact),
                verification_requirements=_default_verification_requirements(),
                scope=database_id or "global",
                source="verified_data_fact",
                updated_at=now,
            )
        recipe_id = _recipe_id(fact_type, request.fact_key, tool_name)
        recipes[recipe_id] = FactRecipe(
            recipe_id=recipe_id,
            fact_type=fact_type,
            name=request.name,
            preferred_tool=tool_name,
            fact_request_template=_stable_fact_request_template(request),
            expected_result_schema={
                "facts": [
                    {
                        "fact_key": request.fact_key,
                        "name": request.name,
                        "fact_type": fact_type,
                        "semantic_class": request.semantic_class,
                        "derivation": request.derivation,
                        "result_shape": request.result_shape,
                        "expected_item_count": request.expected_item_count,
                        "derived_from": request.derived_from,
                    }
                ]
            },
            verification_notes=_default_verification_requirements(),
            scope=database_id or "global",
            source="verified_data_fact",
            updated_at=now,
        )

    for fact in verified_facts:
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
    next_memory = _materialize_card_memory(FactMemory(
        definitions=list(definitions.values()),
        recipes=list(recipes.values()),
        storage_path=str(fact_memory_path(database_id)),
        updated_at=now,
    ))
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
            "recipe_id": "recipe_extreme_max_value",
            "fact_type": "extreme",
            "name": "max_value",
            "preferred_tool": "sql_query",
            "fact_request_template": {"name": "max_value", "fact_type": "extreme", "requirements": {"operator": "max"}},
            "expected_result_schema": {"facts": [{"name": "max_value", "fact_type": "extreme"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_extreme_min_value",
            "fact_type": "extreme",
            "name": "min_value",
            "preferred_tool": "sql_query",
            "fact_request_template": {"name": "min_value", "fact_type": "extreme", "requirements": {"operator": "min"}},
            "expected_result_schema": {"facts": [{"name": "min_value", "fact_type": "extreme"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_extreme_max_time",
            "fact_type": "extreme_time",
            "name": "max_time",
            "preferred_tool": "sql_query",
            "fact_request_template": {"name": "max_time", "fact_type": "extreme_time", "requirements": {"operator": "max"}},
            "expected_result_schema": {"facts": [{"name": "max_time", "fact_type": "extreme_time"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_extreme_min_time",
            "fact_type": "extreme_time",
            "name": "min_time",
            "preferred_tool": "sql_query",
            "fact_request_template": {"name": "min_time", "fact_type": "extreme_time", "requirements": {"operator": "min"}},
            "expected_result_schema": {"facts": [{"name": "min_time", "fact_type": "extreme_time"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
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
    cards = {item.id: item for item in memory.cards}
    details = {item.id: item for item in memory.details}
    return memory.model_copy(
        update={
            "definitions": list(definitions.values()),
            "recipes": list(recipes.values()),
            "cards": list(cards.values()),
            "details": list(details.values()),
        }
    )


def _materialize_card_memory(memory: FactMemory) -> FactMemory:
    cards = {card.id: card for card in memory.cards}
    details = {detail.id: detail for detail in memory.details}
    for definition in memory.definitions:
        card = _card_from_definition(definition)
        cards.setdefault(card.id, card)
        details.setdefault(card.id, _detail_from_definition(definition, card))
    for recipe in memory.recipes:
        card = _card_from_recipe(recipe)
        cards.setdefault(card.id, card)
        details.setdefault(card.id, _detail_from_recipe(recipe, card))
    return memory.model_copy(
        update={
            "cards": list(cards.values()),
            "details": list(details.values()),
        }
    )


def _card_from_definition(definition: FactDefinition) -> MemoryCard:
    fact_type = _safe_id(definition.fact_type)
    return MemoryCard(
        id=f"definition.{fact_type}",
        kind="fact_definition",
        title=definition.fact_type,
        description=definition.description,
        tags=[item for item in [definition.fact_type, *definition.required_evidence, definition.scope] if item],
        updated_at=definition.updated_at,
    )


def _detail_from_definition(definition: FactDefinition, card: MemoryCard) -> MemoryDetail:
    guidance = definition.report_guidance or definition.description
    return MemoryDetail(
        id=card.id,
        card=card,
        fact_request=None,
        guidance=guidance,
        examples=[],
    )


def _card_from_recipe(recipe: FactRecipe) -> MemoryCard:
    return MemoryCard(
        id=recipe_memory_card_id(recipe),
        kind="fact_recipe",
        title=recipe.name,
        description=_recipe_description(recipe),
        tags=[item for item in [recipe.fact_type, recipe.name, recipe.preferred_tool, recipe.scope] if item],
        updated_at=recipe.updated_at,
    )


def recipe_memory_card_id(recipe: FactRecipe) -> str:
    """Return the retrieval-card identity for a persisted Fact recipe."""

    preferred_tool = _safe_id(recipe.preferred_tool)
    fact_type = _safe_id(recipe.fact_type)
    fact_key = ""
    if isinstance(recipe.fact_request_template, dict):
        fact_key = str(recipe.fact_request_template.get("fact_key") or "")
    name = _safe_component(fact_key or recipe.name)
    return f"recipe.{preferred_tool}.{fact_type}.{name}"


def _detail_from_recipe(recipe: FactRecipe, card: MemoryCard) -> MemoryDetail:
    fact_request = None
    if isinstance(recipe.fact_request_template, dict) and recipe.fact_request_template:
        try:
            fact_request = DataFactRequest.model_validate(recipe.fact_request_template)
        except Exception:
            fact_request = None
    return MemoryDetail(
        id=card.id,
        card=card,
        fact_request=fact_request,
        preferred_tool=recipe.preferred_tool,
        guidance="; ".join(recipe.verification_notes or []) or None,
        examples=[],
    )


def _stable_fact_request_template(request: DataFactRequest) -> dict:
    """Strip request-local retrieval diagnostics before long-term persistence."""

    payload = request.model_dump(mode="json", exclude_none=True)
    requirements = dict(payload.get("requirements") or {})
    for key in ("source", "memory_card_ids", "retrieval_reason", "retrieval_confidence"):
        requirements.pop(key, None)
    payload["requirements"] = requirements
    return payload


def _recipe_description(recipe: FactRecipe) -> str:
    if recipe.description and recipe.description.strip():
        return recipe.description.strip()
    name = str(recipe.name or "").replace("_", " ")
    fact_type = str(recipe.fact_type or "").replace("_", " ")
    if name and fact_type and name != fact_type:
        return f"Generate the {name} fact for {fact_type} requests."
    return f"Generate a {fact_type or name} fact from current evidence."


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


def _safe_component(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if readable:
        return readable
    return sha1(str(value).encode("utf-8")).hexdigest()[:12]


def _recipe_id(fact_type: str, fact_key: str, tool_name: str) -> str:
    return f"recipe_{_safe_component(tool_name)}_{_safe_component(fact_type)}_{_safe_component(fact_key)}"


def _required_evidence_for_tool(tool_name: str) -> list[str]:
    if tool_name == "sql_query":
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
        "semantic_class": fact.semantic_class,
        "derivation": fact.derivation,
        "value_shape": fact.value_shape,
        "value_kind": type(fact.value).__name__ if fact.value is not None else "null",
        "item_schema": "list[FactItem]" if fact.items else None,
        "quality_flags": "list[str]",
    }
