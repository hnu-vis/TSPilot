"""Long-term memory for insight definitions and recipes.

This module persists how to understand and generate insight types. It does not
persist concrete numeric insight instances as reusable answers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import re
from pathlib import Path
from typing import Iterable

from schemas.key_insight import (
    KeyInsightRequest,
    InsightDefinition,
    InsightMemory,
    InsightRecipe,
    MemoryCard,
    MemoryDetail,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insight_memory_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "cache_data" / "database" / "insight_memory"


def insight_memory_path(database_id: str | None = None) -> Path:
    name = "global" if not database_id else _safe_id(database_id)
    return insight_memory_dir() / f"{name}.json"


def read_persisted_insight_memory(database_id: str) -> InsightMemory:
    """Read only learned entries physically stored for one database."""

    path = insight_memory_path(database_id)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        payload = {}
    memory = InsightMemory.model_validate(
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


def system_insight_memory() -> InsightMemory:
    """Return code-owned defaults without database-learned entries."""

    return _materialize_card_memory(InsightMemory(
        definitions=[InsightDefinition.model_validate(item) for item in default_insight_definitions()],
        recipes=[InsightRecipe.model_validate(item) for item in default_insight_recipes()],
        storage_path=None,
        updated_at=None,
    ))


def read_insight_memory(database_id: str | None = None) -> InsightMemory:
    """Return runtime-effective Memory: system defaults plus database learning."""

    system = system_insight_memory()
    if not database_id:
        return system
    scoped = read_persisted_insight_memory(database_id)
    return _materialize_card_memory(_dedupe_memory(InsightMemory(
        definitions=[*system.definitions, *scoped.definitions],
        recipes=[*system.recipes, *scoped.recipes],
        cards=[*system.cards, *scoped.cards],
        details=[*system.details, *scoped.details],
        storage_path=scoped.storage_path,
        updated_at=scoped.updated_at,
    )))


def write_insight_memory(memory: InsightMemory, database_id: str | None = None) -> Path:
    if not database_id:
        raise ValueError("Key Insight Memory persistence requires a database scope; system defaults are code-owned.")
    path = insight_memory_path(database_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    persistable = InsightMemory(
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


def prompt_insight_memory_view(database_id: str | None = None) -> dict:
    return memory_cards_view(database_id)


def database_insight_memory_summary(database_id: str) -> dict:
    """Return aggregate metadata for learned memory in one database scope."""

    memory = read_persisted_insight_memory(database_id)
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
        read_insight_memory(database_id)
        if include_system or not database_id
        else read_persisted_insight_memory(database_id)
    )
    cards = memory.cards if max_cards is None else memory.cards[:max_cards]
    definition_cards = [card for card in memory.cards if card.kind == "insight_definition"]
    recipe_cards = [card for card in memory.cards if card.kind == "insight_recipe"]
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
        read_insight_memory(database_id)
        if include_system or not database_id
        else read_persisted_insight_memory(database_id)
    )
    for detail in memory.details:
        if _safe_id(detail.id) == normalized_id:
            return detail
    return None


def memory_details(database_id: str | None, memory_ids: Iterable[str]) -> list[MemoryDetail]:
    memory = read_insight_memory(database_id)
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


def default_insight_definitions() -> list[dict]:
    now = None
    common_verification = _default_verification_requirements()
    return [
        {
            "insight_type": "point_value",
            "description": "A value at a specific boundary, timestamp, category, or row.",
            "required_evidence": ["database_evidence"],
            "preferred_tool": "sql_query",
            "output_schema": {"value": "number|string", "timestamp": "string|null"},
            "verification_requirements": common_verification,
            "report_guidance": "Use for start, end, latest, earliest, or current values.",
            "updated_at": now,
        },
        {
            "insight_type": "extreme",
            "description": "A minimum or maximum value over grounded evidence.",
            "required_evidence": ["database_evidence"],
            "preferred_tool": "sql_query",
            "output_schema": {"value": "number", "operator": "min|max", "row": "object"},
            "verification_requirements": common_verification,
            "report_guidance": "Report the value and, when available, its timestamp or group.",
            "updated_at": now,
        },
        {
            "insight_type": "change",
            "description": "A delta, percentage change, return, or rate between comparable values.",
            "required_evidence": ["database_evidence", "analysis_result"],
            "preferred_tool": "code_interpreter",
            "output_schema": {"start_value": "number", "end_value": "number", "absolute_change": "number", "percentage_change": "number"},
            "verification_requirements": common_verification,
            "report_guidance": "Use code for arithmetic and cite input evidence plus calculation trace.",
            "updated_at": now,
        },
        {
            "insight_type": "seasonality",
            "description": "Recurring temporal pattern or periodic behavior in a time series.",
            "required_evidence": ["time_series", "analysis_result"],
            "preferred_tool": "code_interpreter",
            "output_schema": {"period": "string|null", "strength": "number|null", "method": "string", "limitations": "list"},
            "verification_requirements": common_verification,
            "report_guidance": "Include method, confidence, and limitations; avoid claiming periodicity from sparse data.",
            "updated_at": now,
        },
        {
            "insight_type": "custom",
            "description": "A user- or domain-defined insight not covered by built-in insight types.",
            "required_evidence": ["database_evidence|analysis_result"],
            "preferred_tool": "code_interpreter",
            "output_schema": {"value": "object|string|number|null", "method": "string", "limitations": "list"},
            "verification_requirements": common_verification,
            "report_guidance": "Define the meaning before generating the insight; final claims still require current evidence.",
            "updated_at": now,
        },
    ]


def default_insight_recipes() -> list[dict]:
    return [
        {
            "recipe_id": "recipe_extreme_max_value",
            "insight_type": "extreme",
            "name": "max_value",
            "preferred_tool": "sql_query",
            "insight_request_template": {"name": "max_value", "insight_type": "extreme", "requirements": {"operator": "max"}},
            "expected_result_schema": {"insights": [{"name": "max_value", "insight_type": "extreme"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_extreme_min_value",
            "insight_type": "extreme",
            "name": "min_value",
            "preferred_tool": "sql_query",
            "insight_request_template": {"name": "min_value", "insight_type": "extreme", "requirements": {"operator": "min"}},
            "expected_result_schema": {"insights": [{"name": "min_value", "insight_type": "extreme"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_extreme_max_time",
            "insight_type": "extreme_time",
            "name": "max_time",
            "preferred_tool": "sql_query",
            "insight_request_template": {"name": "max_time", "insight_type": "extreme_time", "requirements": {"operator": "max"}},
            "expected_result_schema": {"insights": [{"name": "max_time", "insight_type": "extreme_time"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_extreme_min_time",
            "insight_type": "extreme_time",
            "name": "min_time",
            "preferred_tool": "sql_query",
            "insight_request_template": {"name": "min_time", "insight_type": "extreme_time", "requirements": {"operator": "min"}},
            "expected_result_schema": {"insights": [{"name": "min_time", "insight_type": "extreme_time"}]},
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        },
        {
            "recipe_id": "recipe_range_change",
            "insight_type": "change",
            "name": "range_change",
            "preferred_tool": "code_interpreter",
            "insight_request_template": {"insight_type": "change", "requirements": {"needs_start_value": True, "needs_end_value": True}},
            "expected_result_schema": {
                "insights": [{"name": "percentage_change", "insight_type": "change"}],
                "metrics": {"start_value": "number", "end_value": "number", "absolute_change": "number", "percentage_change": "number"},
            },
            "verification_notes": _default_verification_requirements(),
            "updated_at": None,
        }
    ]


def _dedupe_memory(memory: InsightMemory) -> InsightMemory:
    definitions = {item.insight_type: item for item in memory.definitions}
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


def _materialize_card_memory(memory: InsightMemory) -> InsightMemory:
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


def _card_from_definition(definition: InsightDefinition) -> MemoryCard:
    insight_type = _safe_id(definition.insight_type)
    return MemoryCard(
        id=f"definition.{insight_type}",
        kind="insight_definition",
        title=definition.insight_type,
        description=definition.description,
        tags=[item for item in [definition.insight_type, *definition.required_evidence, definition.scope] if item],
        updated_at=definition.updated_at,
    )


def _detail_from_definition(definition: InsightDefinition, card: MemoryCard) -> MemoryDetail:
    guidance = definition.report_guidance or definition.description
    return MemoryDetail(
        id=card.id,
        card=card,
        insight_request=None,
        guidance=guidance,
        examples=[],
    )


def _card_from_recipe(recipe: InsightRecipe) -> MemoryCard:
    return MemoryCard(
        id=recipe_memory_card_id(recipe),
        kind="insight_recipe",
        title=recipe.name,
        description=_recipe_description(recipe),
        tags=[item for item in [recipe.insight_type, recipe.name, recipe.preferred_tool, recipe.scope] if item],
        updated_at=recipe.updated_at,
    )


def recipe_memory_card_id(recipe: InsightRecipe) -> str:
    """Return the retrieval-card identity for a persisted Key Insight recipe."""

    preferred_tool = _safe_id(recipe.preferred_tool)
    insight_type = _safe_id(recipe.insight_type)
    insight_key = ""
    if isinstance(recipe.insight_request_template, dict):
        insight_key = str(recipe.insight_request_template.get("insight_key") or "")
    name = _safe_component(insight_key or recipe.name)
    return f"recipe.{preferred_tool}.{insight_type}.{name}"


def _detail_from_recipe(recipe: InsightRecipe, card: MemoryCard) -> MemoryDetail:
    insight_request = None
    if isinstance(recipe.insight_request_template, dict) and recipe.insight_request_template:
        try:
            insight_request = KeyInsightRequest.model_validate(recipe.insight_request_template)
        except Exception:
            insight_request = None
    return MemoryDetail(
        id=card.id,
        card=card,
        insight_request=insight_request,
        preferred_tool=recipe.preferred_tool,
        guidance="; ".join(recipe.verification_notes or []) or None,
        examples=[],
    )


def _recipe_description(recipe: InsightRecipe) -> str:
    if recipe.description and recipe.description.strip():
        return recipe.description.strip()
    name = str(recipe.name or "").replace("_", " ")
    insight_type = str(recipe.insight_type or "").replace("_", " ")
    if name and insight_type and name != insight_type:
        return f"Generate the {name} insight for {insight_type} requests."
    return f"Generate a {insight_type or name} insight from current evidence."


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


def _default_verification_requirements() -> list[str]:
    return [
        "must reference current request evidence",
        "must include producing method",
        "must include calculation trace or output schema for computed values",
        "must not reuse old numeric values as final evidence",
    ]

