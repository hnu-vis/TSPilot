from __future__ import annotations

import json

from core.key_insight import memory as insight_memory
from core.key_insight.contracts import insight_request_contract_error
from schemas.key_insight import KeyInsightRequest, InsightMemory, MemoryCard


def test_code_interpreter_can_produce_atomic_insight_from_database_rows():
    request = KeyInsightRequest(
        insight_key="price.start",
        name="start_price",
        insight_type="point_value",
    )

    assert insight_request_contract_error(request, "code_interpreter") is None


def test_memory_management_view_can_load_all_cards_without_expanding_prompt_view(monkeypatch):
    cards = [
        MemoryCard(
            id=f"recipe.test.{index}",
            kind="insight_recipe",
            title=f"recipe_{index}",
            description="Test recipe.",
        )
        for index in range(30)
    ]
    monkeypatch.setattr(insight_memory, "read_insight_memory", lambda database_id=None: InsightMemory(cards=cards))

    assert len(insight_memory.memory_cards_view()["cards"]) == 24
    management_view = insight_memory.memory_cards_view(max_cards=None)
    assert len(management_view["cards"]) == 30
    assert management_view["summary"]["recipe_count"] == 30


def test_database_view_excludes_system_defaults(monkeypatch):
    learned = MemoryCard(
        id="recipe.sql_query.point_value.learned",
        kind="insight_recipe",
        title="learned",
        description="Learned for this database.",
    )
    monkeypatch.setattr(
        insight_memory,
        "read_persisted_insight_memory",
        lambda database_id: InsightMemory(cards=[learned]),
    )

    scoped = insight_memory.memory_cards_view("demo", max_cards=None, include_system=False)

    assert [card["id"] for card in scoped["cards"]] == [learned.id]


def test_database_memory_summary_counts_only_learned_scope(monkeypatch):
    monkeypatch.setattr(
        insight_memory,
        "read_persisted_insight_memory",
        lambda database_id: InsightMemory.model_validate({
            "definitions": [{
                "insight_type": "database_metric",
                "description": "Learned metric.",
                "source": "verified_key_insight",
                "scope": database_id,
            }],
            "recipes": [{
                "recipe_id": "recipe_learned_demo",
                "insight_type": "database_metric",
                "name": "database metric",
                "preferred_tool": "sql_query",
                "source": "verified_key_insight",
                "scope": database_id,
            }],
            "updated_at": "2026-08-13T10:00:00Z",
        }),
    )

    summary = insight_memory.database_insight_memory_summary("demo")

    assert summary == {
        "definition_count": 1,
        "recipe_count": 1,
        "card_count": 2,
        "updated_at": "2026-08-13T10:00:00Z",
    }


def test_write_database_memory_strips_system_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(insight_memory, "insight_memory_dir", lambda: tmp_path)
    memory = InsightMemory.model_validate({
        "definitions": [
            *insight_memory.default_insight_definitions(),
            {
                "insight_type": "database_metric",
                "description": "Learned metric.",
                "source": "verified_key_insight",
                "scope": "demo",
            },
        ],
        "recipes": [
            *insight_memory.default_insight_recipes(),
            {
                "recipe_id": "recipe_learned_demo",
                "insight_type": "database_metric",
                "name": "database metric",
                "preferred_tool": "code_interpreter",
                "source": "verified_key_insight",
                "scope": "demo",
            },
        ],
    })

    path = insight_memory.write_insight_memory(memory, "demo")
    payload = json.loads(path.read_text())

    assert [item["insight_type"] for item in payload["definitions"]] == ["database_metric"]
    assert [item["recipe_id"] for item in payload["recipes"]] == ["recipe_learned_demo"]
    assert payload["cards"] == []
    assert payload["details"] == []
