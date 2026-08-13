from __future__ import annotations

import json

from core.data_fact import memory as fact_memory
from core.data_fact.contracts import fact_request_contract_error
from schemas.data_fact import DataFact, DataFactRequest, FactEvidenceRef, FactMemory, MemoryCard


def test_fact_memory_persists_only_verified_matching_recipes(monkeypatch):
    written: list[FactMemory] = []
    monkeypatch.setattr(fact_memory, "read_persisted_fact_memory", lambda database_id: FactMemory())
    monkeypatch.setattr(
        fact_memory,
        "write_fact_memory",
        lambda memory, database_id=None: written.append(memory),
    )

    memory = fact_memory.observe_fact_usage(
        database_id="demo",
        tool_name="code_interpreter",
        requests=[
            DataFactRequest(
                fact_key="price.percentage_change",
                name="percentage_change",
                fact_type="difference",
                derived_from=["price.start", "price.end"],
                requirements={
                    "source": "memory",
                    "memory_card_ids": ["recipe.old"],
                    "formula": "(end - start) / start * 100",
                },
            ),
            DataFactRequest(
                fact_key="price.volatility",
                name="volatility",
                fact_type="distribution",
            ),
        ],
        facts=[
            DataFact(
                fact_id="fact_change",
                fact_key="price.percentage_change",
                name="percentage_change",
                fact_type="difference",
                statement="Price increased by 20%.",
                value=20.0,
                method="code_interpreter",
                derived_from=["price.start", "price.end"],
                calculation_trace={"formula": "(end - start) / start * 100"},
                evidence_refs=[FactEvidenceRef(source_type="analysis", source_id="ana_change")],
            )
        ],
    )

    assert written and written[-1] is memory
    assert [recipe.name for recipe in memory.recipes] == ["percentage_change"]
    recipe = memory.recipes[0]
    assert recipe.source == "verified_data_fact"
    assert recipe.fact_request_template["derived_from"] == ["price.start", "price.end"]
    assert "source" not in recipe.fact_request_template["requirements"]
    assert "memory_card_ids" not in recipe.fact_request_template["requirements"]
    assert memory.definitions[0].preferred_tool == "code_interpreter"


def test_code_interpreter_can_produce_atomic_fact_from_database_rows():
    request = DataFactRequest(
        fact_key="price.start",
        name="start_price",
        fact_type="point_value",
    )

    assert fact_request_contract_error(request, "code_interpreter") is None


def test_memory_management_view_can_load_all_cards_without_expanding_prompt_view(monkeypatch):
    cards = [
        MemoryCard(
            id=f"recipe.test.{index}",
            kind="fact_recipe",
            title=f"recipe_{index}",
            description="Test recipe.",
        )
        for index in range(30)
    ]
    monkeypatch.setattr(fact_memory, "read_fact_memory", lambda database_id=None: FactMemory(cards=cards))

    assert len(fact_memory.memory_cards_view()["cards"]) == 24
    management_view = fact_memory.memory_cards_view(max_cards=None)
    assert len(management_view["cards"]) == 30
    assert management_view["summary"]["recipe_count"] == 30


def test_database_view_excludes_system_defaults(monkeypatch):
    learned = MemoryCard(
        id="recipe.sql_query.point_value.learned",
        kind="fact_recipe",
        title="learned",
        description="Learned for this database.",
    )
    monkeypatch.setattr(
        fact_memory,
        "read_persisted_fact_memory",
        lambda database_id: FactMemory(cards=[learned]),
    )

    scoped = fact_memory.memory_cards_view("demo", max_cards=None, include_system=False)

    assert [card["id"] for card in scoped["cards"]] == [learned.id]


def test_database_memory_summary_counts_only_learned_scope(monkeypatch):
    monkeypatch.setattr(
        fact_memory,
        "read_persisted_fact_memory",
        lambda database_id: FactMemory.model_validate({
            "definitions": [{
                "fact_type": "database_metric",
                "description": "Learned metric.",
                "source": "verified_data_fact",
                "scope": database_id,
            }],
            "recipes": [{
                "recipe_id": "recipe_learned_demo",
                "fact_type": "database_metric",
                "name": "database metric",
                "preferred_tool": "sql_query",
                "source": "verified_data_fact",
                "scope": database_id,
            }],
            "updated_at": "2026-08-13T10:00:00Z",
        }),
    )

    summary = fact_memory.database_fact_memory_summary("demo")

    assert summary == {
        "definition_count": 1,
        "recipe_count": 1,
        "card_count": 2,
        "updated_at": "2026-08-13T10:00:00Z",
    }


def test_write_database_memory_strips_system_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(fact_memory, "fact_memory_dir", lambda: tmp_path)
    memory = FactMemory.model_validate({
        "definitions": [
            *fact_memory.default_fact_definitions(),
            {
                "fact_type": "database_metric",
                "description": "Learned metric.",
                "source": "verified_data_fact",
                "scope": "demo",
            },
        ],
        "recipes": [
            *fact_memory.default_fact_recipes(),
            {
                "recipe_id": "recipe_learned_demo",
                "fact_type": "database_metric",
                "name": "database metric",
                "preferred_tool": "code_interpreter",
                "source": "verified_data_fact",
                "scope": "demo",
            },
        ],
    })

    path = fact_memory.write_fact_memory(memory, "demo")
    payload = json.loads(path.read_text())

    assert [item["fact_type"] for item in payload["definitions"]] == ["database_metric"]
    assert [item["recipe_id"] for item in payload["recipes"]] == ["recipe_learned_demo"]
    assert payload["cards"] == []
    assert payload["details"] == []
