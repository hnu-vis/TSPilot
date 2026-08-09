from __future__ import annotations

from core.data_fact import memory as fact_memory
from core.data_fact.contracts import fact_request_contract_error
from schemas.data_fact import DataFact, DataFactRequest, FactEvidenceRef, FactMemory, MemoryCard


def test_fact_memory_persists_only_verified_matching_recipes(monkeypatch):
    written: list[FactMemory] = []
    monkeypatch.setattr(fact_memory, "read_fact_memory", lambda database_id=None: FactMemory())
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


def test_code_interpreter_cannot_replace_atomic_database_fact_without_parents():
    request = DataFactRequest(
        fact_key="price.start",
        name="start_price",
        fact_type="point_value",
    )

    assert "sql_query" in fact_request_contract_error(request, "code_interpreter")


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
