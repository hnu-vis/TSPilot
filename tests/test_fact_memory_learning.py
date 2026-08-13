from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.settings import Settings
from core.data_fact import learning
from core.data_fact.learning import (
    FactLearningOutbox,
    FactMemoryLearner,
    FactMemoryLearningWorker,
    extract_learning_candidates,
    reset_legacy_fact_memory_once,
    separate_fact_memory_scopes_once,
)
from core.data_fact.embedding_store import FactMemoryEmbeddingStore
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.data_fact import (
    DataFact,
    DataFactRequest,
    FactDefinition,
    FactEvidenceRef,
    FactEvent,
    FactMemory,
    FactRecipe,
    MemoryCard,
    MemoryDetail,
)
from schemas.database_context import DatabaseContext
from schemas.output import AnswerClaim, AnswerReference, FinalAnswer


def _terminal_state(tmp_path, *, request_id: str = "req-learning"):
    settings = Settings(tspilot_root=str(tmp_path), conversation_log_enabled=False)
    state = build_request_state(
        ChatRequest(
            message="返回最后一个功率值",
            database_context=DatabaseContext(database_id="energy", database_type="influxdb"),
        ),
        settings,
    )
    state.request_id = request_id
    state.status = "completed"
    request = DataFactRequest(
        fact_key="energy.latest",
        name="latest_energy",
        fact_type="point_value",
        subject="appliances_energy_wh",
        time_range={"start": "2016-01-01", "end": "2016-02-01"},
        dimensions={"building": "A"},
        requirements={
            "time_position": "end",
            "row_filters": {"building": "A"},
            "source": "memory",
            "memory_card_ids": ["old"],
        },
    )
    fact = DataFact(
        fact_id="fact_latest_energy",
        fact_key=request.fact_key,
        name=request.name,
        fact_type=request.fact_type,
        statement="The latest value is 420.",
        value=420,
        subject=request.subject,
        method="sql_query",
        evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi-secret")],
        calculation_trace={"row": {"value": 420}, "value_key": "value", "time_key": "time"},
    )
    state.fact_set.requests = [request]
    state.fact_set.facts = [fact]
    state.fact_events = [FactEvent(iteration=1, tool_name="sql_query", produced_fact_ids=[fact.fact_id])]
    state.final_answer_draft = FinalAnswer(
        summary="最后一个值是 420。",
        claims=[AnswerClaim(claim_id="claim-1", text="最后值", fact_ids=[fact.fact_id])],
        references=[AnswerReference(source_type="fact", source_id=fact.fact_id, label="最后值")],
    )
    return state


def test_terminal_learning_candidate_is_referenced_verified_and_value_free(tmp_path):
    state = _terminal_state(tmp_path)

    candidates = extract_learning_candidates(state)

    assert len(candidates) == 1
    candidate = candidates[0]
    payload = candidate.model_dump(mode="json")
    serialized = json.dumps(payload, ensure_ascii=False)
    assert candidate.database_id == "energy"
    assert candidate.fact_request.time_range is None
    assert candidate.fact_request.dimensions == {"building": "<instance-value-omitted>"}
    assert candidate.fact_request.requirements["row_filters"]["building"] == "<instance-value-omitted>"
    assert candidate.fact_request.requirements["time_position"] == "end"
    assert "420" not in serialized
    assert "evi-secret" not in serialized
    assert "memory_card_ids" not in serialized


def test_unreferenced_partial_or_failed_fact_is_not_queued(tmp_path):
    state = _terminal_state(tmp_path)
    state.final_answer_draft = FinalAnswer(summary="answer")
    assert extract_learning_candidates(state) == []
    state.final_answer_draft = FinalAnswer(
        summary="answer",
        claims=[AnswerClaim(claim_id="c", text="x", fact_ids=["fact_latest_energy"])],
    )
    state.status = "partial"
    assert extract_learning_candidates(state) == []


@pytest.mark.asyncio
async def test_outbox_triggers_at_twenty_and_worker_chunks_llm_batches(tmp_path):
    outbox = FactLearningOutbox(tmp_path / "learning")
    for index in range(20):
        assert outbox.enqueue_request(_terminal_state(tmp_path, request_id=f"req-{index}"))
    assert outbox.due_databases(batch_size=20, max_wait_seconds=600) == ["energy"]

    class _Learner:
        def __init__(self):
            self.outbox = outbox
            self.batch_sizes = []

        async def process(self, jobs):
            self.batch_sizes.append(len(jobs))
            for job in jobs:
                outbox.finish(job, status="completed", reason="test")

    learner = _Learner()
    worker = FactMemoryLearningWorker(learner, batch_size=20, llm_chunk_size=5)
    assert await worker.process_due_once() == 20
    assert learner.batch_sizes == [5, 5, 5, 5]
    assert len(list((tmp_path / "learning" / "completed").glob("*.json"))) == 20


def test_outbox_triggers_when_oldest_candidate_waited_ten_minutes(tmp_path):
    outbox = FactLearningOutbox(tmp_path / "learning")
    job_id = outbox.enqueue_request(_terminal_state(tmp_path))[0]
    path = tmp_path / "learning" / "pending" / f"{job_id}.json"
    payload = json.loads(path.read_text())
    payload["queued_at"] = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    outbox._write_atomic(path, payload)

    assert outbox.due_databases(batch_size=20, max_wait_seconds=600) == ["energy"]


class _EmbeddingProvider:
    model = "test-embedding"

    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _Response:
    def __init__(self, payload):
        self.content = json.dumps(payload)
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class _LearningLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        user_payload = json.loads(messages[1][1])
        if self.calls == 1:
            candidate_id = user_payload["candidates"][0]["candidate_id"]
            return _Response({"abstracted": [{
                "candidate_ids": [candidate_id],
                "definition": {
                    "fact_type": "point_value",
                    "description": "Latest scalar observation for a database metric.",
                    "required_evidence": ["database_evidence"],
                    "preferred_tool": "sql_query",
                    "output_schema": {"value": "number|string"},
                    "verification_requirements": ["must reference current request evidence"],
                    "scope": "energy",
                    "source": "verified_data_fact",
                },
                "recipe": {
                    "recipe_id": "",
                    "fact_type": "point_value",
                    "name": "latest_energy",
                    "preferred_tool": "sql_query",
                    "fact_request_template": {
                        "fact_key": "energy.latest",
                        "name": "latest_energy",
                        "fact_type": "point_value",
                        "subject": "appliances_energy_wh",
                        "requirements": {"time_position": "end"},
                    },
                    "expected_result_schema": {"value": "scalar"},
                    "verification_notes": ["must reference current request evidence"],
                    "scope": "energy",
                    "source": "verified_data_fact",
                    "description": "Generate the latest appliances energy observation.",
                },
            }]})
        item = user_payload["abstracted"][0]
        return _Response({"decisions": [{
            "candidate_ids": item["candidate_ids"],
            "action": "create",
            "reason": "Reusable and distinct.",
            "definition": item["definition"],
            "recipe": item["recipe"],
        }]})


@pytest.mark.asyncio
async def test_batch_learner_abstracts_reviews_and_commits(monkeypatch, tmp_path):
    outbox = FactLearningOutbox(tmp_path / "learning")
    outbox.enqueue_request(_terminal_state(tmp_path))
    jobs = outbox.claim("energy", limit=20, lease_seconds=60)
    written = []
    monkeypatch.setattr(learning, "read_persisted_fact_memory", lambda database_id: FactMemory())
    monkeypatch.setattr(learning, "write_fact_memory", lambda memory, database_id=None: written.append((database_id, memory)))
    learner = FactMemoryLearner(
        llm=_LearningLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=FactMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=outbox,
    )

    await learner.process(jobs)

    assert len(written) == 1
    assert written[0][0] == "energy"
    assert len(written[0][1].recipes) == 1
    recipe = written[0][1].recipes[0]
    assert recipe.recipe_id.startswith("recipe_learned_")
    assert recipe.description == "Generate the latest appliances energy observation."
    assert list((tmp_path / "learning" / "completed").glob("*.json"))


class _RepairingMergeLLM(_LearningLLM):
    async def ainvoke(self, messages):
        self.calls += 1
        user_payload = json.loads(messages[1][1])
        if self.calls == 1:
            candidate_id = user_payload["candidates"][0]["candidate_id"]
            return _Response({"abstracted": [{
                "candidate_ids": [candidate_id],
                "definition": {
                    "fact_type": "point_value",
                    "description": "Latest scalar observation for a database metric.",
                    "preferred_tool": "sql_query",
                },
                "recipe": {
                    "recipe_id": "",
                    "fact_type": "point_value",
                    "name": "Latest appliances value",
                    "preferred_tool": "sql_query",
                    "fact_request_template": {
                        "name": "Latest appliances value",
                        "fact_type": "point_value",
                        "subject": "appliances_energy_wh",
                        "requirements": {"time_position": "end"},
                    },
                },
            }]})
        item = user_payload["abstracted"][0]
        if self.calls == 2:
            return _Response({"decisions": [{
                "candidate_ids": item["candidate_ids"],
                "action": "merge",
                "target_recipe_id": "invented.recipe.id",
                "reason": "Equivalent recipe.",
                "definition": item["definition"],
                "recipe": item["recipe"],
            }]})
        repair_payload = json.loads(messages[-1][1])
        target = repair_payload["allowed_target_recipe_ids_by_candidate"][item["candidate_ids"][0]][0]
        return _Response({"decisions": [{
            "candidate_ids": item["candidate_ids"],
            "action": "merge",
            "target_recipe_id": target,
            "reason": "Equivalent full Fact contract.",
            "definition": item["definition"],
            "recipe": item["recipe"],
        }]})


@pytest.mark.asyncio
async def test_learning_repairs_invented_merge_target_and_updates_existing_recipe(monkeypatch, tmp_path):
    outbox = FactLearningOutbox(tmp_path / "learning")
    outbox.enqueue_request(_terminal_state(tmp_path))
    jobs = outbox.claim("energy", limit=20, lease_seconds=60)
    existing_recipe = FactRecipe(
        recipe_id="recipe_existing_latest_energy",
        fact_type="point_value",
        name="appliances_energy_wh latest observed value",
        preferred_tool="sql_query",
        fact_request_template={
            "name": "appliances_energy_wh latest observed value",
            "fact_type": "point_value",
            "subject": "appliances_energy_wh",
            "requirements": {"time_position": "end"},
        },
        source="verified_data_fact",
        scope="energy",
    )
    persisted = FactMemory(
        definitions=[FactDefinition(fact_type="point_value", description="Latest value")],
        recipes=[existing_recipe],
        cards=[MemoryCard(
            id="recipe.sql_query.point_value.appliances_energy_wh_latest_observed_value",
            kind="fact_recipe",
            title=existing_recipe.name,
            description="Equivalent latest observation recipe.",
        )],
        details=[MemoryDetail(
            id="recipe.sql_query.point_value.appliances_energy_wh_latest_observed_value",
            card=MemoryCard(
                id="recipe.sql_query.point_value.appliances_energy_wh_latest_observed_value",
                kind="fact_recipe",
                title=existing_recipe.name,
                description="Equivalent latest observation recipe.",
            ),
            fact_request=DataFactRequest.model_validate(existing_recipe.fact_request_template),
            preferred_tool="sql_query",
        )],
        updated_at="revision-1",
    )
    written = []
    monkeypatch.setattr(learning, "read_persisted_fact_memory", lambda database_id: persisted)
    monkeypatch.setattr(learning, "write_fact_memory", lambda memory, database_id=None: written.append(memory))
    learner = FactMemoryLearner(
        llm=_RepairingMergeLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=FactMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=outbox,
        neighbor_threshold=0.0,
    )
    await learner.process(jobs)

    assert len(written) == 1
    assert [recipe.recipe_id for recipe in written[0].recipes] == [existing_recipe.recipe_id]
    assert learner.llm.calls == 3
    assert list((tmp_path / "learning" / "completed").glob("*.json"))
    assert not list((tmp_path / "learning" / "failed").glob("*.json"))


def test_neighbors_are_available_to_every_candidate_in_merged_abstraction(tmp_path):
    learner = FactMemoryLearner(
        llm=_LearningLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=FactMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=FactLearningOutbox(tmp_path / "learning"),
    )
    payload = {
        "decisions": [{
            "candidate_ids": ["candidate-a", "candidate-b"],
            "action": "merge",
            "target_recipe_id": "recipe-shared",
            "reason": "Same full contract.",
            "definition": {"fact_type": "point_value", "description": "Latest value"},
            "recipe": {
                "recipe_id": "",
                "fact_type": "point_value",
                "name": "latest value",
                "preferred_tool": "sql_query",
                "fact_request_template": {
                    "name": "latest value",
                    "fact_type": "point_value",
                    "requirements": {"time_position": "end"},
                },
            },
        }]
    }

    decisions = learner._validate_decision_payload(
        payload,
        valid_ids={"candidate-a", "candidate-b"},
        abstracted_ids={"candidate-a", "candidate-b"},
        allowed_targets={
            "candidate-a": ["recipe-shared"],
            "candidate-b": ["recipe-shared"],
        },
        database_id="energy",
    )

    assert decisions[0].target_recipe_id == "recipe-shared"


def test_legacy_reset_backs_up_then_clears_runtime_memory(tmp_path):
    memory_root = tmp_path / "fact_memory"
    embedding_root = tmp_path / "embeddings"
    learning_root = tmp_path / "learning"
    memory_root.mkdir()
    embedding_root.mkdir()
    (memory_root / "demo.json").write_text(json.dumps({"recipes": [{"recipe_id": "legacy"}]}))
    (embedding_root / "legacy.json").write_text("{}")

    assert reset_legacy_fact_memory_once(
        root=learning_root,
        embedding_root=embedding_root,
        memory_root=memory_root,
    ) is True
    payload = json.loads((memory_root / "demo.json").read_text())
    assert payload["recipes"] == []
    assert not list(embedding_root.rglob("*.json"))
    assert list((learning_root / "backups").rglob("demo.json"))
    assert reset_legacy_fact_memory_once(
        root=learning_root,
        embedding_root=embedding_root,
        memory_root=memory_root,
    ) is False


def test_scope_separation_preserves_only_learned_entries(tmp_path):
    memory_root = tmp_path / "fact_memory"
    embedding_root = tmp_path / "embeddings"
    migration_root = tmp_path / "learning"
    memory_root.mkdir()
    embedding_root.mkdir()
    (embedding_root / "cached.json").write_text("{}")
    (memory_root / "demo.json").write_text(json.dumps({
        "definitions": [
            {"fact_type": "point_value", "description": "system", "source": "system"},
            {"fact_type": "domain_value", "description": "learned", "source": "verified_data_fact"},
        ],
        "recipes": [
            {
                "recipe_id": "system_recipe", "fact_type": "extreme", "name": "max",
                "preferred_tool": "sql_query", "source": "system",
            },
            {
                "recipe_id": "learned_recipe", "fact_type": "domain_value", "name": "domain value",
                "preferred_tool": "code_interpreter", "source": "verified_data_fact",
            },
        ],
        "cards": [{"id": "stale", "kind": "fact_recipe", "title": "stale", "description": "stale"}],
        "details": [],
    }))

    assert separate_fact_memory_scopes_once(
        root=migration_root,
        embedding_root=embedding_root,
        memory_root=memory_root,
    ) is True
    payload = json.loads((memory_root / "demo.json").read_text())
    assert [item["fact_type"] for item in payload["definitions"]] == ["domain_value"]
    assert [item["recipe_id"] for item in payload["recipes"]] == ["learned_recipe"]
    assert payload["cards"] == []
    assert not list(embedding_root.glob("*.json"))
    assert list((migration_root / "backups").rglob("demo.json"))
