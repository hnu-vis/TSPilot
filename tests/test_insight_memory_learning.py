from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.settings import Settings
from core.key_insight import learning
from core.key_insight.learning import (
    InsightLearningOutbox,
    InsightLearningSchedule,
    InsightLearningScheduleStore,
    InsightMemoryLearner,
    InsightMemoryLearningWorker,
    extract_learning_candidates,
    reset_legacy_insight_memory_once,
    separate_insight_memory_scopes_once,
)
from core.key_insight.embedding_store import InsightMemoryEmbeddingStore
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.key_insight import (
    KeyInsight,
    KeyInsightRequest,
    InsightDefinition,
    InsightEvidenceRef,
    InsightEvent,
    InsightMemory,
    InsightRecipe,
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
    request = KeyInsightRequest(
        insight_key="energy.latest",
        name="latest_energy",
        insight_type="point_value",
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
    insight = KeyInsight(
        insight_id="insight_latest_energy",
        insight_key=request.insight_key,
        name=request.name,
        insight_type=request.insight_type,
        statement="The latest value is 420.",
        value=420,
        subject=request.subject,
        method="sql_query",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi-secret")],
        calculation_trace={
            "method": "Select the last ordered observation for the requested metric.",
            "row": {"value": 420},
            "value_key": "value",
            "time_key": "time",
        },
    )
    state.insight_set.requests = [request]
    state.insight_set.insights = [insight]
    state.insight_events = [InsightEvent(iteration=1, tool_name="sql_query", produced_insight_ids=[insight.insight_id])]
    state.final_answer_draft = FinalAnswer(
        summary="最后一个值是 420。",
        claims=[AnswerClaim(claim_id="claim-1", text="最后值", insight_ids=[insight.insight_id])],
        references=[AnswerReference(source_type="insight", source_id=insight.insight_id, label="最后值")],
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
    assert candidate.insight_request.time_range is None
    assert candidate.insight_request.dimensions == {"building": "<instance-value-omitted>"}
    assert candidate.insight_request.requirements["row_filters"]["building"] == "<instance-value-omitted>"
    assert candidate.insight_request.requirements["time_position"] == "end"
    assert candidate.calculation_semantics["method"] == (
        "Select the last ordered observation for the requested metric."
    )
    assert "420" not in serialized
    assert "evi-secret" not in serialized
    assert "memory_card_ids" not in serialized


def test_unreferenced_partial_or_failed_insight_is_not_queued(tmp_path):
    state = _terminal_state(tmp_path)
    state.final_answer_draft = FinalAnswer(summary="answer")
    assert extract_learning_candidates(state) == []
    state.final_answer_draft = FinalAnswer(
        summary="answer",
        claims=[AnswerClaim(claim_id="c", text="x", insight_ids=["insight_latest_energy"])],
    )
    state.status = "partial"
    assert extract_learning_candidates(state) == []


@pytest.mark.asyncio
async def test_outbox_triggers_at_twenty_and_worker_chunks_llm_batches(tmp_path):
    outbox = InsightLearningOutbox(tmp_path / "learning")
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
    worker = InsightMemoryLearningWorker(learner, batch_size=20, llm_chunk_size=5)
    assert await worker.process_due_once() == 20
    assert learner.batch_sizes == [5, 5, 5, 5]
    assert len(list((tmp_path / "learning" / "completed").glob("*.json"))) == 20


def test_outbox_triggers_when_oldest_candidate_waited_ten_minutes(tmp_path):
    outbox = InsightLearningOutbox(tmp_path / "learning")
    job_id = outbox.enqueue_request(_terminal_state(tmp_path))[0]
    path = tmp_path / "learning" / "pending" / f"{job_id}.json"
    payload = json.loads(path.read_text())
    payload["queued_at"] = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    outbox._write_atomic(path, payload)

    assert outbox.due_databases(batch_size=20, max_wait_seconds=600) == ["energy"]


def test_learning_schedule_store_persists_validated_runtime_value(tmp_path):
    store = InsightLearningScheduleStore(tmp_path / "learning", default_max_wait_seconds=600)
    assert store.read().max_wait_seconds == 600

    store.write(InsightLearningSchedule(max_wait_seconds=90))

    reloaded = InsightLearningScheduleStore(tmp_path / "learning", default_max_wait_seconds=600)
    assert reloaded.read().max_wait_seconds == 90


@pytest.mark.asyncio
async def test_worker_observes_schedule_changes_without_restart(tmp_path):
    outbox = InsightLearningOutbox(tmp_path / "learning")
    job_id = outbox.enqueue_request(_terminal_state(tmp_path))[0]
    path = tmp_path / "learning" / "pending" / f"{job_id}.json"
    payload = json.loads(path.read_text())
    payload["queued_at"] = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    outbox._write_atomic(path, payload)

    class _Learner:
        def __init__(self):
            self.outbox = outbox

        async def process(self, jobs):
            for job in jobs:
                outbox.finish(job, status="completed", reason="test")

    store = InsightLearningScheduleStore(tmp_path / "learning", default_max_wait_seconds=600)
    worker = InsightMemoryLearningWorker(_Learner(), batch_size=20, schedule_store=store)
    assert await worker.process_due_once() == 0

    store.write(InsightLearningSchedule(max_wait_seconds=1))
    assert await worker.process_due_once() == 1


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
                    "insight_type": "point_value",
                    "description": "Latest scalar observation for a database metric.",
                    "required_evidence": ["database_evidence"],
                    "preferred_tool": "sql_query",
                    "output_schema": {"value": "number|string"},
                    "verification_requirements": ["must reference current request evidence"],
                    "scope": "energy",
                    "source": "verified_key_insight",
                },
                "recipe": {
                    "recipe_id": "",
                    "insight_type": "point_value",
                    "name": "latest_energy",
                    "preferred_tool": "sql_query",
                    "insight_request_template": {
                        "insight_key": "energy.latest",
                        "name": "latest_energy",
                        "insight_type": "point_value",
                        "subject": "appliances_energy_wh",
                        "requirements": {"time_position": "end"},
                    },
                    "expected_result_schema": {"value": "scalar"},
                    "verification_notes": ["must reference current request evidence"],
                    "scope": "energy",
                    "source": "verified_key_insight",
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


class _RejectingAtomicEvidenceLLM:
    def __init__(self):
        self.system_prompt = ""

    async def ainvoke(self, messages):
        self.system_prompt = messages[0][1]
        return _Response({"abstracted": []})


@pytest.mark.asyncio
async def test_learning_semantically_rejects_reusable_atomic_evidence(monkeypatch, tmp_path):
    outbox = InsightLearningOutbox(tmp_path / "learning")
    job_id = outbox.enqueue_request(_terminal_state(tmp_path))[0]
    jobs = outbox.claim("energy", limit=20, lease_seconds=60)
    written = []
    monkeypatch.setattr(learning, "write_insight_memory", lambda memory, database_id=None: written.append(memory))
    llm = _RejectingAtomicEvidenceLLM()
    learner = InsightMemoryLearner(
        llm=llm,
        embedding_provider=_EmbeddingProvider(),
        embedding_store=InsightMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=outbox,
    )

    await learner.process(jobs)

    assert written == []
    rejected = json.loads((tmp_path / "learning" / "rejected" / f"{job_id}.json").read_text())
    assert "request evidence only" in rejected["error_summary"]
    assert "scalar value" in llm.system_prompt
    assert "not a\nhistory of requested outputs" in llm.system_prompt
    assert "concise canonical noun" in llm.system_prompt


def test_learning_requires_one_canonical_name_across_recipe_and_contract(tmp_path):
    learner = InsightMemoryLearner(
        llm=_LearningLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=InsightMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=InsightLearningOutbox(tmp_path / "learning"),
    )
    payload = {
        "abstracted": [{
            "candidate_ids": ["candidate-a"],
            "definition": {"insight_type": "change", "description": "Material directional change."},
            "recipe": {
                "recipe_id": "",
                "insight_type": "change",
                "name": "material directional change",
                "preferred_tool": "code_interpreter",
                "insight_request_template": {
                    "name": "Compute the full-period directional change and explain whether it is material",
                    "insight_type": "change",
                },
            },
        }],
    }

    with pytest.raises(ValueError, match="same canonical Key Insight name"):
        learner._validate_abstracted_payload(
            payload,
            valid_ids={"candidate-a"},
            database_id="energy",
        )


def test_learning_canonicalizes_llm_request_template_to_the_schema(tmp_path):
    learner = InsightMemoryLearner(
        llm=_LearningLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=InsightMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=InsightLearningOutbox(tmp_path / "learning"),
    )
    payload = {
        "abstracted": [{
            "candidate_ids": ["candidate-a"],
            "definition": {"insight_type": "analytical", "description": "Peak volatility regime."},
            "recipe": {
                "recipe_id": "",
                "insight_type": "analytical",
                "name": "peak rolling volatility window",
                "preferred_tool": "code_interpreter",
                "insight_request_template": {
                    "name": "peak rolling volatility window",
                    "insight_type": "analytical",
                    "semantic_class": "volatility_regime",
                    "description": "Model-invented field that is not part of KeyInsightRequest.",
                    "verification_notes": ["Also belongs outside the request template."],
                },
            },
        }],
    }

    result = learner._validate_abstracted_payload(
        payload,
        valid_ids={"candidate-a"},
        database_id="energy",
    )

    assert result[0].recipe.insight_request_template == {
        "name": "peak rolling volatility window",
        "insight_type": "analytical",
        "insight_key": "peak_rolling_volatility_window",
        "derived_from": [],
        "dimensions": {},
        "requirements": {},
        "selection": {},
        "semantic_class": "volatility_regime",
    }


@pytest.mark.asyncio
async def test_batch_learner_abstracts_reviews_and_commits(monkeypatch, tmp_path):
    outbox = InsightLearningOutbox(tmp_path / "learning")
    outbox.enqueue_request(_terminal_state(tmp_path))
    jobs = outbox.claim("energy", limit=20, lease_seconds=60)
    written = []
    monkeypatch.setattr(learning, "read_persisted_insight_memory", lambda database_id: InsightMemory())
    monkeypatch.setattr(learning, "write_insight_memory", lambda memory, database_id=None: written.append((database_id, memory)))
    learner = InsightMemoryLearner(
        llm=_LearningLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=InsightMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=outbox,
    )

    await learner.process(jobs)

    assert len(written) == 1
    assert written[0][0] == "energy"
    assert len(written[0][1].recipes) == 1
    recipe = written[0][1].recipes[0]
    assert recipe.recipe_id.startswith("recipe_learned_")
    assert recipe.description == "Generate the latest appliances energy observation."
    assert recipe.calculation_trace is not None
    assert recipe.calculation_trace.method == "Select the last ordered observation for the requested metric."
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
                    "insight_type": "point_value",
                    "description": "Latest scalar observation for a database metric.",
                    "preferred_tool": "sql_query",
                },
                "recipe": {
                    "recipe_id": "",
                    "insight_type": "point_value",
                    "name": "Latest appliances value",
                    "preferred_tool": "sql_query",
                    "insight_request_template": {
                        "name": "Latest appliances value",
                        "insight_type": "point_value",
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
            "reason": "Equivalent full Insight contract.",
            "definition": item["definition"],
            "recipe": item["recipe"],
        }]})


@pytest.mark.asyncio
async def test_learning_repairs_invented_merge_target_and_updates_existing_recipe(monkeypatch, tmp_path):
    outbox = InsightLearningOutbox(tmp_path / "learning")
    outbox.enqueue_request(_terminal_state(tmp_path))
    jobs = outbox.claim("energy", limit=20, lease_seconds=60)
    existing_recipe = InsightRecipe(
        recipe_id="recipe_existing_latest_energy",
        insight_type="point_value",
        name="appliances_energy_wh latest observed value",
        preferred_tool="sql_query",
        insight_request_template={
            "name": "appliances_energy_wh latest observed value",
            "insight_type": "point_value",
            "subject": "appliances_energy_wh",
            "requirements": {"time_position": "end"},
        },
        source="verified_key_insight",
        scope="energy",
    )
    persisted = InsightMemory(
        definitions=[InsightDefinition(insight_type="point_value", description="Latest value")],
        recipes=[existing_recipe],
        cards=[MemoryCard(
            id="recipe.sql_query.point_value.appliances_energy_wh_latest_observed_value",
            kind="insight_recipe",
            title=existing_recipe.name,
            description="Equivalent latest observation recipe.",
        )],
        details=[MemoryDetail(
            id="recipe.sql_query.point_value.appliances_energy_wh_latest_observed_value",
            card=MemoryCard(
                id="recipe.sql_query.point_value.appliances_energy_wh_latest_observed_value",
                kind="insight_recipe",
                title=existing_recipe.name,
                description="Equivalent latest observation recipe.",
            ),
            insight_request=KeyInsightRequest.model_validate(existing_recipe.insight_request_template),
            preferred_tool="sql_query",
        )],
        updated_at="revision-1",
    )
    written = []
    monkeypatch.setattr(learning, "read_persisted_insight_memory", lambda database_id: persisted)
    monkeypatch.setattr(learning, "write_insight_memory", lambda memory, database_id=None: written.append(memory))
    learner = InsightMemoryLearner(
        llm=_RepairingMergeLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=InsightMemoryEmbeddingStore(tmp_path / "embeddings"),
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
    learner = InsightMemoryLearner(
        llm=_LearningLLM(),
        embedding_provider=_EmbeddingProvider(),
        embedding_store=InsightMemoryEmbeddingStore(tmp_path / "embeddings"),
        outbox=InsightLearningOutbox(tmp_path / "learning"),
    )
    payload = {
        "decisions": [{
            "candidate_ids": ["candidate-a", "candidate-b"],
            "action": "merge",
            "target_recipe_id": "recipe-shared",
            "reason": "Same full contract.",
            "definition": {"insight_type": "point_value", "description": "Latest value"},
            "recipe": {
                "recipe_id": "",
                "insight_type": "point_value",
                "name": "latest value",
                "preferred_tool": "sql_query",
                "insight_request_template": {
                    "name": "latest value",
                    "insight_type": "point_value",
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
    memory_root = tmp_path / "insight_memory"
    embedding_root = tmp_path / "embeddings"
    learning_root = tmp_path / "learning"
    memory_root.mkdir()
    embedding_root.mkdir()
    (memory_root / "demo.json").write_text(json.dumps({"recipes": [{"recipe_id": "legacy"}]}))
    (embedding_root / "legacy.json").write_text("{}")

    assert reset_legacy_insight_memory_once(
        root=learning_root,
        embedding_root=embedding_root,
        memory_root=memory_root,
    ) is True
    payload = json.loads((memory_root / "demo.json").read_text())
    assert payload["recipes"] == []
    assert not list(embedding_root.rglob("*.json"))
    assert list((learning_root / "backups").rglob("demo.json"))
    assert reset_legacy_insight_memory_once(
        root=learning_root,
        embedding_root=embedding_root,
        memory_root=memory_root,
    ) is False


def test_scope_separation_preserves_only_learned_entries(tmp_path):
    memory_root = tmp_path / "insight_memory"
    embedding_root = tmp_path / "embeddings"
    migration_root = tmp_path / "learning"
    memory_root.mkdir()
    embedding_root.mkdir()
    (embedding_root / "cached.json").write_text("{}")
    (memory_root / "demo.json").write_text(json.dumps({
        "definitions": [
            {"insight_type": "point_value", "description": "system", "source": "system"},
            {"insight_type": "domain_value", "description": "learned", "source": "verified_key_insight"},
        ],
        "recipes": [
            {
                "recipe_id": "system_recipe", "insight_type": "extreme", "name": "max",
                "preferred_tool": "sql_query", "source": "system",
            },
            {
                "recipe_id": "learned_recipe", "insight_type": "domain_value", "name": "domain value",
                "preferred_tool": "code_interpreter", "source": "verified_key_insight",
            },
        ],
        "cards": [{"id": "stale", "kind": "insight_recipe", "title": "stale", "description": "stale"}],
        "details": [],
    }))

    assert separate_insight_memory_scopes_once(
        root=migration_root,
        embedding_root=embedding_root,
        memory_root=memory_root,
    ) is True
    payload = json.loads((memory_root / "demo.json").read_text())
    assert [item["insight_type"] for item in payload["definitions"]] == ["domain_value"]
    assert [item["recipe_id"] for item in payload["recipes"]] == ["learned_recipe"]
    assert payload["cards"] == []
    assert not list(embedding_root.glob("*.json"))
    assert list((migration_root / "backups").rglob("demo.json"))
