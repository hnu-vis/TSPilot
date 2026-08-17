"""Asynchronous, value-free learning for database-scoped Key Insight Memory."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from core.key_insight.contracts import insight_request_contract_error
from core.key_insight.embedding import EmbeddingProvider
from core.key_insight.embedding_store import (
    InsightMemoryEmbeddingStore,
    memory_card_embedding_text,
    top_similar_cards,
)
from core.key_insight.memory import (
    insight_memory_dir,
    read_persisted_insight_memory,
    recipe_memory_card_id,
    utc_now_iso,
    write_insight_memory,
)
from schemas.key_insight import (
    KeyInsight,
    KeyInsightRequest,
    InsightDefinition,
    InsightLearningCandidate,
    InsightLearningJob,
    InsightMemory,
    InsightRecipe,
    MemoryCard,
    MemoryDetail,
    normalize_insight_key,
)
from schemas.state import RequestStateModel


_KEY_INSIGHT_MEMORY_SEMANTIC_POLICY = """
Key Insight Memory is a library of reusable ways to discover decision-relevant or explanatory findings; it is not a
history of requested outputs. Admit a candidate only when the reusable pattern produces an interpretation that changes
understanding or supports a decision, such as a material trend, anomalous regime, meaningful comparison, relationship,
ranked opportunity, risk, or context-rich change. A scalar value, timestamp, boundary, row count, raw subset, requested
table/list, forecast endpoint, or generic calculation is supporting evidence or an intermediate metric, not by itself a
Key Insight. Do not admit a candidate merely because it is numerical, verified, cited, complex to compute, or reusable.
Memory stores the insight-discovery pattern, never the one-request conclusion or wording.

For every admitted pattern, rewrite recipe.name and insight_request_template.name as the same concise canonical noun
phrase. The name must identify the stable concept, not repeat the user's instruction, enumerate every returned field,
include request-local dates or values, or read like a sentence. Preserve detailed semantics in description,
requirements, selection, derivation, expected_result_schema, and verification notes. For example, a verbose request for
the dates and standard deviation of the most volatile rolling window should become a compact concept such as
"peak rolling volatility window" while its window definition and returned fields remain in the contract. Omit
non-insights during abstraction; reject any that survive during independent review.
""".strip()


class AbstractedInsightRecipe(BaseModel):
    candidate_ids: list[str]
    definition: InsightDefinition
    recipe: InsightRecipe


class InsightLearningDecision(BaseModel):
    candidate_ids: list[str]
    action: str
    target_recipe_id: str | None = None
    reason: str
    definition: InsightDefinition | None = None
    recipe: InsightRecipe | None = None


class InsightLearningSchedule(BaseModel):
    """Persisted runtime schedule for automatic Key Insight learning."""

    max_wait_seconds: float = Field(gt=0, le=7 * 24 * 60 * 60)


class InsightLearningScheduleStore:
    """Atomically persist the schedule shared by the API and learning worker."""

    def __init__(self, root: Path, *, default_max_wait_seconds: float):
        self.root = Path(root)
        self.path = self.root / "schedule.json"
        self.default = InsightLearningSchedule(max_wait_seconds=default_max_wait_seconds)

    def read(self) -> InsightLearningSchedule:
        if not self.path.exists():
            return self.default.model_copy()
        return InsightLearningSchedule.model_validate_json(self.path.read_text(encoding="utf-8"))

    def write(self, schedule: InsightLearningSchedule | BaseModel | dict[str, Any]) -> InsightLearningSchedule:
        payload = schedule.model_dump(mode="json") if isinstance(schedule, BaseModel) else schedule
        validated = InsightLearningSchedule.model_validate(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(
            json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp.chmod(0o600)
        temp.replace(self.path)
        return validated


def learning_root() -> Path:
    return insight_memory_dir().parent / "insight_memory_learning"


def extract_learning_candidates(request_state: RequestStateModel) -> list[InsightLearningCandidate]:
    """Project terminally-used verified Key Insights into value-free learning candidates."""

    if request_state.status != "completed" or request_state.final_answer_draft is None:
        return []
    database_id = _database_id(request_state)
    if not database_id:
        return []
    referenced = _referenced_insight_ids(request_state.final_answer_draft)
    if not referenced:
        return []
    requests = {item.insight_key: item for item in request_state.insight_set.requests}
    verified = {
        item.insight_key: item
        for item in request_state.insight_set.insights
        if item.status == "verified" and item.insight_key
    }
    verified_references = {
        reference
        for item in verified.values()
        for reference in (item.insight_key, normalize_insight_key(item.insight_id))
        if reference
    }
    produced_in_request = {
        insight_id
        for event in request_state.insight_events
        for insight_id in event.produced_insight_ids
    }
    candidates: list[InsightLearningCandidate] = []
    for insight in verified.values():
        if insight.insight_id not in referenced or insight.method not in {"sql_query", "code_interpreter"}:
            continue
        request = requests.get(insight.insight_key)
        if request is None or insight_request_contract_error(request, insight.method):
            continue
        if insight.insight_id not in produced_in_request or not insight.evidence_refs:
            continue
        if any(normalize_insight_key(parent) not in verified_references for parent in insight.derived_from):
            continue
        candidate_id = _stable_id(request_state.request_id, database_id, insight.insight_key)
        candidates.append(InsightLearningCandidate(
            candidate_id=candidate_id,
            request_id=request_state.request_id,
            database_id=database_id,
            tool_name=insight.method,
            insight_request=_value_free_request(request),
            insight_shape={
                "insight_type": insight.insight_type,
                "value_shape": insight.value_shape,
                "semantic_class": insight.semantic_class,
                "derivation": insight.derivation,
                "has_items": bool(insight.items),
                "item_fields": sorted({key for item in insight.items for key in item.model_dump(exclude_none=True)}),
            },
            evidence_types=sorted({ref.source_type for ref in insight.evidence_refs}),
            dependency_insight_keys=list(insight.derived_from),
            calculation_semantics={
                "derivation": insight.derivation,
                "trace_fields": sorted((insight.calculation_trace or {}).keys()),
            },
            created_at=utc_now_iso(),
        ))
    return candidates


class InsightLearningOutbox:
    STATUSES = ("pending", "processing", "completed", "rejected", "failed")

    def __init__(self, root: Path | None = None):
        self.root = Path(root or learning_root())
        for status in self.STATUSES:
            (self.root / status).mkdir(parents=True, exist_ok=True)
        (self.root / "batches").mkdir(parents=True, exist_ok=True)

    def enqueue_request(self, request_state: RequestStateModel) -> list[str]:
        ids: list[str] = []
        for candidate in extract_learning_candidates(request_state):
            job = InsightLearningJob(
                job_id=candidate.candidate_id,
                candidate=candidate,
                queued_at=utc_now_iso(),
            )
            if self._create_once(self.root / "pending" / f"{job.job_id}.json", job.model_dump(mode="json")):
                ids.append(job.job_id)
        return ids

    def due_databases(self, *, batch_size: int, max_wait_seconds: float) -> list[str]:
        grouped: dict[str, list[InsightLearningJob]] = {}
        self.recover_expired()
        for path in sorted((self.root / "pending").glob("*.json")):
            job = self._read_job(path)
            if job is not None and _parse_time(job.queued_at) <= datetime.now(timezone.utc):
                grouped.setdefault(job.candidate.database_id, []).append(job)
        now = datetime.now(timezone.utc)
        due: list[str] = []
        for database_id, jobs in grouped.items():
            oldest = min(_parse_time(job.queued_at) for job in jobs)
            if len(jobs) >= batch_size or (now - oldest).total_seconds() >= max_wait_seconds:
                due.append(database_id)
        return due

    def claim(self, database_id: str, *, limit: int, lease_seconds: float) -> list[InsightLearningJob]:
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        claimed: list[InsightLearningJob] = []
        for path in sorted((self.root / "pending").glob("*.json")):
            if len(claimed) >= limit:
                break
            job = self._read_job(path)
            if job is None or job.candidate.database_id != database_id:
                continue
            if _parse_time(job.queued_at) > datetime.now(timezone.utc):
                continue
            target = self.root / "processing" / path.name
            try:
                path.replace(target)
            except FileNotFoundError:
                continue
            now = datetime.now(timezone.utc)
            job = job.model_copy(update={
                "status": "processing",
                "attempt_count": job.attempt_count + 1,
                "started_at": now.isoformat().replace("+00:00", "Z"),
                "lease_expires_at": (now + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z"),
                "batch_id": batch_id,
                "failure_stage": None,
                "error_summary": None,
            })
            self._write_atomic(target, job.model_dump(mode="json"))
            claimed.append(job)
        return claimed

    def finish(self, job: InsightLearningJob, *, status: str, reason: str | None = None) -> None:
        source = self.root / "processing" / f"{job.job_id}.json"
        target = self.root / status / source.name
        finished = job.model_copy(update={
            "status": status,
            "completed_at": utc_now_iso(),
            "lease_expires_at": None,
            "error_summary": reason,
        })
        self._write_atomic(source, finished.model_dump(mode="json"))
        source.replace(target)

    def retry_or_fail(self, job: InsightLearningJob, *, stage: str, error: Exception, max_attempts: int) -> None:
        source = self.root / "processing" / f"{job.job_id}.json"
        status = "failed" if job.attempt_count >= max_attempts else "pending"
        target = self.root / status / source.name
        retry_delays = (30, 120, 600)
        next_attempt_at = datetime.now(timezone.utc) + timedelta(
            seconds=retry_delays[min(max(job.attempt_count - 1, 0), len(retry_delays) - 1)]
        )
        updated = job.model_copy(update={
            "status": status,
            "queued_at": next_attempt_at.isoformat().replace("+00:00", "Z") if status == "pending" else job.queued_at,
            "lease_expires_at": None,
            "failure_stage": stage,
            "error_summary": str(error)[:1000],
        })
        self._write_atomic(source, updated.model_dump(mode="json"))
        source.replace(target)

    def recover_expired(self) -> None:
        now = datetime.now(timezone.utc)
        for path in (self.root / "processing").glob("*.json"):
            job = self._read_job(path)
            if job is None or not job.lease_expires_at or _parse_time(job.lease_expires_at) > now:
                continue
            updated = job.model_copy(update={"status": "pending", "lease_expires_at": None, "queued_at": utc_now_iso()})
            self._write_atomic(path, updated.model_dump(mode="json"))
            path.replace(self.root / "pending" / path.name)

    def write_batch_diagnostics(self, batch_id: str, payload: dict) -> None:
        self._write_atomic(self.root / "batches" / f"{batch_id}.json", payload)

    def purge_terminal(self, *, retention_days: int = 30) -> int:
        cutoff = time.time() - max(retention_days, 1) * 86400
        removed = 0
        for directory in ("completed", "rejected", "failed", "batches"):
            for path in (self.root / directory).glob("*.json"):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
        return removed

    def _read_job(self, path: Path) -> InsightLearningJob | None:
        try:
            return InsightLearningJob.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    @staticmethod
    def _write_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp.chmod(0o600)
        temp.replace(path)

    @staticmethod
    def _create_once(path: Path, payload: dict) -> bool:
        if any((path.parent.parent / status / path.name).exists() for status in InsightLearningOutbox.STATUSES):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return True


@dataclass
class InsightMemoryLearner:
    llm: Any
    embedding_provider: EmbeddingProvider
    embedding_store: InsightMemoryEmbeddingStore
    outbox: InsightLearningOutbox
    neighbor_top_k: int = 6
    neighbor_threshold: float = 0.25
    max_attempts: int = 3

    _structured_output_attempts: ClassVar[int] = 2

    async def process(self, jobs: list[InsightLearningJob]) -> None:
        if not jobs:
            return
        parent_batch_id = jobs[0].batch_id or f"batch_{uuid.uuid4().hex[:12]}"
        batch_id = f"{parent_batch_id}_{jobs[0].job_id[-8:]}"
        started = time.perf_counter()
        usage: list[dict] = []
        stage = "llm_abstraction"
        try:
            abstracted, first_usage = await self._abstract(jobs)
            usage.append(first_usage)
            if not abstracted:
                for job in jobs:
                    self.outbox.finish(
                        job,
                        status="rejected",
                        reason="LLM found no reusable Key Insight pattern; the output remains request evidence only.",
                    )
                self.outbox.write_batch_diagnostics(batch_id, {
                    "batch_id": batch_id,
                    "database_id": jobs[0].candidate.database_id,
                    "candidate_count": len(jobs),
                    "abstracted_count": 0,
                    "decision_count": 0,
                    "outcomes": [{"status": "rejected", "candidate_ids": [job.job_id]} for job in jobs],
                    "llm_usage": usage,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "completed_at": utc_now_iso(),
                })
                return
            stage = "embedding_neighbors"
            neighbors, memory_revision = await self._neighbors(jobs[0].candidate.database_id, abstracted)
            stage = "llm_merge_review"
            decisions, second_usage = await self._decide(jobs, abstracted, neighbors)
            usage.append(second_usage)
            stage = "memory_commit"
            outcomes = self._commit(jobs[0].candidate.database_id, decisions, expected_revision=memory_revision)
            by_candidate = {
                candidate_id: (status, reason)
                for status, reason, candidate_ids in outcomes
                for candidate_id in candidate_ids
            }
            for job in jobs:
                status, reason = by_candidate.get(job.candidate.candidate_id, ("rejected", "LLM returned no accepted learning decision."))
                self.outbox.finish(job, status=status, reason=reason)
            self.outbox.write_batch_diagnostics(batch_id, {
                "batch_id": batch_id,
                "database_id": jobs[0].candidate.database_id,
                "candidate_count": len(jobs),
                "abstracted_count": len(abstracted),
                "decision_count": len(decisions),
                "outcomes": [{"status": s, "reason": r, "candidate_ids": ids} for s, r, ids in outcomes],
                "llm_usage": usage,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "completed_at": utc_now_iso(),
            })
        except Exception as exc:
            for job in jobs:
                self.outbox.retry_or_fail(job, stage=stage, error=exc, max_attempts=self.max_attempts)
            self.outbox.write_batch_diagnostics(batch_id, {
                "batch_id": batch_id,
                "database_id": jobs[0].candidate.database_id,
                "candidate_count": len(jobs),
                "failure_stage": stage,
                "error": str(exc)[:1000],
                "llm_usage": usage,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "completed_at": utc_now_iso(),
            })

    async def _abstract(self, jobs: list[InsightLearningJob]) -> tuple[list[AbstractedInsightRecipe], dict]:
        messages = [
            (
                "system",
                (
                    "You curate verified, terminally-used experiences into reusable database-scoped Key Insight "
                    "discovery recipes. Return one JSON object only, with top-level key abstracted. Every abstracted "
                    "item MUST conform exactly to the supplied AbstractedInsightRecipe JSON Schema. definition is an "
                    "InsightDefinition, not an InsightRequest: it requires insight_type and a natural-language "
                    "description. recipe is an InsightRecipe: it requires recipe_id, insight_type, name, "
                    "preferred_tool, and insight_request_template. Copy the abstracted insight_type and canonical "
                    "name into the recipe's top-level fields and its request template. Remove request-instance "
                    "timestamps, values, row filters, evidence ids, SQL, code, retrieval diagnostics, and answer "
                    "wording. Preserve metric/domain subject only when needed inside this database, operation "
                    "semantics, result shape, dimensions by role, dependencies, preferred tool, and verification "
                    "needs. Merge equivalent candidates in this batch. Omit every candidate that is not a genuine "
                    "reusable Key Insight pattern. recipe_id may be an empty string; source must be "
                    "verified_key_insight and scope must be the supplied database_id. Do not invent alternative "
                    "field names.\n\n" + _KEY_INSIGHT_MEMORY_SEMANTIC_POLICY
                ),
            ),
            ("user", json.dumps({
                "database_id": jobs[0].candidate.database_id,
                "output_item_json_schema": AbstractedInsightRecipe.model_json_schema(),
                "candidates": [job.candidate.model_dump(mode="json") for job in jobs],
            }, ensure_ascii=False, default=str)),
        ]
        valid_ids = {job.candidate.candidate_id for job in jobs}
        usage: list[dict] = []
        previous_content: Any = None
        validation_error: Exception | None = None
        for attempt in range(self._structured_output_attempts):
            current_messages = messages
            if attempt:
                current_messages = [
                    *messages,
                    ("assistant", str(previous_content or "")),
                    ("user", json.dumps({
                        "instruction": "Repair the previous JSON so it fully satisfies the schema and constraints. Return the complete corrected object only.",
                        "validation_error": str(validation_error),
                        "valid_candidate_ids": sorted(valid_ids),
                    }, ensure_ascii=False)),
                ]
            response = await self.llm.ainvoke(current_messages)
            usage.append(_response_usage(response))
            previous_content = getattr(response, "content", response)
            try:
                payload = _parse_json_response(previous_content)
                result = self._validate_abstracted_payload(
                    payload,
                    valid_ids=valid_ids,
                    database_id=jobs[0].candidate.database_id,
                )
                return result, {"attempts": usage}
            except Exception as exc:
                validation_error = exc
        raise ValueError(f"Learning abstraction remained invalid after LLM repair: {validation_error}")

    @staticmethod
    def _validate_abstracted_payload(
        payload: dict,
        *,
        valid_ids: set[str],
        database_id: str,
    ) -> list[AbstractedInsightRecipe]:
        items = payload.get("abstracted") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("Learning abstraction response is missing abstracted JSON list.")
        result: list[AbstractedInsightRecipe] = []
        assigned_ids: set[str] = set()
        for raw in items:
            item = AbstractedInsightRecipe.model_validate(raw)
            unknown_ids = set(item.candidate_ids) - valid_ids
            if unknown_ids:
                raise ValueError(f"Abstraction references unknown candidate ids: {sorted(unknown_ids)}")
            item.candidate_ids = list(dict.fromkeys(item.candidate_ids))
            duplicate_ids = assigned_ids.intersection(item.candidate_ids)
            if duplicate_ids:
                raise ValueError(f"Abstraction assigns candidates more than once: {sorted(duplicate_ids)}")
            if not item.candidate_ids:
                raise ValueError("An abstracted recipe must reference at least one candidate.")
            if item.recipe.preferred_tool not in {"sql_query", "code_interpreter"}:
                raise ValueError(f"Unsupported Key Insight recipe tool: {item.recipe.preferred_tool}")
            request = KeyInsightRequest.model_validate(item.recipe.insight_request_template)
            _validate_canonical_recipe_name(item.recipe, request)
            item.recipe.insight_request_template = request.model_dump(mode="json", exclude_none=True)
            contract_error = insight_request_contract_error(request, item.recipe.preferred_tool)
            if contract_error:
                raise ValueError(f"Invalid abstracted Key Insight request contract: {contract_error}")
            item.recipe.scope = database_id
            item.recipe.source = "verified_key_insight"
            item.definition.scope = database_id
            item.definition.source = "verified_key_insight"
            result.append(item)
            assigned_ids.update(item.candidate_ids)
        return result

    async def _neighbors(
        self,
        database_id: str,
        abstracted: list[AbstractedInsightRecipe],
    ) -> tuple[dict[str, list[dict]], str | None]:
        if not abstracted:
            return {}, None
        memory = read_persisted_insight_memory(database_id)
        details = [detail for detail in memory.details if detail.card.kind == "insight_recipe" and detail.insight_request]
        recipe_ids_by_card_id = {
            recipe_memory_card_id(recipe): recipe.recipe_id
            for recipe in memory.recipes
        }
        existing = []
        missing: list[tuple[MemoryDetail, str]] = []
        for detail in details:
            text = memory_card_embedding_text(detail.card, detail.insight_request.model_dump(mode="json", exclude_none=True), detail.guidance)
            cached = self.embedding_store.load(database_id=database_id, model=self.embedding_provider.model, card=detail.card, text=text)
            if cached is None:
                missing.append((detail, text))
            else:
                existing.append(cached)
        if missing:
            vectors = await self.embedding_provider.embed_texts([text for _detail, text in missing])
            if len(vectors) != len(missing):
                raise ValueError("Embedding provider returned an incomplete recipe batch.")
            for (detail, text), vector in zip(missing, vectors):
                existing.append(self.embedding_store.save(
                    database_id=database_id, model=self.embedding_provider.model, card=detail.card,
                    text=text, vector=vector, memory_updated_at=memory.updated_at,
                ))
        query_texts = [_abstracted_embedding_text(item) for item in abstracted]
        query_vectors = await self.embedding_provider.embed_texts(query_texts)
        if len(query_vectors) != len(abstracted):
            raise ValueError("Embedding provider returned an incomplete candidate batch.")
        result: dict[str, list[dict]] = {}
        for item, vector in zip(abstracted, query_vectors):
            hits = top_similar_cards(
                query_vector=vector, card_embeddings=existing,
                top_k=self.neighbor_top_k, score_threshold=self.neighbor_threshold,
            )
            by_id = {detail.id: detail for detail in details}
            item_neighbors = [
                {
                    "target_recipe_id": recipe_ids_by_card_id[hit.card.id],
                    "memory_card_id": hit.card.id,
                    "score": round(float(score), 6),
                    "insight_request": by_id[hit.card.id].insight_request.model_dump(mode="json", exclude_none=True),
                    "description": hit.card.description,
                    "preferred_tool": by_id[hit.card.id].preferred_tool,
                }
                for hit, score in hits
                if hit.card.id in by_id and hit.card.id in recipe_ids_by_card_id
            ]
            for candidate_id in item.candidate_ids:
                result[candidate_id] = item_neighbors
        return result, memory.updated_at

    async def _decide(
        self,
        jobs: list[InsightLearningJob],
        abstracted: list[AbstractedInsightRecipe],
        neighbors: dict[str, list[dict]],
    ) -> tuple[list[InsightLearningDecision], dict]:
        allowed_targets = {
            candidate_id: sorted({
                str(neighbor.get("target_recipe_id") or "").strip()
                for neighbor in candidate_neighbors
                if str(neighbor.get("target_recipe_id") or "").strip()
            })
            for candidate_id, candidate_neighbors in neighbors.items()
        }
        messages = [
            (
                "system",
                (
                    "You are the independent semantic reviewer for abstracted Key Insight recipes. Recheck each "
                    "abstraction against its original candidate before considering nearest neighbors. Return one JSON "
                    "object only, with top-level key decisions. Every item MUST conform exactly to the supplied "
                    "InsightLearningDecision JSON Schema. Produce exactly one decision for every candidate_id present "
                    "in abstracted; a decision may group candidates only when they share the same action and target. "
                    "action must be create, merge, replace, or reject. Reject an abstraction that is merely reusable "
                    "evidence, an intermediate metric, request wording, or answer prose. For accepted items, repair "
                    "verbose names into concise canonical concept labels while preserving full semantics in the other "
                    "contract fields. Equivalent semantics MUST merge into one recipe even when names, languages, or "
                    "insight_key values differ; insight_key presence or spelling is never a reason to create a "
                    "duplicate. Same names with different subject, operation, result shape, dimensions, time "
                    "semantics, or derivation must remain distinct. For create/merge/replace return the complete "
                    "validated InsightDefinition and InsightRecipe, improved when needed. merge/replace requires "
                    "target_recipe_id copied exactly from allowed_target_recipe_ids_by_candidate; never invent or "
                    "transform an id. Use create only after comparing the full contract and determining that none of "
                    "the supplied semantic neighbors is equivalent. Never use historical values as evidence and never "
                    "broaden database-scoped knowledge to global. Do not invent alternative field names.\n\n"
                    + _KEY_INSIGHT_MEMORY_SEMANTIC_POLICY
                ),
            ),
            ("user", json.dumps({
                "database_id": jobs[0].candidate.database_id,
                "output_item_json_schema": InsightLearningDecision.model_json_schema(),
                "original_candidates": [job.candidate.model_dump(mode="json") for job in jobs],
                "abstracted": [item.model_dump(mode="json") for item in abstracted],
                "semantic_neighbors": neighbors,
                "allowed_target_recipe_ids_by_candidate": allowed_targets,
            }, ensure_ascii=False, default=str)),
        ]
        valid_ids = {job.candidate.candidate_id for job in jobs}
        abstracted_ids = {candidate_id for item in abstracted for candidate_id in item.candidate_ids}
        usage: list[dict] = []
        previous_content: Any = None
        validation_error: Exception | None = None
        for attempt in range(self._structured_output_attempts):
            current_messages = messages
            if attempt:
                current_messages = [
                    *messages,
                    ("assistant", str(previous_content or "")),
                    ("user", json.dumps({
                        "instruction": "Repair the previous JSON using only the supplied candidates and allowed target recipe ids. Return the complete corrected object only.",
                        "validation_error": str(validation_error),
                        "allowed_target_recipe_ids_by_candidate": allowed_targets,
                    }, ensure_ascii=False)),
                ]
            response = await self.llm.ainvoke(current_messages)
            usage.append(_response_usage(response))
            previous_content = getattr(response, "content", response)
            try:
                payload = _parse_json_response(previous_content)
                decisions = self._validate_decision_payload(
                    payload,
                    valid_ids=valid_ids,
                    abstracted_ids=abstracted_ids,
                    allowed_targets=allowed_targets,
                    database_id=jobs[0].candidate.database_id,
                )
                return decisions, {"attempts": usage}
            except Exception as exc:
                validation_error = exc
        raise ValueError(f"Learning review remained invalid after LLM repair: {validation_error}")

    @staticmethod
    def _validate_decision_payload(
        payload: dict,
        *,
        valid_ids: set[str],
        abstracted_ids: set[str],
        allowed_targets: dict[str, list[str]],
        database_id: str,
    ) -> list[InsightLearningDecision]:
        items = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("Learning review response is missing decisions JSON list.")
        decisions: list[InsightLearningDecision] = []
        assigned_ids: set[str] = set()
        for raw in items:
            item = InsightLearningDecision.model_validate(raw)
            item.action = item.action.strip().lower()
            unknown_ids = set(item.candidate_ids) - valid_ids
            if unknown_ids:
                raise ValueError(f"Decision references unknown candidate ids: {sorted(unknown_ids)}")
            item.candidate_ids = list(dict.fromkeys(item.candidate_ids))
            if item.action not in {"create", "merge", "replace", "reject"}:
                raise ValueError(f"Unsupported learning action: {item.action}")
            if not item.candidate_ids:
                raise ValueError("A learning decision must reference at least one candidate.")
            duplicate_ids = assigned_ids.intersection(item.candidate_ids)
            if duplicate_ids:
                raise ValueError(f"Candidates received multiple learning decisions: {sorted(duplicate_ids)}")
            if not set(item.candidate_ids).issubset(abstracted_ids):
                raise ValueError("A learning decision may only reference abstracted candidates.")
            if item.action in {"merge", "replace"}:
                target = str(item.target_recipe_id or "").strip()
                allowed_for_group = set.intersection(*(
                    set(allowed_targets.get(candidate_id, [])) for candidate_id in item.candidate_ids
                ))
                if target not in allowed_for_group:
                    raise ValueError(
                        f"Decision target '{target}' is not an allowed semantic neighbor for candidates "
                        f"{item.candidate_ids}; allowed targets are {sorted(allowed_for_group)}"
                    )
            if item.action != "reject":
                if item.recipe is None or item.definition is None:
                    raise ValueError(f"Learning action '{item.action}' requires a complete definition and recipe.")
                request = KeyInsightRequest.model_validate(item.recipe.insight_request_template)
                _validate_canonical_recipe_name(item.recipe, request)
                item.recipe.insight_request_template = request.model_dump(mode="json", exclude_none=True)
                if item.recipe.preferred_tool not in {"sql_query", "code_interpreter"}:
                    raise ValueError(f"Unsupported Key Insight recipe tool: {item.recipe.preferred_tool}")
                contract_error = insight_request_contract_error(request, item.recipe.preferred_tool)
                if contract_error:
                    raise ValueError(f"Invalid decided Key Insight request contract: {contract_error}")
                item.recipe.scope = database_id
                item.recipe.source = "verified_key_insight"
                item.definition.scope = database_id
                item.definition.source = "verified_key_insight"
            decisions.append(item)
            assigned_ids.update(item.candidate_ids)
        missing_ids = abstracted_ids - assigned_ids
        if missing_ids:
            raise ValueError(f"Learning review omitted abstracted candidates: {sorted(missing_ids)}")
        return decisions

    def _commit(
        self,
        database_id: str,
        decisions: list[InsightLearningDecision],
        *,
        expected_revision: str | None,
    ) -> list[tuple[str, str, list[str]]]:
        if not database_id:
            raise ValueError("Automatic Key Insight learning requires a database scope.")
        outcomes: list[tuple[str, str, list[str]]] = []
        with _database_memory_lock(database_id):
            memory = read_persisted_insight_memory(database_id)
            if memory.updated_at != expected_revision:
                raise RuntimeError("Key Insight Memory changed during semantic review; retry against the new revision.")
            definitions = {item.insight_type: item for item in memory.definitions}
            recipes = {item.recipe_id: item for item in memory.recipes}
            for decision in decisions:
                if decision.action == "reject":
                    outcomes.append(("rejected", decision.reason, decision.candidate_ids))
                    continue
                assert decision.recipe is not None and decision.definition is not None
                if decision.action in {"merge", "replace"}:
                    target = str(decision.target_recipe_id or "").strip()
                    if target not in recipes:
                        raise ValueError(f"Learning decision references unknown recipe '{target}'.")
                    recipe_id = target
                else:
                    recipe_id = _learned_recipe_id(decision.recipe)
                recipe = decision.recipe.model_copy(update={
                    "recipe_id": recipe_id,
                    "scope": database_id,
                    "source": "verified_key_insight",
                    "updated_at": utc_now_iso(),
                })
                recipes[recipe_id] = recipe
                current_definition = definitions.get(decision.definition.insight_type)
                if current_definition is None or current_definition.source != "system":
                    definitions[decision.definition.insight_type] = decision.definition.model_copy(update={
                        "scope": database_id,
                        "source": "verified_key_insight",
                        "updated_at": utc_now_iso(),
                    })
                outcomes.append(("completed", decision.reason, decision.candidate_ids))
            next_memory = InsightMemory(
                definitions=list(definitions.values()), recipes=list(recipes.values()),
                storage_path=memory.storage_path, updated_at=utc_now_iso(),
            )
            write_insight_memory(next_memory, database_id)
        return outcomes


class InsightMemoryLearningWorker:
    def __init__(
        self,
        learner: InsightMemoryLearner,
        *,
        batch_size: int = 20,
        max_wait_seconds: float = 600.0,
        poll_seconds: float = 5.0,
        lease_seconds: float = 180.0,
        llm_chunk_size: int = 5,
        schedule_store: InsightLearningScheduleStore | None = None,
    ):
        self.learner = learner
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.llm_chunk_size = max(int(llm_chunk_size), 1)
        self.schedule_store = schedule_store
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="insight-memory-learning-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=max(self.poll_seconds + 1, 2))
            except TimeoutError:
                self._task.cancel()
            self._task = None

    async def run(self) -> None:
        while not self._stop.is_set():
            await self.process_due_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def process_due_once(self) -> int:
        self.learner.outbox.purge_terminal(retention_days=30)
        max_wait_seconds = (
            self.schedule_store.read().max_wait_seconds
            if self.schedule_store is not None
            else self.max_wait_seconds
        )
        processed = 0
        for database_id in self.learner.outbox.due_databases(
            batch_size=self.batch_size, max_wait_seconds=max_wait_seconds,
        ):
            jobs = self.learner.outbox.claim(database_id, limit=self.batch_size, lease_seconds=self.lease_seconds)
            if jobs:
                for start in range(0, len(jobs), self.llm_chunk_size):
                    await self.learner.process(jobs[start:start + self.llm_chunk_size])
                processed += len(jobs)
        return processed


def reset_legacy_insight_memory_once(
    *,
    root: Path | None = None,
    embedding_root: Path | None = None,
    memory_root: Path | None = None,
) -> bool:
    """Back up and remove learned legacy memory once; code defaults remain available."""

    migration_root = Path(root or learning_root())
    marker = migration_root / "migrations" / "insight_memory_v2_reset.json"
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = migration_root / "backups" / timestamp
    backup.mkdir(parents=True, exist_ok=False)
    memory_root = Path(memory_root or insight_memory_dir())
    for path in memory_root.glob("*.json"):
        shutil.copy2(path, backup / path.name)
        InsightLearningOutbox._write_atomic(path, {
            "definitions": [], "recipes": [], "cards": [], "details": [],
            "storage_path": str(path), "updated_at": utc_now_iso(),
        })
    embeddings = Path(embedding_root) if embedding_root is not None else memory_root.parent / "insight_memory_embeddings"
    if embeddings.exists():
        shutil.copytree(embeddings, backup / "insight_memory_embeddings")
        for path in embeddings.rglob("*.json"):
            path.unlink()
    InsightLearningOutbox._write_atomic(marker, {
        "migration": "insight_memory_v2_reset",
        "completed_at": utc_now_iso(),
        "backup_path": str(backup),
    })
    return True


def separate_insight_memory_scopes_once(
    *,
    root: Path | None = None,
    embedding_root: Path | None = None,
    memory_root: Path | None = None,
) -> bool:
    """Remove code-owned defaults from database files while preserving learned entries."""

    migration_root = Path(root or learning_root())
    marker = migration_root / "migrations" / "insight_memory_scope_separation.json"
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = migration_root / "backups" / f"scope-separation-{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    memory_root = Path(memory_root or insight_memory_dir())
    for path in memory_root.glob("*.json"):
        shutil.copy2(path, backup / path.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        definitions = [
            item for item in (payload.get("definitions") or [])
            if isinstance(item, dict) and item.get("source") != "system"
        ]
        recipes = [
            item for item in (payload.get("recipes") or [])
            if isinstance(item, dict) and item.get("source") != "system"
        ]
        InsightLearningOutbox._write_atomic(path, {
            "definitions": definitions,
            "recipes": recipes,
            "cards": [],
            "details": [],
            "storage_path": str(path),
            "updated_at": payload.get("updated_at"),
        })
    embeddings = Path(embedding_root) if embedding_root is not None else memory_root.parent / "insight_memory_embeddings"
    if embeddings.exists():
        shutil.copytree(embeddings, backup / "insight_memory_embeddings")
        for path in embeddings.rglob("*.json"):
            path.unlink()
    InsightLearningOutbox._write_atomic(marker, {
        "migration": "insight_memory_scope_separation",
        "completed_at": utc_now_iso(),
        "backup_path": str(backup),
    })
    return True


def _referenced_insight_ids(answer) -> set[str]:
    result: set[str] = set()
    for claim in answer.claims:
        result.update(_strip_insight_ref(value) for value in claim.insight_ids)
    for reference in answer.references:
        if reference.source_type == "insight" and reference.source_id:
            result.add(_strip_insight_ref(reference.source_id))
    for visualization in answer.visualizations:
        result.update(
            _strip_insight_ref(value.split("#", 1)[0])
            for value in visualization.source_refs
            if value.startswith("insight:")
        )
        result.update(
            _strip_insight_ref(binding.insight_id)
            for binding in visualization.bindings
            if binding.insight_id
        )
    return {value for value in result if value}


def _strip_insight_ref(value: str) -> str:
    text = str(value or "").strip()
    return text[5:] if text.startswith("insight:") else text


def _validate_canonical_recipe_name(recipe: InsightRecipe, request: KeyInsightRequest) -> None:
    """Keep the retrieval label and executable contract on one LLM-authored concept name."""

    recipe_name = str(recipe.name or "").strip()
    request_name = str(request.name or "").strip()
    if not recipe_name or not request_name:
        raise ValueError("A learned Key Insight recipe requires a non-empty canonical concept name.")
    if recipe_name != request_name:
        raise ValueError(
            "The recipe and insight_request_template must use the same canonical Key Insight name."
        )


def _value_free_request(request: KeyInsightRequest) -> KeyInsightRequest:
    payload = request.model_dump(mode="json", exclude_none=True)
    payload.pop("time_range", None)
    payload["dimensions"] = {str(key): "<instance-value-omitted>" for key in (payload.get("dimensions") or {})}
    requirements = dict(payload.get("requirements") or {})
    for key in ("source", "memory_card_ids", "retrieval_reason", "retrieval_confidence"):
        requirements.pop(key, None)
    payload["requirements"] = _redact_requirement_instances(requirements)
    return KeyInsightRequest.model_validate(payload)


def _redact_requirement_instances(value: Any, *, under_filter: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_requirement_instances(item, under_filter=under_filter or str(key).lower() in {"row_filters", "filters"})
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_requirement_instances(item, under_filter=under_filter) for item in value]
    if under_filter:
        return "<instance-value-omitted>"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "<number-omitted>"
    if isinstance(value, str) and _looks_like_time_instance(value):
        return "<time-omitted>"
    return value


def _looks_like_time_instance(value: str) -> bool:
    text = value.strip()
    return bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}:\d{2}", text))


def _database_id(request_state: RequestStateModel) -> str | None:
    if request_state.database_context is not None and request_state.database_context.database_id:
        return str(request_state.database_context.database_id)
    return str(request_state.selected_database) if request_state.selected_database else None


def _stable_id(request_id: str, database_id: str, insight_key: str) -> str:
    digest = hashlib.sha256(f"{request_id}\0{database_id}\0{insight_key}".encode("utf-8")).hexdigest()[:24]
    return f"learn_{digest}"


def _learned_recipe_id(recipe: InsightRecipe) -> str:
    payload = {
        "insight_type": recipe.insight_type,
        "preferred_tool": recipe.preferred_tool,
        "insight_request_template": recipe.insight_request_template,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    return f"recipe_learned_{digest}"


def _abstracted_embedding_text(item: AbstractedInsightRecipe) -> str:
    card = MemoryCard(
        id="candidate",
        kind="insight_recipe",
        title=item.recipe.name,
        description=item.recipe.description or item.definition.description,
        tags=[item.recipe.insight_type, item.recipe.preferred_tool, item.recipe.scope],
    )
    return memory_card_embedding_text(card, item.recipe.insight_request_template, "; ".join(item.recipe.verification_notes))


def _parse_json_response(content: Any) -> dict:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def _response_usage(response: Any) -> dict:
    metadata = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", None) or {}
    return metadata if isinstance(metadata, dict) else {}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@contextmanager
def _database_memory_lock(database_id: str):
    lock_dir = insight_memory_dir() / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", database_id)
    with (lock_dir / f"{safe}.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
