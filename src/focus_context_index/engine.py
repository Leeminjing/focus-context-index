"""Incremental orchestration without Focus database, Loop or Portfolio dependencies."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

from .admission import InMemoryAdmission
from .api import (
    BuildAuthorization,
    BuildResult,
    BuildStats,
    FrozenSnapshot,
    IndexArtifact,
    IndexContracts,
    IndexPolicy,
    IndexProgress,
    PrewarmResult,
    artifact_key,
)
from .contracts import stable_expansion_hash
from .cancellation import complete_before_cancel
from .errors import IndexBuildError
from .index_assembly import assemble_revision_index
from .index_build_contracts import IndexBuildPlan
from .index_scheduling import IndexModelSlots, SegmentLocks
from .interpretation_record import RevisionInterpretationProposal, RevisionInterpretationRecord
from .model_runtime import ManagedModel
from .ports import Admission, IndexStore, ModelAdapter
from .revision_interpretation import INTERPRETATION_PROMPT, RevisionInterpretationBuilder
from .segment_projection import SEGMENT_PROMPT, SegmentProjectionBuilder, SegmentProjectionRecord
from .semantic_grounding import (
    SemanticClaimSupportProposal,
    SemanticGroundingValidator,
    SemanticProjectionProposal,
    SupervisedSemanticClaimSupportVerifier,
)
from .semantic_indexer import ProtocolSafeRevisionSegmenter, RevisionSemanticIndexer
from .stores import MemoryIndexStore

_logger = logging.getLogger(__name__)
ProgressCallback = Callable[[IndexProgress], Awaitable[None] | None]


@dataclass
class _State:
    mode: str = "full"
    cache_hits: int = 0
    projected: int = 0
    completed: int = 0
    total: int = 0
    receipts: list = field(default_factory=list)
    stage_ms: dict = field(default_factory=dict)
    observers: dict = field(default_factory=dict)


@dataclass
class _Shared:
    task: asyncio.Task
    state: _State
    waiters: int = 0


class IndexEngine:
    """Share one instance per event loop to coordinate slots and in-flight work.

    build() publishes only complete artifacts. prewarm() checkpoints local records.
    A supplied baseline is a host-authorized ancestor; this engine independently
    verifies its namespace, Context, contract, message prefix and record inventory.
    """

    def __init__(
        self,
        *,
        models: ModelAdapter,
        store: IndexStore | None = None,
        admission: Admission | None = None,
        concurrency: int = 4,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.models = models
        self.store = store if store is not None else MemoryIndexStore()
        self.admission = admission if admission is not None else InMemoryAdmission()
        self._concurrency = concurrency
        self._slots = IndexModelSlots(concurrency)
        self._segment_locks = SegmentLocks()
        self._inflight: dict[str, _Shared] = {}

    def contracts(self, policy: IndexPolicy | None = None) -> IndexContracts:
        """Expose immutable fingerprints for host persistence and ancestry lookup."""
        policy = policy or IndexPolicy()
        indexer = RevisionSemanticIndexer(
            ProtocolSafeRevisionSegmenter(
                max_messages=policy.segment_max_messages,
                max_estimated_tokens=policy.segment_max_estimated_tokens,
            )
        )
        identities = {
            role: self.models.identity(role) for role in ("projector", "verifier", "interpreter")
        }
        local, synthesis, revision = _contracts(indexer, policy, identities)
        return IndexContracts(
            index_schema_version=indexer.INDEX_SCHEMA_VERSION,
            segmenter_version=indexer.segmenter_version,
            local=local,
            interpretation=synthesis,
            revision=revision,
        )

    async def build(
        self,
        *,
        snapshot: FrozenSnapshot,
        baseline: IndexArtifact | None = None,
        policy: IndexPolicy | None = None,
        authorization: BuildAuthorization | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        return await self._start(snapshot, baseline, policy, authorization, on_progress, False)

    async def prewarm(
        self,
        *,
        snapshot: FrozenSnapshot,
        policy: IndexPolicy | None = None,
        authorization: BuildAuthorization | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> PrewarmResult:
        return await self._start(snapshot, None, policy, authorization, on_progress, True)

    async def _start(self, snapshot, baseline, policy, authorization, observer, background):
        started = perf_counter()
        # A frozen Pydantic model can contain mutable JSON values. Reparse at the
        # public boundary so caller mutations cannot change an in-flight snapshot.
        snapshot = FrozenSnapshot.model_validate_json(snapshot.model_dump_json())
        policy = policy or IndexPolicy()
        authorization = authorization or BuildAuthorization()
        await self.admission.authorize(snapshot, authorization)
        if baseline is not None:
            baseline = IndexArtifact.model_validate_json(baseline.model_dump_json())
        identities = {
            role: self.models.identity(role) for role in ("projector", "verifier", "interpreter")
        }
        if any(not isinstance(v, str) or not v for v in identities.values()):
            raise ValueError("model roles require explicit cache identities")
        key = stable_expansion_hash(
            "authorized-independent-index-build-v1",
            snapshot.fingerprint,
            baseline.index.index_id if baseline else None,
            policy.model_dump(mode="json"),
            authorization.fingerprint,
            identities,
            background,
        )
        shared = self._inflight.get(key)
        joined = shared is not None
        if shared is None:
            state = _State(mode="prewarm" if background else "full")
            task = asyncio.create_task(
                self._execute(
                    snapshot, baseline, policy, authorization, identities, state, background
                )
            )
            shared = self._inflight[key] = _Shared(task, state)
        shared.waiters += 1
        token = uuid4().hex
        if observer is not None:
            shared.state.observers[token] = observer
        try:
            result = await asyncio.shield(shared.task)
            # Every waiter receives its own JSON tree; nested content is never shared.
            copied = type(result).model_validate_json(result.model_dump_json())
        finally:
            shared.state.observers.pop(token, None)
            shared.waiters -= 1
            if shared.waiters == 0:
                if self._inflight.get(key) is shared:
                    del self._inflight[key]
                if not shared.task.done():
                    shared.task.cancel()
                await complete_before_cancel(asyncio.gather(shared.task, return_exceptions=True))
        return copied.model_copy(
            update={
                "stats": copied.stats.model_copy(
                    update={
                        "joined_build": joined,
                        "wait_duration_ms": (perf_counter() - started) * 1000,
                    }
                )
            }
        )

    async def _execute(
        self, snapshot, baseline, policy, authorization, identities, state, background
    ):
        started = perf_counter()
        try:
            indexer = RevisionSemanticIndexer(
                ProtocolSafeRevisionSegmenter(
                    max_messages=policy.segment_max_messages,
                    max_estimated_tokens=policy.segment_max_estimated_tokens,
                )
            )
            local, synthesis, contract = _contracts(indexer, policy, identities)
            frozen = indexer.index(
                source=snapshot.source,
                source_content_hash=snapshot.source_content_hash,
                context_role=snapshot.context_role,
                active_objective=snapshot.active_objective,
                raw_messages=tuple(m.model_dump(mode="json") for m in snapshot.messages),
                projector_version=contract,
            )
            state.total = len(frozen.segments)
            await self._progress(snapshot, state, "preparing")
            key = artifact_key(
                snapshot.fingerprint,
                indexer.segmenter_version,
                contract,
                indexer.INDEX_SCHEMA_VERSION,
            )
            cached = await self.store.get_artifact(key)
            if cached is not None:
                cached = self._validate_artifact(cached, snapshot, frozen)
                state.mode = "prewarm" if background else "exact_hit"
                state.cache_hits = state.completed = state.total
                await self._progress(snapshot, state, "complete")
                stats = self._stats(state, started)
                return (
                    PrewarmResult(records=cached.records, stats=stats)
                    if background
                    else BuildResult(artifact=cached, stats=stats)
                )

            plan, inherited = _plan(frozen, snapshot.namespace, baseline, local)
            if not background:
                state.mode = plan.mode
            state.cache_hits = state.completed = len(inherited)
            by_id = {m.message_id: m for m in frozen.messages}
            segment_messages = {
                s.segment_id: tuple(by_id[i] for i in s.message_ids) for s in frozen.segments
            }

            def model(role, phase):
                return ManagedModel(
                    adapter=self.models,
                    role=role,
                    identity=identities[role],
                    snapshot=snapshot,
                    authorization=authorization,
                    policy=policy,
                    slots=self._slots,
                    admission=self.admission,
                    receipts=state.receipts,
                    phase=phase,
                    background=background,
                )

            local_started = perf_counter()
            records = list(inherited) + [None] * (len(frozen.segments) - len(inherited))
            remaining = iter(enumerate(frozen.segments[len(inherited) :], start=len(inherited)))

            async def worker():
                for ordinal, segment in remaining:
                    cache_key = SegmentProjectionRecord.cache_key_for(segment, local)
                    lock_key = (
                        snapshot.namespace,
                        frozen.source.context_id,
                        cache_key,
                        authorization.fingerprint,
                    )
                    async with self._segment_locks.hold(lock_key):
                        record = await self.store.get_segment(
                            snapshot.namespace, frozen.source.context_id, cache_key
                        )
                        if record is not None:
                            record = SegmentProjectionRecord.model_validate_json(
                                record.model_dump_json()
                            )
                            state.cache_hits += 1
                        else:
                            builder = SegmentProjectionBuilder(
                                frozen.source.context_id,
                                local,
                                model("projector", "local_projection"),
                                model("verifier", "local_verifier"),
                                lambda attempts: None,
                            )
                            record = await builder.build(
                                segment, segment_messages[segment.segment_id]
                            )
                            state.projected += 1
                            record = await self.store.put_segment(snapshot.namespace, record)
                            record = SegmentProjectionRecord.model_validate_json(
                                record.model_dump_json()
                            )
                        if record.context_id != frozen.source.context_id:
                            raise ValueError("local record Context scope integrity mismatch")
                        record.validate_target(segment, segment_messages[segment.segment_id], local)
                        records[ordinal] = record
                    state.completed += 1
                    await self._progress(snapshot, state, "local")

            # A bounded number of workers, not one task for every historical segment.
            tasks = [
                asyncio.create_task(worker())
                for _ in range(min(self._concurrency, len(frozen.segments) - len(inherited)))
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await complete_before_cancel(asyncio.gather(*tasks, return_exceptions=True))
                state.stage_ms["local"] = (perf_counter() - local_started) * 1000
            records = tuple(records)
            if background:
                await self._progress(snapshot, state, "complete")
                return PrewarmResult(records=records, stats=self._stats(state, started))

            await self._progress(snapshot, state, "interpretation")
            interpretation_started = perf_counter()
            try:
                proof = await RevisionInterpretationBuilder(
                    model("interpreter", "interpretation"),
                    model("verifier", "interpretation_verifier"),
                    SimpleNamespace(policy=policy),
                    synthesis,
                    local,
                    lambda attempts: None,
                ).build(frozen, records, plan)
            finally:
                state.stage_ms["interpretation"] = (perf_counter() - interpretation_started) * 1000
            assembly_started = perf_counter()
            index = assemble_revision_index(
                indexer, frozen, records, plan, interpretation=proof, local_contract=local
            )
            artifact = IndexArtifact(
                namespace=snapshot.namespace,
                snapshot_fingerprint=snapshot.fingerprint,
                index=index,
                records=records,
            )
            if any(self.models.identity(role) != identity for role, identity in identities.items()):
                raise ValueError("model configuration changed before index publication")
            winner = await self.store.put_artifact(artifact)
            winner = self._validate_artifact(winner, snapshot, frozen)
            state.stage_ms["assembly_and_store"] = (perf_counter() - assembly_started) * 1000
            await self._progress(snapshot, state, "complete")
            return BuildResult(artifact=winner, stats=self._stats(state, started))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise IndexBuildError(exc, self._stats(state, started)) from exc

    @staticmethod
    def _validate_artifact(artifact, snapshot, frozen):
        artifact = IndexArtifact.model_validate_json(artifact.model_dump_json())
        if (
            artifact.snapshot_fingerprint != snapshot.fingerprint
            or artifact.namespace != snapshot.namespace
            or artifact.index.source != snapshot.source
            or artifact.index.segments != frozen.segments
            or artifact.index.projector_version != frozen.projector_version
            or artifact.index.segmenter_version != frozen.segmenter_version
            or artifact.index.index_schema_version != frozen.index_schema_version
        ):
            raise ValueError("cached artifact target integrity mismatch")
        return artifact

    @staticmethod
    def _stats(state, started):
        def total(name):
            values = [getattr(r, name) for r in state.receipts]
            return None if any(v is None for v in values) else sum(values)

        return BuildStats(
            mode=state.mode,
            local_cache_hits=state.cache_hits,
            projected_segments=state.projected,
            total_segments=state.total,
            model_calls=sum(r.model_calls for r in state.receipts),
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            receipts=tuple(state.receipts),
            stage_ms=dict(state.stage_ms),
            duration_ms=(perf_counter() - started) * 1000,
        )

    @staticmethod
    async def _progress(snapshot, state, phase):
        progress = IndexProgress(
            context_id=snapshot.source.context_id,
            revision_id=snapshot.source.revision_id,
            phase=phase,
            completed_segments=state.completed,
            total_segments=state.total,
        )
        for observer in tuple(state.observers.values()):
            try:
                value = observer(progress)
                if inspect.isawaitable(value):
                    await value
            except Exception:
                _logger.exception("index progress observer failed")


def _plan(target, namespace, baseline, local_contract):
    def full(reason):
        return IndexBuildPlan(
            recomputed_segment_ids=tuple(s.segment_id for s in target.segments),
            full_build_reason=reason,
        ), ()

    if baseline is None:
        return full("no_baseline")
    old = baseline.index
    if namespace != baseline.namespace or target.source.context_id != old.source.context_id:
        return full("different_source_scope")
    if (
        target.index_schema_version != old.index_schema_version
        or target.segmenter_version != old.segmenter_version
        or old.interpretation.local_contract_fingerprint != local_contract
    ):
        return full("incompatible_local_contract")
    if target.source != old.source and target.source.generation <= old.source.generation:
        return full("non_ancestor_generation")
    if target.messages[: len(old.messages)] != old.messages:
        return full("non_append_history")
    reused = []
    for old_segment, new_segment, record in zip(old.segments, target.segments, baseline.records):
        if old_segment != new_segment:
            break
        reused.append(record)
    count = len(reused)
    return IndexBuildPlan(
        mode="incremental",
        baseline_index_id=old.index_id,
        prefix_hash=stable_expansion_hash(
            "execution-prefix-v1", tuple(m.model_dump(mode="json") for m in old.messages)
        ),
        reused_segment_ids=tuple(s.segment_id for s in target.segments[:count]),
        recomputed_segment_ids=tuple(s.segment_id for s in target.segments[count:]),
        full_build_reason=None,
    ), tuple(reused)


def _contracts(indexer, policy, identities):
    local = (
        "loc:"
        + stable_expansion_hash(
            "segment-local-projection-v2",
            SegmentProjectionRecord.SCHEMA_VERSION,
            indexer.INDEX_SCHEMA_VERSION,
            indexer.segmenter_version,
            SEGMENT_PROMPT,
            SupervisedSemanticClaimSupportVerifier.VERSION,
            SemanticGroundingValidator.VERSION,
            SemanticProjectionProposal.model_json_schema(),
            SemanticClaimSupportProposal.model_json_schema(),
            identities["projector"],
            identities["verifier"],
        )[:60]
    )
    synthesis = (
        "syn:"
        + stable_expansion_hash(
            "independent-frozen-evidence-window-v1",
            RevisionInterpretationRecord.SCHEMA_VERSION,
            RevisionInterpretationProposal.model_json_schema(),
            INTERPRETATION_PROMPT,
            identities["interpreter"],
            identities["verifier"],
            policy.model_dump(mode="json"),
            SemanticGroundingValidator.VERSION,
            SupervisedSemanticClaimSupportVerifier.VERSION,
        )[:60]
    )
    contract = (
        "rev:" + stable_expansion_hash("revision-projection-contract-v1", local, synthesis)[:60]
    )
    return local, synthesis, contract
