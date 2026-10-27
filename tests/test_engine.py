import asyncio

import pytest

from focus_context_index import (
    BuildAuthorization,
    InMemoryAdmission,
    IndexArtifact,
    IndexBuildError,
    IndexEngine,
    MemoryIndexStore,
    SQLiteIndexStore,
)
from conftest import Probe, messages, snapshot


def test_cold_append_exact_and_target_evidence(probe):
    async def run():
        engine = IndexEngine(models=probe)
        old = await engine.build(snapshot=snapshot(messages(55)))
        assert old.stats.model_calls == 11
        assert old.index.quality_state == "complete"
        new_snapshot = snapshot(messages(57), generation=2)
        new = await engine.build(snapshot=new_snapshot, baseline=old.artifact)
        assert new.stats.mode == "incremental"
        assert new.stats.local_cache_hits == 4
        assert new.stats.projected_segments == 1
        assert new.stats.model_calls == 3
        assert len(new.index.inheritance.reused_segment_ids) == 4
        assert all(
            ref.source == new_snapshot.source
            for unit in new.index.semantic_units
            for ref in unit.evidence_refs
        )
        assert all(
            ref.source == old.index.source
            for unit in old.index.semantic_units
            for ref in unit.evidence_refs
        )
        exact = await engine.build(snapshot=new_snapshot)
        assert exact.index == new.index
        assert exact.stats.mode == "exact_hit"
        assert exact.stats.model_calls == exact.stats.input_tokens == exact.stats.output_tokens == 0
        assert IndexArtifact.model_validate_json(new.artifact.model_dump_json()) == new.artifact

    asyncio.run(run())


@pytest.mark.parametrize("count,append,reused", [(12, 1, 1), (11, 1, 0), (13, 1, 1), (24, 2, 2)])
def test_append_boundaries(count, append, reused, probe):
    async def run():
        engine = IndexEngine(models=probe)
        first = await engine.build(snapshot=snapshot(messages(count)))
        second = await engine.build(
            snapshot=snapshot(messages(count + append), generation=2), baseline=first.artifact
        )
        assert len(second.index.inheritance.reused_segment_ids) == reused

    asyncio.run(run())


@pytest.mark.parametrize("change", ["body", "tool", "policy", "order", "context", "namespace"])
def test_changed_history_or_scope_cannot_inherit(change, probe):
    async def run():
        engine = IndexEngine(models=probe)
        raw = messages(24)
        first = await engine.build(snapshot=snapshot(raw))
        if change == "body":
            raw[0]["content"] = "corrected"
        if change == "tool":
            raw[0]["name"] = "changed_metadata"
        if change == "policy":
            raw[0]["semantic_policy"] = "reference_only"
        if change == "order":
            raw.reverse()
        target = (
            snapshot(raw + [{"id": "appended", "role": "human", "content": "new"}], generation=2)
            if change not in {"context", "namespace"}
            else None
        )
        # Append uses a new message identity.
        if target is None:
            target = snapshot(
                messages(25),
                generation=2,
                context="other" if change == "context" else "ctx",
                namespace="other" if change == "namespace" else "test",
            )
        result = await engine.build(snapshot=target, baseline=first.artifact)
        assert result.stats.mode == "full"
        assert not result.index.inheritance.reused_segment_ids
        if change in {"context", "namespace"}:
            assert result.stats.local_cache_hits == 0

    asyncio.run(run())


def test_interpreter_change_reuses_local_records_and_changes_proof(probe):
    async def run():
        store = MemoryIndexStore()
        source = snapshot(messages(24))
        first = await IndexEngine(models=probe, store=store).build(snapshot=source)
        next_probe = Probe(identities={"interpreter": "new-interpreter"})
        second = await IndexEngine(models=next_probe, store=store).build(
            snapshot=source, baseline=first.artifact
        )
        assert second.stats.projected_segments == 0
        assert second.stats.local_cache_hits == 2
        assert second.stats.model_calls == 1
        assert second.index.interpretation.record_id != first.index.interpretation.record_id
        assert second.artifact.records == first.artifact.records

    asyncio.run(run())


def test_snapshot_content_participates_in_exact_key_even_if_host_hash_is_reused(probe):
    async def run():
        engine = IndexEngine(models=probe)
        original = snapshot(messages(1))
        first = await engine.build(snapshot=original)
        changed = snapshot([{"id": "m-0", "role": "human", "content": "different"}])
        second = await engine.build(snapshot=changed)
        assert second.stats.mode != "exact_hit"
        assert second.index.index_id != first.index.index_id

    asyncio.run(run())


def test_sqlite_restart_and_local_prewarm(tmp_path, probe):
    async def run():
        path = tmp_path / "index.sqlite"
        source = snapshot(messages(24))
        engine = IndexEngine(models=probe, store=SQLiteIndexStore(path))
        warmed = await engine.prewarm(snapshot=source)
        assert warmed.stats.model_calls == 4
        assert all(role != "interpreter" for role, _ in probe.calls)
        replacement = IndexEngine(models=Probe(), store=SQLiteIndexStore(path))
        built = await replacement.build(snapshot=source)
        assert built.stats.local_cache_hits == 2
        assert built.stats.model_calls == 1
        restarted = await IndexEngine(models=Probe(), store=SQLiteIndexStore(path)).build(
            snapshot=source
        )
        assert restarted.stats.model_calls == 0
        assert restarted.index == built.index

    asyncio.run(run())


def test_failure_keeps_successful_local_records_for_retry(probe):
    async def run():
        async def gate(role, payload):
            if role == "interpreter":
                raise OSError("provider unavailable")

        probe.gate = gate
        engine = IndexEngine(models=probe)
        with pytest.raises(IndexBuildError) as failed:
            await engine.build(snapshot=snapshot(messages(24)))
        assert failed.value.stats.model_calls == 5
        assert not engine.store._artifacts
        assert len(engine.store._segments) == 2
        probe.gate = None
        recovered = await engine.build(snapshot=snapshot(messages(24)))
        assert recovered.stats.model_calls == 1
        assert recovered.stats.local_cache_hits == 2

    asyncio.run(run())


def test_scope_is_checked_before_a_cache_hit_and_budget_zero_can_read(probe):
    async def run():
        engine = IndexEngine(models=probe)
        source = snapshot()
        await engine.build(snapshot=source)
        with pytest.raises(PermissionError):
            await engine.build(
                snapshot=source,
                authorization=BuildAuthorization(
                    owner_id="restricted", allowed_context_ids=("foreign",)
                ),
            )
        cached = await engine.build(
            snapshot=source,
            authorization=BuildAuthorization(owner_id="cached-reader", max_model_calls=0),
        )
        assert cached.stats.model_calls == 0
        with pytest.raises(IndexBuildError) as failure:
            await engine.build(
                snapshot=snapshot(context="new"),
                authorization=BuildAuthorization(owner_id="no-calls", max_model_calls=0),
            )
        assert failure.value.stats.model_calls == 0
        assert not engine.admission.pending_count

    asyncio.run(run())


def test_corrupt_cache_is_rejected(probe):
    async def run():
        engine = IndexEngine(models=probe)
        first = await engine.build(snapshot=snapshot())
        key = first.artifact.cache_key
        engine.store._artifacts[key] = engine.store._artifacts[key].replace(
            "Observed step 0", "tampered"
        )
        with pytest.raises(IndexBuildError, match="integrity|identity"):
            await engine.build(snapshot=snapshot())

    asyncio.run(run())


def test_mutable_caller_input_is_frozen_before_model_work(probe):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def gate(role, payload):
            if role == "projector":
                entered.set()
                await release.wait()

        probe.gate = gate
        source = snapshot([{"id": "m", "role": "human", "content": {"value": "original"}}])
        engine = IndexEngine(models=probe)
        task = asyncio.create_task(engine.build(snapshot=source))
        await asyncio.wait_for(entered.wait(), 2)
        source.messages[0].content["value"] = "changed"
        release.set()
        result = await task
        assert result.index.messages[0].content == {"value": "original"}

    asyncio.run(run())


def test_usage_is_settled_once_for_validation_retry():
    count = 0

    def reply(role, schema, payload):
        nonlocal count
        if role == "projector":
            count += 1
            if count == 1:
                return {"unexpected": True}
        return None

    async def run():
        admission = InMemoryAdmission()
        result = await IndexEngine(models=Probe(reply=reply), admission=admission).build(
            snapshot=snapshot()
        )
        assert result.stats.model_calls == 4
        assert result.stats.input_tokens == 400
        assert len(admission.receipts) == 4
        assert [r.outcome for r in admission.receipts].count("error") == 1
        assert admission.pending_count == 0

    asyncio.run(run())
