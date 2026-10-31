"""Regression cases reported against 62eec0b, exercised through public build()."""

import asyncio
import json

import pytest

from focus_context_index import InMemoryAdmission, IndexBuildError, IndexEngine, IndexPolicy
from conftest import Probe, messages, snapshot


def test_bad_json_response_is_retried_and_each_attempt_is_settled():
    attempts = 0

    def reply(role, schema, payload):
        nonlocal attempts
        if role == "projector":
            attempts += 1
            if attempts == 1:
                return json.loads('{"units":')
        return None

    async def run():
        engine = IndexEngine(models=Probe(reply=reply))
        result = await engine.build(snapshot=snapshot(), policy=IndexPolicy(max_attempts=2))
        assert attempts == 2
        assert result.stats.model_calls == 4
        assert [r.outcome for r in result.stats.receipts] == [
            "error",
            "success",
            "success",
            "success",
        ]
        assert len(engine.admission.receipts) == 4
        assert engine.admission.pending_count == 0

    asyncio.run(run())


def test_host_settlement_parse_failure_does_not_retry_the_model():
    class BrokenSettlement(InMemoryAdmission):
        async def settle(self, call, receipt):
            await super().settle(call, receipt)
            json.loads('{"host_receipt":')

    async def run():
        probe = Probe()
        engine = IndexEngine(models=probe, admission=BrokenSettlement())
        with pytest.raises(IndexBuildError) as failure:
            await engine.build(snapshot=snapshot(), policy=IndexPolicy(max_attempts=2))
        assert isinstance(failure.value.cause, json.JSONDecodeError)
        assert len(probe.calls) == 1
        assert len(engine.admission.receipts) == 1

    asyncio.run(run())


@pytest.mark.parametrize("max_reads, appended", [(0, 1), (1, 2)])
def test_incremental_prefill_respects_the_same_read_limit_as_cold_build(max_reads, appended):
    async def run():
        policy = IndexPolicy(segment_max_messages=1, max_exact_reads=max_reads)
        engine = IndexEngine(models=Probe())
        old = await engine.build(snapshot=snapshot(messages(1)), policy=policy)
        target = snapshot(messages(1 + appended), generation=2)
        cold = await IndexEngine(models=Probe()).build(snapshot=target, policy=policy)
        incremental = await engine.build(snapshot=target, baseline=old.artifact, policy=policy)
        assert incremental.index.semantic_units == cold.index.semantic_units
        assert len(incremental.index.interpretation.provided) <= max_reads
        assert incremental.stats.local_cache_hits == 1

    asyncio.run(run())


@pytest.mark.parametrize("cancellations", [1, 2, 3])
def test_repeated_cancellation_cannot_abandon_an_in_progress_settlement(cancellations):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        settlement_cancelled = False

        class SlowAdmission(InMemoryAdmission):
            async def settle(self, call, receipt):
                nonlocal settlement_cancelled
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    settlement_cancelled = True
                    raise
                await super().settle(call, receipt)

        admission = SlowAdmission()
        engine = IndexEngine(models=Probe(), admission=admission)
        task = asyncio.create_task(engine.build(snapshot=snapshot()))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            for _ in range(cancellations):
                task.cancel()
                for _ in range(10):
                    await asyncio.sleep(0)
            assert not task.done(), "build returned before settlement was durably acknowledged"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert not settlement_cancelled
        assert admission.pending_count == 0
        assert len(admission.receipts) == 1
        assert admission.receipts[0].outcome == "success"
        recovered = await engine.build(snapshot=snapshot())
        assert recovered.index.interpretation.completed

    asyncio.run(run())
