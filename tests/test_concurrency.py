import asyncio

import pytest

from focus_context_index import BuildAuthorization, IndexBuildError, IndexEngine, IndexPolicy
from conftest import Probe, messages, snapshot


@pytest.mark.parametrize("cancel_waiter", [None, 0, 1])
def test_identical_builds_share_work_and_cancel_independently(cancel_waiter):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def gate(role, payload):
            if role == "projector":
                entered.set()
                await release.wait()

        probe = Probe(gate=gate)
        engine = IndexEngine(models=probe)
        source = snapshot()
        tasks = [asyncio.create_task(engine.build(snapshot=source)) for _ in range(2)]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert len(engine._inflight) == 1
            assert next(iter(engine._inflight.values())).waiters == 2
            if cancel_waiter is not None:
                tasks[cancel_waiter].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[cancel_waiter]
                assert probe.active == 1
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = [r for r in results if not isinstance(r, BaseException)]
            assert all(r.index == successful[0].index for r in successful)
            assert len(probe.calls) == len(engine.admission.receipts) == 3
            assert not engine._inflight
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_last_waiter_cancellation_drains_and_restarts_from_local_checkpoints():
    async def run():
        entered = asyncio.Event()

        async def gate(role, payload):
            if role == "interpreter":
                entered.set()
                await asyncio.Event().wait()

        probe = Probe(gate=gate)
        engine = IndexEngine(models=probe)
        task = asyncio.create_task(engine.build(snapshot=snapshot(messages(24))))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe.active == engine.admission.pending_count == 0
        assert not engine._inflight and not engine._segment_locks._entries
        assert len(engine.admission.receipts) == 5
        assert engine.admission.receipts[-1].outcome == "cancelled"
        assert not engine.store._artifacts
        probe.gate = None
        recovered = await engine.build(snapshot=snapshot(messages(24)))
        assert recovered.stats.model_calls == 1
        assert recovered.stats.local_cache_hits == 2

    asyncio.run(run())


def test_slots_are_shared_by_different_contexts_and_workers_are_bounded():
    async def run():
        release = asyncio.Event()
        full = asyncio.Event()

        async def gate(role, payload):
            if probe.active == 2:
                full.set()
            await release.wait()

        probe = Probe(gate=gate)
        engine = IndexEngine(models=probe, concurrency=2)
        tasks = [
            asyncio.create_task(
                engine.build(
                    snapshot=snapshot(messages(30), context=context),
                    policy=IndexPolicy(segment_max_messages=1),
                )
            )
            for context in ("long", "short")
        ]
        try:
            await asyncio.wait_for(full.wait(), 2)
            workers = [
                t for t in asyncio.all_tasks() if t.get_coro().__qualname__.endswith("worker")
            ]
            assert len(workers) <= 4
            assert probe.max_active <= 2
            release.set()
            await asyncio.gather(*tasks)
            assert probe.max_active <= 2
            assert engine.admission.pending_count == 0
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_different_authorizations_do_not_share_inflight_calls_but_adopt_one_winner():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def gate(role, payload):
            if role == "projector":
                if sum(r == "projector" for r, _ in probe.calls) == 2:
                    entered.set()
                await release.wait()

        probe = Probe(gate=gate)
        engine = IndexEngine(models=probe, concurrency=2)
        tasks = [
            asyncio.create_task(
                engine.build(snapshot=snapshot(), authorization=BuildAuthorization(owner_id=owner))
            )
            for owner in ("one", "two")
        ]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert len(engine._inflight) == 2
            release.set()
            a, b = await asyncio.gather(*tasks)
            assert a.index == b.index
            assert a.stats.model_calls == b.stats.model_calls == 3
            assert len(engine.admission.receipts) == 6
            assert {r.owner_id for r in engine.admission.receipts} == {"one", "two"}
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_parallel_failure_cancels_siblings_and_settles_all_started_calls():
    async def run():
        ready = asyncio.Event()

        async def gate(role, payload):
            if role != "projector":
                return
            identity = payload["segments"][0]["messages"][0]["message_id"]
            if identity == "m-1":
                await ready.wait()
                raise OSError("failed segment")
            if identity == "m-2":
                ready.set()
                await asyncio.Event().wait()

        probe = Probe(gate=gate)
        engine = IndexEngine(models=probe, concurrency=3)
        with pytest.raises(IndexBuildError) as failure:
            await engine.build(
                snapshot=snapshot(messages(3)), policy=IndexPolicy(segment_max_messages=1)
            )
        assert not engine.store._artifacts
        assert probe.active == engine.admission.pending_count == 0
        assert not engine._segment_locks._entries
        assert len(failure.value.stats.receipts) == len(engine.admission.receipts)
        assert any(r.outcome == "error" for r in engine.admission.receipts)
        assert any(r.outcome == "cancelled" for r in engine.admission.receipts)

    asyncio.run(run())
