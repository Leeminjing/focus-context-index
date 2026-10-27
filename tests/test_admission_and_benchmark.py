import ast
import asyncio
from pathlib import Path

import pytest

from focus_context_index import BuildAuthorization, IndexBuildError, IndexEngine, IndexPolicy
from benchmarks.run import benchmark
from conftest import Probe, snapshot
from examples.scripted_models import create_adapter


@pytest.mark.parametrize("reported", [False, None])
def test_unknown_provider_tokens_remain_unknown(reported):
    class UnknownUsage(Probe):
        def create(self, role):
            model = super().create(role)
            original = model.invoke

            async def invoke(schema, system, payload):
                result = await original(schema, system, payload)
                model.last_attempt_records = (
                    (
                        {
                            "model_calls": 1,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "usage_reported": False,
                        },
                    )
                    if reported is False
                    else ()
                )
                return result

            model.invoke = invoke
            return model

    async def run():
        engine = IndexEngine(models=UnknownUsage())
        result = await engine.build(snapshot=snapshot())
        assert result.stats.input_tokens is result.stats.output_tokens is None
        assert result.stats.model_calls == 3
        assert len(engine.admission.receipts) == 3
        assert not engine.admission.pending_count

    asyncio.run(run())


def test_output_reserve_includes_actual_model_output_limit():
    class TooLarge(Probe):
        def create(self, role):
            model = super().create(role)
            model.max_output_tokens = 5000
            model.context_window_tokens = 4000
            return model

    async def run():
        probe = TooLarge()
        with pytest.raises(IndexBuildError, match="capacity") as failure:
            await IndexEngine(models=probe).build(
                snapshot=snapshot(), policy=IndexPolicy(output_token_reserve=100)
            )
        assert not probe.calls
        assert failure.value.stats.model_calls == 0

    asyncio.run(run())


def test_parallel_builds_share_one_authorization_budget():
    async def run():
        probe = Probe()
        engine = IndexEngine(models=probe)
        auth = BuildAuthorization(max_model_calls=1)
        results = await asyncio.gather(
            *[
                engine.build(snapshot=snapshot(context=context), authorization=auth)
                for context in ("one", "two")
            ],
            return_exceptions=True,
        )
        assert all(isinstance(r, IndexBuildError) for r in results)
        assert len(probe.calls) == 1
        assert sum(r.model_calls for r in engine.admission.receipts) == 1
        assert engine.admission.pending_count == 0
        assert not engine.store._artifacts

    asyncio.run(run())


def test_configuration_changes_before_publication_fail_closed():
    def reply(role, schema, payload):
        if role == "interpreter":
            probe.identities["projector"] = "changed-configuration"
        return None

    async def run():
        engine = IndexEngine(models=probe)
        with pytest.raises(IndexBuildError, match="configuration changed"):
            await engine.build(snapshot=snapshot())
        assert not engine.store._artifacts
        assert len(engine.admission.receipts) == 3

    probe = Probe(reply=reply)
    asyncio.run(run())


def test_benchmark_seed_cache_hits_and_quality_proxy_are_explicit():
    async def run():
        report = await benchmark(
            factory=create_adapter, mode="offline-scripted", repeats=1, segment_sizes=[12], seed=7
        )
        assert len(report["runs"]) == 9
        exact = [row for row in report["runs"] if row["phase"] == "exact"]
        assert all(row["stats"]["model_calls"] == 0 for row in exact)
        assert all(row["quality"]["human_semantic_review"] == "pending" for row in exact)
        assert report["mode"] == "offline-scripted"
        assert report["corpus_hash"]

    asyncio.run(run())


def test_benchmark_preserves_failure_denominator_and_blocks_dependent_phases():
    async def fail(role, payload):
        raise OSError("provider failure")

    async def run():
        report = await benchmark(
            factory=lambda: Probe(gate=fail),
            mode="test-failure",
            repeats=1,
            segment_sizes=[12],
            seed=7,
        )
        assert len(report["runs"]) == 9
        assert [r["outcome"] for r in report["runs"]].count("failed") == 3
        assert [r["outcome"] for r in report["runs"]].count("blocked") == 6
        assert all(s["samples"] == s["failures"] == 1 for s in report["summary"])

    asyncio.run(run())


def test_core_has_no_focus_application_imports():
    package = Path(__file__).resolve().parents[1] / "src/focus_context_index"
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            assert all(
                name.split(".")[0] not in {"backend", "focus", "sqlalchemy"} for name in names
            ), path
