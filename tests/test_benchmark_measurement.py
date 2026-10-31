import asyncio
import pytest

from benchmarks.run import benchmark
from focus_context_index import IndexEngine
from examples.scripted_models import create_adapter
from benchmarks.contention import benchmark_contention
from benchmarks.review import apply_reviews
from conftest import Probe


def test_benchmark_times_the_entire_await_and_separates_quality_from_execution(monkeypatch):
    original = IndexEngine.build

    async def delayed_return(self, **kwargs):
        result = await original(self, **kwargs)
        await asyncio.sleep(0.02)
        return result

    monkeypatch.setattr(IndexEngine, "build", delayed_return)

    async def run():
        report = await benchmark(
            factory=create_adapter, mode="offline-scripted", repeats=1, segment_sizes=[12], seed=7
        )
        for row in report["runs"]:
            assert row["stats"]["caller_duration_ms"] >= row["stats"]["wait_duration_ms"] + 15
        missing_joint = [
            row for row in report["runs"] if row["quality"]["expected_joint_support_sets"] > 0
        ]
        assert missing_joint
        assert all(row["outcome"] == "ready" for row in missing_joint)
        assert all(row["quality"]["quality_pass"] is False for row in missing_joint)
        assert all(item["quality_passed"] == 0 for item in report["summary"])

    asyncio.run(run())


def test_multiple_contexts_compete_for_one_engine_and_exact_hits_have_no_calls():
    probe = Probe()

    async def run():
        report = await benchmark_contention(
            factory=lambda: probe, context_counts=(1, 4), repeats=1, concurrency=2, seed=7
        )
        assert len(report["runs"]) == 15
        assert all(b["model_calls"] == 0 for b in report["batches"] if b["phase"] == "exact")
        assert all(
            b["short_caller_p95_ms"] is not None
            for b in report["batches"]
            if b["context_count"] == 4
        )
        assert all(b["failures"] == 0 for b in report["batches"])
        assert probe.max_active <= 2
        assert report["real_history_comparison"] == "pending_data"

    asyncio.run(run())


def test_quality_approval_is_bound_to_the_actual_index_and_cannot_override_failed_proxies():
    async def run():
        report = await benchmark(
            factory=create_adapter, mode="offline-scripted", repeats=1, segment_sizes=[12], seed=7
        )
        short = next(r["quality"] for r in report["runs"] if r["case"] == "short_dialogue")
        long = next(
            r["quality"] for r in report["runs"] if r["quality"]["expected_joint_support_sets"]
        )
        assert short["quality_pass"] is None
        reviews = [
            {
                "index_id": q["index_id"],
                "criteria_hash": q["criteria_hash"],
                "reviewer": "test-reviewer",
                "verdict": "approved",
            }
            for q in (short, long)
        ]
        reviewed = apply_reviews(report, reviews)
        matched = [
            r["quality"]
            for r in reviewed["runs"]
            if r.get("quality", {}).get("index_id") == short["index_id"]
        ]
        assert matched and all(q["quality_pass"] is True for q in matched)
        assert all(
            r["quality"]["quality_pass"] is False
            for r in reviewed["runs"]
            if r["quality"]["index_id"] == long["index_id"]
        )
        assert short["quality_pass"] is None  # input measurements and report are immutable
        with pytest.raises(ValueError, match="absent"):
            apply_reviews(report, [{**reviews[0], "criteria_hash": "0" * 64}])

    asyncio.run(run())
