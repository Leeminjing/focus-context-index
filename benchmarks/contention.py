"""Measure 1/4/16 Contexts competing for one Engine; no fairness or speedup claim."""

import argparse
import asyncio
import importlib
import json
import random
from pathlib import Path
from time import perf_counter

from focus_context_index import IndexBuildError, IndexEngine, IndexPolicy
from focus_context_index.contracts import stable_expansion_hash
from .quality import quality_proxy
from .run import corpus, freeze, percentile


async def benchmark_contention(
    *,
    factory,
    context_counts=(1, 4, 16),
    repeats=3,
    concurrency=4,
    seed=20261031,
    mode="offline-scripted",
):
    if repeats < 1 or concurrency < 1 or not context_counts or any(n < 1 for n in context_counts):
        raise ValueError("positive repeats, concurrency and Context counts are required")
    jobs = [(repeat, count) for repeat in range(repeats) for count in context_counts]
    random.Random(seed).shuffle(jobs)
    all_rows, batches = [], []
    for repeat, count in jobs:
        models = factory()
        engine = IndexEngine(models=models, concurrency=concurrency)
        policy = IndexPolicy()
        # One long Context and the remaining short Contexts make head-of-line
        # effects visible. Replica identities isolate content caches per Context.
        cases = []
        for i in range(count):
            template = corpus()[0 if i == 0 else 2]
            cases.append({**template, "name": f"{template['name']}-client-{i}"})
        baselines, blocked = [None] * count, [False] * count
        for phase in ("cold", "append", "exact"):
            batch_started = perf_counter()
            receipts = {}

            async def build_one(i, requested_at):
                case = cases[i]
                target = freeze(
                    case,
                    case["baseline"] if phase == "cold" else case["baseline"] + case["append"],
                    1 if phase == "cold" else 2,
                )
                row = {
                    "repeat": repeat,
                    "context_count": count,
                    "phase": phase,
                    "context_id": target.source.context_id,
                    "case": case["name"],
                    "length_class": "long" if i == 0 else "short",
                    "snapshot_hash": target.fingerprint,
                    "model_identities": {
                        role: models.identity(role)
                        for role in ("projector", "verifier", "interpreter")
                    },
                }
                if blocked[i]:
                    return {**row, "outcome": "blocked", "caller_duration_ms": None, "stats": None}
                try:
                    built = await engine.build(
                        snapshot=target,
                        policy=policy,
                        baseline=baselines[i] if phase == "append" else None,
                    )
                    caller_ms = (perf_counter() - requested_at) * 1000
                    baselines[i] = built.artifact
                    stats = built.stats
                    row.update(
                        outcome="ready",
                        quality=quality_proxy(
                            built.index, case["joint_support_sets"] if phase != "cold" else []
                        ),
                    )
                except IndexBuildError as exc:
                    caller_ms = (perf_counter() - requested_at) * 1000
                    blocked[i] = True
                    stats = exc.stats
                    row.update(outcome="failed", error_type=type(exc.cause).__name__)
                receipts.update({r.call_id: r for r in stats.receipts})
                return {
                    **row,
                    "caller_duration_ms": caller_ms,
                    "stats": stats.model_dump(mode="json", exclude={"receipts"}),
                }

            tasks = [asyncio.create_task(build_one(i, perf_counter())) for i in range(count)]
            try:
                rows = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            batch_ms = (perf_counter() - batch_started) * 1000
            ready = [r for r in rows if r["outcome"] == "ready"]
            short = [r["caller_duration_ms"] for r in ready if r["length_class"] == "short"]
            queue = [r.queue_ms for r in receipts.values()]
            batches.append(
                {
                    "repeat": repeat,
                    "context_count": count,
                    "phase": phase,
                    "batch_duration_ms": batch_ms,
                    "requests": count,
                    "failures": count - len(ready),
                    "quality_passed": sum(r["quality"]["quality_pass"] is True for r in ready),
                    "quality_failed": sum(r["quality"]["quality_pass"] is False for r in ready),
                    "quality_pending": sum(r["quality"]["quality_pass"] is None for r in ready),
                    "short_caller_p50_ms": percentile(short, 0.5) if short else None,
                    "short_caller_p95_ms": percentile(short, 0.95) if short else None,
                    "queue_p95_ms": percentile(queue, 0.95) if queue else 0,
                    "model_calls": sum(r.model_calls for r in receipts.values()),
                    "receipt_count": len(receipts),
                    "input_tokens": _total_tokens(receipts.values(), "input_tokens"),
                    "output_tokens": _total_tokens(receipts.values(), "output_tokens"),
                }
            )
            all_rows.extend(rows)
    return {
        "schema": "context-index-contention-v1",
        "mode": mode,
        "load_source": "frozen synthetic long/short Context replicas; not real user histories",
        "seed": seed,
        "repeats": repeats,
        "concurrency": concurrency,
        "context_counts": list(context_counts),
        "policy": IndexPolicy().model_dump(mode="json"),
        "corpus_hash": stable_expansion_hash("contention-corpus-v1", corpus()),
        "execution_order": [{"repeat": r, "context_count": n} for r, n in jobs],
        "batches": batches,
        "runs": all_rows,
        "real_history_comparison": "pending_data",
    }


def _total_tokens(receipts, field):
    values = [getattr(r, field) for r in receipts]
    return None if any(v is None for v in values) else sum(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="module:factory; default is an offline scripted adapter")
    parser.add_argument("--context-counts", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/contention.json"))
    args = parser.parse_args()
    if args.adapter:
        name, attribute = args.adapter.split(":", 1)
        factory, mode = (
            getattr(importlib.import_module(name), attribute),
            f"external:{args.adapter}",
        )
    else:
        from examples.scripted_models import create_adapter

        factory, mode = create_adapter, "offline-scripted"
    report = asyncio.run(
        benchmark_contention(
            factory=factory,
            mode=mode,
            context_counts=args.context_counts,
            repeats=args.repeats,
            concurrency=args.concurrency,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = sum(b["failures"] for b in report["batches"])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "batches": len(report["batches"]),
                "requests": len(report["runs"]),
                "failures": failures,
                "mode": mode,
                "real_history_comparison": "pending_data",
            }
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
