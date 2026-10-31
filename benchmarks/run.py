"""Repeatable, randomized cold/append/exact benchmark. Default models are offline scripts."""

import argparse
import asyncio
import importlib
import json
import math
import random
import platform
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import focus_context_index
import pydantic

from focus_context_index import (
    BuildStats,
    ContextRevisionRef,
    FrozenSnapshot,
    IndexBuildError,
    IndexEngine,
    IndexPolicy,
)
from focus_context_index.contracts import stable_expansion_hash
from .quality import quality_proxy


def corpus():
    history = [
        {"id": "suspect", "role": "human", "content": "Initially we suspected the connection pool."}
    ]
    history += [
        {
            "id": f"filler-{i}",
            "role": "human",
            "content": f"Unrelated observation {i}: deployment remained stable.",
        }
        for i in range(54)
    ]
    correction = [
        {"id": "reject", "role": "human", "content": "The connection pool was ruled out."},
        {
            "id": "diagnosis",
            "role": "human",
            "content": "The thread-pool deadlock was confirmed as the cause.",
        },
    ]
    tool = [
        {"id": "question", "role": "human", "content": "Investigate the failing service."},
        {
            "id": "call",
            "role": "ai",
            "semantic_policy": "evidence_only",
            "tool_calls": [{"id": "a", "name": "read_logs", "args": {}}],
        },
        {
            "id": "result",
            "role": "tool",
            "tool_call_id": "a",
            "semantic_policy": "evidence_only",
            "content": "Thread worker log detail. " * 800,
        },
        {
            "id": "after",
            "role": "human",
            "content": "Initially this appeared to be a network fault.",
        },
    ]
    return [
        {
            "name": "distant_correction_55_to_57",
            "baseline": history,
            "append": correction,
            "joint_support_sets": [["suspect", "reject", "diagnosis"]],
        },
        {
            "name": "long_tool_exchange",
            "baseline": tool,
            "append": [
                {
                    "id": "fix",
                    "role": "human",
                    "content": "The network fault was ruled out; investigate the worker.",
                }
            ],
            "joint_support_sets": [["after", "fix"]],
        },
        {
            "name": "short_dialogue",
            "baseline": [{"id": "q", "role": "human", "content": "Please check the build."}],
            "append": [{"id": "a", "role": "ai", "content": "The build passed."}],
            "joint_support_sets": [],
        },
    ]


def freeze(case, raw, generation):
    return FrozenSnapshot.from_messages(
        namespace="benchmark",
        source=ContextRevisionRef(
            context_id=case["name"],
            revision_id=f"{case['name']}-{generation}",
            generation=generation,
            execution_thread_id="benchmark-thread",
            checkpoint_id=f"cp-{generation}",
            payload_mode="checkpoint",
        ),
        source_content_hash=stable_expansion_hash("benchmark-host-content", raw),
        raw_messages=raw,
    )


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["segment_size"], row["phase"])].append(row)
    result = []
    for (case, size, phase), group in sorted(groups.items()):
        successful = [r for r in group if r["outcome"] == "ready"]
        qualified = [r for r in successful if r["quality"]["quality_pass"] is True]
        item = {
            "case": case,
            "segment_size": size,
            "phase": phase,
            "samples": len(group),
            "failures": len(group) - len(successful),
            "quality_passed": len(qualified),
            "quality_failed": sum(r["quality"]["quality_pass"] is False for r in successful),
            "quality_pending": sum(r["quality"]["quality_pass"] is None for r in successful),
        }
        # Failed runs stay counted above; quantiles are explicitly conditional on success.
        for metric in ("caller_duration_ms", "wait_duration_ms", "model_calls", "queue_p95_ms"):
            values = [r["stats"][metric] for r in successful]
            item[metric] = (
                {"success_p50": percentile(values, 0.5), "success_p95": percentile(values, 0.95)}
                if values
                else None
            )
        values = [r["stats"]["caller_duration_ms"] for r in qualified]
        item["qualified_caller_duration_ms"] = (
            {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)} if values else None
        )
        result.append(item)
    return result


async def benchmark(*, factory, mode, repeats, segment_sizes, seed):
    if repeats < 1 or not segment_sizes or any(s < 1 for s in segment_sizes):
        raise ValueError("repeats and segment sizes must be positive")
    cases = corpus()
    jobs = [
        (repeat, size, case)
        for repeat in range(repeats)
        for size in segment_sizes
        for case in cases
    ]
    random.Random(seed).shuffle(jobs)
    rows, order = [], []
    for repeat, size, case in jobs:
        models = factory()
        policy = IndexPolicy(segment_max_messages=size)
        engine = IndexEngine(models=models)
        baseline = None
        blocked = False
        snapshots = [
            freeze(case, case["baseline"], 1),
            freeze(case, case["baseline"] + case["append"], 2),
        ]
        order.append({"repeat": repeat, "segment_size": size, "case": case["name"]})
        for phase, target in (
            ("cold", snapshots[0]),
            ("append", snapshots[1]),
            ("exact", snapshots[1]),
        ):
            row = {
                "case": case["name"],
                "repeat": repeat,
                "segment_size": size,
                "phase": phase,
                "snapshot_hash": target.fingerprint,
                "message_count": len(target.messages),
                "model_identities": {
                    role: models.identity(role) for role in ("projector", "verifier", "interpreter")
                },
                "policy": policy.model_dump(mode="json"),
                "concurrency": 4,
            }
            if blocked:
                row.update(
                    outcome="blocked",
                    error="prerequisite build failed",
                    stats={
                        **BuildStats(mode="full").model_dump(mode="json", exclude={"receipts"}),
                        "queue_p95_ms": 0,
                        "receipt_count": 0,
                        "caller_duration_ms": None,
                    },
                )
                rows.append(row)
                continue
            caller_started = perf_counter()
            try:
                result = await engine.build(
                    snapshot=target, policy=policy, baseline=baseline if phase == "append" else None
                )
                caller_duration_ms = (perf_counter() - caller_started) * 1000
                baseline = result.artifact
                row.update(
                    outcome="ready",
                    quality=quality_proxy(
                        result.index, case["joint_support_sets"] if phase != "cold" else []
                    ),
                )
                stats = result.stats
            except IndexBuildError as exc:
                caller_duration_ms = (perf_counter() - caller_started) * 1000
                blocked = True
                row.update(
                    outcome="failed", error_type=type(exc.cause).__name__, error=str(exc)[:500]
                )
                stats = exc.stats
            queues = [receipt.queue_ms for receipt in stats.receipts]
            row["stats"] = {
                **stats.model_dump(mode="json", exclude={"receipts"}),
                "queue_p95_ms": percentile(queues, 0.95) if queues else 0,
                "receipt_count": len(stats.receipts),
                "caller_duration_ms": caller_duration_ms,
            }
            rows.append(row)
    return {
        "schema": "context-index-benchmark-v2",
        "implementation_hash": stable_expansion_hash(
            "index-engine-source-v1",
            {
                p.name: sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(focus_context_index.__file__).parent.glob("*.py"))
            },
        ),
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "pydantic": pydantic.__version__,
            "platform": platform.platform(),
        },
        "mode": mode,
        "quality_note": "Citation/coverage proxies require human semantic review; scripted models cannot measure LLM quality or latency.",
        "seed": seed,
        "repeats": repeats,
        "segment_sizes": segment_sizes,
        "corpus_hash": stable_expansion_hash("context-index-benchmark-corpus-v1", cases),
        "corpus": cases,
        "execution_order": order,
        "runs": rows,
        "summary": summarize(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", help="module:factory returning a ModelAdapter; default is offline"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--segment-sizes", nargs="+", type=int, default=[12, 20, 30])
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/latest.json"))
    parser.add_argument(
        "--require-quality",
        action="store_true",
        help="exit nonzero unless every ready sample passes bound semantic review",
    )
    args = parser.parse_args()
    if args.adapter:
        name, attribute = args.adapter.split(":", 1)
        factory = getattr(importlib.import_module(name), attribute)
        mode = f"external-adapter:{args.adapter}"
    else:
        from examples.scripted_models import create_adapter

        factory, mode = create_adapter, "offline-scripted"
    report = asyncio.run(
        benchmark(
            factory=factory,
            mode=mode,
            repeats=args.repeats,
            segment_sizes=args.segment_sizes,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = sum(r["outcome"] != "ready" for r in report["runs"])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "mode": mode,
                "runs": len(report["runs"]),
                "failures": failures,
            }
        )
    )
    if failures:
        raise SystemExit(1)
    if args.require_quality and any(
        r.get("quality", {}).get("quality_pass") is not True for r in report["runs"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
