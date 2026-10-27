"""Run offline: python -m examples.focus_integration [--store index.sqlite] [--artifact file.json].

make_focus_models(app_config) is the actual Focus factory integration. Only that
function imports Focus; the offline main does not need Focus installed.
"""

import argparse
import asyncio
import json
from pathlib import Path

from focus_context_index import (
    ContextRevisionRef,
    FocusModelAdapter,
    IndexEngine,
    SQLiteIndexStore,
    snapshot_from_focus,
)
from focus_context_index.contracts import stable_expansion_hash

from .scripted_models import create_adapter


def make_focus_models(app_config):
    from backend.app.desktop.patrol.agent_loop.workers.derivation_worker import (
        RoleBoundStructuredModel,
    )
    from backend.app.desktop.patrol.agent_loop.workers.structured_worker import (
        StructuredWorkerModel,
    )

    def factory(role):
        model = RoleBoundStructuredModel(app_config, role, max_attempts=1)
        # Reuse Focus's actual serialized request estimator, including provider schema
        # and role-specific output settings, rather than the default text heuristic.
        estimator = StructuredWorkerModel(app_config, max_output_tokens=model.max_output_tokens)
        model.estimate_request_tokens = estimator.estimate_input_tokens
        return model

    return FocusModelAdapter(
        projector_factory=lambda: factory("semantic_index_projector"),
        verifier_factory=lambda: factory("claim_verifier"),
        interpreter_factory=lambda: factory("semantic_index_interpreter"),
    )


def frozen(raw, generation):
    return snapshot_from_focus(
        namespace="focus-demo",
        revision_ref=ContextRevisionRef(
            context_id="context-1",
            revision_id=f"revision-{generation}",
            generation=generation,
            execution_thread_id="execution-1",
            checkpoint_id=f"checkpoint-{generation}",
            payload_mode="checkpoint",
        ),
        content_hash=stable_expansion_hash("demo-host-content", raw),
        semantic_messages=raw,
        active_objective="Verify independent indexing",
    )


async def run(store=None, artifact_path=None):
    engine = IndexEngine(models=create_adapter(), store=SQLiteIndexStore(store) if store else None)
    history = [
        {"id": f"m-{i}", "role": "human", "content": f"Observed message {i}."} for i in range(55)
    ]
    cold = await engine.build(snapshot=frozen(history, 1))
    history.extend(
        [
            {"id": "m-55", "role": "human", "content": "The early hypothesis was ruled out."},
            {"id": "m-56", "role": "human", "content": "A later diagnosis was confirmed."},
        ]
    )
    target = frozen(history, 2)
    appended = await engine.build(snapshot=target, baseline=cold.artifact)
    exact = await engine.build(snapshot=target)
    if artifact_path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(appended.artifact.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": "offline scripted demonstration; no semantic quality or provider timing claim",
                "cold": cold.stats.model_dump(mode="json", exclude={"receipts"}),
                "append": appended.stats.model_dump(mode="json", exclude={"receipts"}),
                "exact": exact.stats.model_dump(mode="json", exclude={"receipts"}),
                "focus_index_payload": "result.index.model_dump(mode='json')",
            },
            indent=2,
        )
    )
    return appended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    asyncio.run(run(args.store, args.artifact))
