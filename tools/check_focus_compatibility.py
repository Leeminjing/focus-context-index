"""Execute original Focus core modules in isolation, without starting its app or database.

Usage: python tools/check_focus_compatibility.py --focus-root /path/to/Focus
Optional --artifact index-artifact.json validates an engine result in the original
Focus models. --write-golden tests/fixtures/focus-v9.json creates an upstream fixture.
Only imports and application-wide dependency initialization are isolated; the tested
index, segment, grounding and proof implementations are the original source files.
"""

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

COMMIT = "08fe98c28896992c6195912e95b507599b9dcd76"
PREFIX = "backend.app.desktop.patrol.agent_loop.context_expansion"


def module(name):
    if name in sys.modules:
        return sys.modules[name]
    if "." in name:
        module(name.rsplit(".", 1)[0])
    result = types.ModuleType(name)
    result.__path__ = []
    sys.modules[name] = result
    return result


def excerpt(target, path, symbols, prelude=""):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    body = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            name = next((n.id for n in targets if isinstance(n, ast.Name)), None)
        if name in symbols:
            body.append(node)
    namespace = module(target).__dict__
    exec(prelude, namespace)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def load(root):
    base = root / "backend/app/desktop/patrol/agent_loop/context_expansion"
    evolution = "backend.app.desktop.contexts.evolution"
    prelude = (
        "from __future__ import annotations\nimport json\nfrom hashlib import sha256\n"
        "from enum import StrEnum\nfrom typing import Any, Literal, Self\n"
        "from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator\n"
    )
    excerpt(
        evolution,
        root / "backend/app/desktop/contexts/evolution/schemas.py",
        {"ContextRevisionPayloadMode", "_FrozenContract", "ContextRevisionRef"},
        prelude,
    )
    excerpt(
        "backend.app.desktop.contexts.curation",
        root / "backend/app/desktop/contexts/curation/contracts.py",
        {
            "_StrictModel",
            "NamespacedMessageRef",
            "MissionEvidenceRef",
            "RunResultEvidenceRef",
            "MaterialEvidenceRef",
            "WorkspaceEffectEvidenceRef",
            "EvidenceRef",
            "evidence_ref_key",
        },
        prelude + f"from {evolution} import ContextRevisionRef\n",
    )
    excerpt(
        PREFIX + ".contracts",
        base / "contracts.py",
        {
            "_ExpansionModel",
            "stable_expansion_hash",
            "_normalized",
            "SemanticAuthority",
            "SemanticUnitKind",
            "SemanticEvidenceUnit",
        },
        prelude
        + "from backend.app.desktop.contexts.curation import EvidenceRef, evidence_ref_key\n",
    )
    excerpt(
        "backend.app.desktop.patrol.agent_loop.workers.derivation_worker",
        root / "backend/app/desktop/patrol/agent_loop/workers/derivation_worker.py",
        {"StructuredResultValidationError"},
        prelude,
    )
    excerpt(
        "focus.messages.usage",
        root / "backend/packages/harness/focus/messages/usage.py",
        {"estimate_raw_tokens", "_TOKENS_PER_MESSAGE"},
    )
    for name in (
        "index_build_contracts",
        "semantic_grounding",
        "interpretation_record",
        "semantic_index",
        "semantic_indexer",
        "segment_projection",
        "index_assembly",
    ):
        fullname = PREFIX + "." + name
        spec = importlib.util.spec_from_file_location(fullname, base / f"{name}.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = loaded
        spec.loader.exec_module(loaded)
    return (
        module(evolution),
        module(PREFIX + ".semantic_indexer"),
        module(PREFIX + ".semantic_index"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--write-golden", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.focus_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != COMMIT:
        raise SystemExit(f"expected upstream {COMMIT}, got {revision}")
    evolution, indexing, indexes = load(args.focus_root)
    source = evolution.ContextRevisionRef(
        context_id="golden-context",
        revision_id="golden-revision",
        generation=1,
        execution_thread_id="thread",
        checkpoint_id="checkpoint",
        payload_mode="checkpoint",
    )
    raw = [
        {"id": "goal", "role": "human", "content": "验证分组和原始哈希。"},
        {"id": "call", "role": "ai", "tool_calls": [{"id": "a", "name": "test", "args": {}}]},
        {"id": "return", "role": "tool", "tool_call_id": "a", "content": {"passed": True}},
        {
            "id": "reference",
            "role": "human",
            "content": "仅作参考",
            "semantic_policy": "reference_only",
        },
    ]
    indexer = indexing.RevisionSemanticIndexer(
        indexing.ProtocolSafeRevisionSegmenter(max_messages=2)
    )
    index = indexer.index(
        source=source,
        source_content_hash="a" * 64,
        context_role="primary",
        active_objective="parity",
        raw_messages=tuple(raw),
    )
    # Compare the extracted engine's deterministic output with original Focus.
    from focus_context_index.contracts import ContextRevisionRef
    from focus_context_index.semantic_indexer import (
        ProtocolSafeRevisionSegmenter,
        RevisionSemanticIndexer,
    )

    migrated = RevisionSemanticIndexer(ProtocolSafeRevisionSegmenter(max_messages=2)).index(
        source=ContextRevisionRef.model_validate(source.model_dump(mode="json")),
        source_content_hash="a" * 64,
        context_role="primary",
        active_objective="parity",
        raw_messages=tuple(raw),
    )
    assert migrated.model_dump(mode="json") == index.model_dump(mode="json")
    if args.write_golden:
        args.write_golden.parent.mkdir(parents=True, exist_ok=True)
        args.write_golden.write_text(
            json.dumps(
                {
                    "upstream_commit": revision,
                    "raw_messages": raw,
                    "index": index.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.artifact:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        original = indexes.RevisionSemanticIndex.model_validate(payload["index"])
        record_type = module(PREFIX + ".segment_projection").SegmentProjectionRecord
        records = tuple(record_type.model_validate(r) for r in payload["records"])
        original.interpretation.validate_target(original, records)
        assert original.index_id == payload["index"]["index_id"]
    print(
        json.dumps(
            {
                "upstream_commit": revision,
                "deterministic_payload_parity": True,
                "complete_artifact_accepted_by_focus": bool(args.artifact),
            }
        )
    )


if __name__ == "__main__":
    main()
