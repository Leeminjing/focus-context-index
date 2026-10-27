import asyncio
import json
from copy import deepcopy

import pytest

from focus_context_index import ContextRevisionRef, FrozenSnapshot


def messages(count):
    return [
        {"id": f"m-{i}", "role": "human", "content": f"Observed step {i}."} for i in range(count)
    ]


def snapshot(raw=None, *, generation=1, context="ctx", namespace="test", content_hash="a" * 64):
    return FrozenSnapshot.from_messages(
        namespace=namespace,
        source=ContextRevisionRef(
            context_id=context,
            revision_id=f"{context}-r{generation}",
            generation=generation,
            execution_thread_id=f"{context}-thread",
            checkpoint_id=f"cp-{generation}",
            payload_mode="checkpoint",
        ),
        source_content_hash=content_hash,
        raw_messages=messages(5) if raw is None else raw,
    )


class Probe:
    """A deterministic wire-protocol double, not a semantic quality evaluator."""

    def __init__(self, *, gate=None, reply=None, verdict="supported", identities=None):
        self.calls = []
        self.gate = gate
        self.reply = reply
        self.verdict = verdict
        self.identities = identities or {}
        self.active = self.max_active = 0

    def identity(self, role):
        return self.identities.get(role, f"scripted-{role}-v1")

    def create(self, role):
        probe = self

        class Model:
            cache_identity = probe.identity(role)
            context_window_tokens = 1_000_000
            last_attempt_records = ()

            async def invoke(self, schema, system, payload):
                probe.calls.append((role, deepcopy(payload)))
                probe.active += 1
                probe.max_active = max(probe.max_active, probe.active)
                try:
                    if probe.gate:
                        await probe.gate(role, payload)
                    else:
                        await asyncio.sleep(0)
                    result = probe.reply(role, schema, payload) if probe.reply else None
                    if result is not None:
                        return result
                    if schema.__name__ == "SemanticProjectionProposal":
                        return schema(
                            units=[
                                {
                                    "kind": "claim",
                                    "authority": "confirmed",
                                    "statement": str(m["content"]),
                                    "supports": [
                                        {
                                            "message_id": m["message_id"],
                                            "quote": _text(m["content"]),
                                        }
                                    ],
                                }
                                for s in payload["segments"]
                                for m in s["messages"]
                                if m["content"] and m["semantic_policy"] == "index"
                            ]
                        )
                    if schema.__name__ == "SemanticClaimSupportProposal":
                        return schema(
                            assessments=[
                                {
                                    "claim_key": claim["claim_key"],
                                    "verdict": probe.verdict,
                                    "reason": "scripted verdict for a protocol test",
                                }
                                for claim in payload["claims"]
                            ]
                        )
                    return schema(action="complete", units=())
                finally:
                    self.last_attempt_records = (
                        {
                            "model_calls": 1,
                            "input_tokens": 100,
                            "output_tokens": 10,
                        },
                    )
                    probe.active -= 1

        return Model()


def _text(content):
    return (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    )


@pytest.fixture
def probe():
    return Probe()
