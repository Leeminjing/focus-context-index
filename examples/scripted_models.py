"""Offline demonstration only. Responses are scripted; this is not an LLM or a quality judge."""

import asyncio
import json

from focus_context_index import FocusModelAdapter


class ScriptedModel:
    context_window_tokens = 128_000
    max_output_tokens = 4096
    _max_attempts = 1

    def __init__(self, role):
        self.role = role
        self.cache_identity = f"offline-scripted-{role}-v1"
        self.last_attempt_records = ()

    async def invoke(self, schema, system, payload):
        await asyncio.sleep(0)
        self.last_attempt_records = (
            {
                "model_calls": 1,
                # No provider exists in this demo; token usage stays unknown.
                "input_tokens": None,
                "output_tokens": None,
            },
        )
        if schema.__name__ == "SemanticProjectionProposal":
            units = []
            for segment in payload["segments"]:
                for message in segment["messages"]:
                    content = message["content"]
                    text = (
                        content
                        if isinstance(content, str)
                        else json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
                    )
                    if text and message["semantic_policy"] == "index":
                        units.append(
                            {
                                "kind": "claim",
                                "authority": "confirmed",
                                "statement": text[:4000],
                                "supports": [
                                    {
                                        "message_id": message["message_id"],
                                        "quote": text[:8000],
                                    }
                                ],
                            }
                        )
            return schema(units=units)
        if schema.__name__ == "SemanticClaimSupportProposal":
            return schema(
                assessments=[
                    {
                        "claim_key": c["claim_key"],
                        "verdict": "supported",
                        "reason": "Scripted protocol demonstration; not a semantic evaluation.",
                    }
                    for c in payload["claims"]
                ]
            )
        return schema(action="complete", units=())


def create_adapter():
    return FocusModelAdapter(
        projector_factory=lambda: ScriptedModel("projector"),
        verifier_factory=lambda: ScriptedModel("verifier"),
        interpreter_factory=lambda: ScriptedModel("interpreter"),
    )
