import asyncio

import pytest

from focus_context_index import IndexBuildError, IndexEngine, IndexPolicy
from focus_context_index.contracts import stable_expansion_hash
from focus_context_index.interpretation_record import RevisionInterpretationRecord
from conftest import Probe, snapshot

NARRATIVE = [
    {"id": "suspect", "role": "human", "content": "Initially suspected the connection pool."},
    {"id": "unrelated", "role": "human", "content": "An unrelated deployment was recorded."},
    {"id": "reject", "role": "human", "content": "The connection pool was ruled out."},
    {"id": "diagnosis", "role": "human", "content": "The thread-pool deadlock was confirmed."},
]
JOINT = "The connection-pool suspicion was ruled out; the cause was thread-pool deadlock."


def joint_draft(messages=NARRATIVE):
    return {
        "kind": "claim",
        "authority": "confirmed",
        "statement": JOINT,
        "supports": [
            {"message_id": m["id"], "quote": m["content"]}
            for m in messages
            if m["id"] != "unrelated"
        ],
    }


def test_append_reinterprets_correction_with_joint_original_evidence():
    def reply(role, schema, payload):
        if role == "interpreter":
            ids = {m for s in payload["inventory"] for m in s["message_ids"]}
            if "diagnosis" in ids:
                return schema(action="complete", units=(joint_draft(),))
        return None

    async def run():
        probe = Probe(reply=reply)
        engine = IndexEngine(models=probe)
        policy = IndexPolicy(segment_max_messages=1)
        old = await engine.build(snapshot=snapshot(NARRATIVE[:2]), policy=policy)
        result = await engine.build(
            snapshot=snapshot(NARRATIVE, generation=2), baseline=old.artifact, policy=policy
        )
        assert result.stats.local_cache_hits == 2
        assert any(u.statement == JOINT for u in result.index.semantic_units)
        joint = next(u for u in result.index.semantic_units if u.statement == JOINT)
        assert {r.message_id for r in joint.evidence_refs} == {"suspect", "reject", "diagnosis"}
        assert all(ref.source == result.index.source for ref in joint.evidence_refs)
        assert any(
            role == "verifier" and any(c["statement"] == JOINT for c in p["claims"])
            for role, p in probe.calls
        )

    asyncio.run(run())


@pytest.mark.parametrize("verdict", ["unknown", "unsupported"])
def test_exact_quotes_alone_cannot_confirm_claims(verdict):
    async def run():
        result = await IndexEngine(models=Probe(verdict=verdict)).build(
            snapshot=snapshot(NARRATIVE)
        )
        assert result.index.quality_state == "degraded"
        assert not any(u.authority == "confirmed" for u in result.index.semantic_units)
        assert result.index.rejected_units
        assert result.index.fallback_segment_ids

    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["unknown_message", "changed_quote", "reference_decision"])
def test_invalid_local_claims_are_quarantined_with_a_fallback(invalid):
    def reply(role, schema, payload):
        if role == "projector":
            draft = joint_draft(NARRATIVE[:1])
            if invalid == "unknown_message":
                draft["supports"][0]["message_id"] = "foreign"
            elif invalid == "changed_quote":
                draft["supports"][0]["quote"] = "not an original quote"
            else:
                draft["kind"] = "decision"
            return schema(units=(draft,))
        return None

    async def run():
        raw = [dict(NARRATIVE[0])]
        if invalid == "reference_decision":
            raw[0]["semantic_policy"] = "reference_only"
        result = await IndexEngine(models=Probe(reply=reply)).build(snapshot=snapshot(raw))
        assert result.index.quality_state == "degraded"
        assert result.index.rejected_units
        assert not result.index.projected_unit_ids
        if invalid == "reference_decision":
            assert not result.index.semantic_units  # reference cannot create a fallback task

    asyncio.run(run())


def test_original_windows_replace_payload_but_accumulate_read_dependencies():
    rounds = 0

    def reply(role, schema, payload):
        nonlocal rounds
        if role != "interpreter":
            return None
        rounds += 1
        inventory = payload["inventory"]
        if rounds <= 2:
            return schema(action="read", read_segments=(inventory[(rounds - 1) * 2]["segment_id"],))
        return schema(action="complete", units=())

    class WindowProbe(Probe):
        def create(self, role):
            model = super().create(role)
            if role == "interpreter":
                model.context_window_tokens = 1500
                model.estimate_request_tokens = lambda schema, system, payload: (
                    100 + len(payload["segments"]) * 1000
                )
            return model

    async def run():
        probe = WindowProbe(reply=reply)
        result = await IndexEngine(models=probe).build(
            snapshot=snapshot(NARRATIVE[:3]),
            policy=IndexPolicy(segment_max_messages=1, output_token_reserve=100),
        )
        windows = [p for role, p in probe.calls if role == "interpreter"]
        assert [len(p["segments"]) for p in windows] == [0, 1, 1]
        assert all(len(p["read_requests"]) <= 1 for p in windows)
        proof = result.index.interpretation
        assert len(proof.provided) == len(proof.read_requests) == 2
        # Tampering with a read message that was never quoted still invalidates proof.
        payload = proof.model_dump(mode="json")
        payload["provided"][0]["messages"][0]["content"] = "changed without citation"
        payload["record_id"] = stable_expansion_hash(
            proof.SCHEMA_VERSION, {k: v for k, v in payload.items() if k != "record_id"}
        )
        tampered = RevisionInterpretationRecord.model_validate(payload)
        with pytest.raises(ValueError, match="original content integrity"):
            tampered.validate_target(result.index, result.artifact.records)

    asyncio.run(run())


def test_unread_message_cannot_support_a_joint_claim():
    def reply(role, schema, payload):
        if role == "interpreter":
            return schema(action="complete", units=(joint_draft(),))
        return None

    async def run():
        result = await IndexEngine(models=Probe(reply=reply)).build(
            snapshot=snapshot(NARRATIVE),
            policy=IndexPolicy(segment_max_messages=1, max_exact_reads=0),
        )
        assert not result.index.interpretation.provided
        assert not result.index.interpretation.accepted_claim_keys
        assert result.index.interpretation.rejections[0].code == "unknown_message_ref"
        assert not any(u.statement == JOINT for u in result.index.semantic_units)

    asyncio.run(run())


def test_invalid_read_scope_is_correctable_without_mutating_read_dependencies():
    rounds = 0

    def reply(role, schema, payload):
        nonlocal rounds
        if role == "interpreter":
            rounds += 1
            if rounds == 1:
                return schema(action="read", read_segments=("foreign-segment",))
            assert "validation_feedback" in payload
            return schema(action="complete", units=())
        return None

    async def run():
        result = await IndexEngine(models=Probe(reply=reply)).build(snapshot=snapshot(NARRATIVE))
        assert not result.index.interpretation.read_requests
        assert all(s.segment_id != "foreign-segment" for s in result.index.interpretation.provided)
        assert result.stats.model_calls == 4
        assert len([r for r in result.stats.receipts if r.outcome == "error"]) == 1

    asyncio.run(run())


def test_failed_or_unfinished_interpretation_never_publishes_partial_artifact():
    def reply(role, schema, payload):
        if role == "interpreter":
            return schema(action="read", read_segments=(payload["inventory"][0]["segment_id"],))
        return None

    async def run():
        engine = IndexEngine(models=Probe(reply=reply))
        with pytest.raises(IndexBuildError, match="discovery rounds exhausted") as failure:
            await engine.build(
                snapshot=snapshot(NARRATIVE), policy=IndexPolicy(max_planner_model_calls=2)
            )
        assert not engine.store._artifacts
        assert engine.store._segments
        assert failure.value.stats.model_calls == 4

    asyncio.run(run())


@pytest.mark.parametrize(
    "raw",
    [
        [{"id": "call", "role": "ai", "tool_calls": [{"id": "a"}]}],
        [{"id": "result", "role": "tool", "tool_call_id": "a"}],
        [
            {"id": "call", "role": "ai", "tool_calls": [{"id": "a"}, {"id": "a"}]},
            {"id": "result", "role": "tool", "tool_call_id": "a"},
        ],
    ],
)
def test_invalid_tool_protocol_is_rejected_before_any_model_call(raw):
    async def run():
        probe = Probe()
        with pytest.raises(IndexBuildError):
            await IndexEngine(models=probe).build(snapshot=snapshot(raw))
        assert not probe.calls

    asyncio.run(run())
