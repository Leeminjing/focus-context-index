import json
from pathlib import Path

import pytest

from focus_context_index import ContextRevisionRef, FocusModelAdapter, snapshot_from_focus
from focus_context_index.contracts import stable_expansion_hash
from focus_context_index.segment_projection import SegmentProjectionRecord
from focus_context_index.semantic_grounding import (
    SegmentSemanticUnitDraft,
    SemanticClaimSupportAssessment,
)
from focus_context_index.semantic_index import RevisionSemanticIndex
from focus_context_index.semantic_indexer import (
    ProtocolSafeRevisionSegmenter,
    RevisionSemanticIndexer,
)


def test_original_focus_golden_payload_and_hash_are_preserved():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/focus-v9.json").read_text(encoding="utf-8")
    )
    old = RevisionSemanticIndex.model_validate(golden["index"])
    extracted = RevisionSemanticIndexer(ProtocolSafeRevisionSegmenter(max_messages=2)).index(
        source=old.source,
        source_content_hash=old.source_content_hash,
        context_role=old.context_role,
        active_objective=old.active_objective,
        raw_messages=tuple(golden["raw_messages"]),
    )
    assert extracted.model_dump(mode="json") == golden["index"]
    assert extracted.index_id == old.index_id


def test_legacy_projection_payload_keeps_its_original_hash_and_serialization():
    messages = RevisionSemanticIndexer._messages(
        ({"id": "m", "role": "human", "content": "Build passed."},)
    )
    segment = ProtocolSafeRevisionSegmenter().segment(messages)[0]
    draft = SegmentSemanticUnitDraft(
        kind="claim",
        authority="confirmed",
        statement="Build passed.",
        supports=({"message_id": "m", "quote": "Build passed."},),
    )
    record = SegmentProjectionRecord.create(
        context_id="ctx",
        contract="legacy",
        segment=segment,
        messages=messages,
        drafts=(draft,),
        assessments=(
            SemanticClaimSupportAssessment(
                claim_key=draft.claim_key, verdict="supported", reason="yes"
            ),
        ),
    )
    payload = record.model_dump(mode="json")
    payload.pop("grounding_version")
    payload["record_id"] = stable_expansion_hash(
        "segment-projection-record-v1", {k: v for k, v in payload.items() if k != "record_id"}
    )
    legacy = SegmentProjectionRecord.model_validate(payload)
    assert legacy.grounding_version == "support-span-grounding-v1"
    assert legacy.model_dump(mode="json") == payload


def test_focus_adapter_preserves_host_source_positions_and_eligibility():
    source = ContextRevisionRef(
        context_id="ctx",
        revision_id="rev",
        generation=1,
        execution_thread_id="thread",
        checkpoint_id="cp",
        payload_mode="checkpoint",
    )
    raw = [
        {
            "id": "visible",
            "role": "human",
            "content": "hello",
            "source_ordinal": 7,
            "semantic_policy": "reference_only",
        }
    ]
    snapshot = snapshot_from_focus(
        namespace="focus", revision_ref=source, content_hash="a" * 64, semantic_messages=raw
    )
    assert snapshot.source == source
    assert snapshot.messages[0].source_ordinal == 7
    assert snapshot.messages[0].semantic_policy == "reference_only"


def test_focus_factories_require_single_attempt_and_explicit_identity():
    class Invalid:
        _max_attempts = 2
        cache_identity = "invalid"

    adapter = FocusModelAdapter(projector_factory=Invalid, verifier_factory=Invalid)
    with pytest.raises(ValueError, match="max_attempts=1"):
        adapter.identity("projector")

    class MissingIdentity:
        pass

    adapter = FocusModelAdapter(projector_factory=MissingIdentity, verifier_factory=MissingIdentity)
    with pytest.raises(ValueError, match="cache_identity"):
        adapter.identity("projector")
