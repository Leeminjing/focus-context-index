# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
"""Size segmentation by message count and serialized tokens without splitting tools."""

import pytest

from focus_context_index.contracts import stable_expansion_hash
from focus_context_index.semantic_indexer import (
    ProtocolSafeRevisionSegmenter,
    RevisionSemanticIndexer,
)


def split(raw, **options):
    messages = RevisionSemanticIndexer._messages(tuple(raw))
    options.setdefault("max_estimated_tokens", 4096)
    return ProtocolSafeRevisionSegmenter(**options).segment(messages)


def test_long_text_splits_while_short_messages_share_a_segment():
    raw = [
        {"id": "before", "role": "human", "content": "short"},
        {"id": "long", "role": "human", "content": "长" * 5000},
        {"id": "after", "role": "human", "content": "short"},
    ]
    assert [s.message_ids for s in split(raw)] == [("before",), ("long",), ("after",)]
    raw[1]["content"] = "short"
    assert [s.message_ids for s in split(raw)] == [("before", "long", "after")]


@pytest.mark.parametrize("large_field", ["arguments", "result"])
def test_oversize_parallel_tool_exchange_is_atomic_and_isolated(large_field):
    raw = [
        {"id": "before", "role": "human", "content": "short"},
        {
            "id": "call",
            "role": "ai",
            "tool_calls": [
                {
                    "id": "a",
                    "name": "one",
                    "args": {"text": "x" * 20000 if large_field == "arguments" else ""},
                },
                {"id": "b", "name": "two", "args": {}},
            ],
        },
        {
            "id": "ra",
            "role": "tool",
            "tool_call_id": "a",
            "content": "x" * 20000 if large_field == "result" else "ok",
        },
        {"id": "rb", "role": "tool", "tool_call_id": "b", "content": "ok"},
        {"id": "after", "role": "human", "content": "short"},
    ]
    assert [s.message_ids for s in split(raw, max_messages=2)] == [
        ("before",),
        ("call", "ra", "rb"),
        ("after",),
    ]
    # With a generous message count it is still isolated by serialized size.
    assert [s.message_ids for s in split(raw)] == [
        ("before",),
        ("call", "ra", "rb"),
        ("after",),
    ]


def test_count_and_token_limits_both_apply_and_contracts_are_separate():
    raw = [{"id": str(i), "role": "human", "content": "short"} for i in range(5)]
    assert [len(s.message_ids) for s in split(raw, max_messages=2)] == [2, 2, 1]
    legacy = ProtocolSafeRevisionSegmenter(max_estimated_tokens=None)
    assert (
        legacy.fingerprint == "seg:" + stable_expansion_hash("protocol-safe-segmenter-v2", 12)[:60]
    )
    assert (
        len(
            {
                legacy.fingerprint,
                ProtocolSafeRevisionSegmenter(max_estimated_tokens=4096).fingerprint,
                ProtocolSafeRevisionSegmenter(max_estimated_tokens=2000).fingerprint,
                ProtocolSafeRevisionSegmenter(max_messages=20).fingerprint,
            }
        )
        == 4
    )


def test_combined_size_splits_short_messages_and_preserves_append_prefix():
    raw = [{"id": str(i), "role": "human", "content": "证据" * 80} for i in range(8)]
    messages = RevisionSemanticIndexer._messages(tuple(raw))
    size = ProtocolSafeRevisionSegmenter.estimated_tokens(messages[:2])
    segmenter = ProtocolSafeRevisionSegmenter(max_estimated_tokens=size)
    old = segmenter.segment(messages[:6])
    newer = segmenter.segment(messages)
    assert [len(s.message_ids) for s in old] == [2, 2, 2]
    assert newer[: len(old)] == old
    # Structured tool content is counted even without a plain-text body.
    structured = [
        {"id": "structured", "role": "human", "content": [{"type": "text", "text": "长" * 5000}]}
    ]
    assert [s.message_ids for s in split([raw[0], *structured, raw[1]])] == [
        ("0",),
        ("structured",),
        ("1",),
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_token_limit_must_be_positive(limit):
    with pytest.raises(ValueError):
        ProtocolSafeRevisionSegmenter(max_estimated_tokens=limit)
