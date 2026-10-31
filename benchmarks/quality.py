"""Execution completion, mechanical quality proxies and semantic review are distinct."""

from focus_context_index.contracts import stable_expansion_hash


def decide_quality(quality, review=None):
    proxy_pass = (
        quality["quality_state"] == "complete"
        and quality["eligible_message_citation_coverage"] == 1
        and quality["fallback_segments"] == 0
        and quality["quarantined_claims"] == 0
        and quality["found_joint_support_sets"] == quality["expected_joint_support_sets"]
    )
    quality["quality_proxy_pass"] = proxy_pass
    quality["human_semantic_review"] = "pending"
    quality["quality_pass"] = False if not proxy_pass else None
    if review is not None:
        if (
            (review.get("index_id"), review.get("criteria_hash"))
            != (quality["index_id"], quality["criteria_hash"])
            or not review.get("reviewer")
            or review.get("verdict") not in {"approved", "rejected"}
        ):
            raise ValueError(
                "semantic review must bind index_id, criteria_hash, reviewer and verdict"
            )
        quality["human_semantic_review"] = review["verdict"]
        quality["reviewer"] = review["reviewer"]
        quality["quality_pass"] = proxy_pass and review["verdict"] == "approved"
    return quality


def quality_proxy(index, expected_joint_sets, review=None):
    eligible = {m.message_id for m in index.messages if m.semantic_policy == "index"}
    projected = set(index.projected_unit_ids)
    cited = {
        r.message_id
        for u in index.semantic_units
        if u.unit_id in projected
        for r in u.evidence_refs
    }
    proof = index.interpretation
    accepted = set(proof.accepted_claim_keys)
    support_sets = [
        set(s.message_id for s in d.supports) for d in proof.drafts if d.claim_key in accepted
    ]
    quality = {
        "index_id": index.index_id,
        "criteria_hash": stable_expansion_hash(
            "semantic-quality-gate-v1", expected_joint_sets, 1.0
        ),
        "quality_state": index.quality_state,
        "eligible_message_citation_coverage": len(cited & eligible) / len(eligible)
        if eligible
        else 1.0,
        "fallback_segments": len(index.fallback_segment_ids),
        "quarantined_claims": len(index.rejected_units),
        "expected_joint_support_sets": len(expected_joint_sets),
        "found_joint_support_sets": sum(
            any(set(e) <= s for s in support_sets) for e in expected_joint_sets
        ),
        "review_material": {
            "semantic_units": [u.model_dump(mode="json") for u in index.semantic_units],
            "interpretation_drafts": [d.model_dump(mode="json") for d in proof.drafts],
            "rejections": [r.model_dump(mode="json") for r in index.rejected_units],
        },
    }
    return decide_quality(quality, review)
