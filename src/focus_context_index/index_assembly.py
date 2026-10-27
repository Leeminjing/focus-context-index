# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 assemble_revision_index。

输入为冻结 index、局部 records、BuildPlan 与综合 proof；输出为绑定目标 Revision 的完整不可变 index。
具体工作流为重验两类依赖，合并与去重 drafts／verdicts，统一 grounding 并重建覆盖、隔离账本、继承凭据及综合身份。
示例：index = assemble_revision_index(indexer, frozen, records, plan, interpretation=proof)；联合诊断与历史局部证据共存，旧对象不变。
"""

from .index_build_contracts import IndexInheritanceReceipt
from .semantic_index import RevisionSemanticIndex


def assemble_revision_index(
    indexer, frozen, records, plan, *, interpretation=None, local_contract=None
):
    if len(records) != len(frozen.segments):
        raise ValueError("index assembly record inventory incomplete")
    drafts, assessments, rejections = [], [], []
    messages_by_id = {message.message_id: message for message in frozen.messages}
    for segment, record in zip(frozen.segments, records):
        messages = tuple(messages_by_id[identity] for identity in segment.message_ids)
        if record.context_id != frozen.source.context_id:
            raise ValueError("index assembly record context scope mismatch")
        record.validate_target(segment, messages, local_contract or frozen.projector_version)
        accepted = set(record.accepted_claim_keys)
        drafts.extend(d for d in record.drafts if d.claim_key in accepted)
        assessments.extend(record.assessments)
        rejections.extend(record.rejections)
    if interpretation is not None:
        interpretation.validate_target(frozen, records)
        accepted = set(interpretation.accepted_claim_keys)
        drafts.extend(d for d in interpretation.drafts if d.claim_key in accepted)
        assessments.extend(interpretation.assessments)
        rejections.extend(interpretation.rejections)
    drafts = tuple({d.claim_key: d for d in drafts}.values())
    assessments = tuple({a.claim_key: a for a in assessments}.values())
    rejections = tuple({r.rejection_id: r for r in rejections}.values())
    receipt = IndexInheritanceReceipt.model_validate(
        {**plan.model_dump(mode="json"), "record_ids": [r.record_id for r in records]}
    )
    projected = indexer.index(
        source=frozen.source,
        source_content_hash=frozen.source_content_hash,
        context_role=frozen.context_role,
        active_objective=frozen.active_objective,
        raw_messages=tuple(m.model_dump(mode="json") for m in frozen.messages),
        unit_drafts=tuple(drafts),
        claim_support_assessments=tuple(assessments),
        projector_version=frozen.projector_version,
    )
    excluded = {
        "index_id",
        "coverage",
        "inheritance",
        "interpretation",
        "rejected_units",
        "quality_state",
    }
    values = {
        name: getattr(projected, name)
        for name in type(projected).model_fields
        if name not in excluded
    }
    return RevisionSemanticIndex.create(
        **values,
        inheritance=receipt,
        interpretation=interpretation,
        rejected_units=tuple(rejections),
        quality_state="degraded" if rejections or projected.fallback_segment_ids else "complete",
        protocol_exchange_count=projected.coverage.protocol_exchange_count,
    )
