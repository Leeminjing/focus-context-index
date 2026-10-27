# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 SegmentProjectionRecord、SegmentProjectionBuilder 与投影合同 fingerprint。

输入为单个协议闭合 segment、其规范消息及受监督模型；输出为保留 drafts、精确引文、verdict 和隔离结果的不可变记录。
具体工作流为仅向模型提交该段，确定性检查局部 supports 与宿主语义资格，独立验证 confirmed claims，保存原始证据与稳定身份。
新记录保存 grounding_version；旧记录缺失该字段时使用原处置算法与序列化，不改写旧 hash。
SegmentProjectionRecord.cache_key_for 为纯局部合同/段身份函数，创建、完整性验证和缓存读取共用原版本哈希算法。
复用输入为相同 segment；输出为经重新校验的记录，不重新调用模型。示例：record = await builder.build(segment, messages)。
"""

from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator

from .contracts import stable_expansion_hash
from .semantic_grounding import (
    SegmentSemanticUnitDraft,
    SemanticClaimSupportAssessment,
    SemanticGroundingError,
    SemanticGroundingValidator,
    SemanticProjectionProposal,
    SemanticUnitRejection,
    SupervisedSemanticClaimSupportVerifier,
)
from .semantic_index import IndexedMessage, RevisionSegment
from .semantic_indexer import (
    ProtocolSafeRevisionSegmenter,
    RevisionSemanticIndexer,
    SupervisedSegmentSemanticProjector,
)

SEGMENT_PROMPT = (
    "你是无权 semantic_index_projector。只从这个冻结 segment 的原文抽取原子 semantic units。"
    "每个 unit 输出 statement 和 supports；message_id 必须来自本段，quote 必须逐字复制原文。"
    "confirmed statement 必须被给定引文直接支持。不得引用其他段、生成 WorkSpec、查询外部历史或改变状态。"
    "semantic_policy=index 才能提出任务命题；evidence_only 只提供关联证据，reference_only 只提供约束参考，不将两者独立改写为任务要求。"
)


class SegmentProjectionRecord(BaseModel):
    SCHEMA_VERSION: ClassVar[str] = "segment-projection-record-v1"
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    context_id: str
    cache_key: str
    contract_fingerprint: str
    segment: RevisionSegment
    messages: tuple[IndexedMessage, ...]
    drafts: tuple[SegmentSemanticUnitDraft, ...] = ()
    assessments: tuple[SemanticClaimSupportAssessment, ...] = ()
    accepted_claim_keys: tuple[str, ...] = ()
    rejections: tuple[SemanticUnitRejection, ...] = ()
    fallback: bool = False
    model_metadata: tuple[dict[str, Any], ...] = ()
    grounding_version: Literal["support-span-grounding-v1", "support-span-grounding-v2"] = (
        "support-span-grounding-v1"
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        payload = handler(self)
        if self.grounding_version == "support-span-grounding-v1":
            payload.pop("grounding_version", None)
        return payload

    @model_validator(mode="after")
    def require_integrity(self) -> Self:
        expected = self._payload()
        if self.record_id != stable_expansion_hash("segment-projection-record-v1", expected):
            raise ValueError("segment projection record integrity mismatch")
        if self.cache_key != self.cache_key_for(self.segment, self.contract_fingerprint):
            raise ValueError("segment projection cache key mismatch")
        if tuple(m.message_id for m in self.messages) != self.segment.message_ids:
            raise ValueError("segment projection 消息库存不一致")
        if (
            ProtocolSafeRevisionSegmenter.describe_segment(self.segment.ordinal, self.messages)
            != self.segment
        ):
            raise ValueError("segment projection 内容与 segment hash 不一致")
        accepted, rejections, fallback = self._dispositions(
            self.messages, self.drafts, self.assessments, self.grounding_version
        )
        if (accepted, rejections, fallback) != (
            self.accepted_claim_keys,
            self.rejections,
            self.fallback,
        ):
            raise ValueError("segment projection 验证状态不一致")
        return self

    def validate_target(
        self,
        segment: RevisionSegment,
        messages: tuple[IndexedMessage, ...],
        contract: str,
    ) -> None:
        if (
            segment != self.segment
            or messages != self.messages
            or contract != self.contract_fingerprint
        ):
            raise ValueError("segment projection target dependencies mismatch")
        type(self).model_validate(self.model_dump(mode="json"))

    def _payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"record_id"})

    @staticmethod
    def cache_key_for(segment: RevisionSegment, contract: str) -> str:
        return stable_expansion_hash(
            "segment-projection-key-v1", contract, segment.model_dump(mode="json")
        )

    @classmethod
    def create(
        cls,
        *,
        context_id: str,
        contract: str,
        segment: RevisionSegment,
        messages: tuple[IndexedMessage, ...],
        drafts=(),
        assessments=(),
        model_metadata=(),
    ) -> Self:
        accepted, rejected, fallback = cls._dispositions(messages, drafts, assessments)
        payload = {
            "grounding_version": SemanticGroundingValidator.RECORD_VERSION,
            "model_metadata": list(model_metadata),
            "context_id": context_id,
            "cache_key": cls.cache_key_for(segment, contract),
            "contract_fingerprint": contract,
            "segment": segment.model_dump(mode="json"),
            "messages": [m.model_dump(mode="json") for m in messages],
            "drafts": [d.model_dump(mode="json") for d in drafts],
            "assessments": [a.model_dump(mode="json") for a in assessments],
            "accepted_claim_keys": list(accepted),
            "rejections": [r.model_dump(mode="json") for r in rejected],
            "fallback": fallback,
        }
        return cls.model_validate(
            {
                "record_id": stable_expansion_hash("segment-projection-record-v1", payload),
                **payload,
            }
        )

    @staticmethod
    def _dispositions(
        messages, drafts, assessments, grounding_version=SemanticGroundingValidator.RECORD_VERSION
    ):
        contents = RevisionSemanticIndexer.message_contents(messages)
        policies = (
            {message.message_id: message.semantic_policy for message in messages}
            if grounding_version == SemanticGroundingValidator.RECORD_VERSION
            else None
        )
        by_claim = {a.claim_key: a for a in assessments}
        if len(by_claim) != len(assessments) or len({d.claim_key for d in drafts}) != len(drafts):
            raise ValueError("segment projection 重复 claim identity")
        expected = {
            d.claim_key
            for d in drafts
            if d.authority == "confirmed"
            and SemanticGroundingValidator.structurally_valid(contents, d)
        }
        if not set(by_claim).issubset(expected):
            raise ValueError("segment verifier assessment 不在本段 claims 中")
        accepted, rejected, represented = [], [], set()
        for draft in drafts:
            try:
                SemanticGroundingValidator.evaluate_draft(
                    contents, draft, by_claim.get(draft.claim_key), message_policies=policies
                )
                accepted.append(draft.claim_key)
                represented.update(s.message_id for s in draft.supports)
            except SemanticGroundingError as exc:
                rejected.append(
                    SemanticUnitRejection.create(draft, code=exc.code, summary=exc.summary)
                )
        return (
            tuple(sorted(accepted)),
            tuple(sorted(rejected, key=lambda r: r.rejection_id)),
            not {m.message_id for m in messages if m.semantic_policy == "index"}.issubset(
                represented
            ),
        )


class SegmentProjectionBuilder:
    def __init__(
        self,
        context_id: str,
        contract: str,
        projector: Any,
        verifier: Any,
        attempt_sink,
    ) -> None:
        self._context_id = context_id
        self._contract = contract
        self._projector = projector
        self._verifier = verifier
        self._attempt_sink = attempt_sink
        self._model_metadata = []

    def _capture_attempts(self, attempts):
        self._attempt_sink(attempts)
        self._model_metadata.extend(
            a["model_metadata"] for a in attempts if a.get("model_metadata")
        )

    async def build(
        self, segment: RevisionSegment, messages: tuple[IndexedMessage, ...]
    ) -> SegmentProjectionRecord:
        drafts, assessments = (), ()
        if self._projector is not None:
            payload = {
                "segments": (
                    {
                        "segment": segment.model_dump(mode="json"),
                        "messages": tuple(m.model_dump(mode="json") for m in messages),
                    },
                )
            }
            try:
                method = getattr(self._projector, "invoke_validated", None)
                if method is None:
                    proposal = await self._projector.invoke(
                        SemanticProjectionProposal, SEGMENT_PROMPT, payload
                    )
                    SupervisedSegmentSemanticProjector.validate_proposal(proposal)
                else:
                    proposal = await method(
                        SemanticProjectionProposal,
                        SEGMENT_PROMPT,
                        payload,
                        SupervisedSegmentSemanticProjector.validate_proposal,
                    )
                drafts = proposal.units
            finally:
                self._capture_attempts(tuple(getattr(self._projector, "last_attempt_records", ())))
            confirmed = tuple(
                d
                for d in drafts
                if d.authority == "confirmed"
                and SemanticGroundingValidator.structurally_valid(
                    RevisionSemanticIndexer.message_contents(messages), d
                )
            )
            if confirmed and self._verifier is not None:
                verifier = SupervisedSemanticClaimSupportVerifier(self._verifier)
                try:
                    assessments = await verifier.verify(confirmed)
                finally:
                    self._capture_attempts(verifier.attempt_records)
        return SegmentProjectionRecord.create(
            context_id=self._context_id,
            contract=self._contract,
            segment=segment,
            messages=messages,
            drafts=drafts,
            assessments=assessments,
            model_metadata=tuple(self._model_metadata),
        )
