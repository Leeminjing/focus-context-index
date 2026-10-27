# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 RevisionInterpretationProposal、发现库存和不可变 RevisionInterpretationRecord。

输入为完整有序局部证据目录、实际提供的冻结原文、联合 drafts 和独立 verdict；输出为受依赖身份约束的综合 proof。
具体工作流为区分 read／complete 回复，验证库存及原文范围，计算 accepted／quarantined claims，并保存所有理解输入的身份。
新 proof 保存 grounding_version 并检查宿主语义资格；旧 proof 缺少版本时使用原处置与序列化，不改写 identity。
历史 v3/v4/v5 proof 的原消息 schema 继续保持。
示例：record = RevisionInterpretationRecord.create(...); record.validate_target(index, records)。最后引用的消息不代替全部阅读依赖。
"""

from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from .contracts import stable_expansion_hash
from .semantic_grounding import (
    SegmentSemanticUnitDraft,
    SemanticClaimSupportAssessment,
    SemanticGroundingError,
    SemanticGroundingValidator,
    SemanticUnitRejection,
)


class _InterpretationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RevisionInterpretationProposal(_InterpretationModel):
    action: Literal["read", "complete"]
    read_segments: tuple[str, ...] = ()
    units: tuple[SegmentSemanticUnitDraft, ...] = ()

    @model_validator(mode="after")
    def require_valid_response(self) -> Self:
        if self.action == "read" and (not self.read_segments or self.units):
            raise ValueError("interpretation read requires segments and no claims")
        if self.action == "complete" and self.read_segments:
            raise ValueError("completed interpretation cannot request more evidence")
        if len(set(self.read_segments)) != len(self.read_segments):
            raise ValueError("interpretation duplicate read identity")
        if len({u.claim_key for u in self.units}) != len(self.units):
            raise ValueError("interpretation duplicate claim identity")
        return self


class InterpretationHint(_InterpretationModel):
    authority: str
    statement: str
    message_ids: tuple[str, ...]


class InterpretationInventoryEntry(_InterpretationModel):
    segment_id: str
    ordinal: int
    message_ids: tuple[str, ...]
    content_hash: str
    descriptor: str
    record_id: str
    claims: tuple[InterpretationHint, ...]
    rejections: tuple[SemanticUnitRejection, ...]
    fallback: bool


class InterpretationOriginalSegment(_InterpretationModel):
    segment_id: str
    messages: tuple[dict[str, Any], ...]


class RevisionInterpretationRecord(_InterpretationModel):
    SCHEMA_VERSION: ClassVar[str] = "revision-interpretation-record-v1"
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_id: str
    source_content_hash: str
    contract_fingerprint: str
    local_contract_fingerprint: str
    inventory: tuple[InterpretationInventoryEntry, ...]
    provided: tuple[InterpretationOriginalSegment, ...]
    read_requests: tuple[tuple[str, ...], ...] = ()
    drafts: tuple[SegmentSemanticUnitDraft, ...] = ()
    assessments: tuple[SemanticClaimSupportAssessment, ...] = ()
    accepted_claim_keys: tuple[str, ...] = ()
    rejections: tuple[SemanticUnitRejection, ...] = ()
    completed: Literal[True] = True
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
        if self.record_id != stable_expansion_hash(
            self.SCHEMA_VERSION, self.model_dump(mode="json", exclude={"record_id"})
        ):
            raise ValueError("interpretation record integrity mismatch")
        known = {entry.segment_id: entry for entry in self.inventory}
        if len(known) != len(self.inventory) or tuple(e.ordinal for e in self.inventory) != tuple(
            range(len(self.inventory))
        ):
            raise ValueError("interpretation inventory integrity mismatch")
        supplied = {item.segment_id: item for item in self.provided}
        if len(supplied) != len(self.provided) or not set(supplied).issubset(known):
            raise ValueError("interpretation supplied scope integrity mismatch")
        if any(not set(request).issubset(supplied) for request in self.read_requests):
            raise ValueError("interpretation read scope integrity mismatch")
        for item in self.provided:
            if (
                tuple(m.get("message_id") for m in item.messages)
                != known[item.segment_id].message_ids
            ):
                raise ValueError("interpretation original inventory integrity mismatch")
        accepted, rejected = self._dispositions(
            self.provided, self.drafts, self.assessments, self.grounding_version
        )
        if (accepted, rejected) != (self.accepted_claim_keys, self.rejections):
            raise ValueError("interpretation verification integrity mismatch")
        return self

    def validate_target(self, target, records=None) -> None:
        expected_contract = (
            "rev:"
            + stable_expansion_hash(
                "revision-projection-contract-v1",
                self.local_contract_fingerprint,
                self.contract_fingerprint,
            )[:60]
        )
        if target.projector_version != expected_contract:
            raise ValueError("interpretation target contract integrity mismatch")
        if (self.context_id, self.source_content_hash) != (
            target.source.context_id,
            target.source_content_hash,
        ):
            raise ValueError("interpretation target scope mismatch")
        expected = [
            (s.segment_id, s.ordinal, s.message_ids, s.content_hash, s.descriptor)
            for s in target.segments
        ]
        actual = [
            (e.segment_id, e.ordinal, e.message_ids, e.content_hash, e.descriptor)
            for e in self.inventory
        ]
        if actual != expected:
            raise ValueError("interpretation target inventory integrity mismatch")
        messages = {record["message_id"]: record for record in target.message_records()}
        for item in self.provided:
            if tuple(messages[m["message_id"]] for m in item.messages) != item.messages:
                raise ValueError("interpretation original content integrity mismatch")
        if records is not None and tuple(e.record_id for e in self.inventory) != tuple(
            r.record_id for r in records
        ):
            raise ValueError("interpretation local winner dependency integrity mismatch")
        type(self).model_validate(self.model_dump(mode="json"))

    @classmethod
    def create(
        cls,
        *,
        context_id,
        source_content_hash,
        contract_fingerprint,
        local_contract_fingerprint,
        inventory,
        provided,
        read_requests=(),
        drafts=(),
        assessments=(),
    ) -> Self:
        accepted, rejected = cls._dispositions(provided, drafts, assessments)
        payload = {
            "grounding_version": SemanticGroundingValidator.RECORD_VERSION,
            "context_id": context_id,
            "source_content_hash": source_content_hash,
            "contract_fingerprint": contract_fingerprint,
            "local_contract_fingerprint": local_contract_fingerprint,
            "inventory": [e.model_dump(mode="json") for e in inventory],
            "provided": [p.model_dump(mode="json") for p in provided],
            "read_requests": list(read_requests),
            "drafts": [d.model_dump(mode="json") for d in drafts],
            "assessments": [a.model_dump(mode="json") for a in assessments],
            "accepted_claim_keys": list(accepted),
            "rejections": [r.model_dump(mode="json") for r in rejected],
            "completed": True,
        }
        return cls(record_id=stable_expansion_hash(cls.SCHEMA_VERSION, payload), **payload)

    def accepted_units(self, source):
        contents = self._message_contents(self.provided)
        by_claim = {a.claim_key: a for a in self.assessments}
        accepted = set(self.accepted_claim_keys)
        validator = SemanticGroundingValidator()
        policies = self._message_policies(self.provided, self.grounding_version)
        return tuple(
            validator.validate(
                source, contents, draft, by_claim.get(draft.claim_key), message_policies=policies
            )
            for draft in self.drafts
            if draft.claim_key in accepted
        )

    @staticmethod
    def _message_contents(provided):
        import json

        return {
            m["message_id"]: m["content"]
            if isinstance(m["content"], str)
            else json.dumps(m["content"], ensure_ascii=False, sort_keys=True, default=str)
            for item in provided
            for m in item.messages
        }

    @staticmethod
    def _message_policies(provided, grounding_version):
        if grounding_version != SemanticGroundingValidator.RECORD_VERSION:
            return None
        return {
            message["message_id"]: message.get("semantic_policy", "index")
            for original in provided
            for message in original.messages
        }

    @classmethod
    def _dispositions(
        cls,
        provided,
        drafts,
        assessments,
        grounding_version=SemanticGroundingValidator.RECORD_VERSION,
    ):
        contents = cls._message_contents(provided)
        policies = cls._message_policies(provided, grounding_version)
        by_claim = {a.claim_key: a for a in assessments}
        expected = {
            d.claim_key
            for d in drafts
            if d.authority == "confirmed"
            and SemanticGroundingValidator.structurally_valid(contents, d)
        }
        if (
            len(by_claim) != len(assessments)
            or len({d.claim_key for d in drafts}) != len(drafts)
            or not set(by_claim).issubset(expected)
        ):
            raise ValueError("interpretation verdict identity integrity mismatch")
        accepted, rejected = [], []
        for draft in drafts:
            try:
                SemanticGroundingValidator.evaluate_draft(
                    contents, draft, by_claim.get(draft.claim_key), message_policies=policies
                )
                accepted.append(draft.claim_key)
            except SemanticGroundingError as exc:
                rejected.append(
                    SemanticUnitRejection.create(draft, code=exc.code, summary=exc.summary)
                )
        return tuple(sorted(accepted)), tuple(sorted(rejected, key=lambda r: r.rejection_id))
