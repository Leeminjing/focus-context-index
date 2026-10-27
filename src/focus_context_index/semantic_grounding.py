# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 support-span drafts、独立 claim assessments、SemanticGroundingValidator 与受监督 verifier。

输入为冻结消息原文、语义资格、陈述、精确引文及可选独立 verdict；输出为绑定目标 Revision 的 SemanticEvidenceUnit 或隔离原因。
具体工作流为 evaluate_draft 检查引文、宿主资格和 authority，再由 validate 生成来源引用；confirmed 仍必须有独立 supported verdict。
index 可定义任务命题；evidence_only 可产生证据类结果，不能独立定义 decision／question；reference_only 不独立产生 unit。
message_policies=None 仅用于旧 proof 的原算法兼容；新 indexing／projection／interpretation 均传递完整资格。
RECORD_VERSION 标识可读取的 proof 规则；VERSION 参与模型／配置整体 fingerprint，不作为历史 payload schema。
示例：authority = SemanticGroundingValidator.evaluate_draft(contents, draft, assessment, message_policies=policies)。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    SemanticAuthority,
    SemanticEvidenceUnit,
    SemanticUnitKind,
    stable_expansion_hash,
)
from .errors import (
    StructuredResultValidationError,
)
from .contracts import NamespacedMessageRef
from .contracts import ContextRevisionRef

ClaimSupportVerdict = Literal["supported", "unsupported", "unknown"]
SemanticGroundingFailureCode = Literal[
    "semantic_support_error",
    "unknown_message_ref",
    "claim_support_unsupported",
    "claim_support_unknown",
    "claim_support_missing",
    "semantic_eligibility_error",
]


class _GroundingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticSupportSpan(_GroundingModel):
    message_id: str = Field(min_length=1, max_length=512)
    quote: str = Field(min_length=1, max_length=8000)


class SegmentSemanticUnitDraft(_GroundingModel):
    kind: SemanticUnitKind
    authority: SemanticAuthority
    statement: str = Field(min_length=1, max_length=4000)
    supports: tuple[SemanticSupportSpan, ...] = Field(min_length=1)

    @field_validator("statement", mode="before")
    @classmethod
    def normalize_statement(cls, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    @model_validator(mode="after")
    def require_distinct_supports(self):
        identities = [(support.message_id, support.quote) for support in self.supports]
        if len(identities) != len(set(identities)):
            raise ValueError("semantic support span 重复")
        return self

    @property
    def claim_key(self) -> str:
        return stable_expansion_hash(
            "semantic-grounding-claim-v2",
            self.kind,
            self.authority,
            self.statement.casefold(),
            tuple(support.model_dump(mode="json") for support in self.supports),
        )


class SemanticProjectionProposal(_GroundingModel):
    units: tuple[SegmentSemanticUnitDraft, ...]


class SemanticClaimSupportAssessment(_GroundingModel):
    claim_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: ClaimSupportVerdict
    reason: str = Field(min_length=1, max_length=2000)


class SemanticClaimSupportProposal(_GroundingModel):
    assessments: tuple[SemanticClaimSupportAssessment, ...]


class SemanticUnitRejection(_GroundingModel):
    rejection_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    statement: str = Field(min_length=1, max_length=4000)
    message_ids: tuple[str, ...]
    code: SemanticGroundingFailureCode
    summary: str = Field(min_length=1, max_length=2000)
    disposition: Literal["quarantined", "hypothesis"] = "quarantined"

    @classmethod
    def create(
        cls,
        draft: SegmentSemanticUnitDraft,
        *,
        code: SemanticGroundingFailureCode,
        summary: str,
        disposition: Literal["quarantined", "hypothesis"] = "quarantined",
    ) -> SemanticUnitRejection:
        message_ids = tuple(dict.fromkeys(support.message_id for support in draft.supports))
        return cls(
            rejection_id=stable_expansion_hash(
                "semantic-unit-rejection-v1",
                draft.claim_key,
                code,
                summary,
                disposition,
            ),
            claim_key=draft.claim_key,
            statement=draft.statement,
            message_ids=message_ids,
            code=code,
            summary=summary,
            disposition=disposition,
        )


class SemanticGroundingError(ValueError):
    def __init__(
        self,
        code: SemanticGroundingFailureCode,
        summary: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(summary)
        self.code = code
        self.summary = summary
        self.retryable = retryable


class SupervisedSemanticClaimSupportVerifier:
    VERSION = "supervised-semantic-claim-verifier-v1"

    def __init__(self, model) -> None:
        self._model = model

    @property
    def attempt_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(getattr(self._model, "last_attempt_records", ()))

    async def verify(
        self,
        drafts: tuple[SegmentSemanticUnitDraft, ...],
    ) -> tuple[SemanticClaimSupportAssessment, ...]:
        if not drafts:
            return ()
        system = "你是无权 semantic claim verifier。只判断每条 statement 是否被给定冻结引文直接支持；必须逐条返回 supported、unsupported 或 unknown，不得补充证据、改写陈述或读取其他上下文。"
        payload = {
            "claims": tuple(
                {
                    "claim_key": draft.claim_key,
                    "statement": draft.statement,
                    "supports": tuple(
                        support.model_dump(mode="json") for support in draft.supports
                    ),
                }
                for draft in drafts
            )
        }
        expected = {draft.claim_key for draft in drafts}

        def validator(proposal):
            return self._validate_batch(proposal, expected)

        invoke_validated = getattr(self._model, "invoke_validated", None)
        if invoke_validated is None:
            proposal = await self._model.invoke(SemanticClaimSupportProposal, system, payload)
            validator(proposal)
        else:
            proposal = await invoke_validated(
                SemanticClaimSupportProposal,
                system,
                payload,
                validator,
            )
        return tuple(sorted(proposal.assessments, key=lambda item: item.claim_key))

    @staticmethod
    def _validate_batch(
        proposal: SemanticClaimSupportProposal,
        expected: set[str],
    ) -> None:
        actual = [assessment.claim_key for assessment in proposal.assessments]
        if len(actual) == len(set(actual)) and set(actual) == expected:
            return
        affected = next((identity for identity in actual if actual.count(identity) > 1), None)
        if affected is None:
            affected = next(iter(sorted(expected - set(actual))), "claim-support-batch")
        raise StructuredResultValidationError(
            "semantic_support_error",
            "claim support verifier 返回的 claim identities 不完整或重复",
            unit_identity=affected,
            violated_rule="return exactly one assessment for every requested claim_key",
        )


class SemanticGroundingValidator:
    RECORD_VERSION = "support-span-grounding-v2"
    VERSION = RECORD_VERSION

    @classmethod
    def evaluate_draft(cls, message_contents, draft, assessment=None, *, message_policies=None):
        cls._validate_supports(message_contents, draft)
        if message_policies is not None:
            cls.validate_eligibility(
                draft.kind, (support.message_id for support in draft.supports), message_policies
            )
        return cls._validated_authority(draft, assessment)

    @staticmethod
    def validate_eligibility(kind, message_ids, policies):
        selected = {policies.get(identity) for identity in message_ids}
        if not selected or not selected.issubset({"index", "evidence_only", "reference_only"}):
            raise SemanticGroundingError(
                "semantic_eligibility_error", "semantic support 缺少有效宿主资格"
            )
        if "index" in selected:
            return
        evidence_kinds = {
            "claim",
            "hypothesis",
            "implementation_effect",
            "verification_result",
            "failure",
        }
        if "evidence_only" in selected and kind in evidence_kinds:
            return
        raise SemanticGroundingError(
            "semantic_eligibility_error", "参考或证据不能独立定义该任务命题"
        )

    def validate(
        self,
        source: ContextRevisionRef,
        message_contents: Mapping[str, str],
        draft: SegmentSemanticUnitDraft,
        assessment: SemanticClaimSupportAssessment | None = None,
        *,
        message_policies: Mapping[str, str] | None = None,
    ) -> SemanticEvidenceUnit:
        authority = self.evaluate_draft(
            message_contents, draft, assessment, message_policies=message_policies
        )
        message_ids = tuple(dict.fromkeys(support.message_id for support in draft.supports))
        return SemanticEvidenceUnit.create(
            kind=draft.kind,
            authority=authority,
            statement=draft.statement,
            evidence_refs=tuple(
                NamespacedMessageRef(source=source, message_id=message_id)
                for message_id in message_ids
            ),
        )

    @staticmethod
    def structurally_valid(
        message_contents: Mapping[str, str],
        draft: SegmentSemanticUnitDraft,
    ) -> bool:
        try:
            SemanticGroundingValidator._validate_supports(message_contents, draft)
        except SemanticGroundingError:
            return False
        return True

    @staticmethod
    def _validate_supports(
        message_contents: Mapping[str, str],
        draft: SegmentSemanticUnitDraft,
    ) -> None:
        for support in draft.supports:
            content = message_contents.get(support.message_id)
            if content is None:
                raise SemanticGroundingError(
                    "unknown_message_ref",
                    f"semantic support 引用了 Revision 之外的 message: {support.message_id}",
                )
            if support.quote not in content:
                raise SemanticGroundingError(
                    "semantic_support_error",
                    f"semantic support quote 不是冻结 message 原文子串: {support.message_id}",
                )

    @staticmethod
    def _validated_authority(
        draft: SegmentSemanticUnitDraft,
        assessment: SemanticClaimSupportAssessment | None,
    ) -> SemanticAuthority:
        if draft.authority != "confirmed":
            return draft.authority
        if assessment is None or assessment.claim_key != draft.claim_key:
            raise SemanticGroundingError(
                "claim_support_missing",
                "confirmed semantic unit 缺少独立 claim-support assessment",
            )
        if assessment.verdict == "unsupported":
            raise SemanticGroundingError(
                "claim_support_unsupported",
                assessment.reason,
            )
        if assessment.verdict == "unknown":
            raise SemanticGroundingError(
                "claim_support_unknown",
                assessment.reason,
            )
        return "confirmed"
