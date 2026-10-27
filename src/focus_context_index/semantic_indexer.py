# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 ProtocolSafeRevisionSegmenter、SupervisedSegmentSemanticProjector 与 RevisionSemanticIndexer。

输入为冻结 semantic 消息、原始位置、来源/hash 和受监督的 drafts／assessments；输出为协议闭合 segments 与完整目标 Index。
具体工作流为规范消息、将 Tool Exchange 折叠成原子，按实际参数 fingerprint 切段，统一验证引用并补齐 fallback。
可选 max_estimated_tokens 按序列化消息估算体积；同时约束条数，超大单消息／工具交换保持原子。关闭时沿用 v2 切分与 fingerprint。
v9 fallback 只从 index 消息生成正文和来源，所有 drafts 经共享宿主资格校验；混合段的证据／参考仍可解释。
每条 draft 的引文／verdict 独立 grounding，再按最终 unit_id 合并相同事实和 projected 库存；局部／综合 proof 保留原始证据。
describe_segment 是纯 identity 工具；validate_proposal 检查投影 identity 唯一性。旧 projector 保留兼容，新 Portfolio 合并局部及综合 drafts。
示例：index = RevisionSemanticIndexer().index(source=ref, raw_messages=messages, ...)；跨段联合引文绑定同一 Revision，旧 index 不变。
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from .sizing import estimate_raw_tokens

from .contracts import (
    SemanticEvidenceUnit,
)
from .semantic_grounding import (
    SegmentSemanticUnitDraft,
    SemanticClaimSupportAssessment,
    SemanticGroundingError,
    SemanticGroundingValidator,
    SemanticProjectionProposal,
    SemanticUnitRejection,
)
from .semantic_index import (
    IndexedMessage,
    RevisionSegment,
    RevisionSemanticIndex,
    task_fallback_unit,
)
from .errors import (
    StructuredResultValidationError,
)
from .contracts import NamespacedMessageRef
from .contracts import ContextRevisionRef

from .contracts import stable_expansion_hash
from .index_build_contracts import IndexInheritanceReceipt


class SupervisedSegmentSemanticProjector:
    VERSION = "supervised-segment-projector-v2"

    def __init__(self, model) -> None:
        self._model = model

    @property
    def attempt_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(getattr(self._model, "last_attempt_records", ()))

    async def project(
        self,
        index: RevisionSemanticIndex,
    ) -> tuple[SegmentSemanticUnitDraft, ...]:
        system = "你是无权 semantic_index_projector。semantic_policy=index 才能定义任务命题；evidence_only 是关联证据，reference_only 是约束参考，两者不独立提升为任务要求。只从输入冻结 segment 原文抽取原子 semantic units。每个 unit 分别输出自然语言 statement 与一个或多个 supports；每个 support 的 message_id 必须来自输入，quote 必须逐字复制该消息原文。confirmed statement 必须被全部引文共同直接支持。不得生成 WorkSpec、查询外部历史或执行状态变更。"
        payload = {
            "source": index.source.model_dump(mode="json"),
            "segments": tuple(
                {
                    "segment": segment.model_dump(mode="json"),
                    "messages": tuple(
                        message.model_dump(mode="json")
                        for message in index.messages
                        if message.message_id in set(segment.message_ids)
                    ),
                }
                for segment in index.segments
            ),
        }
        invoke_validated = getattr(self._model, "invoke_validated", None)
        if invoke_validated is None:
            proposal = await self._model.invoke(SemanticProjectionProposal, system, payload)
            self.validate_proposal(proposal)
        else:
            proposal = await invoke_validated(
                SemanticProjectionProposal,
                system,
                payload,
                self.validate_proposal,
            )
        return proposal.units

    @staticmethod
    def validate_proposal(proposal: SemanticProjectionProposal) -> None:
        seen: set[str] = set()
        for unit in proposal.units:
            if unit.claim_key in seen:
                raise StructuredResultValidationError(
                    "semantic_support_error",
                    "semantic projector 返回了重复的 unit identity",
                    unit_identity=unit.claim_key,
                    violated_rule="each projected semantic unit identity must be unique within one response",
                )
            seen.add(unit.claim_key)


class ProtocolSafeRevisionSegmenter:
    VERSION = "protocol-safe-segmenter-v3"

    def __init__(self, max_messages: int = 12, *, max_estimated_tokens: int | None = None) -> None:
        if max_messages < 1:
            raise ValueError("segment max_messages 必须为正数")
        if max_estimated_tokens is not None and max_estimated_tokens < 1:
            raise ValueError("segment max_estimated_tokens 必须为正数")
        self._max_messages = max_messages
        self._max_estimated_tokens = max_estimated_tokens

    def segment(self, messages: tuple[IndexedMessage, ...]) -> tuple[RevisionSegment, ...]:
        atoms = self._atoms(messages)
        groups: list[tuple[IndexedMessage, ...]] = []
        current: list[IndexedMessage] = []
        current_tokens = 0
        for atom in atoms:
            atom_tokens = (
                self.estimated_tokens(atom) if self._max_estimated_tokens is not None else 0
            )
            if current and (
                len(current) + len(atom) > self._max_messages
                or (
                    self._max_estimated_tokens is not None
                    and current_tokens + atom_tokens > self._max_estimated_tokens
                )
            ):
                groups.append(tuple(current))
                current = []
                current_tokens = 0
            # A tool exchange (or single large message) is indivisible, even
            # above either limit. Request admission still checks the full input.
            current.extend(atom)
            current_tokens += atom_tokens
        if current:
            groups.append(tuple(current))
        return tuple(self.describe_segment(index, group) for index, group in enumerate(groups))

    @property
    def fingerprint(self) -> str:
        if self._max_estimated_tokens is None:
            return (
                "seg:"
                + stable_expansion_hash("protocol-safe-segmenter-v2", self._max_messages)[:60]
            )
        return (
            "seg:"
            + stable_expansion_hash(
                self.VERSION,
                self._max_messages,
                self._max_estimated_tokens,
                "serialized-cjk1-other4-message12-v1",
            )[:60]
        )

    @staticmethod
    def estimated_tokens(messages: tuple[IndexedMessage, ...]) -> int:
        # Count the serialized payload, including tool arguments, structured
        # results and metadata, with the shared text heuristic. This is a sizing
        # target, not a provider tokenizer or the request admission calculation.
        return sum(
            estimate_raw_tokens(
                json.dumps(
                    message.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                1,
            )
            for message in messages
        )

    def _atoms(
        self, messages: tuple[IndexedMessage, ...]
    ) -> tuple[tuple[IndexedMessage, ...], ...]:
        atoms: list[tuple[IndexedMessage, ...]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            call_ids = {str(call.get("id") or "") for call in message.tool_calls}
            if not call_ids:
                atoms.append((message,))
                index += 1
                continue
            if "" in call_ids:
                raise ValueError("Assistant tool call 缺少 identity")
            exchange = [message]
            remaining = set(call_ids)
            cursor = index + 1
            while cursor < len(messages) and remaining:
                candidate = messages[cursor]
                exchange.append(candidate)
                remaining.discard(candidate.tool_call_id)
                cursor += 1
            if remaining:
                raise ValueError("Assistant tool call 缺少完整 Tool Result")
            atoms.append(tuple(exchange))
            index = cursor
        return tuple(atoms)

    @staticmethod
    def describe_segment(ordinal: int, messages: tuple[IndexedMessage, ...]) -> RevisionSegment:
        payload = tuple(item.model_dump(mode="json") for item in messages)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        descriptor_parts = []
        for message in messages:
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(message.content, ensure_ascii=False, default=str)
            )
            compact = " ".join(content.split())[:240]
            descriptor_parts.append(f"{message.role}:{compact}" if compact else message.role)
        descriptor = " | ".join(descriptor_parts)[:1200] or "empty-content segment"
        return RevisionSegment.create(
            ordinal=ordinal,
            message_ids=tuple(item.message_id for item in messages),
            descriptor=descriptor,
            content_hash=sha256(encoded.encode("utf-8")).hexdigest(),
        )


class RevisionSemanticIndexer:
    INDEX_SCHEMA_VERSION = "revision-semantic-index-v9"
    PROJECTOR_VERSION = "supervised-segment-projector-v2"

    def __init__(
        self,
        segmenter: ProtocolSafeRevisionSegmenter | None = None,
        grounding_validator: SemanticGroundingValidator | None = None,
    ) -> None:
        self._segmenter = segmenter or ProtocolSafeRevisionSegmenter()
        self._grounding = grounding_validator or SemanticGroundingValidator()

    @property
    def segmenter_version(self) -> str:
        return self._segmenter.fingerprint

    def index(
        self,
        *,
        source: ContextRevisionRef,
        source_content_hash: str,
        context_role: str,
        active_objective: str,
        raw_messages: tuple[dict[str, Any], ...],
        unit_drafts: tuple[SegmentSemanticUnitDraft, ...] = (),
        claim_support_assessments: tuple[SemanticClaimSupportAssessment, ...] = (),
        inheritance: IndexInheritanceReceipt | None = None,
        projector_version: str | None = None,
    ) -> RevisionSemanticIndex:
        messages = self._messages(raw_messages)
        segments = self._segmenter.segment(messages)
        units, rejections, projected_ids, fallback_ids = self._units(
            source,
            messages,
            segments,
            unit_drafts,
            claim_support_assessments,
        )
        return RevisionSemanticIndex.create(
            source=source,
            source_content_hash=source_content_hash,
            context_role=context_role or "context",
            active_objective=active_objective or "Current mission",
            messages=messages,
            segments=segments,
            semantic_units=units,
            rejected_units=rejections,
            projected_unit_ids=projected_ids,
            fallback_segment_ids=fallback_ids,
            quality_state="degraded" if rejections or fallback_ids else "complete",
            index_schema_version=self.INDEX_SCHEMA_VERSION,
            segmenter_version=self.segmenter_version,
            projector_version=projector_version or self.PROJECTOR_VERSION,
            inheritance=inheritance,
            protocol_exchange_count=sum(1 for item in messages if item.tool_calls),
        )

    @staticmethod
    def _messages(
        raw_messages: tuple[dict[str, Any], ...],
    ) -> tuple[IndexedMessage, ...]:
        messages = tuple(
            IndexedMessage(
                message_id=str(item.get("id") or item.get("message_id") or ""),
                ordinal=ordinal,
                role=str(item.get("role") or "unknown"),
                content=item.get("content", ""),
                tool_calls=tuple(item.get("tool_calls") or ()),
                tool_call_id=item.get("tool_call_id"),
                name=item.get("name"),
                status=item.get("status"),
                source_item_ids=tuple(item.get("source_item_ids", ())),
                semantic_policy=item.get("semantic_policy", "index"),
                source_ordinal=item.get("source_ordinal", ordinal),
            )
            for ordinal, item in enumerate(raw_messages)
        )
        if any(not item.message_id for item in messages):
            raise ValueError("Revision message 缺少 identity")
        if len({item.message_id for item in messages}) != len(messages):
            raise ValueError("Revision message identity 重复")
        return messages

    def _units(
        self,
        source: ContextRevisionRef,
        messages: tuple[IndexedMessage, ...],
        segments: tuple[RevisionSegment, ...],
        drafts: tuple[SegmentSemanticUnitDraft, ...],
        assessments: tuple[SemanticClaimSupportAssessment, ...],
    ) -> tuple[
        tuple[SemanticEvidenceUnit, ...],
        tuple[SemanticUnitRejection, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        contents = self.message_contents(messages)
        policies = {message.message_id: message.semantic_policy for message in messages}
        by_claim = {assessment.claim_key: assessment for assessment in assessments}
        if len(by_claim) != len(assessments):
            raise ValueError("claim support assessment identity 重复")
        units: dict[str, SemanticEvidenceUnit] = {}
        rejections: list[SemanticUnitRejection] = []
        projected_ids: set[str] = set()
        fallback_ids: list[str] = []
        for draft in drafts:
            try:
                unit = self._grounding.validate(
                    source,
                    contents,
                    draft,
                    by_claim.get(draft.claim_key),
                    message_policies=policies,
                )
                units[unit.unit_id] = unit
                projected_ids.add(unit.unit_id)
            except SemanticGroundingError as exc:
                rejections.append(
                    SemanticUnitRejection.create(
                        draft,
                        code=exc.code,
                        summary=exc.summary,
                    )
                )
        represented = {
            ref.message_id
            for unit in units.values()
            for ref in unit.evidence_refs
            if isinstance(ref, NamespacedMessageRef)
        }
        eligible = {
            message.message_id for message in messages if message.semantic_policy == "index"
        }
        for segment in segments:
            if not set(segment.message_ids) & eligible:
                continue
            if (set(segment.message_ids) & eligible).issubset(represented):
                continue
            fallback_ids.append(segment.segment_id)
            unit = task_fallback_unit(source, segment, messages)
            units[unit.unit_id] = unit
        return (
            tuple(sorted(units.values(), key=lambda item: item.unit_id)),
            tuple(sorted(rejections, key=lambda item: item.rejection_id)),
            tuple(sorted(projected_ids)),
            tuple(sorted(fallback_ids)),
        )

    @staticmethod
    def message_content(message: IndexedMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return json.dumps(message.content, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def message_contents(messages: tuple[IndexedMessage, ...]) -> dict[str, str]:
        return {
            message.message_id: RevisionSemanticIndexer.message_content(message)
            for message in messages
        }
