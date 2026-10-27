# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 RevisionSemanticIndex、覆盖账本和稳定 segment identity。

输入为冻结 Revision、规范消息、segments、验证 units、inheritance receipt 及 interpretation proof；输出为不可变完整索引。
具体工作流为检查恰好一次覆盖、Tool Exchange 闭合、目标来源、质量账本及局部库存，再验证综合全部阅读依赖与联合 units 身份。
旧 v3/v4 无综合 proof 的 payload 保留原 identity 算法；新 rev 合同要求完成综合。
semantic_policy 分离任务命题与证据／参考，source_ordinal 保留原始位置，只有 index 资格的未覆盖消息产生 fallback。
task_fallback_unit 在 v7/v8/v9 只用 index 正文及来源构造假设；v8/v9 校验 projected units 的宿主资格。
旧索引保持其原验证算法读取，不改写历史 identity。
示例：index = RevisionSemanticIndex.create(..., interpretation=proof)；一条诊断共同引用早期怀疑和后来否定。
"""

from __future__ import annotations

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    SemanticEvidenceUnit,
    stable_expansion_hash,
)
from .semantic_grounding import (
    SemanticUnitRejection,
    SemanticGroundingValidator,
)
from .contracts import NamespacedMessageRef
from .contracts import ContextRevisionRef

from .index_build_contracts import IndexInheritanceReceipt
from .interpretation_record import RevisionInterpretationRecord


class _IndexModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexedMessage(_IndexModel):
    message_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    role: str = Field(min_length=1)
    content: Any = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    status: str | None = None
    source_item_ids: tuple[str, ...] = ()
    semantic_policy: Literal["index", "evidence_only", "reference_only"] = "index"
    source_ordinal: int | None = Field(default=None, ge=0)


def task_fallback_unit(
    source: ContextRevisionRef, segment: RevisionSegment, messages: tuple[IndexedMessage, ...]
) -> SemanticEvidenceUnit:
    eligible = [
        message
        for message in messages
        if message.message_id in segment.message_ids and message.semantic_policy == "index"
    ]
    parts = []
    for message in eligible:
        text = (
            message.content
            if isinstance(message.content, str)
            else json.dumps(message.content, ensure_ascii=False, default=str)
        )
        compact = " ".join(text.split())[:240]
        parts.append(f"{message.role}:{compact}" if compact else message.role)
    return SemanticEvidenceUnit.create(
        kind="claim",
        authority="hypothesis",
        statement=" | ".join(parts)[:1200] or "empty-content segment",
        evidence_refs=tuple(
            NamespacedMessageRef(source=source, message_id=message.message_id)
            for message in eligible
        ),
    )


class RevisionSegment(_IndexModel):
    segment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordinal: int = Field(ge=0)
    message_ids: tuple[str, ...] = Field(min_length=1)
    descriptor: str = Field(min_length=1, max_length=1200)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_stable_identity(self) -> Self:
        if len(self.message_ids) != len(set(self.message_ids)):
            raise ValueError("segment message identity 重复")
        expected = stable_expansion_hash(
            "revision-segment",
            self.ordinal,
            self.message_ids,
            self.descriptor,
            self.content_hash,
        )
        if self.segment_id != expected:
            raise ValueError("segment identity 与内容不一致")
        return self

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        message_ids: tuple[str, ...],
        descriptor: str,
        content_hash: str,
    ) -> Self:
        return cls(
            segment_id=stable_expansion_hash(
                "revision-segment",
                ordinal,
                message_ids,
                descriptor,
                content_hash,
            ),
            ordinal=ordinal,
            message_ids=message_ids,
            descriptor=descriptor,
            content_hash=content_hash,
        )


class RevisionCoverageLedger(_IndexModel):
    message_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    covered_message_count: int = Field(ge=0)
    protocol_exchange_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_unique_coverage_inventory(self) -> Self:
        if len(self.message_ids) != len(set(self.message_ids)):
            raise ValueError("coverage ledger message identity 重复")
        if len(self.segment_ids) != len(set(self.segment_ids)):
            raise ValueError("coverage ledger segment identity 重复")
        if self.covered_message_count != len(self.message_ids):
            raise ValueError("coverage ledger count 与 message inventory 不一致")
        return self


class RevisionIndexDescriptor(_IndexModel):
    index_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: ContextRevisionRef
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_role: str = Field(min_length=1, max_length=120)
    active_objective: str = Field(min_length=1, max_length=2000)
    segment_count: int = Field(ge=0)
    semantic_unit_count: int = Field(ge=0)
    index_schema_version: str = Field(min_length=1, max_length=64)
    segmenter_version: str = Field(min_length=1, max_length=64)
    projector_version: str = Field(min_length=1, max_length=64)
    quality_state: Literal["complete", "degraded"] = "complete"
    rejected_unit_count: int = Field(default=0, ge=0)
    projected_unit_count: int = Field(default=0, ge=0)
    fallback_segment_count: int = Field(default=0, ge=0)


class RevisionSemanticIndex(_IndexModel):
    index_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: ContextRevisionRef
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_role: str = Field(min_length=1, max_length=120)
    active_objective: str = Field(min_length=1, max_length=2000)
    messages: tuple[IndexedMessage, ...]
    segments: tuple[RevisionSegment, ...]
    semantic_units: tuple[SemanticEvidenceUnit, ...] = ()
    rejected_units: tuple[SemanticUnitRejection, ...] = ()
    projected_unit_ids: tuple[str, ...] = ()
    fallback_segment_ids: tuple[str, ...] = ()
    quality_state: Literal["complete", "degraded"] = "complete"
    coverage: RevisionCoverageLedger
    index_schema_version: str = Field(min_length=1, max_length=64)
    segmenter_version: str = Field(min_length=1, max_length=64)
    projector_version: str = Field(min_length=1, max_length=64)
    inheritance: IndexInheritanceReceipt | None = None
    interpretation: RevisionInterpretationRecord | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def order_messages(cls, values: Any) -> tuple[Any, ...]:
        return tuple(
            sorted(
                values or (),
                key=lambda item: item.get("ordinal", 0) if isinstance(item, dict) else item.ordinal,
            )
        )

    @field_validator("segments", mode="before")
    @classmethod
    def order_segments(cls, values: Any) -> tuple[Any, ...]:
        return tuple(
            sorted(
                values or (),
                key=lambda item: item.get("ordinal", 0) if isinstance(item, dict) else item.ordinal,
            )
        )

    @model_validator(mode="after")
    def require_complete_valid_index(self) -> Self:
        if not self.source.is_runnable:
            raise ValueError("Revision semantic index source 必须是可精确读取的 runnable Revision")
        message_ids = tuple(item.message_id for item in self.messages)
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("Revision index message identity 重复")
        if tuple(item.ordinal for item in self.messages) != tuple(range(len(self.messages))):
            raise ValueError("Revision index message ordinal 不连续")
        covered = tuple(
            message_id for segment in self.segments for message_id in segment.message_ids
        )
        if covered != message_ids:
            raise ValueError("Revision index segments 未按原序恰好覆盖全部消息")
        if self.coverage.message_ids != message_ids:
            raise ValueError("Revision index coverage ledger 与消息不一致")
        if self.coverage.segment_ids != tuple(item.segment_id for item in self.segments):
            raise ValueError("Revision index coverage ledger 与 segments 不一致")
        self._require_semantic_provenance(set(message_ids))
        self._require_protocol_closure()
        if self.inheritance is not None:
            receipt = self.inheritance
            segment_ids = tuple(s.segment_id for s in self.segments)
            if receipt.reused_segment_ids + receipt.recomputed_segment_ids != segment_ids:
                raise ValueError("index inheritance segment coverage mismatch")
            if len(receipt.record_ids) != len(segment_ids):
                raise ValueError("index inheritance record coverage mismatch")
        self._require_interpretation()
        if self.index_id != self._identity():
            raise ValueError("Revision semantic index identity 与冻结 payload 不一致")
        return self

    def _require_interpretation(self):
        if self.interpretation is None:
            if self.inheritance is not None and self.projector_version.startswith("rev:"):
                raise ValueError("complete revision projection requires interpretation proof")
            return
        self.interpretation.validate_target(self)
        if self.inheritance is None or self.inheritance.record_ids != tuple(
            e.record_id for e in self.interpretation.inventory
        ):
            raise ValueError("interpretation local record inventory integrity mismatch")
        units = {u.unit_id: u for u in self.semantic_units}
        if any(units.get(u.unit_id) != u for u in self.interpretation.accepted_units(self.source)):
            raise ValueError("interpretation accepted unit inventory integrity mismatch")
        rejected = {r.rejection_id for r in self.rejected_units}
        if not {r.rejection_id for r in self.interpretation.rejections}.issubset(rejected):
            raise ValueError("interpretation rejected unit inventory integrity mismatch")

    def _require_semantic_provenance(self, known_message_ids: set[str]) -> None:
        unit_ids = [item.unit_id for item in self.semantic_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Revision index semantic unit identity 重复")
        rejection_ids = [item.rejection_id for item in self.rejected_units]
        if len(rejection_ids) != len(set(rejection_ids)):
            raise ValueError("Revision index semantic rejection identity 重复")
        projected = set(self.projected_unit_ids)
        fallback = set(self.fallback_segment_ids)
        if len(projected) != len(self.projected_unit_ids) or not projected.issubset(unit_ids):
            raise ValueError("projected unit inventory 与 semantic units 不一致")
        segment_ids = {item.segment_id for item in self.segments}
        if len(fallback) != len(self.fallback_segment_ids) or not fallback.issubset(segment_ids):
            raise ValueError("fallback segment inventory 与 segments 不一致")
        if self.quality_state == "complete" and (self.rejected_units or fallback):
            raise ValueError("complete semantic index 不得包含 rejected units 或 segment fallback")
        if self.quality_state == "degraded" and not (self.rejected_units or fallback):
            raise ValueError("degraded semantic index 必须包含 rejected units 或 segment fallback")
        for unit in self.semantic_units:
            for ref in unit.evidence_refs:
                if not isinstance(ref, NamespacedMessageRef):
                    raise TypeError("Revision index semantic unit 只能引用 message evidence")
                if ref.source != self.source or ref.message_id not in known_message_ids:
                    raise ValueError("Revision index semantic unit 引用了 index 之外的 message")
        self._require_fallback_inventory(projected, fallback, set(unit_ids))
        if self.index_schema_version in {
            "revision-semantic-index-v8",
            "revision-semantic-index-v9",
        }:
            policies = {message.message_id: message.semantic_policy for message in self.messages}
            for unit in self.semantic_units:
                if unit.unit_id in projected:
                    SemanticGroundingValidator.validate_eligibility(
                        unit.kind, (ref.message_id for ref in unit.evidence_refs), policies
                    )

    def _require_fallback_inventory(
        self,
        projected: set[str],
        fallback: set[str],
        unit_ids: set[str],
    ) -> None:
        represented = {
            ref.message_id
            for unit in self.semantic_units
            if unit.unit_id in projected
            for ref in unit.evidence_refs
        }
        eligible = {
            message.message_id for message in self.messages if message.semantic_policy == "index"
        }
        expected_fallback = {
            segment.segment_id
            for segment in self.segments
            if set(segment.message_ids) & eligible
            and not (set(segment.message_ids) & eligible).issubset(represented)
        }
        if fallback != expected_fallback:
            raise ValueError("fallback segment inventory 未精确覆盖未投影的来源")
        expected_fallback_units = {
            self._fallback_unit(segment).unit_id
            for segment in self.segments
            if segment.segment_id in fallback
        }
        if unit_ids != projected | expected_fallback_units:
            raise ValueError("semantic units 与 projected/fallback inventory 不一致")

    def _fallback_unit(self, segment: RevisionSegment) -> SemanticEvidenceUnit:
        if self.index_schema_version in {
            "revision-semantic-index-v7",
            "revision-semantic-index-v8",
            "revision-semantic-index-v9",
        }:
            return task_fallback_unit(self.source, segment, self.messages)
        return SemanticEvidenceUnit.create(
            kind="claim",
            authority="hypothesis",
            statement=segment.descriptor,
            evidence_refs=tuple(
                NamespacedMessageRef(source=self.source, message_id=message_id)
                for message_id in segment.message_ids
            ),
        )

    def descriptor(self) -> RevisionIndexDescriptor:
        return RevisionIndexDescriptor(
            index_id=self.index_id,
            source=self.source,
            source_content_hash=self.source_content_hash,
            context_role=self.context_role,
            active_objective=self.active_objective,
            segment_count=len(self.segments),
            semantic_unit_count=len(self.semantic_units),
            index_schema_version=self.index_schema_version,
            segmenter_version=self.segmenter_version,
            projector_version=self.projector_version,
            quality_state=self.quality_state,
            rejected_unit_count=len(self.rejected_units),
            projected_unit_count=len(self.projected_unit_ids),
            fallback_segment_count=len(self.fallback_segment_ids),
        )

    def _require_protocol_closure(self) -> None:
        segment_by_message = {
            message_id: segment.segment_id
            for segment in self.segments
            for message_id in segment.message_ids
        }
        calls: dict[str, str] = {}
        for message in self.messages:
            for call in message.tool_calls:
                call_id = str(call.get("id") or "")
                if not call_id or call_id in calls:
                    raise ValueError("Tool Exchange call identity 缺失或重复")
                calls[call_id] = message.message_id
            if message.tool_call_id:
                assistant_id = calls.get(message.tool_call_id)
                if assistant_id is None:
                    raise ValueError("Tool Result 缺少同 Revision 的 Assistant tool call")
                if segment_by_message[assistant_id] != segment_by_message[message.message_id]:
                    raise ValueError("Tool Exchange 被拆分到不同 segments")
        returned_ids = [item.tool_call_id for item in self.messages if item.tool_call_id]
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("Tool Exchange Result identity 重复")
        returned = set(returned_ids)
        if set(calls) != returned:
            raise ValueError("Assistant tool call 缺少对应 Tool Result")

    def _identity(self) -> str:
        payload = (
            self.source.model_dump(mode="json"),
            self.source_content_hash,
            self.index_schema_version,
            self.segmenter_version,
            self.projector_version,
            self.message_records(),
            tuple(item.model_dump(mode="json") for item in self.segments),
            tuple(item.model_dump(mode="json") for item in self.semantic_units),
            tuple(item.model_dump(mode="json") for item in self.rejected_units),
            self.projected_unit_ids,
            self.fallback_segment_ids,
            self.quality_state,
        )
        if self.inheritance is not None:
            payload += (self.inheritance.model_dump(mode="json"),)
        if self.interpretation is not None:
            payload += (self.interpretation.model_dump(mode="json"),)
        return stable_expansion_hash("revision-semantic-index", *payload)

    def message_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._message_record(item, self.index_schema_version) for item in self.messages
        )

    @staticmethod
    def _message_record(item: IndexedMessage, version: str) -> dict[str, Any]:
        excluded = (
            {"semantic_policy", "source_ordinal", "name", "status", "source_item_ids"}
            if version
            in {
                "revision-semantic-index-v3",
                "revision-semantic-index-v4",
                "revision-semantic-index-v5",
            }
            else set()
        )
        return item.model_dump(mode="json", exclude=excluded)

    @classmethod
    def create(
        cls,
        *,
        source: ContextRevisionRef,
        source_content_hash: str,
        context_role: str,
        active_objective: str,
        messages: tuple[IndexedMessage, ...],
        segments: tuple[RevisionSegment, ...],
        semantic_units: tuple[SemanticEvidenceUnit, ...] = (),
        rejected_units: tuple[SemanticUnitRejection, ...] = (),
        projected_unit_ids: tuple[str, ...] = (),
        fallback_segment_ids: tuple[str, ...] = (),
        quality_state: Literal["complete", "degraded"] = "complete",
        index_schema_version: str,
        segmenter_version: str,
        projector_version: str,
        protocol_exchange_count: int = 0,
        inheritance: IndexInheritanceReceipt | None = None,
        interpretation: RevisionInterpretationRecord | None = None,
    ) -> Self:
        ordered_messages = tuple(sorted(messages, key=lambda item: item.ordinal))
        ordered_segments = tuple(sorted(segments, key=lambda item: item.ordinal))
        ordered_units = tuple(sorted(semantic_units, key=lambda item: item.unit_id))
        ordered_rejections = tuple(sorted(rejected_units, key=lambda item: item.rejection_id))
        ordered_projected_ids = tuple(sorted(projected_unit_ids))
        ordered_fallback_ids = tuple(sorted(fallback_segment_ids))
        coverage = RevisionCoverageLedger(
            message_ids=tuple(item.message_id for item in ordered_messages),
            segment_ids=tuple(item.segment_id for item in ordered_segments),
            covered_message_count=len(ordered_messages),
            protocol_exchange_count=protocol_exchange_count,
        )
        payload = (
            source.model_dump(mode="json"),
            source_content_hash,
            index_schema_version,
            segmenter_version,
            projector_version,
            tuple(cls._message_record(item, index_schema_version) for item in ordered_messages),
            tuple(item.model_dump(mode="json") for item in ordered_segments),
            tuple(item.model_dump(mode="json") for item in ordered_units),
            tuple(item.model_dump(mode="json") for item in ordered_rejections),
            ordered_projected_ids,
            ordered_fallback_ids,
            quality_state,
        )
        if inheritance is not None:
            payload += (inheritance.model_dump(mode="json"),)
        if interpretation is not None:
            payload += (interpretation.model_dump(mode="json"),)
        return cls(
            index_id=stable_expansion_hash("revision-semantic-index", *payload),
            source=source,
            source_content_hash=source_content_hash,
            context_role=context_role,
            active_objective=active_objective,
            messages=ordered_messages,
            segments=ordered_segments,
            semantic_units=ordered_units,
            rejected_units=ordered_rejections,
            projected_unit_ids=ordered_projected_ids,
            fallback_segment_ids=ordered_fallback_ids,
            quality_state=quality_state,
            coverage=coverage,
            index_schema_version=index_schema_version,
            segmenter_version=segmenter_version,
            projector_version=projector_version,
            inheritance=inheritance,
            interpretation=interpretation,
        )


IndexEntryKind = Literal["segment", "semantic_unit"]
