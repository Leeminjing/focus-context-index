# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
from __future__ import annotations
import json
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContextRevisionPayloadMode(StrEnum):
    CHECKPOINT = "checkpoint"
    DEFINITION = "definition"


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextRevisionRef(_FrozenContract):
    context_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    execution_thread_id: str = Field(min_length=1)
    checkpoint_ns: str = ""
    checkpoint_id: str | None = Field(default=None, min_length=1)
    payload_mode: ContextRevisionPayloadMode

    @model_validator(mode="after")
    def _require_checkpoint_for_checkpoint_payload(self) -> Self:
        if self.payload_mode is ContextRevisionPayloadMode.CHECKPOINT and not self.checkpoint_id:
            raise ValueError("checkpoint-backed revision 必须包含 checkpoint_id")
        return self

    @property
    def is_runnable(self) -> bool:
        return self.checkpoint_id is not None

    def checkpoint_config(self) -> dict[str, dict[str, str]]:
        if self.checkpoint_id is None:
            raise ValueError("尚未准备 checkpoint 的 definition revision 不可寻址执行历史")
        return {
            "configurable": {
                "thread_id": self.execution_thread_id,
                "checkpoint_ns": self.checkpoint_ns,
                "checkpoint_id": self.checkpoint_id,
            }
        }


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NamespacedMessageRef(_StrictModel):
    source: ContextRevisionRef
    message_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_checkpoint(self) -> Self:
        if not self.source.is_runnable:
            raise ValueError("来源消息必须属于带精确 checkpoint 的 Context revision")
        return self

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.source.context_id,
            self.source.revision_id,
            self.source.checkpoint_id or "",
            self.message_id,
        )


EvidenceRef = NamespacedMessageRef


def evidence_ref_key(ref: NamespacedMessageRef) -> tuple[str, ...]:
    return ("context_message", *ref.key)


SemanticAuthority = Literal["confirmed", "hypothesis"]


SemanticUnitKind = Literal[
    "decision",
    "claim",
    "hypothesis",
    "unresolved_question",
    "implementation_effect",
    "verification_result",
    "failure",
]


class _ExpansionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def stable_expansion_hash(namespace: str, *parts: Any) -> str:
    payload = json.dumps(
        {"namespace": namespace, "parts": parts},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(value.split()).strip()


class SemanticEvidenceUnit(_ExpansionModel):
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: SemanticUnitKind
    authority: SemanticAuthority
    statement: str = Field(min_length=1, max_length=4000)
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @field_validator("statement", mode="before")
    @classmethod
    def normalize_statement(cls, value: Any) -> str:
        return _normalized(str(value or ""))

    @model_validator(mode="after")
    def require_supported_confirmation(self) -> Self:
        keys = [evidence_ref_key(ref) for ref in self.evidence_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("semantic unit evidence ref 重复")
        if self.authority == "confirmed" and not keys:
            raise ValueError("confirmed semantic unit 必须包含 evidence ref")
        expected = stable_expansion_hash(
            "semantic-evidence-unit",
            self.kind,
            self.authority,
            self.statement.casefold(),
            tuple(sorted(keys)),
        )
        if self.unit_id != expected:
            raise ValueError("semantic unit identity 与内容不一致")
        return self

    @classmethod
    def create(
        cls,
        *,
        kind: SemanticUnitKind,
        authority: SemanticAuthority,
        statement: str,
        evidence_refs: tuple[EvidenceRef, ...] = (),
    ) -> Self:
        normalized = _normalized(statement)
        ordered = tuple(sorted(evidence_refs, key=evidence_ref_key))
        return cls(
            unit_id=stable_expansion_hash(
                "semantic-evidence-unit",
                kind,
                authority,
                normalized.casefold(),
                tuple(evidence_ref_key(ref) for ref in ordered),
            ),
            kind=kind,
            authority=authority,
            statement=normalized,
            evidence_refs=ordered,
        )
