"""Host-neutral inputs and outputs. Historical Focus wire identities remain readable."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ContextRevisionRef, stable_expansion_hash
from .index_assembly import assemble_revision_index
from .segment_projection import SegmentProjectionRecord
from .semantic_index import IndexedMessage, RevisionSemanticIndex
from .semantic_indexer import RevisionSemanticIndexer


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrozenSnapshot(FrozenModel):
    namespace: str = Field(min_length=1)
    source: ContextRevisionRef
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    messages: tuple[IndexedMessage, ...]
    context_role: str = Field(default="context", min_length=1, max_length=120)
    active_objective: str = Field(default="Current mission", min_length=1, max_length=2000)

    @model_validator(mode="after")
    def require_inventory(self) -> Self:
        if not self.source.is_runnable:
            raise ValueError("snapshot requires an exact immutable source address")
        if tuple(m.ordinal for m in self.messages) != tuple(range(len(self.messages))):
            raise ValueError("snapshot message ordinals must be contiguous")
        if len({m.message_id for m in self.messages}) != len(self.messages):
            raise ValueError("snapshot contains duplicate message identities")
        return self

    @classmethod
    def from_messages(
        cls, *, raw_messages: tuple[dict[str, Any], ...] | list[dict[str, Any]], **values: Any
    ) -> Self:
        return cls(messages=RevisionSemanticIndexer._messages(tuple(raw_messages)), **values)

    @property
    def fingerprint(self) -> str:
        # The host hash alone cannot prove which frozen messages were supplied.
        return stable_expansion_hash("frozen-index-snapshot-v1", self.model_dump(mode="json"))


class IndexPolicy(FrozenModel):
    segment_max_messages: int = Field(default=12, ge=1)
    segment_max_estimated_tokens: int | None = Field(default=None, ge=1)
    max_attempts: int = Field(default=2, ge=1)
    max_exact_reads: int | None = Field(default=None, ge=0)
    max_planner_model_calls: int | None = Field(default=8, ge=0)
    max_request_input_tokens: int | None = Field(default=None, ge=1)
    output_token_reserve: int = Field(default=4096, ge=1)


class BuildAuthorization(FrozenModel):
    owner_id: str = Field(default="local", min_length=1)
    grant_id: str = Field(default="default", min_length=1)
    allowed_context_ids: tuple[str, ...] | None = None
    allowed_namespaces: tuple[str, ...] | None = None
    max_model_calls: int | None = Field(default=None, ge=0)
    max_input_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)

    @property
    def fingerprint(self) -> str:
        return stable_expansion_hash("index-authorization-v1", self.model_dump(mode="json"))


class AttemptReceipt(FrozenModel):
    call_id: str
    owner_id: str
    grant_id: str
    namespace: str
    context_id: str
    revision_id: str
    phase: str
    model_identity: str
    outcome: Literal["success", "error", "cancelled"]
    model_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_ms: float = Field(ge=0)
    queue_ms: float = Field(ge=0)
    error_type: str | None = None
    attempt: int = Field(default=1, ge=1)


class IndexContracts(FrozenModel):
    index_schema_version: str
    segmenter_version: str
    local: str
    interpretation: str
    revision: str


class BuildStats(FrozenModel):
    mode: Literal["full", "incremental", "exact_hit", "prewarm"]
    local_cache_hits: int = 0
    projected_segments: int = 0
    total_segments: int = 0
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    receipts: tuple[AttemptReceipt, ...] = ()
    stage_ms: dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0
    wait_duration_ms: float = 0
    joined_build: bool = False


class IndexArtifact(FrozenModel):
    namespace: str = Field(min_length=1)
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    index: RevisionSemanticIndex
    records: tuple[SegmentProjectionRecord, ...]

    @property
    def snapshot(self) -> FrozenSnapshot:
        return FrozenSnapshot(
            namespace=self.namespace,
            source=self.index.source,
            source_content_hash=self.index.source_content_hash,
            messages=self.index.messages,
            context_role=self.index.context_role,
            active_objective=self.index.active_objective,
        )

    @property
    def cache_key(self) -> str:
        return artifact_key(
            self.snapshot_fingerprint,
            self.index.segmenter_version,
            self.index.projector_version,
            self.index.index_schema_version,
        )

    @model_validator(mode="after")
    def require_proofs(self) -> Self:
        if self.snapshot.fingerprint != self.snapshot_fingerprint:
            raise ValueError("artifact snapshot integrity mismatch")
        if self.index.interpretation is None or self.index.inheritance is None:
            raise ValueError("complete artifact requires local and interpretation proofs")
        proof = self.index.interpretation
        # Reassembly also proves that every accepted local claim, verdict and rejection
        # corresponds to the published units, including claims not used by synthesis.
        from .index_build_contracts import IndexBuildPlan

        segmenter = _RecordedSegmenter(self.index.segments, self.index.segmenter_version)
        indexer = RevisionSemanticIndexer(segmenter)
        rebuilt = assemble_revision_index(
            indexer,
            self.index,
            self.records,
            IndexBuildPlan.model_validate(self.index.inheritance.model_dump(mode="json")),
            interpretation=proof,
            local_contract=proof.local_contract_fingerprint,
        )
        if rebuilt != self.index:
            raise ValueError("artifact projection integrity mismatch")
        return self


class _RecordedSegmenter:
    """Replay a validated inventory without guessing parameters from its fingerprint."""

    def __init__(self, segments, fingerprint):
        self._segments = segments
        self.fingerprint = fingerprint

    def segment(self, messages):
        return self._segments


def artifact_key(snapshot_fingerprint: str, segmenter: str, contract: str, schema: str) -> str:
    return stable_expansion_hash(
        "complete-index-cache-v1", snapshot_fingerprint, segmenter, contract, schema
    )


class BuildResult(FrozenModel):
    artifact: IndexArtifact
    stats: BuildStats

    @property
    def index(self) -> RevisionSemanticIndex:
        return self.artifact.index


class PrewarmResult(FrozenModel):
    records: tuple[SegmentProjectionRecord, ...]
    stats: BuildStats


class IndexProgress(FrozenModel):
    context_id: str
    revision_id: str
    phase: Literal["preparing", "local", "interpretation", "complete"]
    completed_segments: int = 0
    total_segments: int = 0
