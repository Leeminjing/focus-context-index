"""Incremental semantic indexes over immutable conversation revisions."""

from .admission import InMemoryAdmission
from .api import (
    AttemptReceipt,
    BuildAuthorization,
    BuildResult,
    BuildStats,
    FrozenSnapshot,
    IndexArtifact,
    IndexContracts,
    IndexPolicy,
    IndexProgress,
    PrewarmResult,
)
from .contracts import ContextRevisionRef, NamespacedMessageRef, SemanticEvidenceUnit
from .engine import IndexEngine
from .errors import IndexBudgetExceeded, IndexBuildError, StructuredResultValidationError
from .focus_adapter import snapshot_from_focus
from .ports import (
    Admission,
    FocusModelAdapter,
    IndexStore,
    ModelAdapter,
    ModelCall,
    StructuredModel,
)
from .semantic_grounding import (
    SegmentSemanticUnitDraft,
    SemanticClaimSupportAssessment,
    SemanticClaimSupportProposal,
    SemanticProjectionProposal,
    SemanticSupportSpan,
)
from .stores import MemoryIndexStore, SQLiteIndexStore

__all__ = [
    "Admission",
    "AttemptReceipt",
    "BuildAuthorization",
    "BuildResult",
    "BuildStats",
    "ContextRevisionRef",
    "FocusModelAdapter",
    "FrozenSnapshot",
    "InMemoryAdmission",
    "IndexArtifact",
    "IndexContracts",
    "IndexBudgetExceeded",
    "IndexBuildError",
    "IndexEngine",
    "IndexPolicy",
    "IndexProgress",
    "IndexStore",
    "MemoryIndexStore",
    "ModelAdapter",
    "ModelCall",
    "NamespacedMessageRef",
    "PrewarmResult",
    "SQLiteIndexStore",
    "SegmentSemanticUnitDraft",
    "SemanticClaimSupportAssessment",
    "SemanticClaimSupportProposal",
    "SemanticEvidenceUnit",
    "SemanticProjectionProposal",
    "SemanticSupportSpan",
    "StructuredModel",
    "StructuredResultValidationError",
    "snapshot_from_focus",
]
