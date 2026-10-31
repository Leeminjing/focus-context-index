"""Small host seams; model instances must represent one physical attempt per invoke."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from .api import AttemptReceipt, BuildAuthorization, FrozenSnapshot, IndexArtifact
from .segment_projection import SegmentProjectionRecord


class StructuredModel(Protocol):
    async def invoke(self, schema: type[BaseModel], system: str, payload: dict) -> Any: ...


class ModelAdapter(Protocol):
    def identity(self, role: str) -> str: ...
    def create(self, role: str) -> StructuredModel: ...


class IndexStore(Protocol):
    async def get_segment(
        self, namespace: str, context_id: str, cache_key: str
    ) -> SegmentProjectionRecord | None: ...

    async def put_segment(
        self, namespace: str, record: SegmentProjectionRecord
    ) -> SegmentProjectionRecord: ...

    async def get_artifact(self, cache_key: str) -> IndexArtifact | None: ...
    async def put_artifact(self, artifact: IndexArtifact) -> IndexArtifact: ...


@dataclass(frozen=True)
class ModelCall:
    call_id: str
    snapshot: FrozenSnapshot
    authorization: BuildAuthorization
    phase: str
    estimated_input_tokens: int
    output_token_reserve: int
    attempt: int = 1


class Admission(Protocol):
    async def authorize(
        self, snapshot: FrozenSnapshot, authorization: BuildAuthorization
    ) -> None: ...
    async def reserve(self, call: ModelCall) -> None: ...
    async def settle(self, call: ModelCall, receipt: AttemptReceipt) -> None: ...


class FocusModelAdapter:
    """Adapt Focus's existing role-bound structured model factories without importing Focus.

    Configure host factories with max_attempts=1 so every provider attempt is admitted
    separately. The engine owns validation retries. Each factory must produce a fresh
    model, with a cache_identity covering provider, model and generation settings.
    """

    def __init__(
        self,
        *,
        projector_factory: Callable[[], Any],
        verifier_factory: Callable[[], Any],
        interpreter_factory: Callable[[], Any] | None = None,
    ):
        self._factories = {
            "projector": projector_factory,
            "verifier": verifier_factory,
            "interpreter": interpreter_factory or projector_factory,
        }

    def identity(self, role: str) -> str:
        return self._identity(self.create(role))

    def create(self, role: str):
        model = self._factories[role]()
        if getattr(model, "_max_attempts", 1) != 1:
            raise ValueError("Focus model factories must use max_attempts=1")
        return model

    @staticmethod
    def _identity(model) -> str:
        identity = getattr(model, "cache_identity", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("models require an explicit, nonempty cache_identity")
        return identity
