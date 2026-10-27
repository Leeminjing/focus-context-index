"""Focus wire adapter. The host supplies the frozen semantic view and lineage proof."""

from collections.abc import Mapping

from .api import FrozenSnapshot
from .contracts import ContextRevisionRef


def snapshot_from_focus(
    *,
    namespace: str,
    revision_ref,
    content_hash: str,
    semantic_messages,
    context_role: str = "context",
    active_objective: str = "Current mission",
) -> FrozenSnapshot:
    """Consume ContextRevisionReader.read(..., 'semantic').messages.

    Read the committed exact Revision, check its frozen content hash and delegation
    scope in the host before calling. Never pass the execution view: it may contain
    control/reasoning messages that have no indexing eligibility.
    """
    payload = (
        dict(revision_ref)
        if isinstance(revision_ref, Mapping)
        else revision_ref.model_dump(mode="json")
    )
    return FrozenSnapshot.from_messages(
        namespace=namespace,
        source=ContextRevisionRef.model_validate(payload),
        source_content_hash=content_hash,
        raw_messages=tuple(semantic_messages),
        context_role=context_role,
        active_objective=active_objective,
    )
