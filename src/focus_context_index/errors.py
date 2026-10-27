# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
from typing import Any


class StructuredResultValidationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        unit_identity: str,
        violated_rule: str,
        details: dict[str, Any] | None = None,
        retry_context: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.unit_identity = unit_identity
        self.violated_rule = violated_rule
        self.details = details
        # Private correction input is never copied into attempt diagnostics.
        self.retry_context = retry_context
        self.retryable = retryable


class IndexBudgetExceeded(ValueError):
    """The host authorization or request capacity does not admit this work."""


class IndexBuildError(RuntimeError):
    """A failed build with the receipts collected before failure; no partial artifact."""

    def __init__(self, cause: Exception, stats):
        super().__init__(str(cause))
        self.cause = cause
        self.stats = stats
