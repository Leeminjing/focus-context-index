"""Process-local authorization and usage accounting; replace at the host budget seam."""

import asyncio
from dataclasses import dataclass, field

from .api import AttemptReceipt, BuildAuthorization, FrozenSnapshot
from .errors import IndexBudgetExceeded
from .ports import ModelCall


@dataclass
class _Account:
    authorization: BuildAuthorization
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    pending: dict[str, ModelCall] = field(default_factory=dict)
    receipts: dict[str, AttemptReceipt] = field(default_factory=dict)


class InMemoryAdmission:
    def __init__(self):
        self._accounts: dict[tuple[str, str], _Account] = {}
        self._lock = asyncio.Lock()

    def _account(self, authorization):
        key = (authorization.owner_id, authorization.grant_id)
        account = self._accounts.setdefault(key, _Account(authorization))
        if account.authorization != authorization:
            raise ValueError("changed authorization requires a new grant_id")
        return account

    async def authorize(self, snapshot: FrozenSnapshot, authorization: BuildAuthorization):
        if (
            authorization.allowed_context_ids is not None
            and snapshot.source.context_id not in authorization.allowed_context_ids
        ):
            raise PermissionError("snapshot is outside the authorized Context scope")
        if (
            authorization.allowed_namespaces is not None
            and snapshot.namespace not in authorization.allowed_namespaces
        ):
            raise PermissionError("snapshot is outside the authorized namespace")
        async with self._lock:
            self._account(authorization)

    def can_reserve(self, call: ModelCall) -> bool:
        """Pure preflight for optional original prefill; does not reserve capacity."""
        account = self._account(call.authorization)
        pending = tuple(account.pending.values())
        usage = (
            (account.calls + len(pending) + 1, call.authorization.max_model_calls),
            (
                account.input_tokens
                + sum(p.estimated_input_tokens for p in pending)
                + call.estimated_input_tokens,
                call.authorization.max_input_tokens,
            ),
            (
                account.output_tokens
                + sum(p.output_token_reserve for p in pending)
                + call.output_token_reserve,
                call.authorization.max_output_tokens,
            ),
        )
        return all(limit is None or value <= limit for value, limit in usage)

    async def reserve(self, call: ModelCall):
        await self.authorize(call.snapshot, call.authorization)
        async with self._lock:
            if not self.can_reserve(call):
                raise IndexBudgetExceeded("authorized model budget exhausted")
            account = self._account(call.authorization)
            if call.call_id in account.pending or call.call_id in account.receipts:
                raise ValueError("model reservation identity already used")
            account.pending[call.call_id] = call

    async def settle(self, call: ModelCall, receipt: AttemptReceipt):
        if (
            receipt.call_id,
            receipt.owner_id,
            receipt.grant_id,
            receipt.namespace,
            receipt.context_id,
            receipt.revision_id,
            receipt.phase,
        ) != (
            call.call_id,
            call.authorization.owner_id,
            call.authorization.grant_id,
            call.snapshot.namespace,
            call.snapshot.source.context_id,
            call.snapshot.source.revision_id,
            call.phase,
        ):
            raise ValueError("usage receipt identity does not match its reservation")
        async with self._lock:
            account = self._account(call.authorization)
            if call.call_id in account.receipts:
                if account.receipts[call.call_id] != receipt:
                    raise ValueError("conflicting usage settlement")
                return
            if account.pending.pop(call.call_id, None) != call:
                raise ValueError("missing model reservation")
            account.receipts[call.call_id] = receipt
            account.calls += receipt.model_calls
            # Unknown provider usage is visible as None in receipts. Admission keeps
            # the estimate/reserve charged so an unreported call never becomes free.
            account.input_tokens += (
                receipt.input_tokens
                if receipt.input_tokens is not None
                else call.estimated_input_tokens
            )
            account.output_tokens += (
                receipt.output_tokens
                if receipt.output_tokens is not None
                else call.output_token_reserve
            )

    @property
    def receipts(self) -> tuple[AttemptReceipt, ...]:
        return tuple(r for a in self._accounts.values() for r in a.receipts.values())

    @property
    def pending_count(self) -> int:
        return sum(len(a.pending) for a in self._accounts.values())
