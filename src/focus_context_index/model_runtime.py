"""Admit, schedule and settle every structured invocation, including failed retries."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from .api import AttemptReceipt
from .cancellation import complete_before_cancel
from .errors import IndexBudgetExceeded, StructuredResultValidationError
from .ports import ModelCall
from .sizing import estimate_raw_tokens


class _ResponseFailure(Exception):
    """Only errors from response decoding/validation may enter the retry loop."""

    def __init__(self, cause):
        self.cause = cause


class ManagedModel:
    def __init__(
        self,
        *,
        adapter,
        role,
        identity,
        snapshot,
        authorization,
        policy,
        slots,
        admission,
        receipts,
        phase,
        background,
    ):
        self._adapter = adapter
        self._role = role
        self._identity = identity
        self._snapshot = snapshot
        self._authorization = authorization
        self._policy = policy
        self._slots = slots
        self._admission = admission
        self._receipts = receipts
        self._phase = phase
        self._background = background
        self.last_attempt_records = ()
        self._sizing_model = self._new_model()

    def _new_model(self):
        model = self._adapter.create(self._role)
        if getattr(model, "cache_identity", None) != self._identity:
            raise ValueError("model configuration changed during frozen build")
        return model

    def _call(self, schema, system, payload, *, call_id="preview", attempt=1):
        text = json.dumps(
            {"schema": schema.model_json_schema(), "system": system, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        # This is the same text heuristic as segmentation. A host model may supply
        # its own full-request estimate to include provider-specific framing.
        estimator = getattr(self._sizing_model, "estimate_request_tokens", None)
        estimated = (
            estimator(schema, system, payload) if estimator else estimate_raw_tokens(text, 2)
        )
        if not isinstance(estimated, int) or estimated < 0:
            raise ValueError("model request estimate must be a nonnegative integer")
        configured_output = getattr(self._sizing_model, "max_output_tokens", None)
        output_reserve = max(
            self._policy.output_token_reserve,
            configured_output
            if isinstance(configured_output, int) and configured_output > 0
            else 0,
        )
        return ModelCall(
            call_id=call_id,
            snapshot=self._snapshot,
            authorization=self._authorization,
            phase=self._phase,
            estimated_input_tokens=int(estimated),
            output_token_reserve=output_reserve,
            attempt=attempt,
        )

    def fits_request(self, schema, system, payload):
        call = self._call(schema, system, payload)
        limit = self._policy.max_request_input_tokens
        if limit is not None and call.estimated_input_tokens > limit:
            return False
        window = getattr(self._sizing_model, "context_window_tokens", None)
        return window is None or call.estimated_input_tokens + call.output_token_reserve <= window

    def fits_available_budget(self, schema, system, payload):
        if not self.fits_request(schema, system, payload):
            return False
        preview = getattr(self._admission, "can_reserve", None)
        return preview is None or preview(self._call(schema, system, payload))

    def require_fit(self, schema, system, payload):
        if not self.fits_request(schema, system, payload):
            raise StructuredResultValidationError(
                "interpretation_read_window",
                "requested complete original window exceeds capacity",
                unit_identity="interpretation-read-window",
                violated_rule="inventory, selected originals, schema and output reserve must fit",
            )

    async def invoke(self, schema, system, payload):
        return await self.invoke_validated(schema, system, payload, lambda value: None)

    async def invoke_validated(self, schema, system, payload, validator):
        self.last_attempt_records = ()
        request = payload
        for attempt in range(self._policy.max_attempts):
            try:
                return await self._attempt(schema, system, request, validator, attempt=attempt + 1)
            except _ResponseFailure as failure:
                exc = failure.cause
                if not getattr(exc, "retryable", True) or attempt + 1 >= self._policy.max_attempts:
                    raise exc from failure
                request = {**payload, "validation_feedback": str(exc)[:1500]}
        raise RuntimeError("structured invocation did not execute")

    async def _attempt(self, schema, system, payload, validator, *, attempt):
        if not self.fits_request(schema, system, payload):
            raise IndexBudgetExceeded("complete model request exceeds input/window capacity")
        queued = perf_counter()
        async with self._slots.slot(background=self._background):
            queue_ms = (perf_counter() - queued) * 1000
            call = self._call(schema, system, payload, call_id=uuid4().hex, attempt=attempt)
            model = self._new_model()
            await self._admission.reserve(call)
            started = perf_counter()
            outcome, error_type = "error", None
            try:
                try:
                    raw = await model.invoke(schema, system, payload)
                    result = schema.model_validate(
                        raw.model_dump(mode="json") if isinstance(raw, BaseModel) else raw
                    )
                    validator(result)
                except (
                    json.JSONDecodeError,
                    ValidationError,
                    StructuredResultValidationError,
                ) as exc:
                    raise _ResponseFailure(exc) from exc
                outcome = "success"
                return result
            except asyncio.CancelledError:
                outcome, error_type = "cancelled", "CancelledError"
                raise
            except BaseException as exc:
                error_type = type(exc.cause if isinstance(exc, _ResponseFailure) else exc).__name__
                raise
            finally:
                rows = tuple(getattr(model, "last_attempt_records", ()))
                counts = [row["model_calls"] for row in rows if "model_calls" in row]
                calls = (
                    sum(counts)
                    if counts and all(isinstance(v, int) and v >= 0 for v in counts)
                    else 1
                )
                receipt = AttemptReceipt(
                    call_id=call.call_id,
                    owner_id=self._authorization.owner_id,
                    grant_id=self._authorization.grant_id,
                    namespace=self._snapshot.namespace,
                    context_id=self._snapshot.source.context_id,
                    revision_id=self._snapshot.source.revision_id,
                    phase=self._phase,
                    model_identity=self._identity,
                    outcome=outcome,
                    model_calls=calls,
                    input_tokens=_tokens(rows, "input_tokens"),
                    output_tokens=_tokens(rows, "output_tokens"),
                    duration_ms=(perf_counter() - started) * 1000,
                    queue_ms=queue_ms,
                    error_type=error_type,
                    attempt=attempt,
                )
                self._receipts.append(receipt)
                self.last_attempt_records += rows
                await complete_before_cancel(self._admission.settle(call, receipt))


def _tokens(rows, key):
    if rows and all(row.get("model_calls") == 0 for row in rows):
        return 0
    if not rows or any(
        row.get("usage_reported") is False or not isinstance(row.get(key), int) or row[key] < 0
        for row in rows
    ):
        return None
    return sum(row[key] for row in rows)
