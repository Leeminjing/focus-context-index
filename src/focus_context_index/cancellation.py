"""Finish required cleanup before propagating any number of caller cancellations."""

import asyncio


async def complete_before_cancel(awaitable):
    """Keep a strong reference and shield every wait, including waits after cancellation.

    This does not make I/O durable or suppress failures. The host must acknowledge
    settlement; until then the caller remains in cleanup. Cancellation of the child
    itself still propagates, and failures are reported rather than silently lost.
    """
    future = asyncio.ensure_future(awaitable)
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(future)
            break
        except asyncio.CancelledError as exc:
            if future.cancelled():
                raise
            cancelled = exc
            if future.done():
                result = future.result()
                break
    if cancelled is not None:
        raise cancelled
    return result
