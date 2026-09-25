"""Week 7-8: retry/backoff helper -- the "recovery logic" half of the JD
language this phase is built to satisfy ("stateful, deterministic, and
fault-tolerant agent workflows").

Scoped narrowly on purpose: the MCP tool call (app/mcp_client.py) is the one
genuinely-fallible *live* call anywhere in this graph -- a subprocess that
hasn't finished starting, a dropped stdio pipe, a slow response. Every other
agent in the pipeline (classification, matching, anomaly detection,
reasoning, persistence) is deterministic and in-process; wrapping those in
retry logic would just be retrying a bug, not recovering from a transient
fault. So this helper isn't sprinkled everywhere -- it wraps exactly the one
call in app/mcp_client.py that has a real failure mode to recover from.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """Raised when every retry attempt has failed.

    Wraps the last underlying error so the caller (and, eventually, an
    audit trail) can see what actually went wrong rather than just "gave up."
    """

    def __init__(self, attempts: int, last_error: BaseException):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Gave up after {attempts} attempt(s); last error: {last_error!r}")


async def with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.25,
    backoff_factor: float = 2.0,
) -> T:
    """Call an async operation with exponential backoff.

    Retries on any Exception: at this layer there's no reliable way to tell
    a transient transport hiccup apart from a real bug in the tool, so this
    is deliberately best-effort recovery -- retry a bounded number of times
    with growing delays, then surface the failure (RetryExhaustedError)
    rather than hang forever or swallow it silently.

    delay before attempt N (N > 1) = base_delay * backoff_factor ** (N - 2)
    e.g. base_delay=0.25, backoff_factor=2.0 -> 0.25s, 0.5s, 1.0s, ...
    """
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (backoff_factor ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%r) -- retrying in %.2fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None  # loop always sets it before falling through
    raise RetryExhaustedError(max_attempts, last_error)
