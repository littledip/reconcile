"""Tests for the Week 7-8 MCP lookup piece: app/retry.py, the
lookup_transaction_by_reference tool itself (app/mcp_server.py), and the
client wrapper that adds retry/backoff around it (app/mcp_client.py).

Runs in mock mode (MONGO_URI unset -> mongomock), which means
app.mcp_client's in-process fallback is the code path actually exercised
here rather than the real stdio subprocess -- see app/mcp_client.py's
module docstring for why that fallback exists and what it stands in for.
The assertions are identical either way: lookup_transaction_by_reference
(the client function) only depends on its session argument exposing
call_tool(name, arguments) -> object with .isError/.content/
.structuredContent, which both _InProcessSession and the real
ClientSession satisfy.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from app.db import db
from app.mcp_client import lookup_session, lookup_transaction_by_reference
from app.mcp_server import lookup_transaction_by_reference as lookup_tool
from app.models.transaction import Transaction, TransactionSource
from app.retry import RetryExhaustedError, with_retry


def _txn(**overrides) -> Transaction:
    base = dict(
        transaction_id="txn_x",
        source=TransactionSource.PROCESSOR_LEDGER,
        merchant_id="merch_1",
        amount=Decimal("40.00"),
        currency="USD",
        timestamp="2026-08-10T00:00:00Z",
        description="Card payment - no order ref",
    )
    base.update(overrides)
    return Transaction.model_validate(base)


def _seed(**fields) -> None:
    """Insert a raw record into the persisted transactions collection,
    mirroring what persist_batch_node would have written for a prior batch.
    """
    base = dict(
        transaction_id="prior_1",
        merchant_id="merch_1",
        amount="40.00",
        timestamp="2026-08-01T00:00:00+00:00",
        description=None,
    )
    base.update(fields)
    db["transactions"].insert_one(base)


# --- app/retry.py -----------------------------------------------------------


def test_with_retry_succeeds_first_try():
    calls = []

    async def op():
        calls.append(1)
        return "ok"

    result = asyncio.run(with_retry(op, max_attempts=3, base_delay=0.001))
    assert result == "ok"
    assert len(calls) == 1


def test_with_retry_succeeds_after_transient_failures():
    calls = []

    async def op():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"

    result = asyncio.run(with_retry(op, max_attempts=5, base_delay=0.001))
    assert result == "ok"
    assert len(calls) == 3


def test_with_retry_exhausts_and_raises():
    calls = []

    async def op():
        calls.append(1)
        raise RuntimeError("always fails")

    with pytest.raises(RetryExhaustedError) as exc_info:
        asyncio.run(with_retry(op, max_attempts=3, base_delay=0.001))

    assert len(calls) == 3
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, RuntimeError)


# --- app/mcp_server.py: lookup_transaction_by_reference tool logic ----------


def test_lookup_tool_finds_duplicate_via_order_id_and_amount():
    _seed(transaction_id="prior_dup", merchant_id="merch_5", amount="99.99", description="order #58213")

    result = lookup_tool(
        transaction_id="txn_new",
        merchant_id="merch_5",
        amount="99.99",
        timestamp="2026-08-10T00:00:00Z",
        description="order #58213 (retry)",
    )

    assert result["found"] is True
    assert result["matched_transaction_id"] == "prior_dup"
    assert result["match_type"] == "duplicate_pair"


def test_lookup_tool_finds_refund_via_equal_and_opposite_amount():
    _seed(transaction_id="prior_payment", merchant_id="merch_6", amount="40.00", timestamp="2026-08-01T00:00:00+00:00")

    result = lookup_tool(
        transaction_id="txn_refund",
        merchant_id="merch_6",
        amount="-40.00",
        timestamp="2026-08-15T00:00:00Z",  # 14 days later, within the 30-day window
        description="Refund - no order ref",
    )

    assert result["found"] is True
    assert result["matched_transaction_id"] == "prior_payment"
    assert result["match_type"] == "refund_pair"


def test_lookup_tool_no_match_outside_window():
    _seed(transaction_id="prior_payment", merchant_id="merch_7", amount="40.00", timestamp="2026-06-01T00:00:00+00:00")

    result = lookup_tool(
        transaction_id="txn_refund",
        merchant_id="merch_7",
        amount="-40.00",
        timestamp="2026-08-15T00:00:00Z",  # ~75 days later, outside the 30-day window
        description="Refund - no order ref",
    )

    assert result["found"] is False
    assert result["matched_transaction_id"] is None


def test_lookup_tool_no_match_different_merchant():
    _seed(transaction_id="prior_payment", merchant_id="merch_other", amount="40.00")

    result = lookup_tool(
        transaction_id="txn_refund",
        merchant_id="merch_8",
        amount="-40.00",
        timestamp="2026-08-10T00:00:00Z",
        description="Refund - no order ref",
    )

    assert result["found"] is False


def test_lookup_tool_excludes_itself():
    """A transaction should never match against its own persisted record
    (relevant once a batch's own transactions are persisted after it runs).
    """
    _seed(transaction_id="txn_self", merchant_id="merch_9", amount="40.00")

    result = lookup_tool(
        transaction_id="txn_self",
        merchant_id="merch_9",
        amount="40.00",
        timestamp="2026-08-10T00:00:00Z",
        description=None,
    )

    assert result["found"] is False


# --- app/mcp_client.py: session + retry-wrapped client call -----------------


def test_client_lookup_session_reports_found():
    _seed(transaction_id="prior_payment", merchant_id="merch_10", amount="40.00", timestamp="2026-08-01T00:00:00+00:00")
    refund = _txn(transaction_id="refund_1", merchant_id="merch_10", amount=Decimal("-40.00"), timestamp="2026-08-10T00:00:00Z")

    async def run():
        async with lookup_session() as session:
            return await lookup_transaction_by_reference(session, refund)

    result = asyncio.run(run())
    assert result["found"] is True
    assert result["matched_transaction_id"] == "prior_payment"


def test_client_lookup_session_reports_not_found():
    txn = _txn(transaction_id="lonely_1", merchant_id="merch_11")

    async def run():
        async with lookup_session() as session:
            return await lookup_transaction_by_reference(session, txn)

    result = asyncio.run(run())
    assert result["found"] is False


def test_client_retries_then_raises_retry_exhausted(monkeypatch):
    """Force every call_tool invocation to fail, and confirm the client
    wrapper retries max_attempts times (via app/retry.py) before giving up
    -- exercising the retry/backoff integration without needing a slow,
    real network timeout to do it.
    """
    from app import mcp_client

    calls = []

    class _AlwaysFailsSession:
        async def call_tool(self, name, arguments):
            calls.append(1)
            raise RuntimeError("simulated transport failure")

    txn = _txn(transaction_id="txn_fail")

    async def run():
        return await mcp_client.lookup_transaction_by_reference(
            _AlwaysFailsSession(), txn, max_attempts=3
        )

    # with_retry's default base_delay is fine here since max_attempts=3 and
    # base_delay=0.25 -> total sleep ~0.75s, acceptable for a single test.
    with pytest.raises(RetryExhaustedError):
        asyncio.run(run())

    assert len(calls) == 3
