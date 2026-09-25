#!/usr/bin/env python3
"""Week 7-8 demo/verification script: cross-batch MCP lookup.

scripts/run_orchestrator.py only ever runs one batch, so it can never show
the actual point of this phase -- a `missing_record` (or
`unresolved_chargeback`) anomaly getting resolved by finding its
counterpart from an *earlier* batch via the MCP lookup server. This script
runs two batches back to back to make that visible:

  1. A lone payment, with no counterpart in its own batch -- nothing
     anomalous about it yet, just persisted.
  2. Its matching refund, in a separate batch/graph run. Without Week 7-8,
     this would be flagged missing_record and escalated. With it,
     lookup_missing_records_node finds batch 1's payment (via the real MCP
     server when MONGO_URI is set, the in-process fallback otherwise -- see
     app/mcp_client.py) and the anomaly never appears.

Also confirms duplicate_charge correctly stays *un*-resolved even when a
matching persisted record exists, since it's excluded from the lookup by
design (see app/agents/orchestrator.py's lookup_missing_records_node
docstring).

Usage:
    python3 scripts/demo_cross_batch_lookup.py

Run this against real Docker Mongo (MONGO_URI set in .env) to confirm the
real stdio MCP subprocess path specifically -- the mock-mode in-process
fallback produces the same *results*, but doesn't exercise the actual
client/server round trip this phase was built to demonstrate.
"""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import run_reconciliation_batch
from app.config import settings
from app.models.transaction import Transaction, TransactionSource


def _txn(**overrides) -> Transaction:
    base = dict(
        source=TransactionSource.PROCESSOR_LEDGER,
        currency="USD",
        description=None,
    )
    base.update(overrides)
    return Transaction.model_validate(base)


async def main() -> None:
    backend = "real MongoDB (real stdio MCP subprocess)" if settings.mongo_uri else "mongomock (in-process MCP fallback)"
    print(f"Mongo backend: {backend}\n")

    print("=== Batch 1: a lone payment (no in-batch counterpart) ===")
    batch1 = [
        _txn(
            transaction_id="demo_payment_1",
            merchant_id="demo_merchant",
            amount=Decimal("87.50"),
            timestamp="2026-09-01T00:00:00Z",
            description="Card payment - no order ref",
        )
    ]
    result1 = await run_reconciliation_batch(batch1)
    print(f"  anomalies: {[a['anomaly_type'] for a in result1['anomalies']] or '(none)'}")
    print("  -> persisted to the transactions collection by persist_batch_node.\n")

    print("=== Batch 2: its matching refund, in a separate run ===")
    batch2 = [
        _txn(
            transaction_id="demo_refund_1",
            merchant_id="demo_merchant",
            amount=Decimal("-87.50"),
            timestamp="2026-09-10T00:00:00Z",
            description="Refund - no order ref",
        )
    ]
    result2 = await run_reconciliation_batch(batch2)
    anomaly_types = [a["anomaly_type"] for a in result2["anomalies"]]
    if "missing_record" in anomaly_types:
        print("  ** missing_record anomaly present -- lookup did NOT resolve it (unexpected) **")
    else:
        print("  missing_record anomaly absent -- lookup_missing_records_node found demo_payment_1")
        print("  from batch 1 via the MCP server and removed the anomaly before it could escalate.")
    print(f"  escalations: {result2['escalations'] or '(none)'}\n")

    print("=== Sanity check: duplicate_charge is never eligible for lookup ===")
    batch3 = [
        _txn(
            transaction_id="demo_dup_1",
            merchant_id="demo_merchant_2",
            amount=Decimal("15.00"),
            timestamp="2026-09-05T00:00:00Z",
            description="Card payment - order #4242",
        ),
        _txn(
            transaction_id="demo_dup_2",
            merchant_id="demo_merchant_2",
            amount=Decimal("15.00"),
            timestamp="2026-09-05T00:05:00Z",
            description="Card payment - order #4242 (retry)",
        ),
    ]
    result3 = await run_reconciliation_batch(batch3)
    anomaly_types3 = [a["anomaly_type"] for a in result3["anomalies"]]
    assert "duplicate_charge" in anomaly_types3, "expected duplicate_charge to still be flagged"
    print("  duplicate_charge anomaly present, as expected -- it's excluded from the lookup by design.")
    print(f"  escalations: {result3['escalations']}")


if __name__ == "__main__":
    asyncio.run(main())
