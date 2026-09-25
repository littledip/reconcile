#!/usr/bin/env python3
"""Week 9-10 demo/verification script: episodic memory (app/memory_store.py).

Neither pytest (mock-mode, isolated per test via tests/conftest.py's
_isolate_chroma fixture) nor the write-tools demo script
(scripts/demo_escalation_queue.py, which resolves anomalies but never looks
at what got remembered) show episodic memory actually doing its job: an
anomaly resolved today should change what a *later*, similar anomaly sees.
This script makes that visible, mirroring demo_cross_batch_lookup.py's
two-batches-back-to-back approach:

  1. Resolves one escalated duplicate_charge anomaly ("approved"), which
     -- per app/mcp_server.py's submit_reconciliation_decision -- writes it
     into episodic memory as a side effect of the same call that writes its
     Mongo audit record.
  2. Runs a second, *different* batch with a similar-but-distinct
     duplicate_charge anomaly (different merchant, amount, order id) through
     the full orchestrator, and shows that reasoning_node
     (app/agents/reasoning_agent.py) attached `episodic_context` citing
     step 1's resolution -- the Reasoning Agent side of the "both
     consumers" design decision.
  3. Calls find_similar_episodes() directly with that same anomaly's
     context -- the same in-process call GET /escalations/{anomaly_id}/similar
     makes -- to show the on-demand, human-reviewer side of that same
     decision.

Persists to the real ./chroma_data directory (or CHROMA_PERSIST_DIR, if
set) like any real run -- not the disposable per-test directory pytest
uses -- so episodes accumulate across runs of this script the same way
persisted transactions already do across runs of run_orchestrator.py /
demo_cross_batch_lookup.py.

Usage:
    python3 scripts/demo_episodic_memory.py
"""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import run_reconciliation_batch
from app.mcp_client import lookup_session, submit_reconciliation_decision
from app.memory_store import find_similar_episodes
from app.models.transaction import Transaction, TransactionSource


def _txn(**overrides) -> Transaction:
    base = dict(source=TransactionSource.PROCESSOR_LEDGER, currency="USD", description=None)
    base.update(overrides)
    return Transaction.model_validate(base)


async def main() -> None:
    print("=== Step 1: resolve an escalated anomaly, seeding one episode ===")
    batch1 = [
        _txn(
            transaction_id="demo_ep_dup_1", merchant_id="demo_merchant_a", amount=Decimal("42.00"),
            timestamp="2026-09-01T00:00:00Z", description="Card payment - order #7001",
        ),
        _txn(
            transaction_id="demo_ep_dup_2", merchant_id="demo_merchant_a", amount=Decimal("42.00"),
            timestamp="2026-09-01T00:05:00Z", description="Card payment - order #7001 (retry)",
        ),
    ]
    result1 = await run_reconciliation_batch(batch1)
    if not result1["escalations"]:
        print("  ** nothing escalated -- unexpected, aborting demo **")
        return
    anomaly_id = result1["escalations"][0]
    print(f"  escalated: {anomaly_id}")

    async with lookup_session() as session:
        await submit_reconciliation_decision(
            session,
            anomaly_id=anomaly_id,
            decision="approved",
            decided_by="demo_script",
            notes="Confirmed duplicate, refunded.",
        )
    print(f"  approved {anomaly_id} -- written into episodic memory as a side effect.\n")

    print("=== Step 2: a similar-but-distinct anomaly, in a fresh batch ===")
    batch2 = [
        _txn(
            transaction_id="demo_ep_dup_3", merchant_id="demo_merchant_b", amount=Decimal("44.00"),
            timestamp="2026-09-05T00:00:00Z", description="Card payment - order #7050",
        ),
        _txn(
            transaction_id="demo_ep_dup_4", merchant_id="demo_merchant_b", amount=Decimal("44.00"),
            timestamp="2026-09-05T00:05:00Z", description="Card payment - order #7050 (retry)",
        ),
    ]
    result2 = await run_reconciliation_batch(batch2)
    duplicate_anomalies = [a for a in result2["anomalies"] if a["anomaly_type"] == "duplicate_charge"]
    if not duplicate_anomalies:
        print("  ** no duplicate_charge anomaly found in batch 2 -- unexpected, aborting demo **")
        return
    anomaly2 = duplicate_anomalies[0]
    print(f"  anomaly: {anomaly2['anomaly_id']} ({anomaly2['anomaly_type']}, severity={anomaly2['severity']})")
    if anomaly2["episodic_context"]:
        print(f"  episodic_context (reasoning_node's own grounding, Reasoning Agent consumer): {anomaly2['episodic_context']}")
    else:
        print(
            "  ** no episodic_context attached -- either step 1's episode fell outside the "
            "similarity cutoff, or something regressed **"
        )

    print("\n=== Step 3: the same query via find_similar_episodes() directly (the on-demand, human-reviewer path -- GET /escalations/{id}/similar) ===")
    similar = find_similar_episodes(
        anomaly_type=anomaly2["anomaly_type"],
        severity=anomaly2["severity"],
        reasoning=anomaly2["reasoning"],
    )
    if not similar:
        print("  (none returned)")
    for s in similar:
        print(f"  {s['anomaly_id']}: {s['decision']} by {s['decided_by']} (distance={s['distance']:.3f}) -- {s['notes']}")


if __name__ == "__main__":
    asyncio.run(main())
