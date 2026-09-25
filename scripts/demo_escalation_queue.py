#!/usr/bin/env python3
"""Week 7-8 demo/verification script: the escalation queue + audit trail
(the write-tools half of Week 7-8's MCP server).

demo_cross_batch_lookup.py exercises the one *read* tool
(lookup_transaction_by_reference). Nothing in this repo exercised the three
*write* tools -- escalate_dispute, list_pending_escalations,
submit_reconciliation_decision -- against real infra end to end until the
Sept 23 real-infra verification session, and that pass was a manual
run_orchestrator.py + curl round trip that left no reusable script behind.
This one does:

  1. Runs the full sample_transactions.json batch through the orchestrator
     (the same data source scripts/run_orchestrator.py uses) -- its
     escalate_node pushes every anomaly with requires_human_review=True
     onto the live Redis escalation_queue via escalate_dispute.
  2. Lists the queue (list_pending_escalations) and prints it -- "before".
  3. Approves the first escalated anomaly and defers the second, via
     submit_reconciliation_decision -- deliberately supplying only
     anomaly_id/decision/decided_by/notes, the same minimal shape POST
     /escalations/{anomaly_id}/decision gets from a human reviewer, so
     this also exercises the tool's Redis-queue-entry enrichment path
     (anomaly_type/transaction_ids/reasoning pulled in from what
     escalate_dispute wrote) rather than just the fully-specified call
     lookup_missing_records_node already makes for auto-resolutions.
  4. Lists the queue again -- "after": the approved anomaly is gone, the
     deferred one is still there ("deferred" is a real audit event, not a
     close-out -- see ReconciliationDecision's docstring).
  5. Reads both permanent audit records back from Mongo's
     reconciliation_decisions collection and prints them.

Needs at least two anomalies with requires_human_review=True in the batch
to demo both approve and defer -- sample_transactions.json currently
produces exactly two (a duplicate_charge and a missing_record). If a future
edit to that fixture leaves fewer than two, the script says so and demos
whatever it can rather than failing outright.

Run this against real Docker Mongo+Redis (MONGO_URI/REDIS_URL set in .env)
to confirm the real stdio MCP subprocess path and a real Redis queue -- the
mock-mode fallback (mongomock/fakeredis) produces the same *results* but
never leaves the current process, so it can't catch anything that only
shows up crossing those real boundaries (see app/mcp_client.py,
app/cache.py).

Usage:
    python3 scripts/demo_escalation_queue.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import run_reconciliation_batch
from app.cache import using_mock_redis
from app.db import db, using_mock_mongo
from app.mcp_client import list_pending_escalations, lookup_session, submit_reconciliation_decision
from app.models.transaction import Transaction

DATA_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "sample_transactions.json"


def _print_queue(label: str, escalations: list[dict]) -> None:
    print(f"  {label}: {len(escalations)} pending")
    for e in escalations:
        print(f"    - {e['anomaly_id']}: {e['anomaly_type']} (severity={e['severity']})")


async def main() -> None:
    mongo_backend = "mongomock (in-memory)" if using_mock_mongo() else "real MongoDB"
    redis_backend = "fakeredis (in-memory)" if using_mock_redis() else "real Redis"
    print(f"Mongo backend: {mongo_backend}")
    print(f"Redis backend: {redis_backend}\n")

    print("=== Step 1: run the sample batch through the orchestrator ===")
    raw = json.loads(DATA_PATH.read_text())
    transactions = [Transaction.model_validate(t) for t in raw]
    result = await run_reconciliation_batch(transactions)
    escalated_ids = result["escalations"]
    print(f"  escalate_node pushed {len(escalated_ids)} anomaly(ies) onto the live queue: {escalated_ids}\n")

    if not escalated_ids:
        print("Nothing escalated -- nothing to demo. (Has sample_transactions.json changed?)")
        return

    async with lookup_session() as session:
        print("=== Step 2: queue before any decisions ===")
        before = await list_pending_escalations(session)
        _print_queue("before", before)
        print()

        to_approve = escalated_ids[0]
        to_defer = escalated_ids[1] if len(escalated_ids) > 1 else None

        print(f"=== Step 3: approve {to_approve}, defer {to_defer or '(only one escalated -- skipping)'} ===")
        approve_result = await submit_reconciliation_decision(
            session,
            anomaly_id=to_approve,
            decision="approved",
            decided_by="demo_script",
            notes="Verified via demo_escalation_queue.py",
        )
        print(f"  {to_approve}: {approve_result}")

        if to_defer:
            defer_result = await submit_reconciliation_decision(
                session,
                anomaly_id=to_defer,
                decision="deferred",
                decided_by="demo_script",
                notes="Verified via demo_escalation_queue.py",
            )
            print(f"  {to_defer}: {defer_result}")
        print()

        print("=== Step 4: queue after decisions ===")
        after = await list_pending_escalations(session)
        _print_queue("after", after)
        remaining_ids = {e["anomaly_id"] for e in after}
        assert to_approve not in remaining_ids, f"{to_approve} should have been removed (approved)"
        if to_defer:
            assert to_defer in remaining_ids, f"{to_defer} should still be queued (deferred, not resolved)"
        print("  confirmed: approved anomaly removed from the queue, deferred anomaly still sitting in it.\n")

    print("=== Step 5: permanent audit records in Mongo (reconciliation_decisions) ===")
    ids_to_check = [to_approve] + ([to_defer] if to_defer else [])
    for record in db["reconciliation_decisions"].find({"anomaly_id": {"$in": ids_to_check}}):
        record.pop("_id", None)
        print(f"  {json.dumps(record, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
