#!/usr/bin/env python3
"""CLI runner for the orchestrator: classify -> match -> detect anomalies ->
lookup missing records (Week 7-8 MCP lookup) -> reason about anomalies
(Week 5-6 graph-grounded enrichment) -> maybe escalate -> persist batch
(Week 7-8), run over the full sample_transactions.json batch.

Usage:
    python3 scripts/run_orchestrator.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import run_reconciliation_batch
from app.config import settings
from app.models.transaction import Transaction

DATA_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "sample_transactions.json"


async def main() -> None:
    """Run the full orchestrator over sample_transactions.json and print results."""
    mode = "LLM (Claude)" if settings.anthropic_api_key else "heuristic (no ANTHROPIC_API_KEY set)"
    print(f"Classification mode: {mode}\n")

    raw = json.loads(DATA_PATH.read_text())
    transactions = [Transaction.model_validate(t) for t in raw]

    result = await run_reconciliation_batch(transactions)

    print("=== Classifications ===")
    for tid, c in result["classifications"].items():
        print(f"  {tid}: {c['transaction_type']} (confidence={c['confidence']:.2f})")

    print("\n=== Matches ===")
    if not result["matches"]:
        print("  (none)")
    for m in result["matches"]:
        print(f"  {m['match_id']}: {m['match_type']} — {m['transaction_ids']} (confidence={m['confidence']:.2f})")
        print(f"      {m['reasoning']}")

    print("\n=== Anomalies ===")
    if not result["anomalies"]:
        print("  (none)")
    for a in result["anomalies"]:
        print(
            f"  {a['anomaly_id']}: {a['anomaly_type']} — {a['transaction_ids']} "
            f"(severity={a['severity']}, confidence={a['confidence']:.2f})"
        )
        print(f"      {a['reasoning']}")

    print("\n=== Escalations (guardrail routing) ===")
    if result["escalations"]:
        print(f"  Routed to escalate node — {len(result['escalations'])} anomaly(ies) need human review:")
        for eid in result["escalations"]:
            print(f"    - {eid}")
    else:
        print("  None flagged — orchestrator routed straight to finish.")


if __name__ == "__main__":
    asyncio.run(main())
