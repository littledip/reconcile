#!/usr/bin/env python3
"""CLI runner for the Week 1 Classification Agent prototype.

Runs every sample transaction in app/data/sample_transactions.json through
the compiled LangGraph graph and prints the classification result for each.
No API server needed — this is the fastest way to confirm the LangGraph
wiring (state -> node -> compiled graph -> invoke) actually works end to end.

Usage:
    python3 scripts/run_prototype.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.classification_agent import classify_transaction
from app.config import settings
from app.models.transaction import Transaction

DATA_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "sample_transactions.json"


def main() -> None:
    mode = "LLM (Claude)" if settings.anthropic_api_key else "heuristic (no ANTHROPIC_API_KEY set — add one to .env for real model calls)"
    print(f"Classification mode: {mode}\n")

    transactions = json.loads(DATA_PATH.read_text())

    for raw in transactions:
        txn = Transaction.model_validate(raw)
        result = classify_transaction(txn)
        print(f"{result.transaction_id}  ->  {result.transaction_type.value}  (confidence={result.confidence:.2f})")
        print(f"    {result.reasoning}\n")


if __name__ == "__main__":
    main()
