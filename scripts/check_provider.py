#!/usr/bin/env python3
"""Answer one question: which classification provider will
scripts/replay_eval_dataset.py (or any live /reconcile call) actually use
right now, on this machine?

Why this exists: GET /health's "classification_mode" field only checks
whether ANTHROPIC_API_KEY is set -- it does NOT reflect
_resolve_provider()'s real priority order, which checks local Ollama
FIRST and prefers it automatically whenever reachable, even with a valid
Anthropic key configured. So /health can say "llm" while every real
classification call is actually going to local Ollama. This script calls
the real resolution + liveness-probe logic directly, so there's no
guessing.

Run with:
    python scripts/check_provider.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# app/ isn't an installed package (no pyproject [project] / editable install --
# same reason eval/run_eval.py does this) -- running this as `python
# scripts/check_provider.py` puts scripts/ on sys.path, not the repo root, so
# `import app...` fails with ModuleNotFoundError without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.classification_agent import _resolve_provider, ollama_reachable, classify_transaction  # noqa: E402
from app.config import settings  # noqa: E402
from app.models.transaction import Transaction  # noqa: E402


def main() -> None:
    print(f"OLLAMA_BASE_URL       = {settings.ollama_base_url}")
    print(f"OLLAMA_MODEL           = {settings.ollama_model}")
    print(f"ANTHROPIC_API_KEY set  = {bool(settings.anthropic_api_key)}")
    print(f"LLM_PROVIDER override  = {settings.llm_provider!r}")
    print()

    reachable = ollama_reachable(settings.ollama_base_url)
    print(f"Ollama reachable at {settings.ollama_base_url}: {reachable}")

    provider = _resolve_provider()
    print(f"_resolve_provider() resolves to: {provider!r}")
    print()

    if provider != "heuristic":
        print(f"Timing one live classification call via {provider!r}...")
        txn = Transaction(
            transaction_id="check_provider_probe",
            source="processor_ledger",
            merchant_id="check_provider_probe_merchant",
            amount=42.00,
            description="Test charge for provider check",
            timestamp="2026-01-01T00:00:00Z",
        )
        start = time.monotonic()
        result = classify_transaction(txn)
        elapsed = time.monotonic() - start
        print(f"  -> {result.transaction_type} (confidence {result.confidence}) in {elapsed:.1f}s")
        print()
        print(f"A 50-transaction batch at this rate: ~{elapsed * 50:.0f}s ({elapsed * 50 / 60:.1f} min)")
    else:
        print("Heuristic mode -- classification is instant, timeouts shouldn't happen.")


if __name__ == "__main__":
    main()
