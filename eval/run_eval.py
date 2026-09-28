#!/usr/bin/env python3
"""Eval framework: v0.1 runner.

Loads eval/eval_dataset.json, runs its 3 batches through the *real*
orchestrator (app/agents/orchestrator.py) in run_order, and scores the
result against ground truth. This is the first cut of the eval framework's
regression gate -- it covers the deterministic dimensions we've designed so
far (classification, matching, anomaly detection, auto-resolution,
graph-RAG citation presence). NOT yet included, by design -- still open
design questions per Progress_Log.md:

  - Drift (comparing this run's scores against a stored baseline)
  - Latency measurement
  - Episodic-memory grounding (deferred to v2, per the design session)
  - A polished report format / CI wiring

Forces heuristic-mode classification for determinism, same as the pytest
suite (see tests/conftest.py) -- this eval is about whether the pipeline's
matching/anomaly/resolution *logic* is correct given a known classification,
not about live model quality. Swapping in a real classification provider to
measure classification accuracy itself is a natural extension once that's
designed (see the Sept 28 design-session notes on Qwen vs. Claude providers).

Run with: python eval/run_eval.py
Needs your real venv (langgraph + whatever Mongo/Redis backend is
configured -- mongomock/fakeredis by default, no Docker required).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "heuristic"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import run_reconciliation_batch  # noqa: E402
from app.db import db  # noqa: E402
from app.models.transaction import Transaction  # noqa: E402

import json  # noqa: E402


def load_dataset(path: str) -> dict:
    return json.loads(Path(path).read_text())


async def run_batch(batch: dict) -> dict:
    txns = [Transaction.model_validate(t) for t in batch["transactions"]]
    state = await run_reconciliation_batch(txns)
    return state


def check_resolution(txn_id: str) -> bool:
    """True if a permanent auto_resolved_via_lookup decision exists for this
    transaction -- the audit record lookup_missing_records_node writes
    (app/agents/orchestrator.py) right before dropping a resolved anomaly.
    """
    return db["reconciliation_decisions"].find_one(
        {"decision": "auto_resolved_via_lookup", "transaction_ids": txn_id}
    ) is not None


async def main() -> None:
    dataset = load_dataset("eval/eval_dataset.json")
    gt = dataset["ground_truth"]

    results: dict[str, dict[str, int]] = {}  # scenario -> {"pass": n, "fail": n}
    failures: list[str] = []

    def record(scenario: str, ok: bool, detail: str) -> None:
        bucket = results.setdefault(scenario, {"pass": 0, "fail": 0})
        bucket["pass" if ok else "fail"] += 1
        if not ok:
            failures.append(detail)

    for batch in sorted(dataset["batches"], key=lambda b: b["run_order"]):
        print(f"--- running {batch['batch_id']} (run_order {batch['run_order']}) ---")
        state = await run_batch(batch)

        final_anomalies_by_txn = {}
        for a in state["anomalies"]:
            for tid in a["transaction_ids"]:
                final_anomalies_by_txn[tid] = a

        for raw_txn in batch["transactions"]:
            txn_id = raw_txn["transaction_id"]
            entry = gt[txn_id]
            scenario = entry["scenario"]

            # Classification
            actual_type = state["classifications"][txn_id]["transaction_type"]
            record(
                f"{scenario}::classification", actual_type == entry["expected_type"],
                f"{txn_id} ({scenario}): expected_type={entry['expected_type']!r} actual={actual_type!r}",
            )

            # Anomaly + resolution
            expected_anomaly = entry.get("expected_anomaly")
            expected_resolution = entry.get("expected_resolution")
            actual_anomaly = final_anomalies_by_txn.get(txn_id)

            if expected_resolution is not None:
                if expected_resolution["outcome"] == "auto_resolved":
                    resolved = check_resolution(txn_id)
                    ok = resolved and actual_anomaly is None
                    record(f"{scenario}::resolution", ok, f"{txn_id} ({scenario}): expected auto_resolved, resolved={resolved}, still_open={actual_anomaly is not None}")
                else:  # not_yet_resolved / unresolved -- anomaly should still be open
                    ok = actual_anomaly is not None and actual_anomaly["anomaly_type"] == expected_anomaly["anomaly_type"]
                    record(f"{scenario}::resolution", ok, f"{txn_id} ({scenario}): expected still-open, actual={actual_anomaly}")
            elif expected_anomaly is None:
                ok = actual_anomaly is None
                record(f"{scenario}::anomaly", ok, f"{txn_id} ({scenario}): expected no anomaly, actual={actual_anomaly}")
            else:
                ok = (
                    actual_anomaly is not None
                    and actual_anomaly["anomaly_type"] == expected_anomaly["anomaly_type"]
                    and actual_anomaly["severity"] == expected_anomaly["severity"]
                )
                record(f"{scenario}::anomaly", ok, f"{txn_id} ({scenario}): expected={expected_anomaly}, actual={actual_anomaly}")

    print("\n=== per-scenario results ===")
    total_pass = total_fail = 0
    for scenario in sorted(results):
        p, f = results[scenario]["pass"], results[scenario]["fail"]
        total_pass += p
        total_fail += f
        marker = "OK" if f == 0 else "FAIL"
        print(f"  [{marker}] {scenario}: {p} passed, {f} failed")

    print(f"\nTOTAL: {total_pass} passed, {total_fail} failed ({total_pass}/{total_pass + total_fail})")
    if failures:
        print("\n=== failure detail ===")
        for f in failures:
            print(f"  {f}")


if __name__ == "__main__":
    asyncio.run(main())
