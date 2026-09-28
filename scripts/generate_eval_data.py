#!/usr/bin/env python3
"""Eval framework: synthetic ground-truth dataset generator.

Produces a frozen, git-committed JSON file (eval/eval_dataset.json by
default) pairing a batch of synthetic transactions with the *correct*
expected outcome for each one -- classification type, whether/how it
matches, whether it should anomaly-flag, and (for cross-batch cases)
whether it should auto-resolve via the MCP lookup.

Design decisions locked in during the eval-framework design session
(Progress_Log.md):
  - Purpose: an ongoing regression gate, not a one-time benchmark.
  - Frozen and committed, not regenerated per run -- reviewable in diffs,
    doesn't depend on this generator staying byte-stable release to release.
  - Deliberately includes cross-batch scenarios (missing_record_earlier_batch
    / missing_record_later_batch) since that's the only way to exercise
    lookup_missing_records_node's order-dependent behavior at all.
  - Ground truth is tagged inline per-transaction with a "scenario" name
    (not derived from ID naming or field-shape, not a separate sidecar
    index) so a regression reads as "duplicate_charge_same_batch broke"
    rather than a bare aggregate percentage.
  - Classification provider used to evaluate this data is a runner concern,
    not this generator's -- see Progress_Log.md. Descriptions below are
    written to be realistic (the correct label should be inferable from
    genuine domain semantics), not reverse-engineered purely to game
    app/agents/classification_agent.py's heuristic keyword list -- it's
    incidental, not the goal, that they also satisfy it.

amount_mismatch: detection lives in app/agents/reconciliation_agent.py's
pass 2 (same merchant, opposite sign, 30-day window) -- a diff within
$1.00 is still treated as a clean refund_pair/chargeback_pair (rounding
noise, no anomaly); a diff beyond that but within 50% of the larger amount
becomes an amount_mismatch_pair match, which app/agents/anomaly_agent.py
turns into an AMOUNT_MISMATCH anomaly (severity "high" if the diff exceeds
20% of the amount, else "medium"). Anything further apart than that isn't
paired at all -- treated as two unrelated transactions rather than forced
into a mismatch.

Usage:
    python scripts/generate_eval_data.py [--out eval/eval_dataset.json]
        [--transactions-per-batch 50] [--num-batches 3]

Only --num-batches 3 / --transactions-per-batch 50 (the current defaults)
are fully supported -- SCENARIO_COUNTS below is hand-tuned for exactly that
shape (see module docstring in the design-session log for why fixed counts
were chosen over proportional scaling for v1). Regenerating overwrites the
frozen file -- only do that intentionally, and review the diff before
committing.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.reason_codes import REASON_CODES

_BASE_DATE = datetime(2026, 9, 1, 9, 0, 0)
_BATCH_DATE_OFFSET_DAYS = 7  # batches 7 days apart -- well inside the 30-day match window
_KNOWN_REASON_CODES = list(REASON_CODES.keys())

# Fixed per-batch instance counts, hand-tuned for num_batches=3 /
# transactions_per_batch=50 (see module docstring). Pulled into one place,
# per the eval-framework design session, specifically so this is the one
# spot that needs to change to expose these as CLI knobs later.
SCENARIO_COUNTS = {
    "duplicate_charge_same_batch": 2,   # pairs, per batch
    "duplicate_charge_near_miss": 1,    # pairs, per batch
    "unresolved_chargeback_known_reason": 2,   # per batch
    "unresolved_chargeback_unknown_reason": 1,  # per batch
    "missing_record_never_resolves": 2,  # per batch
    "missing_record_earlier_batch": 2,  # per applicable (batch, earlier-batch) hop
    "missing_record_later_batch": 2,    # per applicable (batch, later-batch) hop
    "amount_mismatch_clear": 2,         # pairs, per batch
    "amount_mismatch_trivial": 1,       # pairs, per batch
}


class _IdGen:
    """Monotonic counters for transaction ids, merchant ids, and order refs
    -- one counter per kind, shared across the whole generation run, so no
    two scenario instances can ever accidentally collide or cross-match.
    """

    def __init__(self) -> None:
        self._txn = itertools.count(1)
        self._merchant = itertools.count(1)
        self._order = itertools.count(90001)

    def txn_id(self, batch_num: int) -> str:
        return f"eval_b{batch_num}_t{next(self._txn):04d}"

    def merchant_id(self) -> str:
        return f"merch_eval_{next(self._merchant):04d}"

    def order_ref(self) -> str:
        return str(next(self._order))


def _mk_txn(
    txn_id: str,
    *,
    source: str,
    merchant_id: str,
    amount: Decimal,
    timestamp: datetime,
    description: str,
    raw_metadata: dict | None = None,
) -> dict:
    return {
        "transaction_id": txn_id,
        "source": source,
        "merchant_id": merchant_id,
        "amount": str(amount),
        "currency": "USD",
        "timestamp": timestamp.isoformat() + "Z",
        "description": description,
        "raw_metadata": raw_metadata or {},
    }


class Dataset:
    """Accumulates batches + ground_truth as scenario builders run."""

    def __init__(self, num_batches: int) -> None:
        self.num_batches = num_batches
        self.batches: dict[int, list[dict]] = {b: [] for b in range(1, num_batches + 1)}
        self.ground_truth: dict[str, dict] = {}

    def add_txn(self, batch_num: int, txn: dict, gt: dict) -> None:
        self.batches[batch_num].append(txn)
        self.ground_truth[txn["transaction_id"]] = gt

    def batch_id(self, batch_num: int) -> str:
        return f"eval_batch_{batch_num}"

    def batch_timestamp(self, batch_num: int, offset_minutes: int = 0) -> datetime:
        return _BASE_DATE + timedelta(days=_BATCH_DATE_OFFSET_DAYS * (batch_num - 1), minutes=offset_minutes)


# ---------------------------------------------------------------------------
# Scenario builders -- each emits its transaction(s) + ground-truth entries
# directly onto the Dataset.
# ---------------------------------------------------------------------------


def build_duplicate_charge_same_batch(ds: Dataset, ids: _IdGen, batch_num: int) -> None:
    merchant = ids.merchant_id()
    order = ids.order_ref()
    amount = Decimal("75.00")
    ts = ds.batch_timestamp(batch_num)
    original_id = ids.txn_id(batch_num)
    retry_id = ids.txn_id(batch_num)

    original = _mk_txn(
        original_id, source="processor_ledger", merchant_id=merchant, amount=amount,
        timestamp=ts, description=f"Card payment - order #{order}",
        raw_metadata={"card_last4": "1111"},
    )
    retry = _mk_txn(
        retry_id, source="processor_ledger", merchant_id=merchant, amount=amount,
        timestamp=ts + timedelta(seconds=5), description=f"Card payment - order #{order} (retry after timeout)",
        raw_metadata={"card_last4": "1111"},
    )
    shared_anomaly = {"anomaly_type": "duplicate_charge", "severity": "high"}
    ds.add_txn(batch_num, original, {
        "scenario": "duplicate_charge_same_batch", "expected_type": "payment",
        "expected_match": {"match_type": "duplicate_pair", "with": retry_id},
        "expected_anomaly": shared_anomaly,
    })
    ds.add_txn(batch_num, retry, {
        "scenario": "duplicate_charge_same_batch", "expected_type": "duplicate_charge",
        "expected_match": {"match_type": "duplicate_pair", "with": original_id},
        "expected_anomaly": shared_anomaly,
    })


def build_duplicate_charge_near_miss(ds: Dataset, ids: _IdGen, batch_num: int) -> None:
    merchant = ids.merchant_id()
    order = ids.order_ref()
    ts = ds.batch_timestamp(batch_num)
    original_id = ids.txn_id(batch_num)
    retry_id = ids.txn_id(batch_num)

    original = _mk_txn(
        original_id, source="processor_ledger", merchant_id=merchant, amount=Decimal("50.00"),
        timestamp=ts, description=f"Card payment - order #{order}",
        raw_metadata={"card_last4": "2222"},
    )
    # Same order reference, but a different amount (e.g. a surcharge on
    # retry) -- pass 1 of the matcher groups by amount within the order-id
    # group, so this deliberately does NOT form a duplicate_pair. Tests
    # that near-identical-but-not-identical retries don't get over-flagged.
    retry = _mk_txn(
        retry_id, source="processor_ledger", merchant_id=merchant, amount=Decimal("50.75"),
        timestamp=ts + timedelta(seconds=5), description=f"Card payment - order #{order} (retry after timeout, surcharge applied)",
        raw_metadata={"card_last4": "2222"},
    )
    for txn_id, txn, expected_type in ((original_id, original, "payment"), (retry_id, retry, "duplicate_charge")):
        ds.add_txn(batch_num, txn, {
            "scenario": "duplicate_charge_near_miss", "expected_type": expected_type,
            "expected_match": None, "expected_anomaly": None,
        })


def build_unresolved_chargeback_known_reason(ds: Dataset, ids: _IdGen, batch_num: int, code_cycle) -> None:
    merchant = ids.merchant_id()
    ts = ds.batch_timestamp(batch_num)
    txn_id = ids.txn_id(batch_num)
    code = next(code_cycle)
    txn = _mk_txn(
        txn_id, source="processor_ledger", merchant_id=merchant, amount=Decimal("-89.50"),
        timestamp=ts, description="Dispute filed by cardholder - unrecognized charge",
        raw_metadata={"dispute_id": f"dp_eval_{txn_id}", "reason_code": code},
    )
    ds.add_txn(batch_num, txn, {
        "scenario": "unresolved_chargeback_known_reason", "expected_type": "chargeback",
        "expected_match": None,
        "expected_anomaly": {"anomaly_type": "unresolved_chargeback", "severity": "high"},
    })


def build_unresolved_chargeback_unknown_reason(ds: Dataset, ids: _IdGen, batch_num: int) -> None:
    merchant = ids.merchant_id()
    ts = ds.batch_timestamp(batch_num)
    txn_id = ids.txn_id(batch_num)
    txn = _mk_txn(
        txn_id, source="processor_ledger", merchant_id=merchant, amount=Decimal("-64.20"),
        timestamp=ts, description="Dispute filed by cardholder - product not as described",
        raw_metadata={"dispute_id": f"dp_eval_{txn_id}"},  # deliberately no reason_code
    )
    ds.add_txn(batch_num, txn, {
        "scenario": "unresolved_chargeback_unknown_reason", "expected_type": "chargeback",
        "expected_match": None,
        "expected_anomaly": {"anomaly_type": "unresolved_chargeback", "severity": "high"},
    })


def build_missing_record_never_resolves(ds: Dataset, ids: _IdGen, batch_num: int) -> None:
    merchant = ids.merchant_id()
    ts = ds.batch_timestamp(batch_num)
    txn_id = ids.txn_id(batch_num)
    txn = _mk_txn(
        txn_id, source="merchant_order_system", merchant_id=merchant, amount=Decimal("-33.10"),
        timestamp=ts, description="Refund issued for order - item returned",
        raw_metadata={"reason": "customer_return"},
    )
    ds.add_txn(batch_num, txn, {
        "scenario": "missing_record_never_resolves", "expected_type": "refund",
        "expected_match": None,
        "expected_anomaly": {"anomaly_type": "missing_record", "severity": "medium"},
        "expected_resolution": {"outcome": "unresolved"},
    })


def build_missing_record_earlier_batch(ds: Dataset, ids: _IdGen, anomaly_batch: int, counterpart_batch: int) -> None:
    merchant = ids.merchant_id()
    amount = Decimal("112.40")
    counterpart_id = ids.txn_id(counterpart_batch)
    anomaly_id = ids.txn_id(anomaly_batch)

    counterpart = _mk_txn(
        counterpart_id, source="processor_ledger", merchant_id=merchant, amount=amount,
        timestamp=ds.batch_timestamp(counterpart_batch), description="Card payment for merchant services",
        raw_metadata={"card_last4": "3333"},
    )
    anomaly_txn = _mk_txn(
        anomaly_id, source="merchant_order_system", merchant_id=merchant, amount=-amount,
        timestamp=ds.batch_timestamp(anomaly_batch), description="Refund issued - item returned",
        raw_metadata={"reason": "customer_return"},
    )
    ds.add_txn(counterpart_batch, counterpart, {
        "scenario": "missing_record_earlier_batch_counterpart", "expected_type": "payment",
        "expected_match": None, "expected_anomaly": None,
    })
    ds.add_txn(anomaly_batch, anomaly_txn, {
        "scenario": "missing_record_earlier_batch", "expected_type": "refund",
        "expected_match": None,
        "expected_anomaly": {"anomaly_type": "missing_record", "severity": "medium"},
        "expected_resolution": {
            "outcome": "auto_resolved",
            "counterpart_batch": ds.batch_id(counterpart_batch),
            "counterpart_txn": counterpart_id,
        },
    })


def build_missing_record_later_batch(ds: Dataset, ids: _IdGen, anomaly_batch: int, counterpart_batch: int) -> None:
    merchant = ids.merchant_id()
    amount = Decimal("58.75")
    anomaly_id = ids.txn_id(anomaly_batch)
    counterpart_id = ids.txn_id(counterpart_batch)

    anomaly_txn = _mk_txn(
        anomaly_id, source="merchant_order_system", merchant_id=merchant, amount=-amount,
        timestamp=ds.batch_timestamp(anomaly_batch), description="Refund issued - item returned",
        raw_metadata={"reason": "customer_return"},
    )
    counterpart = _mk_txn(
        counterpart_id, source="processor_ledger", merchant_id=merchant, amount=amount,
        timestamp=ds.batch_timestamp(counterpart_batch), description="Card payment for merchant services",
        raw_metadata={"card_last4": "4444"},
    )
    ds.add_txn(anomaly_batch, anomaly_txn, {
        "scenario": "missing_record_later_batch", "expected_type": "refund",
        "expected_match": None,
        "expected_anomaly": {"anomaly_type": "missing_record", "severity": "medium"},
        "expected_resolution": {
            "outcome": "not_yet_resolved",
            "counterpart_batch": ds.batch_id(counterpart_batch),
            "counterpart_txn": counterpart_id,
        },
    })
    ds.add_txn(counterpart_batch, counterpart, {
        "scenario": "missing_record_later_batch_counterpart", "expected_type": "payment",
        "expected_match": None, "expected_anomaly": None,
    })


def build_amount_mismatch(ds: Dataset, ids: _IdGen, batch_num: int, *, trivial: bool) -> None:
    merchant = ids.merchant_id()
    ts = ds.batch_timestamp(batch_num)
    payment_id = ids.txn_id(batch_num)
    refund_id = ids.txn_id(batch_num)
    payment_amount = Decimal("200.00")
    # trivial: 2-cent rounding-level gap -- within reconciliation_agent.py's
    # $1.00 tolerance, so this should match cleanly with no anomaly (it's
    # the false-positive guard: a tiny gap shouldn't get flagged). clear: a
    # real, unmistakable mismatch -- $50 off, well outside the tolerance
    # band but still inside the 50%-of-amount pairing window.
    refund_amount = Decimal("-199.98") if trivial else Decimal("-150.00")

    payment = _mk_txn(
        payment_id, source="processor_ledger", merchant_id=merchant, amount=payment_amount,
        timestamp=ts, description="Card payment for merchant services",
        raw_metadata={"card_last4": "5555"},
    )
    refund = _mk_txn(
        refund_id, source="merchant_order_system", merchant_id=merchant, amount=refund_amount,
        timestamp=ts + timedelta(hours=1), description="Refund issued - item returned",
        raw_metadata={"reason": "customer_return"},
    )
    if trivial:
        # Within tolerance -- a normal, clean match. No anomaly.
        shared_match = {"match_type": "refund_pair"}
        shared_anomaly = None
        scenario = "amount_mismatch_trivial"
    else:
        # Outside tolerance -- reconciliation_agent.py pairs it as
        # amount_mismatch_pair, which anomaly_agent.py turns into a real
        # AMOUNT_MISMATCH anomaly covering both transaction_ids.
        shared_match = {"match_type": "amount_mismatch_pair"}
        shared_anomaly = {"anomaly_type": "amount_mismatch", "severity": "high"}
        scenario = "amount_mismatch_clear"

    ds.add_txn(batch_num, payment, {
        "scenario": scenario, "expected_type": "payment",
        "expected_match": {**shared_match, "with": refund_id},
        "expected_anomaly": shared_anomaly,
    })
    ds.add_txn(batch_num, refund, {
        "scenario": scenario, "expected_type": "refund",
        "expected_match": {**shared_match, "with": payment_id},
        "expected_anomaly": shared_anomaly,
    })


def fill_clean(ds: Dataset, ids: _IdGen, batch_num: int, count: int) -> None:
    """Pad a batch out to its target size with realistic, non-anomalous
    traffic: solo payments, fees (never anomaly-checked), and legitimately
    matched payment/refund pairs (a real refund_pair match -- confirms
    matching a genuine pair doesn't itself get flagged as an anomaly, since
    only duplicate_pair matches ever become anomalies).
    """
    remaining = count
    # ~20% as legitimately-matched refund pairs (rounded down to an even number).
    pair_instances = max(0, (count * 2 // 10) // 2)
    for i in range(pair_instances):
        if remaining < 2:
            break
        merchant = ids.merchant_id()
        amount = Decimal("40.00") + Decimal(i)
        ts = ds.batch_timestamp(batch_num, offset_minutes=10 * i)
        payment_id = ids.txn_id(batch_num)
        refund_id = ids.txn_id(batch_num)
        payment = _mk_txn(
            payment_id, source="processor_ledger", merchant_id=merchant, amount=amount,
            timestamp=ts, description="Card payment for merchant services",
            raw_metadata={"card_last4": "6666"},
        )
        refund = _mk_txn(
            refund_id, source="merchant_order_system", merchant_id=merchant, amount=-amount,
            timestamp=ts + timedelta(days=1), description="Refund issued - item returned",
            raw_metadata={"reason": "customer_return"},
        )
        ds.add_txn(batch_num, payment, {
            "scenario": "clean_matched_refund_pair", "expected_type": "payment",
            "expected_match": {"match_type": "refund_pair", "with": refund_id},
            "expected_anomaly": None,
        })
        ds.add_txn(batch_num, refund, {
            "scenario": "clean_matched_refund_pair", "expected_type": "refund",
            "expected_match": {"match_type": "refund_pair", "with": payment_id},
            "expected_anomaly": None,
        })
        remaining -= 2

    # ~15% fees.
    fee_count = max(0, count * 15 // 100)
    fee_count = min(fee_count, remaining)
    for i in range(fee_count):
        merchant = ids.merchant_id()
        txn_id = ids.txn_id(batch_num)
        txn = _mk_txn(
            txn_id, source="processor_ledger", merchant_id=merchant, amount=Decimal("2.50"),
            timestamp=ds.batch_timestamp(batch_num, offset_minutes=20 * i), description="Monthly platform processing fee",
            raw_metadata={"fee_type": "platform"},
        )
        ds.add_txn(batch_num, txn, {
            "scenario": "clean_payment", "expected_type": "fee",
            "expected_match": None, "expected_anomaly": None,
        })
        remaining -= 1

    # Rest: solo payments, nothing else references them.
    for i in range(remaining):
        merchant = ids.merchant_id()
        txn_id = ids.txn_id(batch_num)
        txn = _mk_txn(
            txn_id, source="processor_ledger", merchant_id=merchant, amount=Decimal("25.00") + Decimal(i),
            timestamp=ds.batch_timestamp(batch_num, offset_minutes=30 * i), description="Card payment for merchant services",
            raw_metadata={"card_last4": "7777"},
        )
        ds.add_txn(batch_num, txn, {
            "scenario": "clean_payment", "expected_type": "payment",
            "expected_match": None, "expected_anomaly": None,
        })


def generate(num_batches: int, transactions_per_batch: int) -> dict:
    if num_batches < 2:
        raise ValueError("num_batches must be >= 2 -- the cross-batch scenarios need at least two batches.")
    if num_batches != 3 or transactions_per_batch != 50:
        print(
            f"WARNING: SCENARIO_COUNTS is hand-tuned for 3 batches / 50 per batch. "
            f"Got {num_batches}/{transactions_per_batch} -- seeded-scenario counts won't scale, "
            f"only the clean-transaction fill will adjust.",
            file=sys.stderr,
        )

    ds = Dataset(num_batches)
    ids = _IdGen()
    code_cycle = itertools.cycle(_KNOWN_REASON_CODES)

    for batch_num in range(1, num_batches + 1):
        for _ in range(SCENARIO_COUNTS["duplicate_charge_same_batch"]):
            build_duplicate_charge_same_batch(ds, ids, batch_num)
        for _ in range(SCENARIO_COUNTS["duplicate_charge_near_miss"]):
            build_duplicate_charge_near_miss(ds, ids, batch_num)
        for _ in range(SCENARIO_COUNTS["unresolved_chargeback_known_reason"]):
            build_unresolved_chargeback_known_reason(ds, ids, batch_num, code_cycle)
        for _ in range(SCENARIO_COUNTS["unresolved_chargeback_unknown_reason"]):
            build_unresolved_chargeback_unknown_reason(ds, ids, batch_num)
        for _ in range(SCENARIO_COUNTS["missing_record_never_resolves"]):
            build_missing_record_never_resolves(ds, ids, batch_num)
        for _ in range(SCENARIO_COUNTS["amount_mismatch_clear"]):
            build_amount_mismatch(ds, ids, batch_num, trivial=False)
        for _ in range(SCENARIO_COUNTS["amount_mismatch_trivial"]):
            build_amount_mismatch(ds, ids, batch_num, trivial=True)

    # Cross-batch scenarios: one hop per consecutive batch pair.
    for anomaly_batch in range(2, num_batches + 1):
        counterpart_batch = anomaly_batch - 1
        for _ in range(SCENARIO_COUNTS["missing_record_earlier_batch"]):
            build_missing_record_earlier_batch(ds, ids, anomaly_batch, counterpart_batch)
    for anomaly_batch in range(1, num_batches):
        counterpart_batch = anomaly_batch + 1
        for _ in range(SCENARIO_COUNTS["missing_record_later_batch"]):
            build_missing_record_later_batch(ds, ids, anomaly_batch, counterpart_batch)

    # Pad every batch out to its target size with clean traffic.
    for batch_num in range(1, num_batches + 1):
        seeded = len(ds.batches[batch_num])
        clean_needed = max(0, transactions_per_batch - seeded)
        fill_clean(ds, ids, batch_num, clean_needed)

    batches_out = [
        {
            "batch_id": ds.batch_id(b),
            "run_order": b,
            "transactions": ds.batches[b],
        }
        for b in range(1, num_batches + 1)
    ]
    return {
        "config": {"transactions_per_batch": transactions_per_batch, "num_batches": num_batches},
        "batches": batches_out,
        "ground_truth": ds.ground_truth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval/eval_dataset.json")
    parser.add_argument("--transactions-per-batch", type=int, default=50)
    parser.add_argument("--num-batches", type=int, default=3)
    args = parser.parse_args()

    data = generate(args.num_batches, args.transactions_per_batch)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n")

    total_txns = sum(len(b["transactions"]) for b in data["batches"])
    scenario_counts: dict[str, int] = {}
    for gt in data["ground_truth"].values():
        scenario_counts[gt["scenario"]] = scenario_counts.get(gt["scenario"], 0) + 1

    print(f"Wrote {out_path} -- {len(data['batches'])} batches, {total_txns} transactions total.")
    print("Per-scenario transaction counts:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")


if __name__ == "__main__":
    main()
