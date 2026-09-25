"""Week 3-4: Reconciliation/Matching Agent.

Pairs up related transactions across the two simulated systems (processor
ledger / merchant order system) so the Anomaly Detection Agent has something
to reason about beyond a single record.

Deliberately deterministic rather than LLM-based — real reconciliation
matching logic needs to be auditable and reproducible (the JD's own language:
"stateful, deterministic, and fault-tolerant agent workflows"). The
Classification and Anomaly Detection agents are where LLM reasoning adds
value (interpreting ambiguous descriptions, writing grounded explanations);
matching itself is closer to a rules/join problem, so it's built that way
here. Still exposed as a LangGraph node (see orchestrator.py) so it
participates in the same stateful, multi-agent workflow.

Two-pass matching strategy:
  1. High-confidence: group by (merchant_id, order reference extracted from
     the description) — transactions that reference the same order and
     amount. Multiple hits here mean a duplicate/retried charge.
  2. Lower-confidence fallback: for transactions not matched above, pair a
     positive-amount payment with an equal-and-opposite-amount refund or
     chargeback for the same merchant within a 30-day window, when no
     shared order reference exists.

Transactions that never get matched (e.g. a chargeback with no traceable
original charge in the batch, or a refund whose original payment isn't in
the batch) are intentionally left unmatched — that's a real signal, not a
bug, and it's exactly what the Anomaly Detection Agent flags next.
"""
from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal

from app.models.reconciliation import TransactionMatch
from app.models.transaction import Transaction, TransactionType

_ORDER_RE = re.compile(r"order #(\d+)", re.IGNORECASE)
_MATCH_WINDOW_DAYS = 30


def _extract_order_id(description: str | None) -> str | None:
    if not description:
        return None
    m = _ORDER_RE.search(description)
    return m.group(1) if m else None


def match_transactions(
    transactions: list[Transaction],
    classifications: dict[str, TransactionType],
) -> list[TransactionMatch]:
    """Public entry point: pair related transactions via order-id matching,
    then equal-and-opposite-amount fallback matching.
    """
    matches: list[TransactionMatch] = []
    matched_ids: set[str] = set()
    match_counter = 1

    # Pass 1: same merchant + same order reference + same amount => duplicate.
    order_groups: dict[tuple[str, str], list[Transaction]] = defaultdict(list)
    for txn in transactions:
        order_id = _extract_order_id(txn.description)
        if order_id:
            order_groups[(txn.merchant_id, order_id)].append(txn)

    for (merchant_id, order_id), group in order_groups.items():
        by_amount: dict[Decimal, list[Transaction]] = defaultdict(list)
        for txn in group:
            by_amount[txn.amount].append(txn)
        for amount, txns in by_amount.items():
            if len(txns) > 1:
                ids = [t.transaction_id for t in txns]
                matches.append(
                    TransactionMatch(
                        match_id=f"match_{match_counter}",
                        transaction_ids=ids,
                        match_type="duplicate_pair",
                        confidence=0.85,
                        reasoning=(
                            f"{len(ids)} transactions reference order #{order_id} for merchant "
                            f"{merchant_id} with identical amount {amount} — consistent with a "
                            f"duplicate or retried charge."
                        ),
                    )
                )
                matched_ids.update(ids)
                match_counter += 1

    # Pass 2: equal-and-opposite amount, same merchant, no shared order ref.
    remaining = [t for t in transactions if t.transaction_id not in matched_ids]
    for payment in remaining:
        if payment.transaction_id in matched_ids:
            continue
        p_type = classifications.get(payment.transaction_id)
        if p_type != TransactionType.PAYMENT or payment.amount <= 0:
            continue
        for other in remaining:
            if other.transaction_id in matched_ids or other.transaction_id == payment.transaction_id:
                continue
            o_type = classifications.get(other.transaction_id)
            if o_type not in (TransactionType.REFUND, TransactionType.CHARGEBACK):
                continue
            if other.merchant_id != payment.merchant_id or other.amount != -payment.amount:
                continue
            days_apart = abs((other.timestamp - payment.timestamp).days)
            if days_apart > _MATCH_WINDOW_DAYS:
                continue

            match_type = "chargeback_pair" if o_type == TransactionType.CHARGEBACK else "refund_pair"
            matches.append(
                TransactionMatch(
                    match_id=f"match_{match_counter}",
                    transaction_ids=[payment.transaction_id, other.transaction_id],
                    match_type=match_type,
                    confidence=0.65,
                    reasoning=(
                        f"Matched by merchant + equal-and-opposite amount ({payment.amount} vs "
                        f"{other.amount}) within {days_apart} day(s); no shared order reference, "
                        f"so this is a lower-confidence heuristic match than an order-id match."
                    ),
                )
            )
            matched_ids.add(payment.transaction_id)
            matched_ids.add(other.transaction_id)
            match_counter += 1
            break

    return matches
