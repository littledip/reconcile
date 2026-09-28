"""Week 3-4: Anomaly Detection Agent.

Consumes the Classification Agent's labels and the Reconciliation Agent's
matches to flag three anomaly types:

  - duplicate_charge: a duplicate_pair match from the Reconciliation Agent.
  - unresolved_chargeback: a chargeback with no matched original transaction
    in this batch — can't verify the dispute against a source charge.
  - missing_record: a refund with no matched original payment in this batch
    — the original charge may be outside the ingestion window, or missing.

Like the Reconciliation Agent, this is heuristic/deterministic for Week 3-4
(auditable rules over the match graph) rather than LLM-based. Grounded,
LLM-generated explanations tied to a knowledge graph are the Reasoning Agent's
job in weeks 5-6 (Graph RAG) — this agent produces the structured anomaly
itself, which the Reasoning Agent will later explain in richer, cited detail.
"""
from __future__ import annotations

from decimal import Decimal

from app.models.reconciliation import Anomaly, AnomalyType, TransactionMatch
from app.models.transaction import Transaction, TransactionType

# Week 11+: threshold for AMOUNT_MISMATCH severity -- a diff exceeding 20% of
# the transaction amount is a "high" severity anomaly, otherwise "medium".
# (Whether the pair gets matched as amount_mismatch_pair at all, vs. treated
# as rounding noise or left unrelated, is decided upstream in
# app/agents/reconciliation_agent.py's pass 2 -- this only grades severity
# for pairs that already cleared that bar.)
_AMOUNT_MISMATCH_HIGH_SEVERITY_PCT = Decimal("0.20")


def _amount_mismatch_severity(match: TransactionMatch, txns_by_id: dict[str, Transaction]) -> str:
    """High if the mismatch is >20% of the larger of the two amounts, else medium."""
    amounts = [abs(txns_by_id[tid].amount) for tid in match.transaction_ids if tid in txns_by_id]
    if len(amounts) < 2:
        return "medium"
    diff = abs(amounts[0] - amounts[1])
    larger = max(amounts)
    if larger and diff / larger > _AMOUNT_MISMATCH_HIGH_SEVERITY_PCT:
        return "high"
    return "medium"


def detect_anomalies(
    transactions: list[Transaction],
    classifications: dict[str, TransactionType],
    matches: list[TransactionMatch],
) -> list[Anomaly]:
    """Public entry point: flag duplicate charges, unresolved chargebacks,
    missing records, and amount mismatches from the classification and match
    results for a batch.
    """
    anomalies: list[Anomaly] = []
    counter = 1
    matched_ids = {tid for m in matches for tid in m.transaction_ids}
    txns_by_id = {t.transaction_id: t for t in transactions}

    for match in matches:
        if match.match_type == "duplicate_pair":
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anomaly_{counter}",
                    anomaly_type=AnomalyType.DUPLICATE_CHARGE,
                    transaction_ids=match.transaction_ids,
                    severity="high",
                    confidence=match.confidence,
                    reasoning=f"Duplicate charge: {match.reasoning}",
                    requires_human_review=True,
                )
            )
            counter += 1
        elif match.match_type == "amount_mismatch_pair":
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anomaly_{counter}",
                    anomaly_type=AnomalyType.AMOUNT_MISMATCH,
                    transaction_ids=match.transaction_ids,
                    severity=_amount_mismatch_severity(match, txns_by_id),
                    confidence=match.confidence,
                    reasoning=f"Amount mismatch: {match.reasoning}",
                    requires_human_review=True,
                )
            )
            counter += 1

    for txn in transactions:
        if txn.transaction_id in matched_ids:
            continue
        txn_type = classifications.get(txn.transaction_id)

        if txn_type == TransactionType.CHARGEBACK:
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anomaly_{counter}",
                    anomaly_type=AnomalyType.UNRESOLVED_CHARGEBACK,
                    transaction_ids=[txn.transaction_id],
                    severity="high",
                    confidence=0.7,
                    reasoning=(
                        f"Chargeback {txn.transaction_id} has no matching original transaction "
                        f"in this batch — cannot verify the dispute against a source charge."
                    ),
                    requires_human_review=True,
                )
            )
            counter += 1

        elif txn_type == TransactionType.REFUND:
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anomaly_{counter}",
                    anomaly_type=AnomalyType.MISSING_RECORD,
                    transaction_ids=[txn.transaction_id],
                    severity="medium",
                    confidence=0.6,
                    reasoning=(
                        f"Refund {txn.transaction_id} has no matching original payment in this "
                        f"batch — the original charge may be outside the ingestion window or missing."
                    ),
                    requires_human_review=True,
                )
            )
            counter += 1

    return anomalies
