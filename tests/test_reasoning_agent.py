"""Tests for the Week 5-6 Reasoning Agent (reason-code graph enrichment)
and the Week 9-10 episodic-context attachment added alongside it.

Runs entirely against the in-memory graph fallback (no NEO4J_URI needed),
same philosophy as the other tests running in heuristic/mock mode. Episodic
tests get their own disposable Chroma directory from tests/conftest.py's
autouse _isolate_chroma fixture, same as everywhere else.
"""
from decimal import Decimal

from app.agents.reasoning_agent import enrich_anomalies
from app.memory_store import write_episode
from app.models.reconciliation import Anomaly, AnomalyType
from app.models.transaction import Transaction, TransactionSource


def _chargeback_txn(reason_code: str | None) -> Transaction:
    raw_metadata = {"dispute_id": "dp_1", "reason_code": reason_code} if reason_code else {}
    return Transaction.model_validate(
        {
            "transaction_id": "txn_cb",
            "source": TransactionSource.PROCESSOR_LEDGER,
            "merchant_id": "merch_1",
            "amount": Decimal("-100.00"),
            "currency": "USD",
            "timestamp": "2026-08-14T00:00:00Z",
            "description": "Dispute filed by cardholder",
            "raw_metadata": raw_metadata,
        }
    )


def _unresolved_chargeback_anomaly() -> Anomaly:
    return Anomaly(
        anomaly_id="anomaly_1",
        anomaly_type=AnomalyType.UNRESOLVED_CHARGEBACK,
        transaction_ids=["txn_cb"],
        severity="high",
        confidence=0.7,
        reasoning="Chargeback txn_cb has no matching original transaction in this batch.",
        requires_human_review=True,
    )


def test_enriches_reasoning_with_known_reason_code():
    txns = [_chargeback_txn(reason_code="10.4")]
    anomalies = [_unresolved_chargeback_anomaly()]

    enriched = enrich_anomalies(anomalies, txns)

    assert len(enriched) == 1
    reasoning = enriched[0].reasoning
    # Grounded, cited detail should now be present -- network, code, category,
    # liable party, and response window, none of which were in the original
    # template string.
    assert "Visa" in reasoning
    assert "10.4" in reasoning
    assert "Fraud" in reasoning
    assert "merchant" in reasoning
    assert "30 days" in reasoning


def test_leaves_reasoning_unchanged_when_no_reason_code_present():
    txns = [_chargeback_txn(reason_code=None)]
    anomalies = [_unresolved_chargeback_anomaly()]

    enriched = enrich_anomalies(anomalies, txns)

    assert enriched[0].reasoning == anomalies[0].reasoning


def test_leaves_reasoning_unchanged_when_reason_code_unknown_to_graph():
    txns = [_chargeback_txn(reason_code="99.9")]  # not in the seed data
    anomalies = [_unresolved_chargeback_anomaly()]

    enriched = enrich_anomalies(anomalies, txns)

    assert enriched[0].reasoning == anomalies[0].reasoning


def test_non_chargeback_anomalies_are_passed_through_untouched():
    anomaly = Anomaly(
        anomaly_id="anomaly_2",
        anomaly_type=AnomalyType.DUPLICATE_CHARGE,
        transaction_ids=["txn_a", "txn_b"],
        severity="high",
        confidence=0.95,
        reasoning="Duplicate charge: same order, same amount.",
        requires_human_review=True,
    )

    enriched = enrich_anomalies([anomaly], transactions=[])

    assert enriched[0].reasoning == anomaly.reasoning


def test_shared_liability_rule_reused_across_networks():
    """10.4 (Visa) and 4837 (Mastercard) are both Fraud-category codes that
    share the fraud_standard liability rule in the seed data -- this is the
    actual point of modeling liability as its own node rather than a
    duplicated property on every reason code.
    """
    from app.graph_db import get_reason_code_context

    visa_fraud = get_reason_code_context("10.4")
    mc_fraud = get_reason_code_context("4837")

    assert visa_fraud.liable_party == mc_fraud.liable_party == "merchant"
    assert visa_fraud.response_window_days == mc_fraud.response_window_days == 30
    assert visa_fraud.network != mc_fraud.network


# --- Week 9-10: episodic_context attachment -----------------------------


def test_attaches_episodic_context_when_a_similar_episode_exists():
    write_episode(
        anomaly_id="past_1",
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Two transactions reference order #500 for merchant merch_1 with identical amount 50.00.",
        decision="approved",
        decided_by="jane_reviewer",
        notes="Confirmed duplicate.",
        decided_at="2026-09-23T12:00:00+00:00",
        transaction_ids=["x"],
    )

    anomaly = Anomaly(
        anomaly_id="anomaly_new",
        anomaly_type=AnomalyType.DUPLICATE_CHARGE,
        transaction_ids=["y", "z"],
        severity="high",
        confidence=0.95,
        reasoning="Two transactions reference order #501 for merchant merch_2 with identical amount 55.00.",
        requires_human_review=True,
    )

    enriched = enrich_anomalies([anomaly], transactions=[])

    assert enriched[0].episodic_context is not None
    assert "approved" in enriched[0].episodic_context
    assert "jane_reviewer" in enriched[0].episodic_context
    # reasoning itself is untouched by episodic grounding -- only
    # episodic_context carries it, per the design session.
    assert enriched[0].reasoning == anomaly.reasoning


def test_episodic_context_stays_none_when_store_is_empty():
    anomaly = Anomaly(
        anomaly_id="anomaly_no_history",
        anomaly_type=AnomalyType.MISSING_RECORD,
        transaction_ids=["w"],
        severity="medium",
        confidence=0.8,
        reasoning="Refund has no matching original payment in this batch.",
        requires_human_review=True,
    )

    enriched = enrich_anomalies([anomaly], transactions=[])

    assert enriched[0].episodic_context is None
