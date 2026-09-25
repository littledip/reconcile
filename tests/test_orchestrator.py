"""Tests for the Week 3-4 Reconciliation and Anomaly Detection agents, the
orchestrator's conditional routing, and the Week 7-8 MCP lookup + persist
nodes. Run in heuristic mode (no ANTHROPIC_API_KEY needed) so these pass in
any environment.

run_reconciliation_batch is async as of Week 7-8 (lookup_missing_records_node
makes a live MCP call) -- tests call it via asyncio.run() rather than
pulling in a pytest-asyncio dependency for one entry point.

Week 9-10 adds one test confirming lookup_missing_records_node's
auto-resolutions also become episodes (app/memory_store.py), same as the
existing audit-record test just above it.
"""
import asyncio
import json
from decimal import Decimal

from app.agents.orchestrator import run_reconciliation_batch
from app.cache import redis_client
from app.db import db
from app.mcp_server import _ESCALATION_QUEUE_KEY
from app.mcp_server import submit_reconciliation_decision as submit_tool
from app.memory_store import find_similar_episodes
from app.models.transaction import Transaction, TransactionSource


def _txn(**overrides) -> Transaction:
    base = dict(
        transaction_id="txn_x",
        source=TransactionSource.PROCESSOR_LEDGER,
        merchant_id="merch_1",
        amount=Decimal("100.00"),
        currency="USD",
        timestamp="2026-08-10T00:00:00Z",
        description="Card payment - order #1",
    )
    base.update(overrides)
    return Transaction.model_validate(base)


def _run(txns):
    return asyncio.run(run_reconciliation_batch(txns))


def test_duplicate_charge_detected_and_escalated():
    txns = [
        _txn(transaction_id="a", description="Card payment - order #500", amount=Decimal("50.00")),
        _txn(transaction_id="b", description="Card payment - order #500 (retry)", amount=Decimal("50.00")),
    ]
    result = _run(txns)

    assert len(result["matches"]) == 1
    assert result["matches"][0]["match_type"] == "duplicate_pair"

    anomaly_types = [a["anomaly_type"] for a in result["anomalies"]]
    assert "duplicate_charge" in anomaly_types
    assert result["escalations"], "duplicate charge should trigger escalation"


def test_unmatched_chargeback_flagged_unresolved():
    txns = [
        _txn(transaction_id="c", description="Dispute filed by cardholder", amount=Decimal("-75.00")),
    ]
    result = _run(txns)

    anomaly_types = [a["anomaly_type"] for a in result["anomalies"]]
    assert "unresolved_chargeback" in anomaly_types
    assert result["escalations"]


def test_unmatched_refund_flagged_missing_record():
    txns = [
        _txn(transaction_id="d", description="Refund issued for returned item", amount=Decimal("-30.00")),
    ]
    result = _run(txns)

    anomaly_types = [a["anomaly_type"] for a in result["anomalies"]]
    assert "missing_record" in anomaly_types
    assert result["escalations"]


def test_matched_payment_and_refund_produce_no_missing_record_anomaly():
    txns = [
        _txn(transaction_id="e", description="Card payment - no order ref", amount=Decimal("40.00")),
        _txn(transaction_id="f", description="Refund - no order ref", amount=Decimal("-40.00")),
    ]
    result = _run(txns)

    assert len(result["matches"]) == 1
    assert result["matches"][0]["match_type"] == "refund_pair"
    anomaly_types = [a["anomaly_type"] for a in result["anomalies"]]
    assert "missing_record" not in anomaly_types


def test_no_anomalies_means_no_escalation():
    txns = [
        _txn(transaction_id="g", description="Monthly platform processing fee", amount=Decimal("2.50")),
    ]
    result = _run(txns)

    assert result["anomalies"] == []
    assert result["escalations"] == []


# --- Week 7-8: lookup_missing_records_node -----------------------------


def test_missing_record_resolved_via_cross_batch_lookup():
    """A refund with no in-batch counterpart is a missing_record anomaly
    (per the test above) -- unless its original payment was persisted by
    an *earlier* batch, in which case lookup_missing_records_node should
    find it and remove the anomaly before it ever reaches escalation.
    """
    payment_batch = [
        _txn(transaction_id="h_payment", merchant_id="merch_h", amount=Decimal("60.00"), description="Card payment - no order ref"),
    ]
    first = _run(payment_batch)
    assert first["anomalies"] == []  # lone payment, nothing anomalous about it yet

    refund_batch = [
        _txn(transaction_id="h_refund", merchant_id="merch_h", amount=Decimal("-60.00"), description="Refund - no order ref"),
    ]
    second = _run(refund_batch)

    anomaly_types = [a["anomaly_type"] for a in second["anomalies"]]
    assert "missing_record" not in anomaly_types
    assert second["escalations"] == []


def test_duplicate_charge_not_eligible_for_lookup():
    """duplicate_charge anomalies come from an already-matched in-batch
    duplicate_pair -- there's nothing "missing" to look up, so they should
    never be resolved by lookup_missing_records_node even if a persisted
    record with the same order id/amount happens to exist.
    """
    db["transactions"].insert_one(
        {
            "transaction_id": "prior_unrelated",
            "merchant_id": "merch_dup",
            "amount": "25.00",
            "timestamp": "2026-08-01T00:00:00+00:00",
            "description": "order #700",
        }
    )
    txns = [
        _txn(transaction_id="dup_1", merchant_id="merch_dup", amount=Decimal("25.00"), description="Card payment - order #700"),
        _txn(transaction_id="dup_2", merchant_id="merch_dup", amount=Decimal("25.00"), description="Card payment - order #700 (retry)"),
    ]
    result = _run(txns)

    anomaly_types = [a["anomaly_type"] for a in result["anomalies"]]
    assert "duplicate_charge" in anomaly_types
    assert result["escalations"]


# --- Week 7-8: persist_batch_node ---------------------------------------


def test_persist_batch_writes_transactions_on_finish_branch():
    txns = [
        _txn(transaction_id="i", merchant_id="merch_i", description="Monthly platform processing fee", amount=Decimal("2.50")),
    ]
    result = _run(txns)

    assert result["escalations"] == []  # took the "finish" branch, not "escalate"
    assert db["transactions"].find_one({"transaction_id": "i"}) is not None


def test_persist_batch_writes_transactions_on_escalate_branch():
    txns = [
        _txn(transaction_id="j", description="Card payment - order #900", amount=Decimal("50.00")),
        _txn(transaction_id="k", description="Card payment - order #900 (retry)", amount=Decimal("50.00")),
    ]
    result = _run(txns)

    assert result["escalations"]  # took the "escalate" branch
    assert db["transactions"].find_one({"transaction_id": "j"}) is not None
    assert db["transactions"].find_one({"transaction_id": "k"}) is not None


# --- Write-tools phase: escalate_node + lookup_missing_records_node audit ---


def test_escalate_node_pushes_onto_redis_queue():
    """A duplicate_charge anomaly requires human review, so escalate_node
    should push it onto the live Redis escalation queue, not just return
    the anomaly ID in state.
    """
    txns = [
        _txn(transaction_id="l", description="Card payment - order #950", amount=Decimal("50.00")),
        _txn(transaction_id="m", description="Card payment - order #950 (retry)", amount=Decimal("50.00")),
    ]
    result = _run(txns)

    assert result["escalations"]
    anomaly_id = result["escalations"][0]
    raw = redis_client.hget(_ESCALATION_QUEUE_KEY, anomaly_id)
    assert raw is not None
    record = json.loads(raw)
    assert record["anomaly_type"] == "duplicate_charge"
    assert record["severity"]


def test_escalated_anomaly_carries_episodic_context_into_the_queue():
    """Closes the gap flagged in Progress_Log.md: reasoning_node was
    already attaching episodic_context to escalated anomalies, but
    escalate_dispute/escalate_node dropped it on the floor before it ever
    reached Redis -- a reviewer could only see it via a separate GET
    /escalations/{anomaly_id}/similar call. Confirm it now rides along in
    the same GET /escalations record.

    First batch resolves and becomes a real episode (via
    submit_reconciliation_decision, the same path a human reviewer's
    decision takes); second batch is a different-merchant/different-order
    duplicate_charge similar enough to match it.
    """
    first_batch = [
        _txn(transaction_id="ec_a", merchant_id="merch_ec1", description="Card payment - order #700", amount=Decimal("70.00")),
        _txn(transaction_id="ec_b", merchant_id="merch_ec1", description="Card payment - order #700 (retry)", amount=Decimal("70.00")),
    ]
    first_result = _run(first_batch)
    first_anomaly_id = first_result["escalations"][0]
    first_record = json.loads(redis_client.hget(_ESCALATION_QUEUE_KEY, first_anomaly_id))

    submit_tool(
        anomaly_id=first_anomaly_id,
        decision="approved",
        decided_by="jane_reviewer",
        notes="Confirmed duplicate, refunded.",
        anomaly_type=first_record["anomaly_type"],
        severity=first_record["severity"],
        transaction_ids=first_record["transaction_ids"],
        reasoning=first_record["reasoning"],
    )

    second_batch = [
        _txn(transaction_id="ec_c", merchant_id="merch_ec2", description="Card payment - order #701", amount=Decimal("72.00")),
        _txn(transaction_id="ec_d", merchant_id="merch_ec2", description="Card payment - order #701 (retry)", amount=Decimal("72.00")),
    ]
    second_result = _run(second_batch)
    second_anomaly_id = second_result["escalations"][0]
    second_record = json.loads(redis_client.hget(_ESCALATION_QUEUE_KEY, second_anomaly_id))

    assert second_record["episodic_context"] is not None
    assert "jane_reviewer" in second_record["episodic_context"]
    assert "approved" in second_record["episodic_context"]


def test_no_escalation_means_nothing_queued():
    txns = [
        _txn(transaction_id="n", description="Monthly platform processing fee", amount=Decimal("2.50")),
    ]
    result = _run(txns)

    assert result["escalations"] == []
    assert redis_client.hlen(_ESCALATION_QUEUE_KEY) == 0


def test_auto_resolved_lookup_writes_audit_record_and_never_queues():
    """The missing_record anomaly that lookup_missing_records_node resolves
    via cross-batch lookup (same scenario as
    test_missing_record_resolved_via_cross_batch_lookup above) should now
    also produce a permanent reconciliation_decisions record -- and never
    touch the escalation queue, since it was resolved before escalation
    was ever considered.
    """
    payment_batch = [
        _txn(transaction_id="o_payment", merchant_id="merch_o", amount=Decimal("60.00"), description="Card payment - no order ref"),
    ]
    _run(payment_batch)

    refund_batch = [
        _txn(transaction_id="o_refund", merchant_id="merch_o", amount=Decimal("-60.00"), description="Refund - no order ref"),
    ]
    result = _run(refund_batch)

    assert result["escalations"] == []
    assert redis_client.hlen(_ESCALATION_QUEUE_KEY) == 0

    record = db["reconciliation_decisions"].find_one({"decision": "auto_resolved_via_lookup"})
    assert record is not None
    assert record["decided_by"] == "system"
    assert record["anomaly_type"] == "missing_record"


def test_auto_resolved_lookup_also_writes_an_episode():
    """Same cross-batch scenario as the audit-record test above -- confirm
    the resolution becomes a queryable episode too, not just a Mongo
    record, and that its severity survived the trip (lookup_missing_records_node
    passes it directly from the Anomaly object).
    """
    payment_batch = [
        _txn(transaction_id="p_payment", merchant_id="merch_p", amount=Decimal("60.00"), description="Card payment - no order ref"),
    ]
    _run(payment_batch)

    refund_batch = [
        _txn(transaction_id="p_refund", merchant_id="merch_p", amount=Decimal("-60.00"), description="Refund - no order ref"),
    ]
    _run(refund_batch)

    record = db["reconciliation_decisions"].find_one({"decision": "auto_resolved_via_lookup", "anomaly_type": "missing_record"})
    assert record is not None
    assert record["severity"]

    matches = find_similar_episodes(
        anomaly_type="missing_record", severity=record["severity"], reasoning=record["reasoning"],
    )
    assert any(m["anomaly_id"] == record["anomaly_id"] for m in matches)
