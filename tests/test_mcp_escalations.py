"""Tests for the write-tools phase of Week 7-8: escalate_dispute,
list_pending_escalations, and submit_reconciliation_decision
(app/mcp_server.py), plus the client wrappers around them
(app/mcp_client.py).

Same mock-mode setup as test_mcp_lookup.py: MONGO_URI/REDIS_URL unset here,
so these run against mongomock and fakeredis directly (the tool functions
themselves, not a real MCP subprocess) -- the tool logic under test is
identical either way, only the transport differs.

Week 9-10 adds a handful of tests confirming submit_reconciliation_decision
also writes an episode to app/memory_store.py (except for "deferred") --
tests/conftest.py's _isolate_chroma fixture gives each test its own
disposable episode store, same isolation idea as the Mongo/Redis fixtures.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.cache import redis_client
from app.db import db
from app.mcp_client import (
    escalate_dispute as client_escalate_dispute,
    list_pending_escalations as client_list_pending_escalations,
    lookup_session,
    submit_reconciliation_decision as client_submit_reconciliation_decision,
)
from app.mcp_server import _ESCALATION_QUEUE_KEY
from app.mcp_server import escalate_dispute as escalate_tool
from app.mcp_server import list_pending_escalations as list_tool
from app.mcp_server import submit_reconciliation_decision as submit_tool
from app.memory_store import find_similar_episodes
from app.models.reconciliation import Anomaly, AnomalyType

# --- app/mcp_server.py: tool logic -------------------------------------


def test_escalate_dispute_writes_redis_hash_entry():
    result = escalate_tool(
        anomaly_id="an_1",
        anomaly_type="duplicate_charge",
        transaction_ids=["a", "b"],
        severity="high",
        reasoning="Two identical charges for order #500.",
    )
    assert result == {"queued": True, "anomaly_id": "an_1"}

    raw = redis_client.hget(_ESCALATION_QUEUE_KEY, "an_1")
    assert raw is not None
    record = json.loads(raw)
    assert record["anomaly_type"] == "duplicate_charge"
    assert record["transaction_ids"] == ["a", "b"]
    assert record["severity"] == "high"
    # No episodic_context passed -- key present, value null, not omitted.
    assert record["episodic_context"] is None


def test_escalate_dispute_stores_episodic_context_when_present():
    """Previously computed by reasoning_node and then discarded for the
    escalated branch -- now carried straight into the queue record so a
    reviewer sees it via plain GET /escalations, not just the separate
    /similar endpoint.
    """
    escalate_tool(
        anomaly_id="an_1b",
        anomaly_type="duplicate_charge",
        transaction_ids=["a2", "b2"],
        severity="high",
        reasoning="Two identical charges for order #501.",
        episodic_context="Similar past decisions: approved by jane_reviewer (Confirmed duplicate, refunded.).",
    )

    record = json.loads(redis_client.hget(_ESCALATION_QUEUE_KEY, "an_1b"))
    assert record["episodic_context"] == "Similar past decisions: approved by jane_reviewer (Confirmed duplicate, refunded.)."


def test_escalate_dispute_overwrites_existing_entry():
    """Re-escalating the same anomaly_id (e.g. a re-run batch) should
    update the queue entry, not error or duplicate it.
    """
    escalate_tool(
        anomaly_id="an_2",
        anomaly_type="missing_record",
        transaction_ids=["c"],
        severity="low",
        reasoning="First pass.",
    )
    escalate_tool(
        anomaly_id="an_2",
        anomaly_type="missing_record",
        transaction_ids=["c"],
        severity="medium",
        reasoning="Updated on re-run.",
    )

    assert redis_client.hlen(_ESCALATION_QUEUE_KEY) == 1
    record = json.loads(redis_client.hget(_ESCALATION_QUEUE_KEY, "an_2"))
    assert record["severity"] == "medium"
    assert record["reasoning"] == "Updated on re-run."


def test_list_pending_escalations_returns_all_queued():
    escalate_tool(
        anomaly_id="an_3", anomaly_type="unresolved_chargeback",
        transaction_ids=["d"], severity="high", reasoning="r1",
    )
    escalate_tool(
        anomaly_id="an_4", anomaly_type="amount_mismatch",
        transaction_ids=["e"], severity="low", reasoning="r2",
    )

    pending = list_tool()["escalations"]
    ids = {p["anomaly_id"] for p in pending}
    assert ids == {"an_3", "an_4"}


def test_list_pending_escalations_empty_when_none_queued():
    assert list_tool() == {"escalations": []}


def test_submit_decision_approved_writes_audit_record_and_dequeues():
    escalate_tool(
        anomaly_id="an_5", anomaly_type="duplicate_charge",
        transaction_ids=["f", "g"], severity="high", reasoning="Duplicate.",
    )

    result = submit_tool(
        anomaly_id="an_5", decision="approved", decided_by="jane_reviewer", notes="Confirmed duplicate, refunded.",
    )

    assert result == {
        "recorded": True,
        "anomaly_id": "an_5",
        "decision": "approved",
        "removed_from_queue": True,
    }
    # Dequeued.
    assert redis_client.hget(_ESCALATION_QUEUE_KEY, "an_5") is None
    # Audit record written, enriched from the queue entry since the caller
    # (mirroring the human-reviewer FastAPI path) didn't pass context directly.
    record = db["reconciliation_decisions"].find_one({"anomaly_id": "an_5"})
    assert record is not None
    assert record["decision"] == "approved"
    assert record["decided_by"] == "jane_reviewer"
    assert record["notes"] == "Confirmed duplicate, refunded."
    assert record["anomaly_type"] == "duplicate_charge"
    assert record["transaction_ids"] == ["f", "g"]
    assert record["reasoning"] == "Duplicate."


def test_submit_decision_deferred_keeps_it_queued():
    escalate_tool(
        anomaly_id="an_6", anomaly_type="missing_record",
        transaction_ids=["h"], severity="medium", reasoning="Needs more info.",
    )

    result = submit_tool(anomaly_id="an_6", decision="deferred", decided_by="jane_reviewer")

    assert result["removed_from_queue"] is False
    # Still queued.
    assert redis_client.hget(_ESCALATION_QUEUE_KEY, "an_6") is not None
    # But the deferral itself is still an audited event.
    record = db["reconciliation_decisions"].find_one({"anomaly_id": "an_6"})
    assert record is not None
    assert record["decision"] == "deferred"


def test_submit_decision_auto_resolved_never_queued_uses_caller_context():
    """lookup_missing_records_node's call site: the anomaly was never
    escalated, so there's no Redis entry to enrich from -- context comes
    entirely from the caller's explicit params, and the HDEL is a no-op.
    """
    result = submit_tool(
        anomaly_id="an_7",
        decision="auto_resolved_via_lookup",
        decided_by="system",
        anomaly_type="missing_record",
        transaction_ids=["i"],
        reasoning="Resolved via cross-batch lookup.",
    )

    assert result["removed_from_queue"] is False
    record = db["reconciliation_decisions"].find_one({"anomaly_id": "an_7"})
    assert record["anomaly_type"] == "missing_record"
    assert record["decided_by"] == "system"


def test_submit_decision_dismissed_dequeues_without_context():
    """A reviewer dismissing something that was, for whatever reason,
    never queued (or already removed) shouldn't error -- HDEL on a
    missing key is a harmless no-op, and the audit record is written with
    whatever context is available (none, here).
    """
    result = submit_tool(anomaly_id="an_8", decision="dismissed", decided_by="jane_reviewer")

    assert result["removed_from_queue"] is False
    record = db["reconciliation_decisions"].find_one({"anomaly_id": "an_8"})
    assert record["decision"] == "dismissed"
    assert record["anomaly_type"] is None


# --- Week 9-10: episodic memory write, alongside the Mongo audit write --


def test_submit_decision_approved_also_writes_an_episode():
    escalate_tool(
        anomaly_id="an_mem_1", anomaly_type="duplicate_charge",
        transaction_ids=["p", "q"], severity="high", reasoning="Duplicate charge for order #700.",
    )

    submit_tool(anomaly_id="an_mem_1", decision="approved", decided_by="jane_reviewer", notes="Confirmed.")

    matches = find_similar_episodes(
        anomaly_type="duplicate_charge", severity="high", reasoning="Duplicate charge for order #700.",
    )
    ids = {m["anomaly_id"] for m in matches}
    assert "an_mem_1" in ids
    match = next(m for m in matches if m["anomaly_id"] == "an_mem_1")
    assert match["decision"] == "approved"
    assert match["decided_by"] == "jane_reviewer"


def test_submit_decision_deferred_never_writes_an_episode():
    """"deferred" is an audit event, not a resolution -- per the design
    session it should never become an episode, unlike every other decision.
    """
    escalate_tool(
        anomaly_id="an_mem_2", anomaly_type="missing_record",
        transaction_ids=["r"], severity="medium", reasoning="Needs more info before a call can be made.",
    )

    submit_tool(anomaly_id="an_mem_2", decision="deferred", decided_by="jane_reviewer")

    matches = find_similar_episodes(
        anomaly_type="missing_record", severity="medium", reasoning="Needs more info before a call can be made.",
    )
    assert all(m["anomaly_id"] != "an_mem_2" for m in matches)


def test_submit_decision_auto_resolved_writes_episode_with_caller_supplied_severity():
    """lookup_missing_records_node's call site supplies severity directly
    (it has the full Anomaly object, never queued) -- confirm severity
    flows all the way through to the stored episode, not just anomaly_type/
    transaction_ids/reasoning.
    """
    submit_tool(
        anomaly_id="an_mem_3",
        decision="auto_resolved_via_lookup",
        decided_by="system",
        anomaly_type="missing_record",
        severity="low",
        transaction_ids=["s"],
        reasoning="Resolved via cross-batch lookup.",
    )

    matches = find_similar_episodes(
        anomaly_type="missing_record", severity="low", reasoning="Resolved via cross-batch lookup.",
    )
    match = next(m for m in matches if m["anomaly_id"] == "an_mem_3")
    assert match["severity"] == "low"
    assert match["decision"] == "auto_resolved_via_lookup"


# --- app/mcp_client.py: wrapper functions over a session --------------


def _anomaly(**overrides) -> Anomaly:
    base = dict(
        anomaly_id="an_client_1",
        anomaly_type=AnomalyType.DUPLICATE_CHARGE,
        transaction_ids=["x", "y"],
        severity="high",
        confidence=0.9,
        reasoning="Two identical charges.",
        requires_human_review=True,
    )
    base.update(overrides)
    return Anomaly.model_validate(base)


def test_client_escalate_dispute_queues_via_session():
    anomaly = _anomaly()

    async def run():
        async with lookup_session() as session:
            return await client_escalate_dispute(session, anomaly)

    result = asyncio.run(run())
    assert result == {"queued": True, "anomaly_id": "an_client_1"}
    assert redis_client.hget(_ESCALATION_QUEUE_KEY, "an_client_1") is not None


def test_client_escalate_dispute_passes_episodic_context_through():
    """The client wrapper used to drop anomaly.episodic_context on the
    floor when building the tool call -- confirm it now reaches Redis.
    """
    anomaly = _anomaly(
        anomaly_id="an_client_1b",
        episodic_context="Similar past decisions: approved by jane_reviewer (Confirmed duplicate, refunded.).",
    )

    async def run():
        async with lookup_session() as session:
            return await client_escalate_dispute(session, anomaly)

    asyncio.run(run())
    record = json.loads(redis_client.hget(_ESCALATION_QUEUE_KEY, "an_client_1b"))
    assert record["episodic_context"] == "Similar past decisions: approved by jane_reviewer (Confirmed duplicate, refunded.)."


def test_client_list_and_submit_round_trip_via_session():
    anomaly = _anomaly(anomaly_id="an_client_2")

    async def run():
        async with lookup_session() as session:
            await client_escalate_dispute(session, anomaly)
            pending = await client_list_pending_escalations(session)
            decision = await client_submit_reconciliation_decision(
                session,
                anomaly_id="an_client_2",
                decision="dismissed",
                decided_by="jane_reviewer",
                notes="False positive.",
            )
            return pending, decision

    pending, decision = asyncio.run(run())
    assert any(p["anomaly_id"] == "an_client_2" for p in pending)
    assert decision["removed_from_queue"] is True

    record = db["reconciliation_decisions"].find_one({"anomaly_id": "an_client_2"})
    assert record["decision"] == "dismissed"
    # Enriched from the queue entry the client's escalate_dispute call wrote.
    assert record["anomaly_type"] == "duplicate_charge"
