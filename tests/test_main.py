"""Tests for the write-tools phase's FastAPI surface: GET /escalations and
POST /escalations/{anomaly_id}/decision (app/main.py).

Uses FastAPI's TestClient (httpx under the hood), same mock-mode backends
as every other test in this suite. Both endpoints go through
app.mcp_client's lookup_session() -- the same path the orchestrator uses --
so exercising them here also covers that "for consistency" design choice
end to end, not just the tool functions in isolation
(test_mcp_escalations.py) or the orchestrator's own calls
(test_orchestrator.py).

Week 9-10 adds tests for GET /escalations/{anomaly_id}/similar -- the one
endpoint in this module that does NOT go through lookup_session() (it
calls app/memory_store.py in-process instead, per the design session).
"""
from fastapi.testclient import TestClient

from app.main import app
from app.mcp_server import escalate_dispute as escalate_tool
from app.mcp_server import submit_reconciliation_decision as submit_tool

client = TestClient(app)


def test_get_escalations_empty_when_none_queued():
    resp = client.get("/escalations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_escalations_returns_queued_items():
    escalate_tool(
        anomaly_id="ep_1", anomaly_type="duplicate_charge",
        transaction_ids=["a", "b"], severity="high", reasoning="Duplicate charge.",
    )

    resp = client.get("/escalations")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["anomaly_id"] == "ep_1"


def test_get_escalations_includes_episodic_context_field():
    """GET /escalations is a plain passthrough of the Redis record (see
    app.main's list_escalations), so once escalate_dispute stores
    episodic_context, no endpoint-side change is needed for it to show up
    here -- this just confirms that passthrough actually holds.
    """
    escalate_tool(
        anomaly_id="ep_1c", anomaly_type="duplicate_charge",
        transaction_ids=["a", "b"], severity="high", reasoning="Duplicate charge.",
        episodic_context="Similar past decisions: approved by jane_reviewer (Confirmed duplicate, refunded.).",
    )

    resp = client.get("/escalations")
    assert resp.status_code == 200
    body = resp.json()
    item = next(i for i in body if i["anomaly_id"] == "ep_1c")
    assert item["episodic_context"] == "Similar past decisions: approved by jane_reviewer (Confirmed duplicate, refunded.)."


def test_post_decision_approved_removes_from_queue():
    escalate_tool(
        anomaly_id="ep_2", anomaly_type="unresolved_chargeback",
        transaction_ids=["c"], severity="medium", reasoning="Chargeback filed.",
    )

    resp = client.post(
        "/escalations/ep_2/decision",
        json={"decision": "approved", "decided_by": "jane_reviewer", "notes": "Confirmed."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["removed_from_queue"] is True

    # It's gone from the live queue now.
    assert client.get("/escalations").json() == []


def test_post_decision_deferred_keeps_it_in_queue():
    escalate_tool(
        anomaly_id="ep_3", anomaly_type="missing_record",
        transaction_ids=["d"], severity="low", reasoning="Needs more info.",
    )

    resp = client.post(
        "/escalations/ep_3/decision",
        json={"decision": "deferred", "decided_by": "jane_reviewer"},
    )
    assert resp.status_code == 200
    assert resp.json()["removed_from_queue"] is False

    # Still shows up under GET /escalations for later review.
    pending_ids = {item["anomaly_id"] for item in client.get("/escalations").json()}
    assert "ep_3" in pending_ids


def test_similar_returns_404_for_an_unqueued_anomaly_id():
    resp = client.get("/escalations/not_a_real_id/similar")
    assert resp.status_code == 404


def test_similar_returns_matching_past_episodes():
    # A resolved episode already in memory from an earlier decision...
    submit_tool(
        anomaly_id="ep_past", decision="approved", decided_by="jane_reviewer",
        anomaly_type="duplicate_charge", severity="high",
        transaction_ids=["m"], reasoning="Two transactions reference order #500 for merchant merch_1 with identical amount 50.00.",
    )
    # ...and a new anomaly, similar enough, currently sitting in the queue.
    escalate_tool(
        anomaly_id="ep_current", anomaly_type="duplicate_charge",
        transaction_ids=["n", "o"], severity="high",
        reasoning="Two transactions reference order #501 for merchant merch_2 with identical amount 55.00.",
    )

    resp = client.get("/escalations/ep_current/similar")
    assert resp.status_code == 200
    body = resp.json()
    ids = {item["anomaly_id"] for item in body}
    assert "ep_past" in ids


def test_similar_returns_empty_list_when_nothing_close_enough():
    escalate_tool(
        anomaly_id="ep_lonely", anomaly_type="missing_record",
        transaction_ids=["q"], severity="low", reasoning="Monthly platform processing fee with no counterpart expected.",
    )

    resp = client.get("/escalations/ep_lonely/similar")
    assert resp.status_code == 200
    assert resp.json() == []
