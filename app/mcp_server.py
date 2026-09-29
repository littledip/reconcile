"""Week 7-8: standalone MCP server for live, beyond-the-current-batch
transaction lookups.

Why a real MCP server rather than a plain function call: weeks 5-6 already
covered *knowledge retrieval* (static reason-code reference data, via Graph
RAG -- app/graph_db.py, app/agents/reasoning_agent.py). What's left, and
what target JDs mean by "hands-on MCP server development," is *tool use*: a
live lookup beyond what's already in the graph's state, reached over a
genuine client/server boundary -- a separate process, spoken to over the MCP
protocol -- not an in-process helper wearing an MCP-shaped API. See
Progress_Log.md for the fuller reasoning.

This server exposes exactly one tool, lookup_transaction_by_reference: given
one transaction from the batch currently being processed, look for a
counterpart among transactions persisted by *previous* batch runs (the
`transactions` collection written by the new persist_batch node in
app/agents/orchestrator.py). "Missing" from the current batch doesn't always
mean missing altogether -- the original charge for a refund, or the
disputed charge for a chargeback, may simply have landed in an earlier
batch. That's exactly the gap this tool closes.

Deliberately deterministic, not LLM-based -- same rationale as the
Reconciliation Agent (app/agents/reconciliation_agent.py), which this tool
mirrors: a real reconciliation match needs to be auditable and reproducible,
not a matter of model judgment. In fact the matching strategy here is the
*same* two-pass strategy (order-id-and-amount, then equal-and-opposite-
amount-within-30-days) -- the only thing that changes crossing this
client/server boundary is which pool of transactions gets searched
(persisted history vs. the in-memory current batch).

Run directly for manual testing:
    python3 -m app.mcp_server
In production use, the orchestrator spawns this as a subprocess per batch
run and talks to it as an MCP client (see app/mcp_client.py) -- it is never
imported directly by orchestrator.py.

Write-tools phase (still Week 7-8's scope, built after the read-only
lookup tool above and the real-Mongo verification pass -- see
Progress_Log.md) adds three more tools, all governing the escalation
queue/audit trail the project's README describes ("submit reconciliation
decisions, escalate disputes"):

  - escalate_dispute: pushes one anomaly onto a live queue -- a Redis hash
    (`escalation_queue`) keyed by anomaly_id -- for human review. Called by
    escalate_node (app/agents/orchestrator.py), once per anomaly with
    requires_human_review=True.

  - list_pending_escalations: read-only, HGETALLs that same hash. Backs
    the GET /escalations FastAPI endpoint.

  - submit_reconciliation_decision: writes a permanent audit record (see
    ReconciliationDecision, app/models/reconciliation.py) to a Mongo
    collection (`reconciliation_decisions`), and removes the anomaly from
    the Redis queue -- unless the decision is "deferred", which keeps it
    queued for a later reviewer while still recording that someone looked
    at it. Two callers: lookup_missing_records_node, automatically, right
    before dropping an anomaly it resolved via live lookup
    (decision="auto_resolved_via_lookup"); and the human-reviewer path,
    POST /escalations/{anomaly_id}/decision (decision="approved" /
    "dismissed" / "deferred"). The reviewer path only supplies
    decision/decided_by/notes -- this tool fills in anomaly_type/severity/
    transaction_ids/reasoning from the anomaly's own queue entry when one
    exists, so the caller never has to re-supply context escalate_dispute
    already recorded.

  Redis is the live/working queue; Mongo is the permanent record once
  something is resolved off it -- same split the project already uses
  Redis (app/cache.py, otherwise unused before this) and Mongo for
  elsewhere.

Week 9-10 adds one more write inside submit_reconciliation_decision, right
after the Mongo insert: app/memory_store.py's write_episode(), for every
decision except "deferred". This is the one write that crosses out of this
module's own write-tools boundary into the episodic-memory module -- see
that module's docstring and Progress_Log.md's Sept 23 design-session entry
for why retrieval stays in-process there while the write happens here,
alongside the audit record it can never be allowed to drift out of sync
with.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.cache import ESCALATION_EVENTS_CHANNEL, redis_client
from app.db import db
from app.memory_store import write_episode
from app.models.reconciliation import ReconciliationDecision

mcp = FastMCP("reconcile-lookup")

_ORDER_RE = re.compile(r"order #(\d+)", re.IGNORECASE)
_MATCH_WINDOW_DAYS = 30
_ESCALATION_QUEUE_KEY = "escalation_queue"
_DECISIONS_COLLECTION = "reconciliation_decisions"


def _extract_order_id(description: str | None) -> str | None:
    if not description:
        return None
    m = _ORDER_RE.search(description)
    return m.group(1) if m else None


def _parse_timestamp(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _not_found(reasoning: str) -> dict:
    return {"found": False, "matched_transaction_id": None, "match_type": None, "reasoning": reasoning}


@mcp.tool()
def lookup_transaction_by_reference(
    transaction_id: str,
    merchant_id: str,
    amount: str,
    timestamp: str,
    description: str | None = None,
) -> dict:
    """Look up a counterpart for one transaction among previously persisted
    batches (i.e. outside the batch currently being processed).

    Args:
        transaction_id: the transaction being looked up (excluded from its
            own search results).
        merchant_id: restricts the search to the same merchant.
        amount: the transaction's amount, as a string (e.g. "42.50") to
            preserve Decimal precision across the MCP JSON boundary.
        timestamp: ISO-8601 timestamp, used for the 30-day match window in
            the equal-and-opposite-amount fallback pass.
        description: used to extract an order reference (e.g. "order #123")
            for the higher-confidence first pass.

    Returns:
        {"found": bool, "matched_transaction_id": str | None,
         "match_type": "duplicate_pair" | "refund_pair" | None,
         "reasoning": str}
    """
    try:
        parsed_amount = Decimal(amount)
    except InvalidOperation:
        return _not_found(f"Could not parse amount {amount!r} -- treating as not found.")

    parsed_timestamp = _parse_timestamp(timestamp)
    if parsed_timestamp is None:
        return _not_found(f"Could not parse timestamp {timestamp!r} -- treating as not found.")

    order_id = _extract_order_id(description)
    collection = db["transactions"]

    # Pass 1: same merchant + same order reference + same amount.
    if order_id:
        for candidate in collection.find({"merchant_id": merchant_id}):
            if candidate.get("transaction_id") == transaction_id:
                continue
            if _extract_order_id(candidate.get("description")) != order_id:
                continue
            try:
                candidate_amount = Decimal(str(candidate.get("amount")))
            except InvalidOperation:
                continue
            if candidate_amount != parsed_amount:
                continue
            return {
                "found": True,
                "matched_transaction_id": candidate["transaction_id"],
                "match_type": "duplicate_pair",
                "reasoning": (
                    f"Found {candidate['transaction_id']} in a previously persisted batch, "
                    f"referencing the same order #{order_id} for merchant {merchant_id} with "
                    f"identical amount {parsed_amount}."
                ),
            }

    # Pass 2: equal-and-opposite amount, same merchant, within the match window.
    for candidate in collection.find({"merchant_id": merchant_id}):
        if candidate.get("transaction_id") == transaction_id:
            continue
        try:
            candidate_amount = Decimal(str(candidate.get("amount")))
        except InvalidOperation:
            continue
        if candidate_amount != -parsed_amount:
            continue
        candidate_timestamp = _parse_timestamp(candidate.get("timestamp"))
        if candidate_timestamp is None:
            continue
        if abs((parsed_timestamp - candidate_timestamp).days) > _MATCH_WINDOW_DAYS:
            continue
        return {
            "found": True,
            "matched_transaction_id": candidate["transaction_id"],
            "match_type": "refund_pair",
            "reasoning": (
                f"Found {candidate['transaction_id']} in a previously persisted batch for merchant "
                f"{merchant_id}, with equal-and-opposite amount ({candidate_amount} vs "
                f"{parsed_amount}) within {_MATCH_WINDOW_DAYS} days."
            ),
        }

    return _not_found("No counterpart found in previously persisted batches.")


@mcp.tool()
def escalate_dispute(
    anomaly_id: str,
    anomaly_type: str,
    transaction_ids: list[str],
    severity: str,
    reasoning: str,
    episodic_context: Optional[str] = None,
) -> dict:
    """Push one anomaly onto the live escalation queue for human review.

    Writes a JSON record into a Redis hash (`escalation_queue`) keyed by
    anomaly_id. Overwrites any existing entry for the same anomaly_id --
    re-escalating (e.g. a re-run batch) updates the queued record rather
    than erroring, the same idempotency idea persist_batch_node's upsert
    already uses.

    Args:
        anomaly_id: the Anomaly's own id.
        anomaly_type: one of the AnomalyType values (as a string).
        transaction_ids: the anomaly's source transaction id(s).
        severity: "low" | "medium" | "high".
        reasoning: the anomaly's (possibly graph-RAG-enriched) explanation.
        episodic_context: reasoning_node's similar-past-decision summary
            (app/memory_store.py), if any cleared the similarity cutoff --
            None otherwise. Was previously computed by reasoning_node for
            every anomaly and then discarded for the escalated branch
            specifically (a reviewer only saw it via a separate GET
            /escalations/{anomaly_id}/similar call); now carried straight
            through so GET /escalations shows it inline instead.

    Returns:
        {"queued": True, "anomaly_id": str}
    """
    record = {
        "anomaly_id": anomaly_id,
        "anomaly_type": anomaly_type,
        "transaction_ids": transaction_ids,
        "severity": severity,
        "reasoning": reasoning,
        "episodic_context": episodic_context,
        "escalated_at": datetime.now(timezone.utc).isoformat(),
    }
    redis_client.hset(_ESCALATION_QUEUE_KEY, anomaly_id, json.dumps(record))

    # Week 11+: tell any connected SSE clients the queue changed. See
    # app/cache.py's ESCALATION_EVENTS_CHANNEL and app/main.py's
    # subscriber task / GET /escalations/stream.
    redis_client.publish(ESCALATION_EVENTS_CHANNEL, "queue_changed")

    return {"queued": True, "anomaly_id": anomaly_id}


@mcp.tool()
def list_pending_escalations() -> dict:
    """Return every anomaly currently sitting in the live escalation queue.

    Read-only; backs the GET /escalations FastAPI endpoint.

    Returns:
        {"escalations": [<queue record>, ...]} -- wrapped in an object
        rather than a bare list, since MCP's structuredContent must be a
        JSON object (the client wrapper, app/mcp_client.py, unwraps this
        back to a plain list for callers).
    """
    raw = redis_client.hgetall(_ESCALATION_QUEUE_KEY)
    return {"escalations": [json.loads(v) for v in raw.values()]}


@mcp.tool()
def submit_reconciliation_decision(
    anomaly_id: str,
    decision: str,
    decided_by: str,
    notes: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    severity: Optional[str] = None,
    transaction_ids: Optional[list[str]] = None,
    reasoning: Optional[str] = None,
) -> dict:
    """Write a permanent audit record for one anomaly's resolution,
    embed it as an episode in episodic memory (Week 9-10, unless
    decision="deferred"), and -- unless the decision is "deferred" --
    remove it from the live escalation queue.

    Two callers:
      - lookup_missing_records_node, automatically, right before dropping
        an anomaly it resolved via live lookup. Passes anomaly_type/
        severity/transaction_ids/reasoning directly (it already has the
        full Anomaly object) with decision="auto_resolved_via_lookup",
        decided_by="system". Never queued, so there's nothing to enrich
        from Redis.
      - POST /escalations/{anomaly_id}/decision, for a human reviewer
        closing something off the queue with decision="approved" /
        "dismissed" / "deferred". Only decision/decided_by/notes are
        required here -- anomaly_type/severity/transaction_ids/reasoning
        are pulled from the anomaly's own queue entry (written by
        escalate_dispute) when the caller doesn't supply them directly.

    "deferred" keeps the anomaly queued -- it's a real audit event (someone
    reviewed it) without a resolution yet, not a close-out. Re-escalating
    to a different reviewer is out of scope for now. "deferred" is also
    the one decision that never becomes an episode -- same reasoning,
    nothing resolved yet to remember.

    Returns:
        {"recorded": True, "anomaly_id": str, "decision": str,
         "removed_from_queue": bool}
    """
    queued_raw = redis_client.hget(_ESCALATION_QUEUE_KEY, anomaly_id)
    queued = json.loads(queued_raw) if queued_raw else None

    resolved_anomaly_type = anomaly_type or (queued.get("anomaly_type") if queued else None)
    resolved_severity = severity or (queued.get("severity") if queued else None)
    resolved_transaction_ids = transaction_ids or (queued.get("transaction_ids") if queued else [])
    resolved_reasoning = reasoning or (queued.get("reasoning") if queued else None)

    record = ReconciliationDecision(
        anomaly_id=anomaly_id,
        decision=decision,
        decided_by=decided_by,
        notes=notes,
        anomaly_type=resolved_anomaly_type,
        severity=resolved_severity,
        transaction_ids=resolved_transaction_ids,
        reasoning=resolved_reasoning,
        decided_at=datetime.now(timezone.utc),
    )
    db[_DECISIONS_COLLECTION].insert_one(record.model_dump(mode="json"))

    # Episodic memory write (Week 9-10) -- co-located with the audit
    # insert above so the two can't drift out of sync. Skipped for
    # "deferred" (nothing resolved yet) and for the rare case of no
    # anomaly context at all (e.g. dismissing an anomaly_id that was
    # never queued and no context was supplied directly) -- an episode
    # with no real anomaly_type/reasoning would just be noise in the
    # store, not a useful memory.
    if decision != "deferred" and resolved_anomaly_type:
        write_episode(
            anomaly_id=anomaly_id,
            anomaly_type=resolved_anomaly_type,
            severity=resolved_severity,
            reasoning=resolved_reasoning,
            decision=decision,
            decided_by=decided_by,
            notes=notes,
            decided_at=record.decided_at.isoformat(),
            transaction_ids=resolved_transaction_ids,
        )

    removed = False
    if decision != "deferred":
        removed = bool(redis_client.hdel(_ESCALATION_QUEUE_KEY, anomaly_id))
        # Only publish when the queue actually shrank -- "deferred" keeps
        # the same record queued (nothing for a reviewer's list to catch),
        # and the auto-resolved-via-lookup caller was never queued at all
        # (removed is always False there), so this naturally publishes
        # only for a real human close-out.
        if removed:
            redis_client.publish(ESCALATION_EVENTS_CHANNEL, "queue_changed")

    return {
        "recorded": True,
        "anomaly_id": anomaly_id,
        "decision": decision,
        "removed_from_queue": removed,
    }


if __name__ == "__main__":
    mcp.run()
