"""Schemas for the Reconciliation/Matching and Anomaly Detection agents (Week 3-4).

These consume the Classification Agent's output (Week 1-2) and operate over
a *batch* of transactions from both simulated systems, rather than one
transaction in isolation — matching is inherently a cross-record operation.

Week 7-8 (write-tools phase) adds ReconciliationDecision: the permanent
audit-record shape written by the submit_reconciliation_decision MCP tool
(app/mcp_server.py) — see that module for the escalation-queue/decision
design.
"""
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransactionMatch(BaseModel):
    match_id: str
    transaction_ids: list[str]
    match_type: Literal["duplicate_pair", "refund_pair", "chargeback_pair", "amount_mismatch_pair"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AnomalyType(str, Enum):
    DUPLICATE_CHARGE = "duplicate_charge"
    UNRESOLVED_CHARGEBACK = "unresolved_chargeback"
    MISSING_RECORD = "missing_record"
    AMOUNT_MISMATCH = "amount_mismatch"


class Anomaly(BaseModel):
    anomaly_id: str
    anomaly_type: AnomalyType
    transaction_ids: list[str]
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    requires_human_review: bool = False
    # Week 9-10: similar-past-decision context from episodic memory
    # (app/memory_store.py), set by reasoning_agent.py when the store has
    # anything similar enough to be useful. Deliberately separate from
    # `reasoning` rather than appended onto it -- `reasoning` is what
    # eventually gets embedded as *this* anomaly's own future episode
    # (via submit_reconciliation_decision -> write_episode), and folding
    # "here's what happened on other past anomalies" into that text would
    # contaminate it for anyone querying against this one later. None
    # until reasoning_agent.py sets it; still None if no similar episode
    # clears the similarity cutoff.
    episodic_context: Optional[str] = None


class ReconciliationDecision(BaseModel):
    """Permanent audit record written to Mongo (`reconciliation_decisions`)
    by submit_reconciliation_decision, one per anomaly close-out.

    Two call sites, two provenances:
      - lookup_missing_records_node calls this automatically, with
        decision="auto_resolved_via_lookup", decided_by="system", and the
        anomaly's own context (it already has the full Anomaly object).
      - POST /escalations/{anomaly_id}/decision calls this for a human
        reviewer closing something off the queue ("approved" / "dismissed"
        / "deferred"). anomaly_type/severity/transaction_ids/reasoning are
        optional here since the reviewer only supplies decision/decided_by/
        notes -- the tool enriches from the Redis queue entry
        escalate_dispute wrote, when one exists for that anomaly_id.

    "deferred" is the one decision that does NOT remove the anomaly from
    the Redis escalation queue -- it's a real audit event (someone looked
    at it) without a resolution yet. Re-escalating to a different reviewer
    is out of scope for now.

    Week 9-10 adds `severity`, alongside the existing anomaly_type/
    transaction_ids/reasoning trio -- app/memory_store.py's episodic
    embedding needs it (severity + anomaly_type + reasoning is the whole
    embedded-context template), and it wasn't otherwise carried from
    Anomaly onto this permanent record. Same enrichment path as the other
    three: supplied directly by lookup_missing_records_node (it has the
    full Anomaly object), pulled from the Redis queue entry for the
    human-reviewer path.
    """

    anomaly_id: str
    decision: Literal["auto_resolved_via_lookup", "approved", "dismissed", "deferred"]
    decided_by: str
    notes: Optional[str] = None
    anomaly_type: Optional[AnomalyType] = None
    severity: Optional[Literal["low", "medium", "high"]] = None
    transaction_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    decided_at: datetime
