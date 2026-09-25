"""Week 3-4 + Week 5-6 + Week 7-8: the orchestrator graph.

Routes a batch of transactions through the agent pipeline, then applies a
guardrail: if anomaly detection flags anything requiring human review,
route to an escalation node before finishing; otherwise finish directly.
This is the smallest real example of the JD's "guardrails, routing, and
recovery logic" — a confidence/severity-based conditional edge, not just a
linear pipeline.

    classify_all -> match -> detect_anomalies -> lookup_missing_records
        -> reason_about_anomalies
            --[escalate?]--> escalate -> persist_batch -> END
            --[no escalate?]--> persist_batch -> END

`reason_about_anomalies` is the Week 5-6 addition: it's a node (not a
conditional edge -- see Progress_Log.md Aug 28, 2026 for why that distinction
matters), inserted before the existing escalation routing, which enriches
anomaly reasoning with grounded, cited explanations retrieved from the
reason-code knowledge graph (app/graph_db.py) wherever a dispute reason code
is available. It doesn't change which anomalies exist or whether they
escalate -- only the quality of the explanation attached to them.

Week 7-8 adds three things:

  - lookup_missing_records: a node (async -- it makes a live MCP call, see
    app/mcp_client.py) between detect_anomalies and reason_about_anomalies.
    For anomalies that might have a counterpart *outside* the current batch
    (missing_record, unresolved_chargeback -- not duplicate_charge, which
    is generated from an already-matched in-batch duplicate_pair and has
    nothing "missing" to look up, see app/agents/anomaly_agent.py), it
    calls the custom MCP lookup server (app/mcp_server.py) for live lookups
    beyond the current batch's scope -- tool use, distinct from the
    knowledge retrieval reason_about_anomalies does. A found counterpart
    resolves the anomaly and removes it from the anomaly set entirely;
    auditing that removal is deferred to a future write-tools pass
    (submit_reconciliation_decision / escalate_dispute -- see
    Progress_Log.md), out of scope here. The MCP tool call is wrapped in
    retry/backoff (app/retry.py) -- the one genuinely-fallible live call
    anywhere in this graph, unlike the deterministic in-process agents
    around it.

  - persist_batch: a node after reason_about_anomalies, on *both* branches
    of the escalation routing (escalate and finish alike, so every
    transaction is persisted regardless of outcome -- full auditability).
    It writes this batch's transactions to MongoDB so a *future* batch's
    lookup_missing_records call can find them. Because persist_batch runs
    strictly after lookup_missing_records within the same graph execution,
    this batch's own transactions are never in Mongo yet when the lookup
    fires -- no separate current-batch-ID exclusion filter needed, an
    emergent consequence of this ordering.

  - app/mcp_server.py: a standalone MCP server, not an in-process
    abstraction -- a genuine client/server boundary reached over the MCP
    protocol, matching what "hands-on MCP server development" means in
    target JDs.

Write-tools phase (the deferred third piece of Week 7-8) makes two of
those nodes write, not just read:

  - escalate_node is now async: it opens its own MCP session and calls
    escalate_dispute once per anomaly with requires_human_review=True,
    pushing it onto the live Redis-backed escalation queue
    (app/mcp_server.py) before persist_batch runs.

  - lookup_missing_records_node now calls submit_reconciliation_decision
    (in the same session it already has open for the lookup itself) right
    before dropping each anomaly it resolves -- decision=
    "auto_resolved_via_lookup", decided_by="system" -- so that resolution
    is a permanent, queryable Mongo record instead of a silent removal
    from the anomaly set.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.anomaly_agent import detect_anomalies
from app.agents.classification_agent import classify_transaction
from app.agents.reasoning_agent import enrich_anomalies
from app.agents.reconciliation_agent import match_transactions
from app.db import db
from app.mcp_client import (
    escalate_dispute,
    lookup_session,
    lookup_transaction_by_reference,
    submit_reconciliation_decision,
)
from app.models.reconciliation import Anomaly, AnomalyType, TransactionMatch
from app.models.transaction import ClassificationResult, Transaction


class OrchestratorState(TypedDict):
    transactions: list[dict]
    classifications: dict[str, dict]
    matches: list[dict]
    anomalies: list[dict]
    escalations: list[str]


def classify_all_node(state: OrchestratorState) -> OrchestratorState:
    """Run every transaction in the batch through the Classification Agent."""
    classifications: dict[str, dict] = {}
    for raw in state["transactions"]:
        txn = Transaction.model_validate(raw)
        result = classify_transaction(txn)
        classifications[txn.transaction_id] = result.model_dump(mode="json")
    return {**state, "classifications": classifications}


def match_node(state: OrchestratorState) -> OrchestratorState:
    """Run the batch through the Reconciliation/Matching Agent."""
    transactions = [Transaction.model_validate(t) for t in state["transactions"]]
    class_types = {
        tid: ClassificationResult.model_validate(c).transaction_type
        for tid, c in state["classifications"].items()
    }
    matches = match_transactions(transactions, class_types)
    return {**state, "matches": [m.model_dump(mode="json") for m in matches]}


def anomaly_node(state: OrchestratorState) -> OrchestratorState:
    """Run the batch through the Anomaly Detection Agent."""
    transactions = [Transaction.model_validate(t) for t in state["transactions"]]
    class_types = {
        tid: ClassificationResult.model_validate(c).transaction_type
        for tid, c in state["classifications"].items()
    }
    matches = [TransactionMatch.model_validate(m) for m in state["matches"]]
    anomalies = detect_anomalies(transactions, class_types, matches)
    return {**state, "anomalies": [a.model_dump(mode="json") for a in anomalies]}


def reasoning_node(state: OrchestratorState) -> OrchestratorState:
    """Week 5-6: enrich anomaly reasoning with grounded, cited explanations
    retrieved from the reason-code knowledge graph, wherever a dispute
    reason code is available on the anomaly's source transaction(s).

    Runs after detect_anomalies and before the escalation routing decision
    -- it only rewrites the `reasoning` text on existing anomalies, it
    never adds, removes, or changes the type/severity of an anomaly, so it
    has no effect on _route_after_anomalies below.
    """
    transactions = [Transaction.model_validate(t) for t in state["transactions"]]
    anomalies = [Anomaly.model_validate(a) for a in state["anomalies"]]
    enriched = enrich_anomalies(anomalies, transactions)
    return {**state, "anomalies": [a.model_dump(mode="json") for a in enriched]}


async def lookup_missing_records_node(state: OrchestratorState) -> OrchestratorState:
    """Week 7-8: call the MCP lookup server for anomalies that might have a
    counterpart outside the current batch, and drop any that are resolved.

    Only missing_record and unresolved_chargeback anomalies are eligible --
    duplicate_charge anomalies come from an already-matched in-batch
    duplicate_pair and have nothing "missing" to look up (see
    app/agents/anomaly_agent.py). Each of those two types is generated from
    exactly one unmatched transaction, so `anomaly.transaction_ids[0]` is
    always the right transaction to look up.

    A RetryExhaustedError from the MCP client (app/mcp_client.py) is
    allowed to propagate rather than being treated as "not found" -- an
    infra failure after retries is a real failure, not a business
    conclusion that no counterpart exists.
    """
    lookup_types = {AnomalyType.MISSING_RECORD, AnomalyType.UNRESOLVED_CHARGEBACK}
    anomalies = [Anomaly.model_validate(a) for a in state["anomalies"]]
    to_check = [a for a in anomalies if a.anomaly_type in lookup_types]

    if not to_check:
        return state

    transactions_by_id = {t["transaction_id"]: Transaction.model_validate(t) for t in state["transactions"]}
    resolved: list[Anomaly] = []

    async with lookup_session() as session:
        for anomaly in to_check:
            txn = transactions_by_id.get(anomaly.transaction_ids[0])
            if txn is None:
                continue
            result = await lookup_transaction_by_reference(session, txn)
            if result["found"]:
                resolved.append(anomaly)

        # Same session as the lookups above: audit every resolution as a
        # permanent Mongo record before dropping it from the anomaly set.
        # These were never escalated, so there's nothing to remove from
        # the Redis queue -- submit_reconciliation_decision handles that
        # (a no-op HDEL) on its own.
        for anomaly in resolved:
            await submit_reconciliation_decision(
                session,
                anomaly_id=anomaly.anomaly_id,
                decision="auto_resolved_via_lookup",
                decided_by="system",
                anomaly_type=anomaly.anomaly_type.value,
                severity=anomaly.severity,
                transaction_ids=anomaly.transaction_ids,
                reasoning=anomaly.reasoning,
            )

    resolved_ids = {a.anomaly_id for a in resolved}
    remaining = [a for a in anomalies if a.anomaly_id not in resolved_ids]
    return {**state, "anomalies": [a.model_dump(mode="json") for a in remaining]}


def persist_batch_node(state: OrchestratorState) -> OrchestratorState:
    """Week 7-8: persist this batch's transactions to MongoDB so a future
    batch's lookup_missing_records_node call can find them as a
    counterpart. Runs after reason_about_anomalies on both branches of the
    escalation routing -- every transaction is persisted regardless of
    outcome, not just the happy path.

    Upserts on transaction_id so re-invoking a batch (e.g. retried after an
    earlier failure) doesn't create duplicates in the collection.
    """
    collection = db["transactions"]
    for raw in state["transactions"]:
        collection.replace_one({"transaction_id": raw["transaction_id"]}, raw, upsert=True)
    return state


async def escalate_node(state: OrchestratorState) -> OrchestratorState:
    """Guardrail landing spot for anomalies that need a human in the loop.

    Async since the write-tools phase: opens its own MCP session and calls
    escalate_dispute once per anomaly with requires_human_review=True,
    pushing each onto the live Redis-backed escalation queue
    (app/mcp_server.py) for a human reviewer to pick up via GET
    /escalations. Still returns the anomaly IDs in state too, so the
    caller (CLI script / API response) has them immediately without a
    separate round trip.
    """
    to_escalate = [
        Anomaly.model_validate(a) for a in state["anomalies"] if a.get("requires_human_review")
    ]
    if not to_escalate:
        return {**state, "escalations": []}

    async with lookup_session() as session:
        for anomaly in to_escalate:
            await escalate_dispute(session, anomaly)

    return {**state, "escalations": [a.anomaly_id for a in to_escalate]}


def _route_after_anomalies(state: OrchestratorState) -> str:
    if any(a.get("requires_human_review") for a in state["anomalies"]):
        return "escalate"
    return "finish"


def build_graph():
    """Compile the classify -> match -> detect anomalies -> lookup missing
    records -> reason -> (maybe escalate) -> persist orchestrator graph.
    """
    graph = StateGraph(OrchestratorState)
    graph.add_node("classify_all", classify_all_node)
    graph.add_node("match", match_node)
    graph.add_node("detect_anomalies", anomaly_node)
    graph.add_node("lookup_missing_records", lookup_missing_records_node)
    graph.add_node("reason_about_anomalies", reasoning_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("persist_batch", persist_batch_node)

    graph.set_entry_point("classify_all")
    graph.add_edge("classify_all", "match")
    graph.add_edge("match", "detect_anomalies")
    graph.add_edge("detect_anomalies", "lookup_missing_records")
    graph.add_edge("lookup_missing_records", "reason_about_anomalies")
    graph.add_conditional_edges(
        "reason_about_anomalies",
        _route_after_anomalies,
        {"escalate": "escalate", "finish": "persist_batch"},
    )
    graph.add_edge("escalate", "persist_batch")
    graph.add_edge("persist_batch", END)

    return graph.compile()


_compiled_graph = build_graph()


async def run_reconciliation_batch(transactions: list[Transaction]) -> OrchestratorState:
    """Public entry point: run a batch of transactions through the full
    classify -> match -> detect anomalies -> lookup missing records ->
    reason -> (maybe escalate) -> persist pipeline.

    Async since Week 7-8: lookup_missing_records_node makes a live MCP
    call. Callers (FastAPI's /reconcile, scripts/run_orchestrator.py, the
    test suite) all await/asyncio.run this now -- see Progress_Log.md.
    """
    initial_state: OrchestratorState = {
        "transactions": [t.model_dump(mode="json") for t in transactions],
        "classifications": {},
        "matches": [],
        "anomalies": [],
        "escalations": [],
    }
    return await _compiled_graph.ainvoke(initial_state)
