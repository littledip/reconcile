"""Week 5-6: the Reasoning/Explanation Agent.

Upgrades the Anomaly Detection Agent's template-string reasoning into
grounded, cited explanations for anomalies where a dispute reason code is
available -- currently just unresolved_chargeback, since that's the only
anomaly type whose source transaction(s) carry a `reason_code` in
raw_metadata (see app/data/sample_transactions.json, txn_1004).

Like the Classification Agent, this is a genuine retrieval step -- it
queries the reason-code knowledge graph (app/graph_db.py) rather than
generating the explanation purely from a language model's own knowledge,
which is what makes the result "grounded": the liability party, evidence
requirement, and response window are looked up facts with a traceable
source (the graph), not an LLM's best guess at what a chargeback code
"probably" means.

Anomalies with no available reason code (duplicate_charge, missing_record,
or an unresolved_chargeback whose transaction happens to have no
reason_code in metadata) are passed through with their original reasoning
unchanged -- this agent only enriches when it actually has grounding to
add, rather than fabricating detail it doesn't have.
"""
from __future__ import annotations

from app.graph_db import get_reason_code_context
from app.memory_store import find_similar_episodes
from app.models.reconciliation import Anomaly, AnomalyType
from app.models.transaction import Transaction


def enrich_anomalies(anomalies: list[Anomaly], transactions: list[Transaction]) -> list[Anomaly]:
    """Public entry point: ground unresolved_chargeback anomalies in cited
    reason-code context from the knowledge graph (Week 5-6), then attach
    similar-past-decision context from episodic memory (Week 9-10) to
    every anomaly -- two independent enrichment passes over the same
    anomaly, in that order.
    """
    txns_by_id = {t.transaction_id: t for t in transactions}
    enriched: list[Anomaly] = []

    for anomaly in anomalies:
        current = anomaly

        if current.anomaly_type == AnomalyType.UNRESOLVED_CHARGEBACK:
            context = _lookup_reason_code(current, txns_by_id)
            if context is not None:
                grounded_reasoning = (
                    f"Chargeback {', '.join(current.transaction_ids)} has no matching original "
                    f"transaction in this batch -- cannot verify the dispute against a source "
                    f"charge. {context.network} reason code {context.code} "
                    f"({context.description}, category: {context.category}): per network "
                    f"liability rules, the {context.liable_party} is liable and must submit "
                    f"evidence ({context.evidence_required}) within "
                    f"{context.response_window_days} days."
                )
                current = current.model_copy(update={"reasoning": grounded_reasoning})

        current = _attach_episodic_context(current)
        enriched.append(current)

    return enriched


def _attach_episodic_context(anomaly: Anomaly) -> Anomaly:
    """Week 9-10: set `episodic_context` from similar past reconciliation
    decisions (app/memory_store.py), when the store has anything similar
    enough to clear the similarity cutoff. Applies to every anomaly this
    agent processes, not just chargebacks -- unlike the graph-RAG grounding
    above, episodic similarity isn't scoped to one anomaly type.

    Deliberately writes to `episodic_context`, never to `reasoning` --
    `reasoning` is what later gets embedded as this anomaly's own future
    episode (once it's decided, via submit_reconciliation_decision ->
    write_episode); folding "here's what happened on other past anomalies"
    into that same text would contaminate it for whoever queries against
    this one next. See Progress_Log.md's Sept 23 design-session entry.
    """
    episodes = find_similar_episodes(
        anomaly_type=anomaly.anomaly_type.value,
        severity=anomaly.severity,
        reasoning=anomaly.reasoning,
    )
    if not episodes:
        return anomaly

    summary = "; ".join(
        f"{e['decision']} by {e['decided_by']}" + (f" ({e['notes']})" if e.get("notes") else "")
        for e in episodes
    )
    return anomaly.model_copy(update={"episodic_context": f"Similar past decisions: {summary}."})


def _lookup_reason_code(anomaly: Anomaly, txns_by_id: dict[str, Transaction]):
    for txn_id in anomaly.transaction_ids:
        txn = txns_by_id.get(txn_id)
        if txn is None:
            continue
        reason_code = txn.raw_metadata.get("reason_code")
        if reason_code is None:
            continue
        context = get_reason_code_context(reason_code)
        if context is not None:
            return context
    return None
