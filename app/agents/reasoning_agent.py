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

    # Sept 30 design note: episodic_context started as a flat "decision by
    # reviewer" line per episode, joined with "; ". That reads as noise more
    # than signal -- anomaly_agent.py's reasoning templates are formulaic
    # enough (they mostly vary by transaction_id) that several retrieved
    # episodes routinely render identically, and a reviewer has to mentally
    # tally "ok, 2 of these were auto-resolved, 1 was approved" themselves.
    # This renders that tally directly: episodes are grouped by decision
    # (closest-first, i.e. find_similar_episodes()'s own order), each group
    # reports its count and who decided it, and reviewer names are collapsed
    # only within a group so "by system" doesn't drown out "by John" or vice
    # versa. Full per-episode detail (including distance and notes) is still
    # available via "Show similar past decisions" in the UI -- this is a
    # summary on top of that, not a replacement for it.
    groups: dict[str, dict] = {}
    order: list[str] = []
    for e in episodes:
        decision = e["decision"]
        if decision not in groups:
            groups[decision] = {"count": 0, "reviewers": []}
            order.append(decision)
        groups[decision]["count"] += 1
        groups[decision]["reviewers"].append(e["decided_by"])

    total = len(episodes)
    case_word = "case" if total == 1 else "cases"

    def render_reviewers(names: list[str]) -> str:
        uniq = list(dict.fromkeys(names))  # dedupe, preserve first-seen order
        if len(uniq) == 1:
            return uniq[0]
        if len(uniq) == 2:
            return f"{uniq[0]} and {uniq[1]}"
        return f"{uniq[0]}, {uniq[1]}, and {len(uniq) - 2} other{'s' if len(uniq) > 3 else ''}"

    if len(order) == 1:
        decision = order[0]
        who = render_reviewers(groups[decision]["reviewers"])
        qualifier = "all " if total > 1 else ""
        summary = f"{total} similar past {case_word}: {qualifier}{_humanize_decision(decision)}, by {who}."
    else:
        parts = []
        for decision in order:
            group = groups[decision]
            who = render_reviewers(group["reviewers"])
            count_prefix = f"{group['count']} " if group["count"] > 1 else ""
            parts.append(f"{count_prefix}{_humanize_decision(decision)} by {who}")
        summary = f"{total} similar past {case_word}: " + ", ".join(parts) + "."

    # No "Similar past decisions:" prefix here -- the UI (EscalationDetail.tsx)
    # already labels this field "Similar past context:" before rendering it,
    # so a second label here just duplicated the framing.
    return anomaly.model_copy(update={"episodic_context": summary})


def _humanize_decision(decision: str) -> str:
    """Render a stored decision value (e.g. "auto_resolved_via_lookup") as
    a readable phrase for display in episodic_context. Purely cosmetic --
    the underlying decision values stored in memory_store.py and returned
    from GET /escalations/{id}/similar are left untouched."""
    return decision.replace("_", " ")


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
