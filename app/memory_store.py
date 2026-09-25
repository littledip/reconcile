"""Week 9-10: episodic memory -- a Chroma-backed store of past
reconciliation decisions, retrievable by semantic similarity rather than
just by anomaly_id.

Full design rationale lives in Progress_Log.md's Sept 23 "Week 9-10
kickoff: episodic memory design session" entry; this module implements it.
The short version: the Neo4j reason-code graph (app/graph_db.py) is this
project's *semantic* memory -- static, timeless facts. This module is the
*episodic* half -- specific past decisions, retrievable by meaning. The raw
material already exists, structurally intact, in ReconciliationDecision
(app/models/reconciliation.py) / the `reconciliation_decisions` Mongo
collection; this module makes it retrievable by similarity instead of only
by exact anomaly_id.

Client/server boundary (revisiting the same knowledge-retrieval-vs-tool-use
split Week 5-6 drew for Neo4j vs. the cross-batch MCP lookup): retrieval
here stays in-process, mirroring app/graph_db.py's shape, since it's
grounding-by-meaning, not "check whether X exists beyond this batch." The
*write* path is the one exception -- app/mcp_server.py's
submit_reconciliation_decision calls write_episode() right after its Mongo
insert, so the audit record and the embedded episode can never drift out
of sync with each other.

Embedded text vs. metadata split (the actual point of this design): only
anomaly-side fields that exist *before* a decision -- anomaly_type,
severity, reasoning -- ever influence what gets matched. build_episode_context()
is the one template used identically at write-time (from a resolved
ReconciliationDecision's own anomaly-side fields) and at query-time (from
an anomaly still being reasoned about, with no decision yet) -- so a live
query and a stored episode are always comparable in the same embedding
space. Decision-side fields (decision/decided_by/notes/decided_at) ride
along as unembedded Chroma metadata: returned once a match is found, never
part of the similarity comparison.

Embedding model: Chroma's built-in local default (all-MiniLM-L6-v2, via
Sentence Transformers/onnxruntime) -- no API key, downloads once on first
use, consistent with every other backend's "runs with nothing configured"
property. Voyage AI (Anthropic's recommended third-party embeddings
provider, since Anthropic has no first-party embeddings API) is a noted
future upgrade, not built here.

Client/storage: one PersistentClient used everywhere -- real runs and
tests alike -- rather than a second EphemeralClient code path. Chroma
doesn't need a mock the way Mongo/Redis do (no Docker dependency to work
around), so a second client implementation would only add a mock-mode
parity risk (the exact lesson from the Sept 22 write-tools session) with no
corresponding benefit. Tests get a disposable temp directory instead of a
different client class -- see reset_client_for_tests() below and
tests/conftest.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb

from app.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PERSIST_DIR = _PROJECT_ROOT / "chroma_data"
_COLLECTION_NAME = "reconciliation_episodes"

_DEFAULT_TOP_K = 3
# Chroma's default embedding space uses cosine *distance* (0 = identical,
# 2 = opposite) -- not a similarity score. 0.8 is an empirically-picked
# cutoff, not a guess: a genuinely-similar duplicate_charge episode scored
# ~0.30 against a same-type/different-merchant query in manual testing,
# while an unrelated missing_record episode scored ~1.27 against the same
# query. 0.8 sits well clear of both, but is a first-pass default -- worth
# revisiting once there's real accumulated data to tune against (see the
# eval framework, still to be scoped).
_DEFAULT_MAX_DISTANCE = 0.8

_client: Optional["chromadb.ClientAPI"] = None


def _get_client() -> "chromadb.ClientAPI":
    global _client
    if _client is None:
        persist_dir = Path(settings.chroma_persist_dir) if settings.chroma_persist_dir else _DEFAULT_PERSIST_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_dir))
    return _client


def reset_client_for_tests() -> None:
    """Test-only: drop the cached client so the next call to _get_client()
    rebuilds against whatever settings.chroma_persist_dir currently points
    at. Lets tests/conftest.py redirect storage to a disposable per-test
    temp directory without a second client implementation to maintain --
    see this module's docstring for why that's the design, not just a
    convenience.
    """
    global _client
    _client = None


def _get_collection():
    return _get_client().get_or_create_collection(_COLLECTION_NAME)


def build_episode_context(anomaly_type: str, severity: str, reasoning: str) -> str:
    """The one template used to build the text Chroma embeds and searches
    against -- identically at write time and query time (see module
    docstring). Deliberately excludes every decision-side field.

    `reasoning` is already rich in practice: anomaly_agent.py's template
    reasoning names the merchant/amount/order reference, and
    reasoning_agent.py's graph-RAG grounding (when it applies) adds the
    cited liability detail on top of that -- this function doesn't need to
    reconstruct transaction context separately, it's already there.
    """
    return f"{severity} severity {anomaly_type}: {reasoning}"


def write_episode(
    *,
    anomaly_id: str,
    anomaly_type: str,
    severity: Optional[str],
    reasoning: Optional[str],
    decision: str,
    decided_by: str,
    notes: Optional[str],
    decided_at: str,
    transaction_ids: list[str],
) -> None:
    """Embed and store one resolved anomaly as an episode.

    Called from submit_reconciliation_decision (app/mcp_server.py) right
    after its Mongo audit-record write, for every decision except
    "deferred" (an audit event with no real resolution yet -- would only
    add noise to the store, per the design session).

    Uses anomaly_id as the Chroma document id: upsert rather than add, so
    calling this twice for the same anomaly_id (shouldn't normally happen
    once resolved, but matches the upsert-on-conflict idempotency already
    used elsewhere in this project -- persist_batch_node, escalate_dispute)
    overwrites rather than duplicating.

    severity/reasoning are Optional to match ReconciliationDecision's own
    typing (both are None-able there), but an episode with no severity/
    reasoning has nothing meaningful to embed -- callers should have real
    values in every actual code path (submit_reconciliation_decision
    always resolves severity/reasoning, from the caller or the Redis queue
    entry, before this is called); a missing value here falls back to
    "unknown" rather than raising, so a future caller with genuinely
    incomplete context degrades instead of breaking the write-tools path.
    """
    collection = _get_collection()
    # Chroma metadata values must be str/int/float/bool -- not list --
    # so transaction_ids is comma-joined going in and split back out in
    # find_similar_episodes(). Transaction IDs in this project never
    # contain commas (see app/models/transaction.py), so this is lossless.
    collection.upsert(
        ids=[anomaly_id],
        documents=[build_episode_context(anomaly_type, severity or "unknown", reasoning or "")],
        metadatas=[
            {
                "anomaly_id": anomaly_id,
                "anomaly_type": anomaly_type,
                "severity": severity or "unknown",
                "decision": decision,
                "decided_by": decided_by,
                "notes": notes or "",
                "decided_at": decided_at,
                "transaction_ids": ",".join(transaction_ids),
            }
        ],
    )


def find_similar_episodes(
    *,
    anomaly_type: str,
    severity: str,
    reasoning: str,
    k: int = _DEFAULT_TOP_K,
    max_distance: float = _DEFAULT_MAX_DISTANCE,
) -> list[dict]:
    """Return up to k past episodes similar to the given (not-yet-decided)
    anomaly context, using the same build_episode_context() template used
    at write time -- see module docstring for why that symmetry matters.

    Applies a minimum-similarity cutoff rather than always returning k
    results regardless of quality: 0 to k results, never padded out with
    weak matches (per the design session). max_distance is a cutoff on
    Chroma's cosine-distance scale (0 = identical, 2 = opposite), not a
    similarity score -- lower is more similar.

    Returns each match as a dict: anomaly_id, anomaly_type, severity,
    decision, decided_by, notes, decided_at, transaction_ids (split back
    out of its comma-joined metadata form), and distance.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_text = build_episode_context(anomaly_type, severity, reasoning)
    result = collection.query(query_texts=[query_text], n_results=min(k, count))

    matches: list[dict] = []
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for doc_id, meta, distance in zip(ids, metadatas, distances):
        if distance > max_distance:
            continue
        raw_txn_ids = meta.get("transaction_ids") or ""
        matches.append(
            {
                "anomaly_id": meta.get("anomaly_id", doc_id),
                "anomaly_type": meta.get("anomaly_type"),
                "severity": meta.get("severity"),
                "decision": meta.get("decision"),
                "decided_by": meta.get("decided_by"),
                "notes": meta.get("notes") or None,
                "decided_at": meta.get("decided_at"),
                "transaction_ids": raw_txn_ids.split(",") if raw_txn_ids else [],
                "distance": distance,
            }
        )
    return matches
