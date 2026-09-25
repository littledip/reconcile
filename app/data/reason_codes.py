"""Representative reason-code knowledge-graph seed data (Week 5-6).

Deliberately a small, representative subset (2 card networks, 3 dispute
categories, 5 reason codes, 3 liability rules) rather than full real-world
card-network compliance-doc fidelity -- see
03_Knowledge/AI/Reconcile_Portfolio/GraphDB_Best_Practices.md in the
SecondBrain vault for the scope reasoning.

This is the single source of truth for the graph's content: scripts/seed_graph.py
uses it to populate real Neo4j, and app/graph_db.py uses the exact same
structure as the in-memory fallback when NEO4J_URI isn't set -- so mock and
real modes can never silently drift out of sync with each other.

Schema (see GraphDB_Best_Practices.md for the modeling rationale):

    (ReasonCode)-[:DEFINED_BY]->(CardNetwork)
    (ReasonCode)-[:BELONGS_TO]->(DisputeCategory)
    (ReasonCode)-[:GOVERNED_BY]->(LiabilityRule)

Liability rules are intentionally shared across multiple reason codes where
that reflects reality (e.g. most card-absent fraud codes carry the same
merchant-liable, proof-of-authorization-required rule regardless of network)
-- that reuse is the actual point of modeling this as a graph rather than a
flat table.
"""
from __future__ import annotations

LIABILITY_RULES: dict[str, dict] = {
    "fraud_standard": {
        "rule_id": "fraud_standard",
        "liable_party": "merchant",
        "evidence_required": (
            "Proof of cardholder authorization (AVS/CVV match or 3-D Secure "
            "result) and delivery confirmation"
        ),
        "response_window_days": 30,
    },
    "auth_error_standard": {
        "rule_id": "auth_error_standard",
        "liable_party": "issuer",
        "evidence_required": "None -- issuer-side authorization error, no merchant action required",
        "response_window_days": 10,
    },
    "processing_error_standard": {
        "rule_id": "processing_error_standard",
        "liable_party": "merchant",
        "evidence_required": "Corrected transaction record or proof of original valid authorization",
        "response_window_days": 20,
    },
}

CARD_NETWORKS: list[str] = ["Visa", "Mastercard"]

DISPUTE_CATEGORIES: list[str] = ["Fraud", "Authorization Error", "Processing Error"]

REASON_CODES: dict[str, dict] = {
    "10.4": {
        "code": "10.4",
        "description": "Other Fraud -- Card-Absent Environment",
        "network": "Visa",
        "category": "Fraud",
        "liability_rule": "fraud_standard",
    },
    "11.3": {
        "code": "11.3",
        "description": "Card-Present Fraud",
        "network": "Visa",
        "category": "Fraud",
        "liability_rule": "fraud_standard",
    },
    "12.1": {
        "code": "12.1",
        "description": "Late Presentment",
        "network": "Visa",
        "category": "Processing Error",
        "liability_rule": "processing_error_standard",
    },
    "4837": {
        "code": "4837",
        "description": "No Cardholder Authorization",
        "network": "Mastercard",
        "category": "Fraud",
        "liability_rule": "fraud_standard",
    },
    "4853": {
        "code": "4853",
        "description": "Cardholder Dispute -- Not Elsewhere Classified",
        "network": "Mastercard",
        "category": "Authorization Error",
        "liability_rule": "auth_error_standard",
    },
}
