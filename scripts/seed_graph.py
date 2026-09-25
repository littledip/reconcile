"""Seed the reason-code knowledge graph into a real Neo4j instance.

Only meaningful once NEO4J_URI is set and pointed at a running Neo4j (see
docker-compose.yml) -- the in-memory fallback in app/graph_db.py builds its
lookup directly from app/data/reason_codes.py at import time and never
needs seeding. Idempotent: uses MERGE throughout, safe to re-run.

Run with: python scripts/seed_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.data.reason_codes import CARD_NETWORKS, DISPUTE_CATEGORIES, LIABILITY_RULES, REASON_CODES

_CONSTRAINTS = [
    "CREATE CONSTRAINT reason_code_unique IF NOT EXISTS FOR (r:ReasonCode) REQUIRE r.code IS UNIQUE",
    "CREATE CONSTRAINT card_network_unique IF NOT EXISTS FOR (n:CardNetwork) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT dispute_category_unique IF NOT EXISTS FOR (c:DisputeCategory) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT liability_rule_unique IF NOT EXISTS FOR (l:LiabilityRule) REQUIRE l.rule_id IS UNIQUE",
]

_MERGE_NETWORK = "MERGE (n:CardNetwork {name: $name})"
_MERGE_CATEGORY = "MERGE (c:DisputeCategory {name: $name})"
_MERGE_LIABILITY_RULE = """
MERGE (l:LiabilityRule {rule_id: $rule_id})
SET l.liable_party = $liable_party,
    l.evidence_required = $evidence_required,
    l.response_window_days = $response_window_days
"""
_MERGE_REASON_CODE = """
MERGE (r:ReasonCode {code: $code})
SET r.description = $description
WITH r
MATCH (n:CardNetwork {name: $network})
MERGE (r)-[:DEFINED_BY]->(n)
WITH r
MATCH (c:DisputeCategory {name: $category})
MERGE (r)-[:BELONGS_TO]->(c)
WITH r
MATCH (l:LiabilityRule {rule_id: $liability_rule})
MERGE (r)-[:GOVERNED_BY]->(l)
"""


def seed() -> None:
    """Idempotently write the reason-code knowledge graph into Neo4j."""
    if settings.neo4j_uri is None:
        print(
            "NEO4J_URI is not set -- nothing to seed. The app falls back to an "
            "in-memory lookup built directly from app/data/reason_codes.py, so "
            "this script only matters once you're running against real Neo4j "
            "(see docker-compose.yml)."
        )
        sys.exit(0)

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with driver.session() as session:
        for stmt in _CONSTRAINTS:
            session.run(stmt)

        for name in CARD_NETWORKS:
            session.run(_MERGE_NETWORK, name=name)

        for name in DISPUTE_CATEGORIES:
            session.run(_MERGE_CATEGORY, name=name)

        for rule in LIABILITY_RULES.values():
            session.run(_MERGE_LIABILITY_RULE, **rule)

        for reason in REASON_CODES.values():
            session.run(_MERGE_REASON_CODE, **reason)

    driver.close()
    print(
        f"Seeded {len(CARD_NETWORKS)} card networks, {len(DISPUTE_CATEGORIES)} "
        f"dispute categories, {len(LIABILITY_RULES)} liability rules, and "
        f"{len(REASON_CODES)} reason codes into Neo4j."
    )


if __name__ == "__main__":
    seed()
