"""Neo4j connection + reason-code retrieval (Week 5-6).

Same fallback philosophy as app/db.py and app/cache.py: no Docker in the
initial build environment, so this falls back to an in-memory lookup built
from the exact same seed data (app/data/reason_codes.py) that would seed a
real Neo4j instance, whenever NEO4J_URI isn't set. Swap in real Neo4j by
setting NEO4J_URI (see docker-compose.yml) and running scripts/seed_graph.py
once -- no code changes needed, get_reason_code_context() has the same
signature and return type either way.

The one Cypher query this module runs is the multi-hop traversal designed
in the Week 5-6 schema session:

    MATCH (r:ReasonCode {code: $code})-[:DEFINED_BY]->(n:CardNetwork),
          (r)-[:BELONGS_TO]->(c:DisputeCategory),
          (r)-[:GOVERNED_BY]->(l:LiabilityRule)
    RETURN r.code, r.description, n.name, c.name,
           l.liable_party, l.evidence_required, l.response_window_days

Per the "write your queries first" principle in GraphDB_Best_Practices.md,
this query is what the schema in app/data/reason_codes.py was designed
around, not the other way around.
"""
from __future__ import annotations

from app.config import settings
from app.data.reason_codes import LIABILITY_RULES, REASON_CODES
from app.models.graph import ReasonCodeContext

_USING_MOCK = settings.neo4j_uri is None

_REASON_CODE_QUERY = """
MATCH (r:ReasonCode {code: $code})-[:DEFINED_BY]->(n:CardNetwork),
      (r)-[:BELONGS_TO]->(c:DisputeCategory),
      (r)-[:GOVERNED_BY]->(l:LiabilityRule)
RETURN r.code AS code, r.description AS description,
       n.name AS network, c.name AS category,
       l.liable_party AS liable_party, l.evidence_required AS evidence_required,
       l.response_window_days AS response_window_days
"""

if not _USING_MOCK:
    from neo4j import GraphDatabase

    _driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
else:
    _driver = None


def using_mock_graph() -> bool:
    return _USING_MOCK


def get_reason_code_context(code: str) -> ReasonCodeContext | None:
    """Run the multi-hop reason-code traversal and return the grounding
    context for the Reasoning Agent, or None if the code isn't in the graph.
    """
    if _USING_MOCK:
        return _get_reason_code_context_mock(code)

    with _driver.session() as session:
        record = session.run(_REASON_CODE_QUERY, code=code).single()
        if record is None:
            return None
        return ReasonCodeContext.model_validate(dict(record))


def _get_reason_code_context_mock(code: str) -> ReasonCodeContext | None:
    """In-memory equivalent of the Cypher traversal above, built from the
    same seed data a real Neo4j instance would be loaded with -- so mock
    and real modes always return identical results for the same input.
    """
    reason = REASON_CODES.get(code)
    if reason is None:
        return None
    rule = LIABILITY_RULES[reason["liability_rule"]]
    return ReasonCodeContext(
        code=reason["code"],
        description=reason["description"],
        network=reason["network"],
        category=reason["category"],
        liable_party=rule["liable_party"],
        evidence_required=rule["evidence_required"],
        response_window_days=rule["response_window_days"],
    )
