"""Schema for data retrieved from the reason-code knowledge graph (Week 5-6).

This is what a single Cypher traversal returns -- not what's stored in
Neo4j itself (that's the node/relationship schema documented in
app/data/reason_codes.py and the SecondBrain GraphDB_Best_Practices.md
note). This model is the *retrieval* contract the Reasoning Agent depends
on, identical whether the data came from real Neo4j or the in-memory
fallback in app/graph_db.py.
"""
from pydantic import BaseModel


class ReasonCodeContext(BaseModel):
    code: str
    description: str
    network: str
    category: str
    liable_party: str
    evidence_required: str
    response_window_days: int
