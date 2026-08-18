"""FastAPI skeleton — Week 1 scope: health check + a single endpoint wrapping
the Classification Agent. Ingestion, Reconciliation/Matching, Anomaly
Detection, and Reasoning agents (and the orchestrator that routes between
them) land in weeks 3-4+ per the project README timeline.
"""
from fastapi import FastAPI, HTTPException

from app.agents.classification_agent import classify_transaction
from app.cache import using_mock_redis
from app.config import settings
from app.db import using_mock_mongo
from app.models.transaction import ClassificationResult, Transaction

app = FastAPI(title="Reconcile", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mongo_backend": "mongomock (in-memory)" if using_mock_mongo() else "mongodb",
        "redis_backend": "fakeredis (in-memory)" if using_mock_redis() else "redis",
        "classification_mode": "llm" if settings.anthropic_api_key else "heuristic (no ANTHROPIC_API_KEY set)",
    }


@app.post("/classify", response_model=ClassificationResult)
def classify(transaction: Transaction) -> ClassificationResult:
    try:
        return classify_transaction(transaction)
    except Exception as exc:  # pragma: no cover - week 1 skeleton, broad on purpose
        raise HTTPException(status_code=500, detail=str(exc)) from exc
