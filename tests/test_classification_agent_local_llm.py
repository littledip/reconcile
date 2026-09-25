"""Real-LLM smoke test for the Classification Agent's local-model path
(Week 9-10 follow-up -- see Progress_Log.md).

Every other test in this suite forces heuristic mode
(tests/conftest.py's autouse _force_heuristic_classification), by design --
fast, deterministic, free, and what test_classification_agent.py's own
exact-output assertions already depend on. This is the one deliberate
exception: it pins llm_provider to "local" and asserts the real
langchain-ollama/ChatOllama code path -- structured output binding,
transaction_id enforcement, the graph wiring -- works end to end against a
live model, not just the heuristic fallback.

Skips (rather than fails) if Ollama isn't reachable or no model is
configured, since there's no meaningful mock for what a live model actually
returns -- the same reasoning the Docker-dependent parts of this project
would lean on if they didn't have mongomock/fakeredis available. This is
the one test in the whole suite that requires Ollama to be running.
"""
from decimal import Decimal

import pytest

from app.agents import classification_agent
from app.agents.classification_agent import classify_transaction, ollama_reachable
from app.config import settings
from app.models.transaction import Transaction, TransactionSource, TransactionType


def _txn(**overrides) -> Transaction:
    base = dict(
        transaction_id="txn_local_llm_smoke",
        source=TransactionSource.PROCESSOR_LEDGER,
        merchant_id="merch_1",
        amount=Decimal("100.00"),
        currency="USD",
        timestamp="2026-08-10T00:00:00Z",
        description="Card payment - order #1",
    )
    base.update(overrides)
    return Transaction.model_validate(base)


@pytest.fixture
def _force_local(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "local")
    classification_agent.reset_provider_cache_for_tests()
    yield
    classification_agent.reset_provider_cache_for_tests()


def test_local_qwen_returns_well_formed_result(_force_local):
    if not settings.ollama_model:
        pytest.skip("OLLAMA_MODEL not configured -- set it in .env to run this smoke test.")
    if not ollama_reachable(settings.ollama_base_url):
        pytest.skip(f"Ollama not reachable at {settings.ollama_base_url} -- start it to run this smoke test.")

    result = classify_transaction(_txn())

    assert result.transaction_id == "txn_local_llm_smoke"
    assert isinstance(result.transaction_type, TransactionType)
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning
