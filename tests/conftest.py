"""Shared test fixtures.

Week 7-8 note: persist_batch_node (app/agents/orchestrator.py) writes real
data into the `transactions` collection now, and lookup_missing_records_node
reads it back -- so any test that runs the orchestrator needs a clean
collection beforehand, or an earlier test's persisted data leaks into a
later test's "not found"/"found" expectations. app.db's `db` handle is a
module-level singleton shared across the whole pytest process (mongomock in
this environment, see app/db.py), so isolation needs an explicit reset
rather than relying on process boundaries.

Write-tools phase adds the same problem for two more module-level
singletons: the `reconciliation_decisions` Mongo collection and the
`escalation_queue` Redis hash (app/cache.py's `redis_client`, fakeredis in
this environment) -- both written by escalate_node/
lookup_missing_records_node/submit_reconciliation_decision and read back
across tests, so they need the same clean-before-and-after treatment.

Week 9-10 adds a different flavor of the same problem for
app/memory_store.py's Chroma client: it's a module-level singleton too
(lazily built, see _get_client()), and unlike Mongo/Redis it has no
separate mock library to isolate against -- the design session (Sept 23)
deliberately chose one PersistentClient implementation everywhere over a
second EphemeralClient code path, specifically to avoid a mock-mode parity
risk. Isolation here means redirecting settings.chroma_persist_dir to a
disposable per-test temp directory (pytest's own tmp_path fixture) and
dropping the cached client so the next call rebuilds against it --
memory_store.reset_client_for_tests() exists for exactly this.
"""
import pytest

from app import memory_store
from app.agents import classification_agent
from app.cache import redis_client
from app.config import settings
from app.db import db
from app.mcp_server import _ESCALATION_QUEUE_KEY


@pytest.fixture(autouse=True)
def _clean_transactions_collection():
    db["transactions"].delete_many({})
    yield
    db["transactions"].delete_many({})


@pytest.fixture(autouse=True)
def _clean_escalation_state():
    db["reconciliation_decisions"].delete_many({})
    redis_client.delete(_ESCALATION_QUEUE_KEY)
    yield
    db["reconciliation_decisions"].delete_many({})
    redis_client.delete(_ESCALATION_QUEUE_KEY)


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path, monkeypatch):
    """Redirect episodic memory to a disposable per-test directory (see
    module docstring) instead of whatever CHROMA_PERSIST_DIR/./chroma_data
    a real run would use -- otherwise every test run would read and write
    the same on-disk episode store, the same leakage problem the fixtures
    above solve for Mongo/Redis.
    """
    persist_dir = str(tmp_path / "chroma_test")
    # Both the setattr (this process) and setenv (any subprocess spawned via
    # the real MCP stdio path -- see app/mcp_client.py's _server_params(),
    # which does env = dict(os.environ)) are needed: when MONGO_URI is real,
    # submit_reconciliation_decision runs inside a spawned subprocess that
    # re-imports app.config.settings fresh from its own inherited OS
    # environment, which a monkeypatch.setattr() in this process never
    # reaches. Without setenv, that subprocess's write_episode() writes to
    # the real chroma_data/ dir while this test's own find_similar_episodes()
    # queries the isolated tmp_path dir -- a mismatch that only surfaces
    # against real infra, not mock mode.
    monkeypatch.setattr(settings, "chroma_persist_dir", persist_dir)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", persist_dir)
    memory_store.reset_client_for_tests()
    yield
    memory_store.reset_client_for_tests()


@pytest.fixture(autouse=True)
def _force_heuristic_classification(monkeypatch):
    """Force heuristic-mode classification for the whole suite, regardless
    of what's actually configured/running on the machine executing tests.

    Without this, two different things could silently flip tests into real
    model calls: a real ANTHROPIC_API_KEY in .env (needed for the app's own
    real usage), or -- now that classify_node prefers local Qwen whenever
    Ollama's reachable -- simply having Ollama running locally for whatever
    other reason while the suite happens to execute. test_classification_agent.py
    and test_orchestrator.py's own docstrings already document the
    assumption that they run in heuristic mode; this fixture is what
    actually enforces it.

    classify_node never runs inside the MCP subprocess (only
    submit_reconciliation_decision/write_episode do -- see mcp_server.py),
    so unlike _isolate_chroma above, a plain monkeypatch.setattr is enough
    here; there's no subprocess re-import of settings to reach.
    """
    monkeypatch.setattr(settings, "llm_provider", "heuristic")
    classification_agent.reset_provider_cache_for_tests()
    yield
    classification_agent.reset_provider_cache_for_tests()
