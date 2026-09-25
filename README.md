# Reconcile

Autonomous transaction reconciliation & anomaly detection platform — portfolio project built to develop hands-on agentic AI experience (LangGraph orchestration, Graph RAG, MCP server development, episodic memory) for agentic-AI-architect roles.

Full project plan/architecture: see SecondBrain `03_Knowledge/AI/Reconcile_Portfolio/README.md`.
Progress log (dated, session-by-session): see SecondBrain `03_Knowledge/AI/Reconcile_Portfolio/Progress_Log.md`.

## Status

Weeks 1–10 built and verified against real infra (MongoDB, Redis, Neo4j, a live MCP client/server round trip, Chroma). Eval framework (grounding accuracy / latency / false-positive-negative rate / drift) is the one remaining piece — not yet scoped.

## Architecture

A LangGraph orchestrator runs each batch of transactions through a fixed pipeline, branching only at the escalation decision:

```
classify_all → match → detect_anomalies → lookup_missing_records → reason → ─┬─ escalate ─┐
                                                                               └─ (skip)  ──┴─→ persist_batch
```

- **Classification** (`app/agents/classification_agent.py`) — classifies each transaction (`payment` / `refund` / `chargeback` / `duplicate_charge` / `fee`). Three interchangeable providers — see [Classification model providers](#classification-model-providers) below.
- **Reconciliation/Matching** (`app/agents/reconciliation_agent.py`) — finds duplicate and refund/chargeback pairs within the batch.
- **Anomaly Detection** (`app/agents/anomaly_agent.py`) — flags `duplicate_charge`, `unresolved_chargeback`, `missing_record`, and `amount_mismatch` anomalies from the match results.
- **Cross-batch lookup** (`app/mcp_client.py` / `app/mcp_server.py`) — `missing_record` and `unresolved_chargeback` anomalies get one real MCP tool call (stdio client/server) to check whether their counterpart transaction showed up in a *different* batch; a hit auto-resolves the anomaly and writes a permanent audit record, no human involved.
- **Reasoning** (`app/agents/reasoning_agent.py`) — grounds each anomaly's explanation two ways: cited liability detail from the Neo4j reason-code graph (`app/graph_db.py`, Graph RAG) for chargebacks with a dispute reason code, and similar-past-decision context from episodic memory (`app/memory_store.py`, Chroma) for every anomaly type where something similar enough has been decided before.
- **Escalation** — anything still needing a human goes onto a live Redis-backed queue (`escalate_dispute`), including its episodic-memory context inline, for a reviewer to act on via the API.
- **Persistence** — every transaction in the batch is upserted to MongoDB regardless of outcome, so a future batch's lookup step can find it as a counterpart.

A decision on an escalated anomaly (`submit_reconciliation_decision`) writes a permanent Mongo audit record and — for every decision except `deferred` — embeds it as a new episode in episodic memory, closing the loop for future similar anomalies.

## Stack

- Python 3.10+, LangGraph, LangChain
- Classification model: Anthropic Claude, a local model via Ollama, or a rule-based heuristic fallback — auto-selected, see below
- FastAPI (API layer)
- MongoDB (transaction records, decision audit trail) — `mongomock` for local dev without Docker; real Mongo via `MONGO_URI`
- Redis (live escalation queue) — `fakeredis` for local dev without Docker; real Redis via `REDIS_URL`
- Neo4j (reason-code knowledge graph, Graph RAG) — in-memory fallback for local dev without Docker; real Neo4j via `NEO4J_URI`
- Chroma (episodic memory — vector store of past reconciliation decisions) — always local (SQLite-backed), no Docker dependency
- MCP (`mcp` / FastMCP) — a real stdio client/server boundary for cross-batch lookups and write tools, with an in-process fallback when Mongo is mocked

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY, and MONGO_URI/REDIS_URL/NEO4J_URI if you have Docker running
```

`docker compose up -d` brings up real MongoDB, Redis, and Neo4j if you want to run against real backends instead of the in-memory fallbacks (see [Local dev without Docker](#local-dev-without-docker)).

## Running it

**FastAPI app:**
```bash
uvicorn app.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check; also reports whether Mongo/Redis are real or mocked. |
| `POST /classify` | Runs one transaction through the Classification Agent. |
| `POST /reconcile` | Runs a batch of transactions through the full orchestrator pipeline above. |
| `GET /escalations` | Lists everything currently on the live escalation queue, including episodic-memory context. |
| `POST /escalations/{anomaly_id}/decision` | A human reviewer's decision (`approved` / `dismissed` / `deferred`) — writes the audit record and, unless deferred, a new episode. |
| `GET /escalations/{anomaly_id}/similar` | On-demand: similar past decisions for one pending anomaly, queried directly from episodic memory. |

**CLI / scripts** (no API server needed):

| Script | What it demonstrates |
|---|---|
| `scripts/run_prototype.py` | The original Week 1 single-agent Classification Agent prototype. |
| `scripts/run_orchestrator.py` | One full batch through the orchestrator pipeline. |
| `scripts/demo_cross_batch_lookup.py` | The MCP lookup tool resolving a `missing_record` anomaly across two separate batches. |
| `scripts/demo_escalation_queue.py` | The write tools end to end: escalate, list, approve one, defer another, read back the audit records. |
| `scripts/demo_episodic_memory.py` | Resolving an anomaly seeds an episode; a similar anomaly in a later batch gets grounded by it, both via the Reasoning Agent and the `/similar` endpoint's own code path. |

## Classification model providers

`app/agents/classification_agent.py` auto-selects a provider each run, in
priority order: a local model via Ollama (whenever reachable) -> real
Anthropic (Claude, if `ANTHROPIC_API_KEY` is set) -> a rule-based heuristic
fallback (no live model at all). `LLM_PROVIDER` in `.env` forces one of
`local` / `anthropic` / `heuristic` explicitly, skipping auto-detection --
this is what the test suite uses so it never depends on what's actually
running on the machine executing it (see `tests/conftest.py`).

### Swapping the local (Ollama) model

Config lives entirely in `.env` -- no code changes needed to point at a
different model.

1. **Get a GGUF for the new model.** Download one directly (e.g. from
   Hugging Face) or export one from Unsloth if it's a model you fine-tuned
   yourself.

2. **Register it with Ollama** (skip if it's an official model you can
   `ollama pull` instead):
   ```bash
   cat > Modelfile <<'EOF'
   FROM /absolute/path/to/new-model.gguf
   EOF
   ollama create <new-model-name> -f Modelfile
   ollama list   # confirm it shows up, and note the exact name Ollama stored
   ```

3. **Update `.env`:**
   ```
   OLLAMA_MODEL=<new-model-name>
   ```
   Use the exact string `ollama list` shows, not the GGUF filename.

4. **Restart whatever's running.** `_resolve_provider()` caches the
   auto-detected provider per-process, but a plain restart (the app, or a
   fresh `pytest` invocation) re-resolves against the new `.env` value --
   no manual cache-clearing needed.

5. **Re-run the real-LLM smoke test before trusting it:**
   ```bash
   pytest tests/test_classification_agent_local_llm.py -v
   ```
   This is the step that actually matters, not a formality -- it's what
   catches a mismatched chat template or a model that doesn't support
   tool-calling/structured output in Ollama, either of which would
   otherwise silently produce bad `ClassificationResult`s rather than a
   clean error.

6. **If step 5 fails on structured output specifically** (a real
   tool-calling/schema error, not just a bad classification): that model
   doesn't support `.with_structured_output()` via Ollama. There's no
   fallback built for that today -- `_local_classify()` would need a
   prompt-for-JSON + manual Pydantic-validation path instead.

## Local dev without Docker

`app/db.py`, `app/cache.py`, and `app/graph_db.py` fall back to `mongomock` / `fakeredis` / an in-memory reason-code graph automatically whenever `MONGO_URI` / `REDIS_URL` / `NEO4J_URI` aren't set — the whole pipeline runs with nothing but an `ANTHROPIC_API_KEY` (or nothing at all, in heuristic mode). Chroma (episodic memory) needs no such fallback — it's local either way, no Docker or account required. `docker-compose.yml` brings up real MongoDB, Redis, and Neo4j when you want to test against real backends, including the genuine MCP stdio client/server round trip (the in-process fallback only kicks in when Mongo is mocked).

## Testing

```bash
pytest
```

66 tests, mock-mode by default (no Docker or Ollama required — one test skips cleanly if Ollama isn't reachable). Every test forces heuristic-mode classification regardless of what's configured, so the suite never makes a real Anthropic or Ollama call except the one deliberate smoke test (`test_classification_agent_local_llm.py`). Set `MONGO_URI`/`REDIS_URL`/`NEO4J_URI` (with `docker compose up -d`) to run the same suite against real infra instead — same tests, real backends, including the genuine MCP subprocess round trip.

## Layout

```
reconcile/
  app/
    main.py                        FastAPI app, routes
    config.py                      env/settings
    db.py                          Mongo connection (real or mongomock)
    cache.py                       Redis connection (real or fakeredis)
    graph_db.py                    Neo4j connection (real or in-memory reason-code graph)
    memory_store.py                Episodic memory (Chroma) — write/query past decisions
    mcp_client.py                  MCP client wrappers (stdio, with in-process fallback)
    mcp_server.py                  MCP tool server: lookup + write tools (FastMCP)
    retry.py                       Shared retry helper for MCP calls
    agents/
      classification_agent.py      Classifies a transaction (Anthropic / Ollama / heuristic)
      reconciliation_agent.py      Finds duplicate/refund/chargeback matches within a batch
      anomaly_agent.py             Flags anomalies from match results
      reasoning_agent.py           Grounds anomaly explanations (Graph RAG + episodic memory)
      orchestrator.py              LangGraph orchestrator wiring the pipeline together
    models/
      transaction.py               Pydantic transaction/classification schema
      reconciliation.py            Anomaly / ReconciliationDecision schema
      graph.py                     Reason-code graph node/edge schema
    data/
      sample_transactions.json
      reason_codes.py              Seed data for the Neo4j reason-code graph
  scripts/
    run_prototype.py               CLI runner for the Week 1 single-agent prototype
    run_orchestrator.py            CLI runner for one full batch through the pipeline
    seed_graph.py                  Loads reason_codes.py into Neo4j
    demo_cross_batch_lookup.py     Demonstrates the MCP lookup tool across batches
    demo_escalation_queue.py       Demonstrates the write tools + live escalation queue
    demo_episodic_memory.py        Demonstrates episodic memory grounding a later anomaly
  tests/
    test_classification_agent.py
    test_classification_agent_local_llm.py
    test_orchestrator.py
    test_mcp_lookup.py
    test_mcp_escalations.py
    test_memory_store.py
    test_reasoning_agent.py
    test_main.py
    conftest.py                    Shared fixtures — Mongo/Redis/Chroma isolation, heuristic-mode enforcement
  requirements.txt
  docker-compose.yml
  .env.example
```
