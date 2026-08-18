# Reconcile

Autonomous transaction reconciliation & anomaly detection platform — portfolio project built to develop hands-on agentic AI experience (LangGraph orchestration, Graph RAG, MCP servers) for agentic-AI-architect roles.

Full project plan/architecture: see SecondBrain `03_Knowledge/AI/Reconcile_Portfolio/README.md`.
Progress log: see SecondBrain `03_Knowledge/AI/Reconcile_Portfolio/Progress_Log.md`.

## Status: Week 1 — environment setup, LangGraph fundamentals, single-agent prototype, FastAPI skeleton

## Stack

- Python 3.10+, LangGraph, LangChain (Anthropic Claude as primary model)
- FastAPI (API layer)
- MongoDB (transaction records) — `mongomock` used for local dev since Docker isn't available in this build environment; swap to a real MongoDB via `MONGO_URI` when running locally with Docker
- Redis (state/queues) — `fakeredis` used for local dev for the same reason; swap to real Redis via `REDIS_URL` when available
- Neo4j + Graph RAG, custom MCP server — planned for weeks 5–8, not yet built

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run the Week 1 single-agent prototype

```bash
python3 scripts/run_prototype.py
```

This runs a minimal LangGraph graph with one node — a **Classification Agent** that reads a sample transaction and classifies it (e.g., `payment`, `refund`, `chargeback`, `duplicate_charge`) with reasoning. It's intentionally small: the goal this week is proving the LangGraph plumbing (state schema, node, compiled graph, invocation) works end to end before adding more agents/orchestration in weeks 3–4.

## Run the FastAPI skeleton

```bash
uvicorn app.main:app --reload
```

`GET /health` — liveness check, also reports whether it's using real or mock Mongo/Redis backends.
`POST /classify` — runs a transaction through the Classification Agent via HTTP.

## Local dev without Docker

This build environment doesn't have Docker available, so `app/db.py` and `app/cache.py` fall back to `mongomock` / `fakeredis` in-memory implementations automatically when `MONGO_URI` / `REDIS_URL` aren't set. A `docker-compose.yml` is included for running against real MongoDB + Redis once this is pulled down to a machine with Docker (e.g. for weeks 3+ multi-agent work, where persistence across runs starts to matter).

## Layout

```
reconcile/
  app/
    main.py            FastAPI app, routes
    config.py           env/settings
    db.py                Mongo connection (real or mongomock)
    cache.py             Redis connection (real or fakeredis)
    agents/
      classification_agent.py   LangGraph single-agent prototype
    models/
      transaction.py     Pydantic transaction schema
    data/
      sample_transactions.json
  scripts/
    run_prototype.py     CLI runner for the agent, no API server needed
  tests/
    test_classification_agent.py
  requirements.txt
  docker-compose.yml
  .env.example
```
