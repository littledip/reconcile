"""FastAPI app.

Week 1 scope: health check + a single endpoint wrapping the Classification
Agent. Week 3-4 adds /reconcile, which runs a batch of transactions through
the full orchestrator graph (classify -> match -> detect anomalies ->
maybe escalate). Reasoning Agent / Graph RAG endpoints land in weeks 5-6.

Write-tools phase adds GET /escalations and POST
/escalations/{anomaly_id}/decision -- the human-reviewer side of the
escalation queue escalate_node now writes to. Both go through the same MCP
client path as the orchestrator (app/mcp_client.py's lookup_session()),
per the project's own "read and write tools belong together, reached the
same way" design -- no separate in-process code path for these.

Week 9-10 adds GET /escalations/{anomaly_id}/similar -- on demand, not
inlined into GET /escalations (per the design session). Unlike the two
endpoints above, this one calls app/memory_store.py directly, in-process,
rather than through lookup_session() -- episodic retrieval follows the
same knowledge-retrieval-vs-tool-use split Week 5-6 drew for the reason-code
graph, not the MCP write-tools boundary (see Progress_Log.md's Sept 23
Week 9-10 design-session entry).
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.classification_agent import classify_transaction
from app.agents.orchestrator import run_reconciliation_batch
from app.cache import ESCALATION_EVENTS_CHANNEL, redis_client, using_mock_redis
from app.config import settings
from app.db import using_mock_mongo
from app.graph_db import using_mock_graph
from app.mcp_client import list_pending_escalations, lookup_session, submit_reconciliation_decision
from app.memory_store import find_similar_episodes
from app.models.transaction import ClassificationResult, Transaction

# Week 11+ (push notifications -- see the Sept 29 design doc "Escalation
# Queue: Polling to Push"): one background task, started for the life of
# the process (lifespan below), holds the Redis subscription and fans
# each "queue changed" message out to every currently-connected SSE
# client. Each client's own asyncio.Queue is added when GET
# /escalations/stream opens and discarded when it closes.
#
# Deliberately outside the lookup_session()/MCP boundary every other
# escalation read/write in this file goes through: pub/sub is a long-lived
# stream, not a request/response tool call, so it doesn't fit that model
# -- but it still reuses the exact same redis_client (real or fakeredis,
# per app/cache.py) that app/mcp_server.py's tools publish to. That
# client is synchronous, and pubsub.get_message() blocks, so each read
# runs in a thread pool executor rather than on the event loop.
_stream_subscribers: set[asyncio.Queue] = set()


async def _broadcast_queue_changes() -> None:
    pubsub = redis_client.pubsub()
    pubsub.subscribe(ESCALATION_EVENTS_CHANNEL)
    loop = asyncio.get_running_loop()
    try:
        while True:
            message = await loop.run_in_executor(
                None, lambda: pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            )
            if message is not None:
                # Coalesce: each client's queue holds at most one pending
                # "go refetch" signal. The event only ever means "go look
                # again" (never carries the change itself, Section 4 of
                # the design doc), so collapsing a burst of near-together
                # changes into one delivered event loses nothing -- one
                # refetch after the burst picks up all of them -- and this
                # also protects against a slow/backgrounded tab building up
                # a large backlog of identical events while it's away.
                for queue in list(_stream_subscribers):
                    if queue.empty():
                        queue.put_nowait({"type": "queue_changed"})
    finally:
        pubsub.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_broadcast_queue_changes())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Reconcile", version="0.1.0", lifespan=lifespan)

# Week 11 (evaluator UI): the React app (ui/, served by Vite's dev server)
# calls this API directly from the browser -- no Node backend/proxy layer,
# per the Sept 28 design session ("Node is frontend tooling only"). CORS
# is the only backend change that decision requires. Wide open for local
# dev -- this app has no auth model at all yet (single-evaluator, no
# login), so there's no session/cookie boundary to protect here regardless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecisionRequest(BaseModel):
    decision: str
    decided_by: str
    notes: Optional[str] = None


class ReconcileRequest(BaseModel):
    transactions: list[Transaction]


class ReconcileResponse(BaseModel):
    classifications: dict[str, ClassificationResult]
    matches: list[dict]
    anomalies: list[dict]
    escalations: list[str]


@app.get("/health")
def health() -> dict:
    """Report which backends (Mongo, Redis, graph) are live vs. mocked, and
    whether classification is running in LLM or heuristic mode.
    """
    return {
        "status": "ok",
        "mongo_backend": "mongomock (in-memory)" if using_mock_mongo() else "mongodb",
        "redis_backend": "fakeredis (in-memory)" if using_mock_redis() else "redis",
        "graph_backend": "in-memory (reason_codes.py)" if using_mock_graph() else "neo4j",
        "classification_mode": "llm" if settings.anthropic_api_key else "heuristic (no ANTHROPIC_API_KEY set)",
    }


@app.post("/classify", response_model=ClassificationResult)
def classify(transaction: Transaction) -> ClassificationResult:
    """Classify a single transaction via the Classification Agent."""
    try:
        return classify_transaction(transaction)
    except Exception as exc:  # pragma: no cover - week 1 skeleton, broad on purpose
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reconcile", response_model=ReconcileResponse)
async def reconcile(request: ReconcileRequest) -> ReconcileResponse:
    """Run a batch of transactions through the full orchestrator graph.

    Async since Week 7-8, when lookup_missing_records_node started making a
    live MCP call as part of the graph -- FastAPI runs async route
    handlers natively, so this is just `async def` + `await`.
    """
    try:
        result = await run_reconciliation_batch(request.transactions)
    except Exception as exc:  # pragma: no cover - week 3-4 skeleton, broad on purpose
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ReconcileResponse(
        classifications={
            tid: ClassificationResult.model_validate(c) for tid, c in result["classifications"].items()
        },
        matches=result["matches"],
        anomalies=result["anomalies"],
        escalations=result["escalations"],
    )


@app.get("/escalations")
async def list_escalations() -> list[dict]:
    """Every anomaly currently sitting in the live escalation queue,
    awaiting a human reviewer's decision.
    """
    try:
        async with lookup_session() as session:
            return await list_pending_escalations(session)
    except Exception as exc:  # pragma: no cover - mirrors /reconcile's broad handler
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/escalations/{anomaly_id}/decision")
async def decide_escalation(anomaly_id: str, request: DecisionRequest) -> dict:
    """A human reviewer closes out (or defers) one queued escalation.

    decision is "approved" | "dismissed" | "deferred" -- "deferred" leaves
    the anomaly in the queue for later review while still recording that
    it was looked at; the other two remove it. anomaly_type/
    transaction_ids/reasoning aren't required here -- the underlying tool
    pulls them from the anomaly's own queue entry (written when
    escalate_node queued it).
    """
    try:
        async with lookup_session() as session:
            return await submit_reconciliation_decision(
                session,
                anomaly_id=anomaly_id,
                decision=request.decision,
                decided_by=request.decided_by,
                notes=request.notes,
            )
    except Exception as exc:  # pragma: no cover - mirrors /reconcile's broad handler
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/escalations/{anomaly_id}/similar")
async def similar_escalations(anomaly_id: str) -> list[dict]:
    """On-demand, for a human reviewer: past reconciliation decisions
    similar to one anomaly currently sitting in the escalation queue.

    Not inlined into GET /escalations -- the reviewer asks for this
    explicitly, per anomaly (design session, Sept 23). Looks the anomaly
    up in the live queue to get its own anomaly_type/severity/reasoning
    (the same fields build_episode_context() uses everywhere else), then
    queries episodic memory in-process -- see app/memory_store.py and this
    module's own docstring for why this one doesn't go through
    lookup_session() the way /escalations and its decision endpoint do.
    """
    try:
        async with lookup_session() as session:
            pending = await list_pending_escalations(session)
    except Exception as exc:  # pragma: no cover - mirrors /reconcile's broad handler
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    queued = next((item for item in pending if item["anomaly_id"] == anomaly_id), None)
    if queued is None:
        raise HTTPException(status_code=404, detail=f"No pending escalation for anomaly_id={anomaly_id!r}")

    return find_similar_episodes(
        anomaly_type=queued["anomaly_type"],
        severity=queued["severity"],
        reasoning=queued["reasoning"],
    )


@app.get("/escalations/stream")
async def escalations_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events: push {"type": "queue_changed"} whenever the
    escalation queue changes, so the UI (ui/src/hooks/useEscalations.ts)
    can react immediately instead of polling GET /escalations on a timer.
    See the Sept 29 design doc ("Escalation Queue: Polling to Push") for
    the full design.

    This endpoint never sends the queue's contents itself -- only "go
    look" (a thin invalidate signal, Section 4 of the design doc). The
    client is expected to call GET /escalations on every message,
    including right after the connection (re)opens, which is what covers
    anything that changed while it was disconnected.

    Each connection gets its own asyncio.Queue, filled by the
    _broadcast_queue_changes background task above. A 15s idle heartbeat
    comment line keeps intermediary proxies from timing the connection
    out while the queue is quiet.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _stream_subscribers.add(queue)

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            _stream_subscribers.discard(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
