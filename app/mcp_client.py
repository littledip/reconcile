"""Week 7-8: MCP client for the lookup server (app/mcp_server.py).

Spawns app/mcp_server.py as a subprocess over stdio and calls its
lookup_transaction_by_reference tool -- a genuine client/server round trip
over the MCP protocol, not an in-process function call. That boundary is
the actual point of this piece (see app/mcp_server.py's module docstring
for the fuller rationale), and it's what actually runs -- and what's
demonstrated -- whenever a real, shared Mongo is configured (MONGO_URI set,
see app/db.py).

Mock-mode wrinkle: mongomock is an in-memory store scoped to a single
process. persist_batch_node (app/agents/orchestrator.py) writes through the
parent process's `db` handle; a real subprocess spawned here would get its
*own*, empty mongomock instance with no way to see that data -- confirmed
by hand (batch 1 persists a payment, batch 2's matching refund still comes
back "not found"). Rather than let the lookup silently produce wrong
answers whenever Docker isn't running, _InProcessSession below calls the
same tool function directly, in-process, against the same db handle,
so mock-mode lookups stay correct for local demos without Docker. See
Progress_Log.md for the full reasoning; this is the one place in the
project where "no Docker" changes *which code path runs*, not just which
backend it talks to.

The orchestrator (app/agents/orchestrator.py) is the only caller of
lookup_transaction_by_reference. One session is opened per orchestrator
node call -- not one per transaction -- since a batch may have several
anomalies to look up and spawning a fresh subprocess per lookup would be
wasteful. Retry/backoff (app/retry.py) wraps each individual tool call,
since the real stdio path is the one genuinely-fallible live call in the
graph: a subprocess still starting up, a dropped pipe, a slow response --
unlike the deterministic, in-process agents around it. (The in-process
fallback has no such failure mode of its own, but is wrapped the same way
for a uniform call site.)

Write-tools phase adds three more wrapper functions -- escalate_dispute,
list_pending_escalations, submit_reconciliation_decision -- mirroring this
same pattern (retry-wrapped, parse the tool's JSON payload) for the three
new tools in app/mcp_server.py. Callers: the orchestrator (escalate_node,
lookup_missing_records_node), same as above, plus the new /escalations
FastAPI endpoints (app/main.py), which open their own session via
lookup_session() -- same mock/real branching, no separate code path needed
for these tools.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

from app.config import settings
from app.models.reconciliation import Anomaly
from app.models.transaction import Transaction
from app.retry import with_retry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _InProcessSession:
    """Mock-mode stand-in for ClientSession -- see module docstring.

    Matches just enough of ClientSession's surface (an async call_tool
    returning something with .isError / .content / .structuredContent) for
    lookup_transaction_by_reference below to treat both interchangeably.
    """

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        from app import mcp_server  # local import: this module must stay importable without a Mongo connection

        tool_fn = getattr(mcp_server, name)
        try:
            payload = tool_fn(**arguments)
        except Exception as exc:  # noqa: BLE001 - mirror a real tool-call error, not a Python exception
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], isError=True)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload,
            isError=False,
        )


LookupSession = Union[ClientSession, _InProcessSession]


def _server_params() -> StdioServerParameters:
    # Inherit the parent's environment (so ANTHROPIC_API_KEY etc. still flow
    # through if the server ever needs it) but pin MONGO_URI explicitly so
    # the subprocess resolves the same real Mongo the parent is using,
    # rather than whatever a separate .env search from the subprocess's cwd
    # would pick up.
    env = dict(os.environ)
    env["MONGO_URI"] = settings.mongo_uri or ""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=str(_PROJECT_ROOT),
        env=env,
    )


@asynccontextmanager
async def lookup_session() -> AsyncIterator[LookupSession]:
    """Open one lookup session for the duration of the `async with` block --
    reused across every lookup call made within it.

    Real stdio MCP subprocess when a real Mongo is configured; in-process
    fallback when mocked (see module docstring for why).
    """
    if settings.mongo_uri is None:
        yield _InProcessSession()
        return

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def lookup_transaction_by_reference(
    session: LookupSession,
    txn: Transaction,
    *,
    max_attempts: int = 3,
) -> dict:
    """Call the lookup tool for one transaction, with retry/backoff around
    the call itself.

    Returns the tool's parsed JSON payload:
        {"found": bool, "matched_transaction_id": str | None,
         "match_type": str | None, "reasoning": str}
    """

    async def _call() -> dict:
        return await _call_tool(
            session,
            "lookup_transaction_by_reference",
            {
                "transaction_id": txn.transaction_id,
                "merchant_id": txn.merchant_id,
                "amount": str(txn.amount),
                "timestamp": txn.timestamp.isoformat(),
                "description": txn.description,
            },
        )

    return await with_retry(_call, max_attempts=max_attempts)


async def _call_tool(session: LookupSession, name: str, arguments: dict) -> dict:
    """Shared plumbing for every tool wrapper in this module: call `name`,
    raise on a tool-level error, and return its parsed JSON payload.
    """
    result = await session.call_tool(name, arguments)
    if result.isError:
        text = result.content[0].text if result.content else "unknown error"
        raise RuntimeError(f"MCP tool error from {name}: {text}")
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


async def escalate_dispute(
    session: LookupSession,
    anomaly: Anomaly,
    *,
    max_attempts: int = 3,
) -> dict:
    """Push one anomaly onto the live escalation queue. Called by
    escalate_node, once per anomaly with requires_human_review=True.

    Passes anomaly.episodic_context through unchanged (None if
    reasoning_node found nothing similar enough) -- previously dropped
    here, so a reviewer could only see it via a separate GET
    /escalations/{anomaly_id}/similar call rather than inline.

    Returns the tool's payload: {"queued": True, "anomaly_id": str}
    """

    async def _call() -> dict:
        return await _call_tool(
            session,
            "escalate_dispute",
            {
                "anomaly_id": anomaly.anomaly_id,
                "anomaly_type": anomaly.anomaly_type.value,
                "transaction_ids": anomaly.transaction_ids,
                "severity": anomaly.severity,
                "reasoning": anomaly.reasoning,
                "episodic_context": anomaly.episodic_context,
            },
        )

    return await with_retry(_call, max_attempts=max_attempts)


async def list_pending_escalations(
    session: LookupSession,
    *,
    max_attempts: int = 3,
) -> list[dict]:
    """Return every anomaly currently sitting in the live escalation
    queue. Backs the GET /escalations FastAPI endpoint.

    Unwraps the tool's {"escalations": [...]} payload (wrapped in an
    object since MCP's structuredContent must be a JSON object, not a bare
    array) back to a plain list for callers.
    """

    async def _call() -> list[dict]:
        result = await _call_tool(session, "list_pending_escalations", {})
        return result["escalations"]

    return await with_retry(_call, max_attempts=max_attempts)


async def submit_reconciliation_decision(
    session: LookupSession,
    *,
    anomaly_id: str,
    decision: str,
    decided_by: str,
    notes: str | None = None,
    anomaly_type: str | None = None,
    severity: str | None = None,
    transaction_ids: list[str] | None = None,
    reasoning: str | None = None,
    max_attempts: int = 3,
) -> dict:
    """Write a permanent audit record for one anomaly's resolution, embed
    it as an episode in episodic memory (Week 9-10, unless
    decision="deferred"), and -- unless decision="deferred" -- remove it
    from the live escalation queue.

    anomaly_type/severity/transaction_ids/reasoning are optional: pass them
    when the caller already has the full Anomaly object in hand (e.g.
    lookup_missing_records_node); omit them for the human-reviewer path
    (POST /escalations/{anomaly_id}/decision) and the tool will pull that
    context from the anomaly's own queue entry instead.
    """

    async def _call() -> dict:
        return await _call_tool(
            session,
            "submit_reconciliation_decision",
            {
                "anomaly_id": anomaly_id,
                "decision": decision,
                "decided_by": decided_by,
                "notes": notes,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "transaction_ids": transaction_ids,
                "reasoning": reasoning,
            },
        )

    return await with_retry(_call, max_attempts=max_attempts)
