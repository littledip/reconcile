"""Redis connection.

Same story as db.py: falls back to fakeredis (in-memory, API-compatible)
whenever REDIS_URL isn't set, since Docker isn't available in the initial
build environment. Swap in real Redis by setting REDIS_URL later.
"""
from app.config import settings

_USING_MOCK = settings.redis_url is None

if _USING_MOCK:
    import fakeredis

    redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
else:
    import redis

    redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def using_mock_redis() -> bool:
    return _USING_MOCK


# Week 11+ (push notifications, replacing the evaluator UI's poll -- see
# the Sept 29 design doc "Escalation Queue: Polling to Push"): the channel
# app/mcp_server.py's escalate_dispute/submit_reconciliation_decision
# PUBLISH to right after they write the escalation_queue hash, and that
# app/main.py's background subscriber task listens on to fan each change
# out over SSE. Named here, alongside the connection itself, since both
# the publisher and the subscriber already import redis_client from this
# module and need to agree on the same channel name.
ESCALATION_EVENTS_CHANNEL = "escalation_events"
