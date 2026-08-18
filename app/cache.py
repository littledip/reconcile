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
