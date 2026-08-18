"""Mongo connection.

No Docker in the initial build environment, so this falls back to mongomock
(an in-memory, API-compatible Mongo client) whenever MONGO_URI isn't set.
Swap in a real MongoDB by setting MONGO_URI once running somewhere with
Docker (see docker-compose.yml) or a hosted Mongo instance — no code changes
needed, same pymongo-style interface either way.
"""
from app.config import settings

_USING_MOCK = settings.mongo_uri is None

if _USING_MOCK:
    import mongomock

    _client = mongomock.MongoClient()
else:
    from pymongo import MongoClient

    _client = MongoClient(settings.mongo_uri)

db = _client["reconcile"]


def using_mock_mongo() -> bool:
    return _USING_MOCK
