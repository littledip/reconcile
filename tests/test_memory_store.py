"""Tests for the Week 9-10 episodic memory store (app/memory_store.py).

Isolation comes from tests/conftest.py's autouse _isolate_chroma fixture --
every test here gets its own disposable Chroma directory, so writes in one
test never leak into another's similarity search.
"""
from app.memory_store import build_episode_context, find_similar_episodes, write_episode


def test_build_episode_context_excludes_decision_fields():
    """The embedded text template only ever draws on anomaly-side fields
    -- decision/decided_by/notes/decided_at have no way to influence it,
    by construction (the function doesn't even accept them).
    """
    text = build_episode_context("duplicate_charge", "high", "Two charges for order #500.")
    assert text == "high severity duplicate_charge: Two charges for order #500."


def test_find_similar_episodes_empty_store_returns_empty_list():
    assert find_similar_episodes(anomaly_type="duplicate_charge", severity="high", reasoning="anything") == []


def test_write_then_find_returns_the_written_episode():
    write_episode(
        anomaly_id="ep_1",
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Two transactions reference order #500 for merchant merch_1 with identical amount 50.00.",
        decision="approved",
        decided_by="jane_reviewer",
        notes="Confirmed duplicate, refunded.",
        decided_at="2026-09-23T12:00:00+00:00",
        transaction_ids=["a", "b"],
    )

    results = find_similar_episodes(
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Two transactions reference order #999 for merchant merch_9 with identical amount 75.00.",
    )

    assert len(results) == 1
    match = results[0]
    assert match["anomaly_id"] == "ep_1"
    assert match["anomaly_type"] == "duplicate_charge"
    assert match["severity"] == "high"
    assert match["decision"] == "approved"
    assert match["decided_by"] == "jane_reviewer"
    assert match["notes"] == "Confirmed duplicate, refunded."
    assert match["transaction_ids"] == ["a", "b"]
    assert match["distance"] >= 0.0


def test_find_similar_episodes_excludes_dissimilar_ones_via_threshold():
    """A genuinely unrelated episode (different anomaly type, unrelated
    text) shouldn't be padded into the results just to hit k -- the
    minimum-similarity cutoff should exclude it, per the design session.
    """
    write_episode(
        anomaly_id="ep_close",
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Two transactions reference order #500 for merchant merch_1 with identical amount 50.00.",
        decision="approved",
        decided_by="jane_reviewer",
        notes=None,
        decided_at="2026-09-23T12:00:00+00:00",
        transaction_ids=["a"],
    )
    write_episode(
        anomaly_id="ep_far",
        anomaly_type="missing_record",
        severity="low",
        reasoning="Monthly platform processing fee with no counterpart expected.",
        decision="dismissed",
        decided_by="jane_reviewer",
        notes=None,
        decided_at="2026-09-23T12:00:00+00:00",
        transaction_ids=["z"],
    )

    results = find_similar_episodes(
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Two transactions reference order #501 for merchant merch_2 with identical amount 60.00.",
        max_distance=0.8,
    )

    ids = {r["anomaly_id"] for r in results}
    assert "ep_close" in ids
    assert "ep_far" not in ids


def test_find_similar_episodes_respects_k():
    for i in range(5):
        write_episode(
            anomaly_id=f"ep_k_{i}",
            anomaly_type="duplicate_charge",
            severity="high",
            reasoning="Two transactions reference the same order and amount.",
            decision="approved",
            decided_by="jane_reviewer",
            notes=None,
            decided_at="2026-09-23T12:00:00+00:00",
            transaction_ids=[f"t{i}"],
        )

    results = find_similar_episodes(
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Two transactions reference the same order and amount.",
        k=2,
        max_distance=2.0,  # wide open, so k is the only thing limiting results here
    )
    assert len(results) == 2


def test_write_episode_upserts_on_repeated_anomaly_id():
    write_episode(
        anomaly_id="ep_upsert",
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="First pass.",
        decision="approved",
        decided_by="jane_reviewer",
        notes=None,
        decided_at="2026-09-23T12:00:00+00:00",
        transaction_ids=["a"],
    )
    write_episode(
        anomaly_id="ep_upsert",
        anomaly_type="duplicate_charge",
        severity="high",
        reasoning="Updated pass.",
        decision="dismissed",
        decided_by="jane_reviewer",
        notes="Actually a false positive.",
        decided_at="2026-09-23T13:00:00+00:00",
        transaction_ids=["a"],
    )

    results = find_similar_episodes(anomaly_type="duplicate_charge", severity="high", reasoning="Updated pass.")
    matching = [r for r in results if r["anomaly_id"] == "ep_upsert"]
    assert len(matching) == 1
    assert matching[0]["decision"] == "dismissed"
    assert matching[0]["notes"] == "Actually a false positive."
