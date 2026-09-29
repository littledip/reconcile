import { useCallback, useEffect, useState } from "react";
import { escalationsStreamUrl, listEscalations } from "../api";
import type { EscalatedAnomaly } from "../types";

// Redis Pub/Sub (app/main.py's subscriber task) has no replay -- a
// message fires once, to whoever's connected at that instant. This bounds
// staleness to a minute even if a push is silently dropped, e.g. during
// the gap between a dropped connection and EventSource's own reconnect.
// See the "Escalation Queue: Polling to Push" design doc (Sept 29) for
// the full reasoning; down from the old 10s poll interval since it's now
// a safety net, not the primary signal.
const SAFETY_NET_POLL_MS = 60_000;

/** Push-driven via GET /escalations/stream (SSE), not a poll timer.
 * Refetches on every "queue changed" event, and again each time the
 * stream (re)connects (EventSource.onopen fires on the first connect and
 * after every auto-reconnect) -- that second part is what picks up
 * anything missed while disconnected, since the browser's EventSource
 * reconnects on its own but Redis Pub/Sub never replays what it missed.
 * Also exposes an immediate refetch() -- used right after a decision is
 * submitted, so the list updates without waiting for the next event. */
export function useEscalations() {
  const [escalations, setEscalations] = useState<EscalatedAnomaly[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    try {
      const data = await listEscalations();
      setEscalations(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();

    const source = new EventSource(escalationsStreamUrl());
    source.onopen = refetch;
    source.onmessage = refetch;

    const intervalId = setInterval(refetch, SAFETY_NET_POLL_MS);

    return () => {
      source.close();
      clearInterval(intervalId);
    };
  }, [refetch]);

  return { escalations, loading, error, refetch };
}
