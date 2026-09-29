import type {
  DecisionRequest,
  DecisionResponse,
  EscalatedAnomaly,
  SimilarEpisode,
} from "./types";

// Points at uvicorn (app/main.py) directly -- no proxy layer, per the
// "Node is just frontend tooling" decision (Progress_Log.md, Sept 28
// UI design session). Override via ui/.env if FastAPI runs elsewhere.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${init?.method ?? "GET"} ${path} failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function listEscalations(): Promise<EscalatedAnomaly[]> {
  return request<EscalatedAnomaly[]>("/escalations");
}

/** GET /escalations/stream (SSE) -- pushes {"type": "queue_changed"}
 * whenever the escalation queue changes; see ui/src/hooks/useEscalations.ts. */
export function escalationsStreamUrl(): string {
  return `${API_BASE_URL}/escalations/stream`;
}

export function getSimilarEpisodes(anomalyId: string): Promise<SimilarEpisode[]> {
  return request<SimilarEpisode[]>(`/escalations/${encodeURIComponent(anomalyId)}/similar`);
}

export function submitDecision(
  anomalyId: string,
  body: DecisionRequest,
): Promise<DecisionResponse> {
  return request<DecisionResponse>(`/escalations/${encodeURIComponent(anomalyId)}/decision`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
