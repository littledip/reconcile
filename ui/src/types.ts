// Mirrors app/main.py's response/request shapes exactly -- see that
// module's docstrings for the design rationale behind each endpoint.

export type Severity = "low" | "medium" | "high";
export type Decision = "approved" | "dismissed" | "deferred";

/** One item from GET /escalations -- a queue record written by
 * escalate_dispute (app/mcp_server.py). */
export interface EscalatedAnomaly {
  anomaly_id: string;
  anomaly_type: string;
  transaction_ids: string[];
  severity: Severity;
  reasoning: string;
  episodic_context: string | null;
  escalated_at: string; // ISO 8601
}

/** POST /escalations/{id}/decision request body. */
export interface DecisionRequest {
  decision: Decision;
  decided_by: string;
  notes?: string;
}

/** POST /escalations/{id}/decision response. */
export interface DecisionResponse {
  recorded: boolean;
  anomaly_id: string;
  decision: string;
  removed_from_queue: boolean;
}

/** One item from GET /escalations/{id}/similar -- shape returned by
 * app/memory_store.py's find_similar_episodes(). */
export interface SimilarEpisode {
  anomaly_id: string;
  anomaly_type: string;
  severity: Severity;
  decision: string;
  decided_by: string;
  notes: string | null;
  decided_at: string;
  transaction_ids: string[];
  distance: number;
}
