import { useState } from "react";
import { getSimilarEpisodes, submitDecision } from "../api";
import { getDecisionHint } from "../decisionHelp";
import type { Decision, EscalatedAnomaly, SimilarEpisode } from "../types";

interface Props {
  anomaly: EscalatedAnomaly;
  reviewerName: string;
  onDecided: () => void;
}

const DECISIONS: Decision[] = ["approved", "dismissed", "deferred"];

// Parent (App.tsx) mounts this with key={anomaly.anomaly_id} -- switching
// the selected anomaly remounts the component fresh, so per-anomaly form/
// lookup state (below) starts clean with no reset-on-prop-change effect
// needed. This is the React-docs-recommended pattern for exactly this
// case (resetting state when a prop changes) -- an effect doing the same
// reset was flagged by oxlint's react(set-state-in-effect) rule, and the
// key approach is the actual fix, not a workaround for the lint warning.
export function EscalationDetail({ anomaly, reviewerName, onDecided }: Props) {
  const [similar, setSimilar] = useState<SimilarEpisode[] | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState<string | null>(null);

  const [decision, setDecision] = useState<Decision>("approved");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  async function handleShowSimilar() {
    setSimilarLoading(true);
    setSimilarError(null);
    try {
      setSimilar(await getSimilarEpisodes(anomaly.anomaly_id));
    } catch (err) {
      setSimilarError(err instanceof Error ? err.message : String(err));
    } finally {
      setSimilarLoading(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!reviewerName.trim()) {
      setSubmitError("Enter your name above before submitting a decision.");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      await submitDecision(anomaly.anomaly_id, {
        decision,
        decided_by: reviewerName.trim(),
        notes: notes.trim() || undefined,
      });
      onDecided();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="escalation-detail">
      <h2>{anomaly.anomaly_type.replace(/_/g, " ")}</h2>
      <dl className="detail-meta">
        <dt>Severity</dt>
        <dd className={`severity-badge severity-${anomaly.severity}`}>{anomaly.severity}</dd>
        <dt>Transactions</dt>
        <dd>{anomaly.transaction_ids.join(", ")}</dd>
        <dt>Escalated</dt>
        <dd>{new Date(anomaly.escalated_at).toLocaleString()}</dd>
      </dl>

      <p className="reasoning">{anomaly.reasoning}</p>

      {anomaly.episodic_context && (
        <p className="episodic-context">
          <strong>Similar past context:</strong> {anomaly.episodic_context}
        </p>
      )}

      {/* anomaly.episodic_context is set iff find_similar_episodes() found
          anything at escalation time (app/agents/reasoning_agent.py) -- the
          exact same query this button's on-demand /similar call re-runs
          live. Gating on it here means we never show a "Show similar past
          decisions" button that's guaranteed to come back empty. */}
      {anomaly.episodic_context && (
        <div className="similar-section">
          <button type="button" onClick={handleShowSimilar} disabled={similarLoading}>
            {similarLoading ? "Loading..." : "Show similar past decisions"}
          </button>
          {similarError && <p className="error-text">{similarError}</p>}
          {similar && similar.length === 0 && <p className="empty-state">No similar past decisions found.</p>}
          {similar && similar.length > 0 && (
            <ul className="similar-list">
              {similar.map((ep) => (
                <li key={ep.anomaly_id}>
                  <strong>{ep.decision.replace(/_/g, " ")}</strong> by {ep.decided_by}
                  {ep.notes ? ` -- ${ep.notes}` : ""}
                  <span className="similar-distance"> (distance {ep.distance.toFixed(2)})</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <form className="decision-form" onSubmit={handleSubmit}>
        <label>
          Decision
          <select value={decision} onChange={(e) => setDecision(e.target.value as Decision)}>
            {DECISIONS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <p className="decision-hint">{getDecisionHint(anomaly.anomaly_type, decision)}</p>
        <label>
          Notes
          <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} />
        </label>
        {submitError && <p className="error-text">{submitError}</p>}
        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting..." : "Submit decision"}
        </button>
      </form>
    </div>
  );
}
