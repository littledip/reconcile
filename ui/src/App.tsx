import { useMemo, useState } from "react";
import "./App.css";
import { EscalationDetail } from "./components/EscalationDetail";
import { EscalationList } from "./components/EscalationList";
import { useEscalations } from "./hooks/useEscalations";
import { useReviewerName } from "./hooks/useReviewerName";
import { SORT_LABELS, sortEscalations, type SortOption } from "./sort";

const SORT_OPTIONS: SortOption[] = ["relevance", "priority", "date_asc"];

function App() {
  const { escalations, loading, error, refetch } = useEscalations();
  const { name: reviewerName, setName: setReviewerName } = useReviewerName();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<SortOption>("relevance");

  const sortedEscalations = useMemo(
    () => sortEscalations(escalations, sortBy),
    [escalations, sortBy],
  );
  const selected = escalations.find((e) => e.anomaly_id === selectedId) ?? null;

  function handleDecided() {
    setSelectedId(null);
    refetch();
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Reconcile -- Escalation Review</h1>
        <label className="reviewer-name">
          Your name
          <input
            type="text"
            value={reviewerName}
            onChange={(e) => setReviewerName(e.target.value)}
            placeholder="e.g. John"
          />
        </label>
      </header>

      {error && <p className="error-text">Couldn't reach the API: {error}</p>}

      <main className="app-main">
        <section className="queue-pane">
          <div className="queue-pane-header">
            <h2>Pending ({escalations.length})</h2>
            <label className="sort-control">
              Sort
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value as SortOption)}>
                {SORT_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {SORT_LABELS[opt]}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {loading ? <p>Loading...</p> : (
            <EscalationList
              escalations={sortedEscalations}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
        </section>
        <section className="detail-pane">
          {selected ? (
            <EscalationDetail
              key={selected.anomaly_id}
              anomaly={selected}
              reviewerName={reviewerName}
              onDecided={handleDecided}
            />
          ) : (
            <p className="empty-state">Select an anomaly from the queue to review it.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
