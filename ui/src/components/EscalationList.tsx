import type { EscalatedAnomaly } from "../types";

interface Props {
  escalations: EscalatedAnomaly[];
  selectedId: string | null;
  onSelect: (anomalyId: string) => void;
}

const SEVERITY_LABEL: Record<string, string> = {
  high: "High",
  medium: "Medium",
  low: "Low",
};

export function EscalationList({ escalations, selectedId, onSelect }: Props) {
  if (escalations.length === 0) {
    return <p className="empty-state">Queue is empty -- nothing waiting on review.</p>;
  }

  return (
    <ul className="escalation-list">
      {escalations.map((item) => (
        <li key={item.anomaly_id}>
          <button
            type="button"
            className={`escalation-row severity-${item.severity} ${
              item.anomaly_id === selectedId ? "selected" : ""
            }`}
            onClick={() => onSelect(item.anomaly_id)}
          >
            <span className="anomaly-type">{item.anomaly_type.replace(/_/g, " ")}</span>
            <span className={`severity-badge severity-${item.severity}`}>
              {SEVERITY_LABEL[item.severity] ?? item.severity}
            </span>
            <span className="txn-ids">{item.transaction_ids.join(", ")}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
