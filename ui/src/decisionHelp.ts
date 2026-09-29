import type { Decision } from "./types";

// The `decision` field is a single flat vocabulary (approved / dismissed /
// deferred) applied identically regardless of anomaly_type -- there is no
// per-type remediation logic anywhere in the backend. That's fine for
// storage, but "approved" means something different for a duplicate charge
// than it does for a chargeback or a missing record, and a reviewer
// shouldn't have to guess. This maps (anomaly_type, decision) to a short
// clarifying line shown next to the decision select -- UI-only, no effect
// on what's actually submitted.
const HINTS: Record<string, Record<Decision, string>> = {
  duplicate_charge: {
    approved: "Approved = confirmed genuine duplicate -- same charge posted twice.",
    dismissed: "Dismissed = not a duplicate -- the charges are legitimately separate.",
    deferred: "Deferred = need more information before ruling on the duplicate.",
  },
  unresolved_chargeback: {
    approved: "Approved = dispute upheld -- the chargeback is valid.",
    dismissed: "Dismissed = dispute rejected -- the chargeback does not hold up.",
    deferred: "Deferred = need more information before ruling on the dispute.",
  },
  missing_record: {
    approved: "Approved = confirmed legitimate despite no matching record being found.",
    dismissed: "Dismissed = not legitimate -- treat as an unresolved exception.",
    deferred: "Deferred = need more information before resolving the missing record.",
  },
  amount_mismatch: {
    approved: "Approved = the mismatch is expected/explained -- confirmed correct.",
    dismissed: "Dismissed = the mismatch is not acceptable -- treat as an error.",
    deferred: "Deferred = need more information before resolving the mismatch.",
  },
};

const DEFAULT_HINTS: Record<Decision, string> = {
  approved: "Approved = confirmed the system's diagnosis is correct.",
  dismissed: "Dismissed = the diagnosis does not hold up.",
  deferred: "Deferred = need more information before deciding.",
};

export function getDecisionHint(anomalyType: string, decision: Decision): string {
  return (HINTS[anomalyType] ?? DEFAULT_HINTS)[decision];
}
