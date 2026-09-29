import type { EscalatedAnomaly, Severity } from "./types";

export type SortOption = "relevance" | "priority" | "date_asc";

export const SORT_LABELS: Record<SortOption, string> = {
  relevance: "Relevance",
  priority: "Priority",
  date_asc: "Date (oldest first)",
};

const SEVERITY_RANK: Record<Severity, number> = { high: 2, medium: 1, low: 0 };

/** Returns a new, sorted array -- never mutates the input (escalations
 * comes straight from useEscalations' polled state). */
export function sortEscalations(items: EscalatedAnomaly[], sortBy: SortOption): EscalatedAnomaly[] {
  const sorted = [...items];

  switch (sortBy) {
    case "priority":
      sorted.sort((a, b) => SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity]);
      break;

    case "date_asc":
      sorted.sort(
        (a, b) => new Date(a.escalated_at).getTime() - new Date(b.escalated_at).getTime(),
      );
      break;

    case "relevance":
      // Urgency (Sept 28 UI design session): severity ranks first (a
      // high-severity item outranks a low one regardless of age); within
      // the same severity, the one that's been waiting longest surfaces
      // first. This is a lexicographic comparator, not a blended numeric
      // score -- it produces the same practical ranking ("a high-severity
      // item from 3 days ago outranks a low one from an hour ago")
      // without an arbitrary weighting constant to justify.
      sorted.sort((a, b) => {
        const severityDiff = SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity];
        if (severityDiff !== 0) return severityDiff;
        return new Date(a.escalated_at).getTime() - new Date(b.escalated_at).getTime();
      });
      break;
  }

  return sorted;
}
