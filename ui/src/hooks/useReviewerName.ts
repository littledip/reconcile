import { useState } from "react";

const STORAGE_KEY = "reconcile.reviewerName";

/** Single-evaluator identity: no login, just a name remembered in
 * localStorage for convenience across reloads (Progress_Log.md, Sept 28
 * UI design session -- "assume a single evaluator for now"). Wrapped in
 * try/catch since localStorage can throw (private browsing, blocked
 * storage) -- degrades to an in-memory-only name rather than crashing. */
export function useReviewerName() {
  const [name, setNameState] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? "";
    } catch {
      return "";
    }
  });

  const setName = (value: string) => {
    setNameState(value);
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch {
      // best-effort only -- see module docstring
    }
  };

  return { name, setName };
}
