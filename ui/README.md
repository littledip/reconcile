# Reconcile -- Escalation Review UI

React + TypeScript (via Vite) single-evaluator UI for the human review side
of the escalation queue -- lists what's pending (`GET /escalations`), lets
a reviewer pull similar past decisions on demand
(`GET /escalations/{id}/similar`), and submit a decision
(`POST /escalations/{id}/decision`). Design rationale: Progress_Log.md's
Sept 28 UI design session.

No Node backend/proxy layer -- this talks to `app/main.py` (FastAPI)
directly from the browser. Node here is just the build/dev tooling.

## Running it

```bash
npm install
npm run dev
```

Needs the FastAPI app running separately (`uvicorn app.main:app --reload`
from the repo root) -- `app/main.py` allows CORS from
`http://localhost:5173` (Vite's default dev port) specifically for this.

Override the API URL with `VITE_API_BASE_URL` in a `.env` (see
`.env.example`) if FastAPI is running somewhere other than
`localhost:8000`.

## Design notes

- **No login / no user model** -- single evaluator assumed for now. The
  reviewer's name is a plain text field, remembered in `localStorage`
  (`src/hooks/useReviewerName.ts`) purely for convenience across reloads,
  not an auth mechanism.
- **Polling, not websockets** -- the queue view refetches `GET /escalations`
  every 10s (`src/hooks/useEscalations.ts`). Simple, and the endpoint
  already supports being called repeatedly.
- **Similar-past-decisions is on-demand**, not shown automatically --
  matches how the backend itself was designed (Sept 23 session).
- **No extra state-management dependencies** -- plain `fetch` + hooks. The
  scope here doesn't justify React Query/Redux/etc.

## Layout

```
ui/
  src/
    types.ts                       TS interfaces mirroring app/main.py's request/response shapes
    api.ts                         fetch wrappers for the 3 escalation endpoints
    hooks/
      useEscalations.ts            Polls GET /escalations every 10s
      useReviewerName.ts           localStorage-backed reviewer name, no auth
    components/
      EscalationList.tsx           Queue list (left pane)
      EscalationDetail.tsx         Anomaly detail + similar-episodes lookup + decision form (right pane)
    App.tsx                        Ties the above together
```
