#!/usr/bin/env python3
"""Replay eval/eval_dataset.json's batches against a *live, running*
server, over HTTP -- so the queue's real reviewer UI and SSE push have
something realistic to show, rather than a single hand-typed curl
transaction.

Deliberately different from eval/run_eval.py, which this script is not a
replacement for:

  - run_eval.py imports the orchestrator and calls it in-process, forces
    heuristic classification for determinism, and scores the result
    against ground truth. It's the regression gate.
  - This script only speaks HTTP to POST /reconcile on whatever server is
    already running -- it uses that server's own classification provider
    (live Anthropic, local Ollama, or heuristic, whatever LLM_PROVIDER
    that process resolved to), doesn't score anything, and doesn't care
    what's inside the response beyond a short progress line. Its only job
    is to put realistic traffic through the live queue.

Note both would work here -- run_eval.py talks to the same real Redis/
Mongo your server uses (it never overrides MONGO_URI/REDIS_URL), so its
escalations show up in the live queue too. This script exists for when
you want that traffic to go through the server's real classification
provider instead of the eval run's forced heuristic one, or just want a
smaller, repeatable "drive some live demo traffic" tool that isn't
bundled with scoring/assertions.

Idempotency note: transaction_ids in eval_dataset.json are fixed, so a
re-run re-persists the same transactions (persist_batch_node upserts --
harmless). anomaly_ids are NOT fixed across runs -- they're regenerated
per batch each time (anomaly_1, anomaly_2, ...) -- so a second replay
overwrites any still-open escalation from the first run under the same
anomaly_id. Same idempotency escalate_dispute already documents for a
re-run batch; fine for a demo, just don't expect two replays to leave
twice as many items in the queue.

Live classification is sequential, one Anthropic call per transaction
(app/agents/orchestrator.py's classify_all_node is a plain for-loop, not
batched/parallel) -- a 50-transaction batch can comfortably take a couple
of minutes against the real API, not seconds. --timeout below defaults
generously for that; lower it if you're running heuristic/local-model
classification instead, where a batch finishes in well under a second.

Run with:
    python scripts/replay_eval_dataset.py
    python scripts/replay_eval_dataset.py --delay 5
    python scripts/replay_eval_dataset.py --batch 1 --timeout 60   # heuristic/local mode: fast, short timeout is fine
    python scripts/replay_eval_dataset.py --base-url http://localhost:8000 --dataset eval/eval_dataset.json

Needs a server already running (uvicorn app.main:app --reload) -- this
script only makes HTTP calls, no app imports, so it has no dependency on
your venv beyond the standard library.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dataset(path: str) -> dict:
    return json.loads(Path(path).read_text())


def post_batch(base_url: str, transactions: list[dict], timeout: float) -> dict:
    body = json.dumps({"transactions": transactions}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/reconcile",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Couldn't reach {base_url} -- is the server running? "
            f"(uvicorn app.main:app --reload)\n{exc}"
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"POST /reconcile failed ({exc.code}): {detail}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/eval_dataset.json", help="path to the generated dataset")
    parser.add_argument("--base-url", default="http://localhost:8000", help="the running server's base URL")
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="seconds to pause between batches, so the UI/SSE push is easy to watch (default: 2.0)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="replay only this one batch's run_order (default: all batches, in run_order)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds to wait for each batch's POST /reconcile to finish -- generous by default "
        "since live classification is sequential and slow; lower it for heuristic/local-model mode (default: 300.0)",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    batches = sorted(dataset["batches"], key=lambda b: b["run_order"])
    if args.batch is not None:
        batches = [b for b in batches if b["run_order"] == args.batch]
        if not batches:
            raise SystemExit(f"No batch with run_order={args.batch} in {args.dataset}")

    print(f"Replaying {len(batches)} batch(es) from {args.dataset} against {args.base_url}")

    for i, batch in enumerate(batches):
        transactions = batch["transactions"]
        print(f"  -> {batch['batch_id']} (run_order {batch['run_order']}, {len(transactions)} transactions)...", end=" ", flush=True)
        result = post_batch(args.base_url, transactions, args.timeout)
        num_anomalies = len(result.get("anomalies", []))
        num_escalations = len(result.get("escalations", []))
        print(f"{num_anomalies} anomaly(ies), {num_escalations} escalated")

        if i < len(batches) - 1 and args.delay > 0:
            time.sleep(args.delay)

    print("Done. Check the evaluator UI (or GET /escalations) for what landed in the queue.")


if __name__ == "__main__":
    main()
