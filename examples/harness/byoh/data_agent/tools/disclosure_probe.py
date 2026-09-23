"""Records whether an agent actually files a compliance disclosure.

`../harbor-server` hands each rollout a `COMPLIANCE_ENDPOINT` and a bearer token, and the task
instruction tells the agent to POST a disclosure there when the data it inspected is sensitive.
Whether a given model *does* that, from prose alone, is the open question the whole
compliance-grading design rests on -- and it is not something the grader can answer, because a
model that never calls the tool and a model that correctly decided not to call it produce the same
silence.

This stands in for `../rle`'s real endpoint during a local run: same route, same auth header, but
it records every call (and every *rejected* call) instead of grading. Run it, point
`COMPLIANCE_ENDPOINT` at it, run some tasks, then read the summary.

    python tools/disclosure_probe.py --port 8899

Not part of the deployed system; a measurement tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

_LOCK = threading.Lock()
_CALLS: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
        auth = self.headers.get("authorization", "")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"_unparseable": raw[:400]}

        record = {
            "at": datetime.now(timezone.utc).isoformat(),
            "path": self.path,
            "rollout": self.headers.get("x-rollout-id", ""),
            # Recorded as a boolean, never stored: the point is whether the agent sent one.
            "authorized": auth.lower().startswith("bearer ") and len(auth) > 10,
            # Correlation key for concurrent batches. The agent's `curl` sends no rollout header,
            # so the only thing tying a disclosure to its task is the per-rollout bearer token.
            # Fingerprinted rather than stored, so a real RLE token never lands in the log.
            "token_fp": hashlib.sha256(auth.split(" ", 1)[-1].encode()).hexdigest()[:12] if auth else "",
            "body": body,
        }
        with _LOCK:
            _CALLS.append(record)

        print(
            f"  [disclosure] {self.path} auth={'ok' if record['authorized'] else 'MISSING'} "
            f"columns={body.get('columns_reported')}",
            flush=True,
        )
        payload = json.dumps({"recorded": True}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        """`/_calls` returns what has been recorded, so a test run can assert on it."""
        with _LOCK:
            payload = json.dumps(_CALLS).encode()
        self.send_response(200 if self.path == "/_calls" else 404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        """Silences the default per-request stderr line; do_POST prints what matters."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"disclosure probe listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\nrecorded {len(_CALLS)} disclosure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
