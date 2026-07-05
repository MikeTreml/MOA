"""E2E: the full production path under real uvicorn (not TestClient).

Starts the backend + fake upstream, runs a hybrid workflow over HTTP, streams
the SSE pipeline live, then reconnects a SECOND time to the finished run and
confirms the buffer replays including the terminal events. Exit 0 on PASS.

    python e2e/verify_backend_sse.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "e2e"))

import _fake_upstream  # noqa: E402

UP, API = 8772, 8098
os.environ["LOCALAPPDATA"] = tempfile.mkdtemp()
WORKROOT = tempfile.mkdtemp()


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{API}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def stream_kinds(run_id):
    kinds = []
    with urllib.request.urlopen(f"http://127.0.0.1:{API}/api/runs/{run_id}/events", timeout=30) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("event:"):
                kinds.append(line.split(":", 1)[1].strip())
                if kinds[-1] == "end":
                    break
    return kinds


def main():
    up = _fake_upstream.start(UP)

    import uvicorn
    from workbench.api import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=API, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(2.5)

    profile = {"name": "Fake", "provider": "openai-compatible", "base_url": f"http://127.0.0.1:{UP}/v1",
               "memory_cap_gb": 80, "workflow": "hybrid", "worker_models": ["fa", "fb"],
               "aggregator_model": "fg", "evaluator_model": "fe", "max_iterations": 1,
               "allowed_roots": [WORKROOT]}
    run_id = post("/api/runs/start", {"prompt": "Build a thing", "profile": profile})["run_id"]
    live = stream_kinds(run_id)
    replay = stream_kinds(run_id)
    with urllib.request.urlopen(f"http://127.0.0.1:{API}/api/runs/{run_id}") as r:
        persisted = json.loads(r.read())

    server.should_exit = True
    up.shutdown()

    print("live kinds  :", live)
    print("replay kinds:", replay)
    print("stages      :", [s["stage"] for s in persisted["trace"]])
    print("final_output:", repr(persisted["final_output"])[:50])
    ok = ("complete" in live and live[-1] == "end" and "token" in live
          and "complete" in replay and replay[-1] == "end"
          and len(persisted["trace"]) >= 4 and persisted["final_output"])
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
