"""Serve the full Workbench stack for the browser reconnect test:
built UI (ui/dist) + backend + fake upstream, with a pre-seeded active profile
that points at the fake upstream. Run this, then run recon_test.cjs.

    python e2e/serve_stack.py            # then, in another shell:
    NODE_PATH=ui/node_modules node e2e/recon_test.cjs

Requires the UI to be built (cd ui && npm run build).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "e2e"))

import _fake_upstream  # noqa: E402

UP, API = 8773, 8099

appdata = tempfile.mkdtemp()
workroot = tempfile.mkdtemp()
os.environ["LOCALAPPDATA"] = appdata

# Pre-seed the active profile -> Fake (openai-compatible @ fake upstream).
moa_dir = Path(appdata) / "MoAWorkbench"
moa_dir.mkdir(parents=True, exist_ok=True)
profile = {
    "name": "Fake", "provider": "openai-compatible", "base_url": f"http://127.0.0.1:{UP}/v1",
    "memory_cap_gb": 80, "workflow": "hybrid", "worker_models": ["fa", "fb"],
    "aggregator_model": "fg", "evaluator_model": "fe", "max_iterations": 1,
    "allowed_roots": [workroot],
}
(moa_dir / "profiles.json").write_text(json.dumps([profile]), encoding="utf-8")
(moa_dir / "active_profile.txt").write_text("Fake", encoding="utf-8")

# token_delay makes the stream slow enough that a mid-stream drop is realistic.
_fake_upstream.start(UP, token_delay=0.25)

import uvicorn  # noqa: E402
from workbench.api import create_app  # noqa: E402

if not (REPO / "ui" / "dist").exists():
    print("ui/dist not found — run `cd ui && npm run build` first.", file=sys.stderr)
    sys.exit(2)

print(f"stack up: UI+API http://127.0.0.1:{API}  fake upstream :{UP}", flush=True)
uvicorn.run(create_app(), host="127.0.0.1", port=API, log_level="warning")
