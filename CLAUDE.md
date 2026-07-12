# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of Together AI's **Mixture-of-Agents (MoA)** reference implementation, extended with two things the upstream repo does not have:

1. A **provider abstraction layer** (`providers.py`) that routes all LLM calls through an OpenAI-compatible client. The Workbench defaults to a local **LM Studio / OpenAI-compatible** server (`MOA_BASE_URL`, default `http://127.0.0.1:1234/v1`) — LM Studio serves several models from one endpoint, which a mixture needs; a llama.cpp `llama-server` (one model per process) also works at any base URL.
2. A local **"Workbench"** — a FastAPI backend (`workbench/`) plus a React/Vite GUI (`ui/`) for running hybrid MoA workflows (orchestrate → parallel workers → synthesize → evaluate → refine) with human-approved file edits.

The original MoA CLI demo and the academic evaluation harness (AlpacaEval / MT-Bench / FLASK) are still present and still work.

This is a **Windows / PowerShell** workspace. The venv lives at `.venv\Scripts\python.exe`; prefer it over a bare `python`.

## Commands

```powershell
# Run the whole Workbench (starts backend + Vite GUI if not already listening, opens browser)
.\scripts\start_workbench.ps1
# -> http://127.0.0.1:5173 ; logs under %LOCALAPPDATA%\MoAWorkbench\logs

# Backend only
.\.venv\Scripts\python.exe -m uvicorn workbench.api:app --host 127.0.0.1 --port 8008

# GUI only
cd ui; npm run dev          # dev server on 127.0.0.1:5173
npm run build               # tsc + vite build -> ui/dist (backend serves this at "/" if present)

# Interactive MoA CLI demo (typer)
.\.venv\Scripts\python.exe bot.py

# One-file MoA examples
.\.venv\Scripts\python.exe moa.py             # 2-layer
.\.venv\Scripts\python.exe advanced-moa.py    # 3+ layer (MOA_LAYERS env)
```

### Tests

Tests are **`unittest`**-based (not pytest-native, though pytest can run them):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"   # all
.\.venv\Scripts\python.exe -m unittest test_providers            # one module
.\.venv\Scripts\python.exe -m unittest test_workbench.WorkerFanoutTests.test_roster_larger_than_subtasks_pads_to_roster_size  # one test
```

Test files: `test_providers.py`, `test_workbench.py` (workflow stages, roster-driven worker fan-out, reasoning-strip/degeneration guards), `test_workbench_api.py`, `test_workbench_graph.py` (graph engine), `test_workbench_security.py` (Host guard, path-traversal ids, server-authoritative roots, apply/reject status), `test_workbench_storage.py` (atomic writes, corruption recovery, concurrent-save race), `test_workbench_filesafety.py` (stale-diff/TOCTOU, newline preservation), `test_workbench_lifecycle.py` (bounded registry, stopped-run persistence, worker sibling-cancel). The API tests use FastAPI's `TestClient` and mock the model server + the provider; they redirect storage by patching `LOCALAPPDATA` to a temp dir. (`tests.py` is the upstream evaluation harness test and expects API keys — not part of the local suite.)

Note: constructing a real OpenAI/httpx client triggers SSL init that is blocked in some sandboxes; tests that need a client mock `providers.openai.OpenAI` rather than building one.

### Evaluation harness (upstream, requires Together/OpenAI keys)

`bash run_eval_alpaca_eval.sh` / `run_eval_mt_bench.sh` / `run_eval_flask.sh`. These drive `generate_for_*.py` + `eval_mt_bench.py` against the vendored `alpaca_eval/` and `FastChat/` packages (install with `pip install -e .` per the README).

## Provider configuration (the central abstraction)

**Everything LLM goes through `providers.py`.** Nothing else should construct an OpenAI client directly. Key functions: `get_provider_config`, `create_client` / `create_async_client`, `generate_chat_completion[_async]`, `generate_text_with_retries`, `generate_image` (text-to-image via `/v1/images/generations`), `get_default_model`, `get_default_reference_models`, `get_default_image_model`.

Provider is chosen by `MOA_PROVIDER` (aliases normalized in `_normalize_provider`): `openai-compatible` (default; aliases `local`/`llamacpp`/`custom`), `together`, `openai`, `omp`, `atomic`. There is no `lmstudio` provider — LM Studio is just an OpenAI-compatible server, so use `openai-compatible` with its base URL.

Config resolves from env via a first-match helper, so these matter:
- `MOA_PROVIDER`, `MOA_MODEL`, `MOA_REFERENCE_MODELS` (comma-separated), `MOA_BASE_URL`, `MOA_API_KEY`
- provider-specific keys: `TOGETHER_API_KEY`, `OPENAI_API_KEY`/`OPENAI_BASE_URL`, `OMP_API_KEY`/`OMP_BASE_URL`, `LM_STUDIO_*`
- `DEBUG=1` enables loguru debug logging

Notes:
- `omp` and `openai-compatible` **require** a base URL or they raise `ValueError`.
- `atomic` is an intentional placeholder — `create_client` raises `NotImplementedError` pointing at `atomic_agents_adapter.py`. Don't treat it as functional.
- API keys are **never** persisted (not in profiles, not in storage) — always read from env.
- Sync clients are cached by resolved config (`_sync_client_cache`); async clients are created per call and closed in a `finally` (never cache them — they bind to an event loop). Clients set `timeout=MOA_TIMEOUT` (default 120s) and `max_retries=0`. `generate_text_with_retries` only retries *transient* errors (`_is_retryable`: 429/5xx/timeouts/connection) with backoff+jitter and **raises** on exhaustion — it no longer returns `None`.

## Workbench architecture (`workbench/` + `ui/`)

Pipeline lives in `workbench/workflow.py::run_workflow`. Profile's `workflow` field selects the shape:
- `parallel_subtask`: orchestrator splits the task into exactly `len(worker_models)` subtasks (padded/trimmed by `_fit_subtasks`, capped at 8) → workers run in parallel (`asyncio.gather`, round-robined across `worker_models`, titled `Worker N: …`) → synthesize. No eval loop.
- `hybrid` (default): the above **plus** an evaluate→refine loop up to `max_iterations`.
- `iterative_evaluator`: skip workers; synthesize directly from the prompt, then evaluate→refine.
- `graph`: run a **custom pre-built flow graph** from `Profile.graph` (`workbench/graph.py` + `run_graph_workflow`). A `FlowGraph` is `nodes` (each `llm` or `fanout`, with `depends_on`, a prompt template referencing `{input}`/`{{node_id}}`/`{item}`, and a saved `lane`/`order` layout) plus an `output` node id. The executor validates (unique ids, deps exist, data-DAG acyclic — `validate_graph`), runs in topological order, and expands `fanout` nodes to one parallel instance per item of an upstream list output (capped at 8). A `gate` node scores the draft on configurable `checks` (each `{term, min}` — freeform terms like scope/direction/best_practices) and, when any term falls short, loops back to `loop_to` (an ancestor) re-running the loop body (`loop_body` = nodes downstream of `loop_to` and upstream of the gate) until every check passes or `max_loops` is hit; the gate's `loop_to` is a control back-edge (not a data dependency) so it doesn't make the data DAG cyclic. A refine node reaches the gate feedback via `{{gate_id}}` and its own prior draft via `{{self_id}}`. Loop iterations stack in the pop-out by incrementing `order`. **Conditional routing:** a node with `when_node`/`when_equals` runs only if that node's output contains the label (case-insensitive; `when_node` is an ordering dependency); a node whose every dependency was skipped is skipped too — so an untaken branch's whole tail drops out while a join with at least one live input still runs. Together the engine supports DAG + fan-out + gated refine loops + conditional branches. Authoring is either JSON or a **visual builder** (`ui/src/GraphBuilder.tsx`, Profiles tab, Visual/JSON toggle): drag nodes on a lane/order grid, click a node's ◦ handle then a target to wire a dependency, click a dependency edge to remove it, edit all node fields (incl. gate checks / loop_to / fanout over / branch when) in a per-node editor, and ★ sets the output. Both modes edit the same `profile.graph`; "Validate & tidy" runs `/api/graph/validate` and applies the auto-layout. It reuses the same trace/streaming machinery and emits each node's `lane`/`order` in the `stage_start` event, so the pop-out graph renders from the authored layout (data-driven, not auto-laid). `POST /api/graph/validate` returns validation errors + an auto-laid (`auto_layout`, lane = dependency depth) graph for the authoring UI (Profiles tab, when workflow is "graph").

The evaluate/refine loop relies on models returning JSON. `provider_complete` requests `response_format={"type":"json_object"}` for orchestration/evaluation and falls back gracefully if the provider rejects it. `extract_json` is deliberately lenient (handles fenced blocks and stray prose).

Module map:
- `workflow.py` — the orchestration stages (`make_subtasks`, `run_worker`, `synthesize`, `evaluate`, `refine`) and `TraceStep` recording.
- `api.py` — FastAPI app factory `create_app()`; REST under `/api/*`; serves `ui/dist` at `/` when built. CORS allows the Vite dev origin, and a `TrustedHostMiddleware` rejects non-loopback `Host` headers (DNS-rebinding guard; extend via `MOA_WORKBENCH_ALLOWED_HOSTS`). `/api/files/read`, `/api/files/list`, and run/change ids are server-validated — never trust client-supplied `allowed_roots` or ids for filesystem paths. Setting `MOA_WORKBENCH_TOKEN` gates all `/api/*` behind a shared secret (Authorization: Bearer / `X-MoA-Token` header / `?token=` query param — the last is how the header-less `EventSource` authenticates); off by default (loopback + Host guard is the baseline). The UI stores the token in `localStorage` and prompts on the first 401.
- The **pop-out flow graph** (`ui/src/FlowGraph.tsx`, served at `activity.html?run_id=…`) renders a run as a live left-to-right flowchart: one node per agent laid out by stage, **real flow edges** (SVG beziers with arrowheads; animated while the target agent runs) drawn from each step's upstream step ids — every stage's `stage_start` event and `TraceStep.metadata` carry `deps` (workers ← orchestrator, synthesizer ← workers, evaluator ⇄ refiner, and `depends_on`/fanout wiring for custom graphs) — plus a pulse on active agents, per-agent timer + approx token count, overall timer / active count, and click-an-agent-to-see-its-feed. It's SSE-driven (reuses `useRunActivity`) and **stays open until the user closes it**. Per-agent windows (serena-style): `activity.html?run_id=…&step_id=…` renders one agent's feed alone (window title = agent title); every graph node / stage card has an open-in-window button, and the Run tab's "Pop out each agent" toggle auto-opens one window per agent as it starts (SSE replay means a mid-run window still gets full history).
- `run_registry.py` — in-memory `RunRegistry`/`RunSession` backing **streaming** runs. Not persisted (lives for the process only); completed runs are still saved to `storage`.
- `schemas.py` — Pydantic models (`Profile`, `RunRequest`, `RunRecord`, `FileChange`, `TraceStep`, `ModelPlan`, …). `Profile` is the unit of config: provider, base_url, the three model roles (worker/aggregator/evaluator), `max_iterations`, and `allowed_roots`.
- `storage.py` — JSON-file persistence under `%LOCALAPPDATA%\MoAWorkbench` (`profiles.json`, `active_profile.txt`, `runs/`, `changes/`). No database.
- `file_safety.py` — **security boundary.** Every file read/write resolves the path against `allowed_roots` and raises `PathOutsideAllowedRoots` otherwise. Preserve this guard; do not add file I/O that bypasses `resolve_allowed_path`. `constrain_roots(requested, authoritative)` lets API callers narrow but never widen the server-trusted roots; `apply_file_change` re-reads the target and raises `FileChangedOnDisk` (→ 409) if it drifted from the proposal (TOCTOU), and reads/writes with `newline=""` to preserve exact bytes.
- `provider_models.py` — provider-agnostic model discovery: `list_models(base_url)` and `server_status(base_url)` read the **active profile's** OpenAI-compatible `/v1/models` (llama.cpp, OpenAI, any compatible server). No CLI. (The old LM Studio `lms`-CLI module + `model_planner.py` were removed — the tool is now provider-agnostic; the Models tab and status reflect whatever endpoint the active profile points at.)
- `provider_presets.py` — seed presets (llama.cpp, OMP, Together, generic, atomic) surfaced in the Profiles tab.

Cloud/API models: `Profile.cloud_models` is a registry of `CloudModel {alias, provider, model, base_url?}` entries. Anywhere a model name appears (workers/aggregator/evaluator/graph-node `model`), an alias routes that one call to its own provider via `workflow.resolve_model_route` — so a flow can mix local llama.cpp workers with a cloud aggregator. API keys always come from env (`OPENAI_API_KEY`, `TOGETHER_API_KEY`, `OMP_API_KEY`, `MOA_API_KEY`) and are never persisted. Cloud models have no local footprint; the old memory-cap/size accounting was removed with the LM Studio planner (`Profile.memory_cap_gb` remains in the schema for compat but nothing reads it).

### Run lifecycle (streaming)

Runs are asynchronous and streamed, not request/response:
- `POST /api/runs/start` launches `run_workflow` as a background asyncio task and returns a `run_id` immediately.
- `GET /api/runs/{run_id}/events` is an **SSE** endpoint (`text/event-stream`). The `RunSession` buffers every event — `stage_start`, streamed `token`, each completed `TraceStep`, and the terminal result — so a late subscriber replays what it missed, then receives live events. Subscribing/disconnecting never affects the run (viewers are pure observers).
- `POST /api/runs/{run_id}/stop` cancels the task and **persists the partial trace** so a stopped run still appears in history; `GET /api/runs/{run_id}` fetches the final record. Other routes: `/api/runs` (list/create records), `DELETE /api/runs/{id}`, `DELETE /api/profiles/{name}`, `/api/files/read`, `/api/file-changes/{id}/apply|reject`. Model discovery is `/api/status` + `/api/models`, both reading the active profile's `/v1/models`.

Saved flows as callable agents: a `Profile` is a named, persisted flow (its `workflow` field selects the shape). `GET /api/flows` lists them; `POST /api/flows/{name}/run` resolves the flow **server-side** and runs it synchronously from just `{prompt, context_files?}` (the caller never ships the config), returning the `RunRecord`; `POST /api/flows/{name}/run-stream` does the same but returns a `run_id` to subscribe to via `/api/runs/{id}/events`. Both share `_launch_run`/`run_workflow` with the ad-hoc run endpoints and honor the optional `MOA_WORKBENCH_TOKEN`. The Profiles tab shows a copy-able curl for the active flow. (A flow may be any of the four shapes, including a custom `graph`.)

Image generation: `POST /api/images` (`{prompt, n, size, model?}`) calls `providers.generate_image` through the **active profile's** provider/base_url and returns `{images:[{b64_json,url}]}` (b64 preferred so the GUI embeds it directly). The Images tab renders and downloads them; `Profile.image_model` (or `MOA_IMAGE_MODEL`) sets the model. Works against OpenAI/DALL·E, Together (FLUX), or any local server exposing `/v1/images/generations`.

The `RunSession` event buffer is a bounded `deque(maxlen)` and each SSE subscriber has a capped queue — a stalled viewer is dropped as a slow consumer rather than growing memory. `storage.py` writes atomically (temp + fsync + `os.replace`), tolerates a corrupt `profiles.json`/run file (quarantine + rebuild / skip), and serializes read-modify-write with an `RLock`.

### File-change flow (how the GUI edits files)

The synthesizer may emit a fenced JSON block `{"file_changes":[{"path","proposed_content"}]}`. `extract_file_changes` turns these into `FileChange` records (each carries a unified diff and the originating `allowed_roots`) with status `proposed`. **Nothing is written to disk** until the user calls `POST /api/file-changes/{id}/apply`, which rewrites the **entire file** with `proposed_content` (whole-file replacement, not a patch).

## Gotchas

- The Workbench default profile and the CLI demos (`moa.py`, `advanced-moa.py`, `bot.py`) all default to **`openai-compatible`** at `MOA_BASE_URL` (LM Studio, :1234). There is **no** `lmstudio` provider *name* — `_normalize_provider` raises `ValueError` for it (pinned by `test_providers.ProviderConfigTests.test_lmstudio_provider_is_no_longer_supported`); LM Studio is reached via `openai-compatible`, it's just an OpenAI-compatible server.
- Parallel workers need server-side concurrency: enable parallel requests in LM Studio's server settings, or for llama.cpp start `llama-server` with `--parallel N` (default is one slot → concurrent requests serialize). The Workbench probes llama.cpp's `/props` for `total_slots` and warns in the Run tab when the roster exceeds it; LM Studio exposes no such endpoint, so no warning is possible there.
- Multi-model mixtures need LM Studio (or multiple servers/cloud aliases): one `llama-server` process hosts one model. This is why the default moved back to LM Studio (2026-07).
- The worker roster drives the fan-out: `run_workflow` asks the orchestrator for exactly `len(worker_models)` subtasks (capped at `MAX_WORKERS=8`) and pads/trims to match (`_fit_subtasks`), so Worker ×N really runs N instances. Worker/evaluator/refiner titles carry instance/iteration numbers.
- `provider_complete` strips `<think>…</think>` reasoning from feeds and outputs (`strip_reasoning`/`ThinkStreamFilter`) and retries once with hotter sampling when `looks_degenerate` flags an instruction-echo loop; `Profile.frequency_penalty`/`presence_penalty` pass through to the provider.
- `bot.py` uses HuggingFace `datasets` + multiprocessing (`num_proc`) to fan out reference-model calls — that's why `datasets` is a dependency for a CLI chat demo.
- Provider config is `frozen` dataclass; pass overrides via function args, not mutation.
