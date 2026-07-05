# End-to-end verification harnesses

These exercise the real code paths that the `unittest` suite can't (they need a
running server, real HTTP, or a browser). They stand up a **fake
OpenAI-compatible upstream** (`_fake_upstream.py`) so no LM Studio / model is
required. They are **not** collected by `unittest discover` (the `verify_`
prefix and the non-package directory keep them out).

All use the local venv: `.venv\Scripts\python.exe`.

## 1. Provider async-client leak + streaming
```powershell
.\.venv\Scripts\python.exe e2e\verify_provider_no_leak.py
```
Runs the real hybrid workflow through `providers.py` and asserts every async
client is **created and closed** (no connection-pool leak) with real SSE
streaming. Expect `created=N closed=N`, `unclosed warns: 0`, `RESULT: PASS`.

## 2. Backend SSE under real uvicorn (live + replay-on-reconnect)
```powershell
.\.venv\Scripts\python.exe e2e\verify_backend_sse.py
```
Drives the full pipeline over HTTP under a real uvicorn server, then connects a
second time to the finished run and confirms the buffer replays including the
terminal events. Expect `RESULT: PASS`.

## 3. Browser SSE reconnect (Playwright + Chromium)
```powershell
cd ui; npm run build; cd ..
.\.venv\Scripts\python.exe e2e\serve_stack.py        # leave running
# in another shell:
$env:NODE_PATH="ui/node_modules"; node e2e\recon_test.cjs
```
Loads the built UI in headless Chromium, starts a run, **drops the first SSE
connection** to force the hook's reconnect, then asserts the UI shows
"Reconnecting…", recovers, and finishes with the correct, non-duplicated output.
Expect `events connections: 2`, `reconnecting shown: true`, `RESULT: PASS`.

## 4. Browser image generation (Playwright + Chromium)
With the same `serve_stack.py` running (its fake upstream also serves
`/v1/images/generations`):
```powershell
$env:NODE_PATH="ui/node_modules"; node e2e\image_test.cjs
```
Opens the Images tab, generates 2 images, and asserts real `<img>` elements
render from `data:image/png;base64` sources with download links.
Expect `images rendered: 2`, `RESULT: PASS`.

## 5. Flow-graph pop-out (Playwright + Chromium)
With `serve_stack.py` running:
```powershell
$env:NODE_PATH="ui/node_modules"; node e2e\graph_test.cjs
```
Starts a flow, opens `activity.html?run_id=…`, and asserts the graph renders one
node per agent with live stats (overall timer / active count / agent count),
pulses while running, and reveals an agent's feed on click.
Expect `graph nodes: 5`, `RESULT: PASS`.

## 6. Custom graph flow (Playwright + Chromium)
With `serve_stack.py` running:
```powershell
$env:NODE_PATH="ui/node_modules"; node e2e\graph_flow_test.cjs
```
Saves a custom `graph` flow (plan → fanout research → synthesize), runs it, and
asserts it executes server-side and renders in the pop-out from its saved layout
— fan-out expanding to real per-item nodes across the correct lanes.
Expect `node titles: Plan | Research 1 | Research 2 | Research 3 | Synthesize`, `RESULT: PASS`.

## 7. Refine-loop graph flow (Playwright + Chromium)
With `serve_stack.py` running (its fake scores gate prompts: fail twice, then pass):
```powershell
$env:NODE_PATH="ui/node_modules"; node e2e\loop_flow_test.cjs
```
Runs a graph with a gate that scores the draft on settable terms (scope /
direction) and loops back to the refine node until they pass. Asserts the
pop-out shows the loop iterating.
Expect `node titles: Draft | Revise | Revise (2) | Revise (3) | Check | Check (2) | Check (3)`, `RESULT: PASS`.

## 8. Conditional branch graph flow (Playwright + Chromium)
With `serve_stack.py` running:
```powershell
$env:NODE_PATH="ui/node_modules"; node e2e\branch_flow_test.cjs
```
Runs a graph with a router node; only the matching branch runs and a join merges
it. Asserts the untaken branch is skipped (absent from the pop-out).
Expect `node titles: Route | Taken | Deliver`, `RESULT: PASS`.

## 9. Visual graph builder (Playwright + Chromium)
With `serve_stack.py` running:
```powershell
$env:NODE_PATH="ui/node_modules"; node e2e\builder_test.cjs
```
Opens the Profiles tab of a graph flow and checks the visual builder renders one
node per graph node, that clicking a node opens its editor, that "Add node" adds
one, and that a fan-out `over` edge draws.
Expect `initial nodes: 3`, `after add: 4`, `RESULT: PASS`.

> Note: constructing an OpenAI/httpx client triggers SSL init that some sandboxes
> block; run these in a normal shell if a sandbox interferes.
