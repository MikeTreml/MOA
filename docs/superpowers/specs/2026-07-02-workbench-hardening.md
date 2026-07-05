# MoA Workbench Hardening — 2026-07-02

Goal chosen by the user: **harden what exists** (backend + `providers.py`), fixing
**everything found** by the audit, with a regression test per fix. Scope excluded the
React UI (its findings are logged below as follow-ups).

Delivered as five layered commits on branch `harden/workbench-backend`, each green.
Total: 59 unittest tests (was 26).

## Layer 1 — Security boundary
- `TrustedHostMiddleware` rejects non-loopback `Host` headers → blocks DNS-rebinding
  against this file-writing localhost API (`MOA_WORKBENCH_ALLOWED_HOSTS` to extend).
- `/api/files/read` intersects client `allowed_roots` with the union of saved profile
  roots (`constrain_roots`): callers may narrow scope, never widen it.
- Run/change ids validated `^(run|change|step)_[0-9a-f]{32}$` before touching the FS
  (blocks `..%5C` traversal).
- `default_allowed_roots` off-by-one fixed (`parents[2]`→`parents[1]`): repo, not its
  parent.
- apply/reject guard on `change.status` (409).

## Layer 2 — Storage durability
- `_atomic_write_text` (temp + fsync + `os.replace`) for every JSON/pointer write.
- Corrupt `profiles.json` quarantined + default rebuilt; corrupt run files skipped.
- `RLock` serializes read-modify-write (profiles, change→run sync).
- Added `delete_run`, `delete_profile` (guards the last profile).

## Layer 3 — File-change apply safety
- Staleness/TOCTOU guard: `apply_file_change` refuses (`FileChangedOnDisk`→409) if the
  target drifted from `original_content`.
- `newline=""` read/write preserves exact bytes (no LF→CRLF); CRLF originals round-trip.
- Context reads bounded to `max_bytes+1` (no whole-file slurp).
- `extract_file_changes` logs dropped proposals instead of swallowing.

## Layer 4 — Run lifecycle & registry
- Bounded event `deque(maxlen)` + capped subscriber queues; slow viewers dropped.
- Stopping/failing a run persists its partial trace.
- Failed/cancelled worker cancels siblings; their exceptions are retrieved.
- Atomic `attach()` (replay snapshot + subscribe); stream closes dropped viewers.
- `DELETE /api/runs/{id}`, `DELETE /api/profiles/{name}`.

## Layer 5 — Providers hardening
- `generate_text_with_retries`: retry only transient errors (`_is_retryable`) with
  backoff+jitter; raise on exhaustion instead of returning `None`.
- Async clients closed per call (no pool leak); sync clients cached; `timeout` +
  `max_retries=0`.
- Empty-choices guards in `get_completion_text` / `stream_text_chunks` /
  `stream_chat_completion_async`.
- DEBUG log reuses resolved config (no re-resolve crash for omp/openai-compatible).
- CLI: `bot.py` references char-iteration bug fixed + worker resilience + stream guard;
  `advanced-moa.py` `MOA_LAYERS` clamped to ≥2; `utils.py` mutable-default removed.

## Layer 6 — UI hardening (added under "repair all")
- SSE reconnect with capped backoff + deterministic replay rebuild (no double
  tokens); defensive parsing; "reconnecting" indicator.
- Backend-down boot error screen with Retry (was eternal "Loading…").
- History mid-run no longer orphans the live run; Refresh surfaces failures.
- Context files read from the textarea on Run even without "Preview context".
- Header pill only claims online/offline for LM Studio; remote → neutral label.
- Number inputs keep value/min when cleared; unsaved-edit confirm on switch.
- Files-tab "Save roots"; Apply confirmation (whole-file overwrite); stale load
  results cleared on switch.
- Profile delete, run delete, copy-final-output, Ctrl/Cmd+Enter to run.

## End-to-end verification (fake OpenAI-compatible upstream)
LM Studio's server wasn't running, so a fake OpenAI-compatible upstream stood in
to exercise the real code paths that otherwise need a live model. All three
harnesses are preserved under `e2e/` (see `e2e/README.md`) and pass:
- **Async-client leak fix** (`e2e/verify_provider_no_leak.py`) — real hybrid
  workflow through `providers.py`; every async client created *and* closed
  (5 → 5, zero unclosed warnings) with real SSE token streaming.
- **Full pipeline under real uvicorn** (`e2e/verify_backend_sse.py`, not
  TestClient) — orchestrator → 2 workers → synthesizer → evaluator streamed live
  over SSE; a **second connection replayed the entire buffer** including the
  terminal `complete`/`end`.
- **Browser SSE reconnect** (`e2e/recon_test.cjs`, headless Chromium via
  Playwright) — started a run, **dropped the first `/events` connection**, and
  the UI showed "Reconnecting…", reconnected (2 connections), replayed, and
  finished with the correct, **non-duplicated** output (status Complete).

Every behavior originally flagged as "needs a live run" is now verified.

## Follow-ups still open (nice-to-have, not bugs)
- Native file/folder picker for context files and allowed roots (still typed).
- Per-stage timing / token counts in the trace UI.
- `apply-all` / `reject-all` batch endpoints for a run's changes.
- SPA deep-link fallback for `StaticFiles(html=True)`.
- Profile rename/duplicate.
- Optional API auth token (localhost + Host guard is the current boundary).
