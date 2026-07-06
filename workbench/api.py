from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .file_safety import (
    FileChangedOnDisk,
    PathOutsideAllowedRoots,
    apply_file_change,
    constrain_roots,
    list_directory,
    read_context_files,
    reject_file_change,
)
from providers import generate_image

from .lmstudio import list_models, load_models, server_status
from .graph import auto_layout, validate_graph
from .model_planner import plan_models
from .provider_presets import provider_presets
from .run_registry import RunRegistry, RunSession
from .schemas import Evaluation, FlowGraph, ModelPlan, Profile, RunRecord, RunRequest
from .schemas import new_id
from .storage import WorkbenchStorage
from .workflow import run_workflow


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class ModelPlanRequest(BaseModel):
    memory_cap_gb: float = 80
    models: list[dict[str, Any]] | None = None


class LoadPlanRequest(BaseModel):
    models: list[str] = Field(default_factory=list)
    plan: ModelPlan | None = None


class FileReadRequest(BaseModel):
    paths: list[str]
    # Advisory only — the server intersects this with the profile's saved roots
    # (see /api/files/read). A client can narrow scope here but never widen it.
    allowed_roots: list[str] = Field(default_factory=list)


class FileListRequest(BaseModel):
    path: str | None = None


class RenameRequest(BaseModel):
    new_name: str


class ImageRequest(BaseModel):
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    model: str | None = None


class FlowRunRequest(BaseModel):
    """Input for invoking a saved flow by name — just the prompt (and optional
    context files); the flow's provider/models/shape come from the saved profile."""
    prompt: str
    context_files: list[str] = Field(default_factory=list)


class GraphCheckRequest(BaseModel):
    graph: FlowGraph


def create_app() -> FastAPI:
    app = FastAPI(title="MoA Workbench", version="0.1.0")
    storage = WorkbenchStorage()
    registry = RunRegistry()

    # Host allowlist. This API can read and write local files, so it must only
    # answer requests addressed to the loopback host. It rejects requests whose
    # Host header is an attacker-controlled domain (DNS rebinding) even though
    # they arrive on 127.0.0.1. "testserver" is Starlette's TestClient default.
    allowed_hosts = ["localhost", "127.0.0.1", "[::1]", "testserver"]
    extra_hosts = os.environ.get("MOA_WORKBENCH_ALLOWED_HOSTS", "")
    allowed_hosts.extend(host.strip() for host in extra_hosts.split(",") if host.strip())
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # Optional shared-secret gate for /api/*. Off by default (loopback + Host
    # guard is the baseline); set MOA_WORKBENCH_TOKEN to require it, e.g. when
    # exposing the port. The token may arrive as an Authorization: Bearer header,
    # an X-MoA-Token header, or a ?token= query param — the last is the only way
    # a browser EventSource (which can't set headers) can authenticate the SSE
    # stream.
    api_token = os.environ.get("MOA_WORKBENCH_TOKEN", "").strip()
    if api_token:

        @app.middleware("http")
        async def require_api_token(request: Request, call_next):
            path = request.url.path
            if request.method == "OPTIONS" or not path.startswith("/api/"):
                return await call_next(request)
            header = request.headers.get("authorization", "")
            provided = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-moa-token", "")
            if not provided:
                provided = request.query_params.get("token", "")
            if not hmac.compare_digest(provided, api_token):
                return JSONResponse({"detail": "Invalid or missing API token."}, status_code=401)
            return await call_next(request)

    allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    extra_origins = os.environ.get("MOA_WORKBENCH_CORS_ORIGINS", "")
    allowed_origins.extend(
        origin.strip() for origin in extra_origins.split(",") if origin.strip()
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/lmstudio/status")
    def lmstudio_status():
        profile = storage.get_active_profile()
        return server_status(profile.base_url)

    @app.get("/api/models")
    def models():
        profile = storage.get_active_profile()
        return {"models": list_models(), "loaded": server_status(profile.base_url)["loaded_models"]}

    @app.get("/api/provider-presets")
    def presets():
        return {"presets": provider_presets()}

    @app.post("/api/model-plan")
    def model_plan(request: ModelPlanRequest):
        return plan_models(request.models or list_models(), request.memory_cap_gb)

    @app.post("/api/models/load-plan")
    def load_plan(request: LoadPlanRequest):
        model_keys = request.models
        if request.plan:
            model_keys = request.plan.worker_models + [
                request.plan.aggregator_model,
                request.plan.evaluator_model,
            ]
        return {"results": load_models([model for model in model_keys if model])}

    @app.get("/api/profiles")
    def profiles():
        return {
            "profiles": storage.list_profiles(),
            "active": storage.get_active_profile().name,
        }

    @app.post("/api/profiles")
    def save_profile(profile: Profile):
        saved = storage.save_profile(profile)
        storage.set_active_profile(saved.name)
        return saved

    @app.post("/api/profiles/{name}/rename")
    def rename_profile(name: str, request: RenameRequest):
        try:
            return storage.rename_profile(name, request.new_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/profiles/{name}")
    def delete_profile(name: str):
        try:
            storage.delete_profile(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "deleted", "name": name}

    @app.get("/api/runs")
    def runs():
        return {"runs": storage.list_runs()}

    @app.post("/api/runs")
    async def create_run(request: RunRequest):
        try:
            record = await run_workflow(request)
        except PathOutsideAllowedRoots as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        storage.save_run(record)
        return record

    def _launch_run(request: RunRequest) -> str:
        """Start a workflow as a background task and return its run id. Shared by
        the ad-hoc /runs/start path and the saved-flow /flows/{name}/run-stream
        path so both stream identically and persist partial work on stop/failure."""
        run_id = new_id("run")
        session = registry.create(run_id)

        def persist_partial(reason: str) -> None:
            try:
                partial = RunRecord(
                    id=run_id,
                    profile_name=request.profile.name,
                    prompt=request.prompt,
                    workflow=request.profile.workflow,
                    trace=list(session.trace_steps),
                    final_output="",
                    evaluation=Evaluation(status="REVISE", feedback=reason, score=0),
                    context_files=request.context_files,
                )
                storage.save_run(partial)
            except Exception:  # persistence is best-effort; never mask the stop
                pass

        async def runner() -> None:
            try:
                record = await run_workflow(
                    request,
                    on_step=session.on_step,
                    on_event=session.on_event,
                    run_id=run_id,
                )
                storage.save_run(record)
                session.finish(record)
            except asyncio.CancelledError:
                persist_partial("Run stopped by user.")
                session.stop()
                raise
            except NotImplementedError as exc:
                session.fail(501, str(exc))
            except (ValueError, PathOutsideAllowedRoots) as exc:
                session.fail(400, str(exc))
            except Exception as exc:  # surface any failure to subscribers
                persist_partial(f"Run failed: {exc}")
                session.fail(500, str(exc))

        session.task = asyncio.create_task(runner())
        return run_id

    @app.post("/api/runs/start")
    async def start_run(request: RunRequest):
        """Kick a run off as a background task and return its id immediately.

        The run executes independently of any client connection, so an Activity
        view can subscribe (or be closed) without affecting the agents' work."""
        return {"run_id": _launch_run(request), "status": "running"}

    @app.post("/api/graph/validate")
    def graph_validate(request: GraphCheckRequest):
        """Authoring helper: report problems and return a tidy auto-layout (lanes
        = dependency depth) the author can accept or nudge. Pure — runs nothing."""
        errors = validate_graph(request.graph)
        # Don't run layout on a rejected (e.g. oversized) graph.
        graph = request.graph if errors else auto_layout(request.graph)
        return {"errors": errors, "graph": graph}

    @app.get("/api/flows")
    def flows():
        """Saved flows = named profiles, surfaced by their agent-facing shape."""
        return {
            "flows": [
                {"name": p.name, "workflow": p.workflow, "provider": p.provider}
                for p in storage.list_profiles()
            ]
        }

    @app.post("/api/flows/{name}/run")
    async def run_flow(name: str, request: FlowRunRequest):
        """Call a saved flow like a whole agent: name + input -> final record.
        Resolves the flow server-side (the caller never ships the config) and
        returns the completed run synchronously."""
        try:
            profile = storage.get_profile(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Flow not found: {name}") from exc
        run_request = RunRequest(prompt=request.prompt, profile=profile, context_files=request.context_files)
        try:
            record = await run_workflow(run_request)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (ValueError, PathOutsideAllowedRoots) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        storage.save_run(record)
        return record

    @app.post("/api/flows/{name}/run-stream")
    async def run_flow_stream(name: str, request: FlowRunRequest):
        """Same as /flows/{name}/run but streamed: returns a run_id to subscribe
        to via /api/runs/{run_id}/events."""
        try:
            profile = storage.get_profile(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Flow not found: {name}") from exc
        run_request = RunRequest(prompt=request.prompt, profile=profile, context_files=request.context_files)
        return {"run_id": _launch_run(run_request), "status": "running"}

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(run_id: str):
        """Explicitly cancel an in-flight run. This is a deliberate user action,
        distinct from closing a (view-only) Activity window."""
        session = registry.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"No active run: {run_id}")
        if session.task is not None and not session.task.done():
            session.task.cancel()
        return {"run_id": run_id, "status": "stopping"}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        """Server-sent events for a run: replays everything so far, then streams
        new activity live until the run ends. Read-only and multi-subscriber,
        with a periodic keepalive so idle connections don't time out."""
        session = registry.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"No active run: {run_id}")

        async def event_stream():
            queue, replay = session.attach()
            try:
                if session.replay_truncated:
                    yield _sse(
                        "notice",
                        {"detail": "Some earlier events were dropped from replay (buffer cap reached)."},
                    )
                for kind, data in replay:
                    yield _sse(kind, data)
                # A finished run's replay already ends with the terminal event.
                if replay and replay[-1][0] == "end":
                    return
                while True:
                    try:
                        kind, data = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        # If the run is over and nothing is queued, we're done —
                        # this also releases a viewer that was dropped as a slow
                        # consumer and would otherwise wait forever.
                        if session.finished and queue.empty():
                            yield _sse("end", None)
                            break
                        yield ": keepalive\n\n"
                        continue
                    yield _sse(kind, data)
                    if kind == "end":
                        break
            finally:
                session.unsubscribe(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return storage.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        try:
            storage.delete_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "run_id": run_id}

    @app.post("/api/file-changes/{change_id}/apply")
    def apply_change(change_id: str):
        try:
            change = storage.get_file_change(change_id)
            if change.status != "proposed":
                raise HTTPException(
                    status_code=409,
                    detail=f"Change already {change.status}; only proposed changes can be applied.",
                )
            applied = apply_file_change(change)
            storage.save_file_change(applied)
            return applied
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileChangedOnDisk as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PathOutsideAllowedRoots as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/file-changes/{change_id}/reject")
    def reject_change(change_id: str):
        try:
            change = storage.get_file_change(change_id)
            if change.status == "applied":
                raise HTTPException(
                    status_code=409, detail="Change already applied; cannot reject."
                )
            rejected = reject_file_change(change)
            storage.save_file_change(rejected)
            return rejected
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _resolve_many(run_id: str, action: str) -> dict:
        """Apply or reject every still-proposed change of a run. One change
        failing (e.g. a stale diff) is recorded and skipped, never aborting the
        batch, so the user gets a per-file report."""
        try:
            record = storage.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        results = []
        for change in record.file_changes:
            if change.status != "proposed":
                continue
            try:
                fresh = storage.get_file_change(change.id)
                # Re-check the authoritative status (the run's inlined copy may be
                # stale if the change was applied/rejected via another endpoint).
                if fresh.status != "proposed":
                    continue
                if action == "apply":
                    storage.save_file_change(apply_file_change(fresh))
                else:
                    storage.save_file_change(reject_file_change(fresh))
                results.append({"id": change.id, "path": change.path, "ok": True, "detail": action})
            except (FileChangedOnDisk, PathOutsideAllowedRoots) as exc:
                results.append({"id": change.id, "path": change.path, "ok": False, "detail": str(exc)})
            except KeyError as exc:
                results.append({"id": change.id, "path": change.path, "ok": False, "detail": str(exc)})
        return {"results": results, "applied": sum(1 for r in results if r["ok"])}

    @app.post("/api/runs/{run_id}/file-changes/apply-all")
    def apply_all(run_id: str):
        return _resolve_many(run_id, "apply")

    @app.post("/api/runs/{run_id}/file-changes/reject-all")
    def reject_all(run_id: str):
        return _resolve_many(run_id, "reject")

    @app.post("/api/files/read")
    def read_files(request: FileReadRequest):
        # The security boundary is server-defined: constrain whatever the client
        # asked for to the roots saved in local profiles. A caller cannot read
        # outside them no matter what allowed_roots it sends.
        roots = constrain_roots(request.allowed_roots, storage.authoritative_roots())
        try:
            return {"files": read_context_files(request.paths, roots)}
        except (OSError, PathOutsideAllowedRoots) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/images")
    def images(request: ImageRequest):
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Image prompt is required.")
        profile = storage.get_active_profile()
        try:
            result = generate_image(
                prompt=request.prompt,
                model=request.model or profile.image_model or None,
                n=request.n,
                size=request.size,
                provider=profile.provider,
                base_url=profile.base_url,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:  # provider/network/model errors -> 502
            raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}") from exc
        return {"images": result, "model": request.model or profile.image_model or None}

    @app.post("/api/files/list")
    def list_files(request: FileListRequest):
        # Read-only directory browse, constrained to the saved profile roots.
        try:
            return list_directory(request.path, storage.authoritative_roots())
        except (OSError, PathOutsideAllowedRoots) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    ui_dist = Path(__file__).resolve().parents[1] / "ui" / "dist"
    if ui_dist.exists():
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")

    return app


app = create_app()
