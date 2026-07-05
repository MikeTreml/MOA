import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class WorkbenchApiTests(unittest.TestCase):
    def make_client(self, appdata):
        from fastapi.testclient import TestClient
        from workbench.api import create_app

        with patch.dict(os.environ, {"LOCALAPPDATA": appdata}):
            return TestClient(create_app())

    def test_models_endpoint_returns_mocked_lmstudio_models(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            model = {
                "type": "llm",
                "modelKey": "llama-3.2-1b-instruct",
                "sizeBytes": 712575975,
                "trainedForToolUse": True,
                "maxContextLength": 131072,
            }
            with patch("workbench.api.list_models", return_value=[model]), patch(
                "workbench.api.server_status",
                return_value={"loaded_models": [], "running": True},
            ):
                response = client.get("/api/models")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["models"][0]["modelKey"], model["modelKey"])

    def test_run_endpoint_returns_complete_trace_from_mocked_workflow(self):
        from workbench.schemas import Evaluation, Profile, RunRecord, TraceStep

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            record = RunRecord(
                profile_name="Default LM Studio",
                prompt="hello",
                workflow="hybrid",
                final_output="done",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
                trace=[
                    TraceStep(
                        stage="orchestrator",
                        title="Break into subtasks",
                        model="model",
                        input="hello",
                        output="{}",
                    )
                ],
            )
            payload = {
                "prompt": "hello",
                "profile": Profile(allowed_roots=[str(Path.cwd())]).model_dump(mode="json"),
            }
            with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
                response = client.post("/api/runs", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["final_output"], "done")
            self.assertEqual(body["trace"][0]["stage"], "orchestrator")

    def _create_run_with_change(self, client, workroot):
        from workbench.file_safety import create_file_change
        from workbench.schemas import Evaluation, RunRecord

        target = Path(workroot) / "out.txt"
        target.write_text("old", encoding="utf-8")
        record = RunRecord(
            profile_name="p",
            prompt="hi",
            workflow="hybrid",
            final_output="done",
            evaluation=Evaluation(status="PASS", feedback="ok", score=1),
        )
        change = create_file_change(
            run_id=record.id,
            path=str(target),
            proposed_content="new content",
            allowed_roots=[workroot],
        )
        record.file_changes = [change]
        with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
            response = client.post(
                "/api/runs", json={"prompt": "hi", "profile": {"allowed_roots": [workroot]}}
            )
        self.assertEqual(response.status_code, 200)
        return target, record.id, change.id

    def test_apply_change_updates_run_history_and_writes_file(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workroot:
            client = self.make_client(appdata)
            target, run_id, change_id = self._create_run_with_change(client, workroot)

            applied = client.post(f"/api/file-changes/{change_id}/apply")
            self.assertEqual(applied.status_code, 200)
            self.assertEqual(applied.json()["status"], "applied")
            self.assertEqual(target.read_text(encoding="utf-8"), "new content")

            run = client.get(f"/api/runs/{run_id}").json()
            self.assertEqual(run["file_changes"][0]["status"], "applied")

    def test_reject_change_updates_run_history_without_writing_file(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workroot:
            client = self.make_client(appdata)
            target, run_id, change_id = self._create_run_with_change(client, workroot)

            rejected = client.post(f"/api/file-changes/{change_id}/reject")
            self.assertEqual(rejected.status_code, 200)
            self.assertEqual(rejected.json()["status"], "rejected")
            self.assertEqual(target.read_text(encoding="utf-8"), "old")

            run = client.get(f"/api/runs/{run_id}").json()
            self.assertEqual(run["file_changes"][0]["status"], "rejected")

    def test_apply_missing_change_returns_404(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.post("/api/file-changes/change_does_not_exist/apply")
            self.assertEqual(response.status_code, 404)

    def test_run_with_no_models_returns_400(self):
        from workbench.schemas import Profile

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            payload = {
                "prompt": "hello",
                "profile": Profile(
                    worker_models=[],
                    aggregator_model="",
                    evaluator_model="",
                    allowed_roots=[str(Path.cwd())],
                ).model_dump(mode="json"),
            }
            response = client.post("/api/runs", json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn("no models", response.text.lower())

    def test_start_run_streams_live_activity_then_persists(self):
        from workbench.schemas import Evaluation, RunRecord, TraceStep

        async def fake_workflow(request, complete_fn=None, on_step=None, run_id=None, on_event=None):
            for stage, title in [("orchestrator", "Break into subtasks"), ("worker", "Inspect")]:
                if on_event is not None:
                    on_event("stage_start", {"id": f"s_{stage}", "stage": stage, "title": title, "model": "m"})
                step = TraceStep(stage=stage, title=title, model="m", output="o")
                if on_step is not None:
                    on_step(step)
            return RunRecord(
                id=run_id or "run_x",
                profile_name="p",
                prompt=request.prompt,
                workflow="hybrid",
                final_output="done",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            with patch("workbench.api.run_workflow", new=fake_workflow):
                start = client.post(
                    "/api/runs/start",
                    json={"prompt": "hello", "profile": {"allowed_roots": [str(Path.cwd())]}},
                )
                self.assertEqual(start.status_code, 200)
                run_id = start.json()["run_id"]

                events = []
                with client.stream("GET", f"/api/runs/{run_id}/events") as response:
                    for line in response.iter_lines():
                        if line.startswith("event:"):
                            events.append(line.split(":", 1)[1].strip())

                self.assertIn("step", events)
                self.assertIn("complete", events)
                self.assertEqual(events[-1], "end")

                persisted = client.get(f"/api/runs/{run_id}").json()
                self.assertEqual(persisted["final_output"], "done")
                self.assertEqual(persisted["id"], run_id)

    def test_events_for_unknown_run_returns_404(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.get("/api/runs/run_missing/events")
            self.assertEqual(response.status_code, 404)

    def test_stop_unknown_run_returns_404(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            self.assertEqual(client.post("/api/runs/run_missing/stop").status_code, 404)

    def test_stop_run_cancels_and_emits_stopped(self):
        async def slow_workflow(request, complete_fn=None, on_step=None, run_id=None, on_event=None):
            if on_event is not None:
                on_event("stage_start", {"id": "s1", "stage": "worker", "title": "work", "model": "m"})
            await asyncio.sleep(30)  # cancelled by the stop endpoint before this returns
            raise AssertionError("workflow should have been cancelled")

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            with patch("workbench.api.run_workflow", new=slow_workflow):
                run_id = client.post(
                    "/api/runs/start",
                    json={"prompt": "x", "profile": {"allowed_roots": [str(Path.cwd())]}},
                ).json()["run_id"]
                stop = client.post(f"/api/runs/{run_id}/stop")
                self.assertEqual(stop.status_code, 200)

                kinds = []
                with client.stream("GET", f"/api/runs/{run_id}/events") as response:
                    for line in response.iter_lines():
                        if line.startswith("event:"):
                            kinds.append(line.split(":", 1)[1].strip())

                self.assertIn("stopped", kinds)
                self.assertEqual(kinds[-1], "end")

    def test_graph_validate_returns_errors_and_layout(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            # Valid chain a -> b, output b: no errors, lanes assigned.
            good = {"graph": {"nodes": [{"id": "a"}, {"id": "b", "depends_on": ["a"]}], "output": "b"}}
            response = client.post("/api/graph/validate", json=good)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["errors"], [])
            lanes = {n["id"]: n["lane"] for n in body["graph"]["nodes"]}
            self.assertEqual(lanes["a"], 0)
            self.assertEqual(lanes["b"], 1)

            # Invalid: unknown output.
            bad = {"graph": {"nodes": [{"id": "a"}], "output": "z"}}
            errors = client.post("/api/graph/validate", json=bad).json()["errors"]
            self.assertTrue(errors)

    def test_flows_list_returns_saved_flows(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.get("/api/flows")
            self.assertEqual(response.status_code, 200)
            flows = response.json()["flows"]
            self.assertTrue(flows)
            self.assertIn("workflow", flows[0])
            self.assertIn("name", flows[0])

    def test_run_flow_by_name_resolves_profile_server_side(self):
        from workbench.schemas import Evaluation, RunRecord

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            name = client.get("/api/profiles").json()["active"]
            record = RunRecord(
                profile_name=name, prompt="hi", workflow="hybrid", final_output="agent result",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            captured = {}

            async def fake_workflow(request, **kwargs):
                captured["profile_name"] = request.profile.name
                captured["prompt"] = request.prompt
                return record

            with patch("workbench.api.run_workflow", new=fake_workflow):
                # Caller sends ONLY name + prompt — no config in the body.
                response = client.post(f"/api/flows/{name}/run", json={"prompt": "solve X"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["final_output"], "agent result")
            self.assertEqual(captured["profile_name"], name)  # resolved server-side
            self.assertEqual(captured["prompt"], "solve X")

    def test_run_flow_unknown_name_returns_404(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.post("/api/flows/DoesNotExist/run", json={"prompt": "x"})
            self.assertEqual(response.status_code, 404)

    def test_run_flow_stream_by_name_streams_then_persists(self):
        from workbench.schemas import Evaluation, RunRecord, TraceStep

        async def fake_workflow(request, complete_fn=None, on_step=None, run_id=None, on_event=None):
            if on_event is not None:
                on_event("stage_start", {"id": "s1", "stage": "worker", "title": "w", "model": "m"})
            step = TraceStep(stage="worker", title="w", model="m", output="o")
            if on_step is not None:
                on_step(step)
            return RunRecord(
                id=run_id or "run_x", profile_name=request.profile.name, prompt=request.prompt,
                workflow="hybrid", final_output="streamed",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            name = client.get("/api/profiles").json()["active"]
            with patch("workbench.api.run_workflow", new=fake_workflow):
                start = client.post(f"/api/flows/{name}/run-stream", json={"prompt": "go"})
                self.assertEqual(start.status_code, 200)
                run_id = start.json()["run_id"]
                kinds = []
                with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
                    for line in resp.iter_lines():
                        if line.startswith("event:"):
                            kinds.append(line.split(":", 1)[1].strip())
                self.assertIn("complete", kinds)
                self.assertEqual(kinds[-1], "end")
                self.assertEqual(client.get(f"/api/runs/{run_id}").json()["final_output"], "streamed")

    def test_images_endpoint_returns_generated_images(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            with patch(
                "workbench.api.generate_image",
                return_value=[{"b64_json": "QUJD", "url": None}],
            ):
                response = client.post("/api/images", json={"prompt": "a red cube", "n": 1, "size": "512x512"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["images"][0]["b64_json"], "QUJD")

    def test_images_endpoint_rejects_empty_prompt(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.post("/api/images", json={"prompt": "   "})
            self.assertEqual(response.status_code, 400)

    def test_images_endpoint_surfaces_provider_failure_as_502(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            with patch("workbench.api.generate_image", side_effect=RuntimeError("no image model")):
                response = client.post("/api/images", json={"prompt": "a cat"})
            self.assertEqual(response.status_code, 502)

    def test_provider_presets_include_omp_and_atomic_placeholder(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.get("/api/provider-presets")

        self.assertEqual(response.status_code, 200)
        preset_ids = {item["id"] for item in response.json()["presets"]}
        self.assertIn("omp-chatgpt-claude", preset_ids)
        self.assertIn("atomic-placeholder", preset_ids)

    def test_atomic_provider_returns_not_implemented_response(self):
        from workbench.schemas import Profile

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            payload = {
                "prompt": "hello",
                "profile": Profile(
                    name="Atomic",
                    provider="atomic",
                    worker_models=["atomic-worker"],
                    aggregator_model="atomic-orchestrator",
                    evaluator_model="atomic-evaluator",
                    allowed_roots=[str(Path.cwd())],
                ).model_dump(mode="json"),
            }
            response = client.post("/api/runs", json=payload)

        self.assertEqual(response.status_code, 501)
        self.assertIn("Atomic Agents", response.text)


if __name__ == "__main__":
    unittest.main()
