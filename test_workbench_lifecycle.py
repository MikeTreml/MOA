import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RunRegistryTests(unittest.TestCase):
    def test_slow_subscriber_is_dropped_not_unbounded(self):
        async def scenario():
            from workbench.run_registry import RunSession, MAX_SUBSCRIBER_QUEUE

            session = RunSession("run_" + "0" * 32)
            queue = session.subscribe()
            # Publish far more than the queue can hold without anyone draining it.
            for i in range(MAX_SUBSCRIBER_QUEUE + 50):
                session.on_event("token", {"id": "s", "text": str(i)})
            # The queue never exceeds its cap, and the slow subscriber was dropped.
            self.assertLessEqual(queue.qsize(), MAX_SUBSCRIBER_QUEUE)
            self.assertEqual(len(session.subscribers), 0)

        asyncio.run(scenario())

    def test_events_buffer_is_bounded(self):
        async def scenario():
            from workbench.run_registry import RunSession

            session = RunSession("run_" + "0" * 32)
            session.events = type(session.events)(maxlen=10)  # shrink for the test
            for i in range(100):
                session.on_event("token", {"text": str(i)})
            self.assertLessEqual(len(session.events), 10)
            self.assertTrue(session.replay_truncated)

        asyncio.run(scenario())

    def test_trim_evicts_finished_but_keeps_running(self):
        from workbench.run_registry import RunRegistry

        registry = RunRegistry(max_sessions=2)
        a = registry.create("run_" + "a" * 32)
        a.status = "complete"
        b = registry.create("run_" + "b" * 32)
        b.status = "running"
        registry.create("run_" + "c" * 32)  # triggers trim; evicts finished 'a'
        self.assertIsNone(registry.get("run_" + "a" * 32))
        self.assertIsNotNone(registry.get("run_" + "b" * 32))


class StopPersistenceTests(unittest.TestCase):
    def make_client(self, appdata):
        from fastapi.testclient import TestClient
        from workbench.api import create_app

        with patch.dict(os.environ, {"LOCALAPPDATA": appdata}):
            return TestClient(create_app())

    def test_stopped_run_persists_partial_trace(self):
        from workbench.schemas import TraceStep

        async def slow_workflow(request, complete_fn=None, on_step=None, run_id=None, on_event=None):
            # Emit one completed step, then hang until cancelled.
            step = TraceStep(stage="worker", title="partial", model="m", output="did some work")
            if on_step is not None:
                on_step(step)
            await asyncio.sleep(30)
            raise AssertionError("should have been cancelled")

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            with patch("workbench.api.run_workflow", new=slow_workflow):
                run_id = client.post(
                    "/api/runs/start",
                    json={"prompt": "x", "profile": {"allowed_roots": [str(Path.cwd())]}},
                ).json()["run_id"]
                self.assertEqual(client.post(f"/api/runs/{run_id}/stop").status_code, 200)

                # Drain events so the stop propagates through the runner.
                with client.stream("GET", f"/api/runs/{run_id}/events") as response:
                    for _ in response.iter_lines():
                        pass

                persisted = client.get(f"/api/runs/{run_id}")
                self.assertEqual(persisted.status_code, 200)
                body = persisted.json()
                self.assertEqual(body["trace"][0]["title"], "partial")

    def test_delete_profile_endpoint(self):
        from workbench.schemas import Profile

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            client.get("/api/profiles")  # seed the default
            client.post("/api/profiles", json=Profile(name="Extra").model_dump(mode="json"))

            deleted = client.delete("/api/profiles/Extra")
            self.assertEqual(deleted.status_code, 200)
            names = {p["name"] for p in client.get("/api/profiles").json()["profiles"]}
            self.assertNotIn("Extra", names)

            # Deleting down to the last profile is refused with 409.
            remaining = client.get("/api/profiles").json()["profiles"]
            while len(remaining) > 1:
                client.delete(f"/api/profiles/{remaining[0]['name']}")
                remaining = client.get("/api/profiles").json()["profiles"]
            last = remaining[0]["name"]
            self.assertEqual(client.delete(f"/api/profiles/{last}").status_code, 409)

    def test_apply_all_reports_per_change_and_tolerates_stale(self):
        from unittest.mock import AsyncMock

        from workbench.file_safety import create_file_change
        from workbench.schemas import Evaluation, RunRecord

        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workroot:
            client = self.make_client(appdata)
            good = Path(workroot) / "good.txt"
            good.write_text("old", encoding="utf-8")
            stale = Path(workroot) / "stale.txt"
            stale.write_text("original", encoding="utf-8")

            record = RunRecord(
                profile_name="p", prompt="hi", workflow="hybrid", final_output="x",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            record.file_changes = [
                create_file_change(run_id=record.id, path=str(good), proposed_content="new", allowed_roots=[workroot]),
                create_file_change(run_id=record.id, path=str(stale), proposed_content="new2", allowed_roots=[workroot]),
            ]
            with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
                client.post("/api/runs", json={"prompt": "hi", "profile": {"allowed_roots": [workroot]}})

            # Make the second change stale on disk after proposal.
            stale.write_text("edited by hand", encoding="utf-8")

            result = client.post(f"/api/runs/{record.id}/file-changes/apply-all")
            self.assertEqual(result.status_code, 200)
            body = result.json()
            self.assertEqual(body["applied"], 1)  # good applied, stale skipped
            self.assertEqual(good.read_text(encoding="utf-8"), "new")
            self.assertEqual(stale.read_text(encoding="utf-8"), "edited by hand")
            failed = [r for r in body["results"] if not r["ok"]]
            self.assertEqual(len(failed), 1)

    def test_reject_all_marks_all_rejected(self):
        from unittest.mock import AsyncMock

        from workbench.file_safety import create_file_change
        from workbench.schemas import Evaluation, RunRecord

        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workroot:
            client = self.make_client(appdata)
            target = Path(workroot) / "a.txt"
            target.write_text("x", encoding="utf-8")
            record = RunRecord(
                profile_name="p", prompt="hi", workflow="hybrid", final_output="x",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            record.file_changes = [
                create_file_change(run_id=record.id, path=str(target), proposed_content="n", allowed_roots=[workroot]),
            ]
            with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
                client.post("/api/runs", json={"prompt": "hi", "profile": {"allowed_roots": [workroot]}})

            result = client.post(f"/api/runs/{record.id}/file-changes/reject-all")
            self.assertEqual(result.status_code, 200)
            run = client.get(f"/api/runs/{record.id}").json()
            self.assertEqual(run["file_changes"][0]["status"], "rejected")
            self.assertEqual(target.read_text(encoding="utf-8"), "x")  # untouched

    def test_delete_run_then_404(self):
        from unittest.mock import AsyncMock

        from workbench.schemas import Evaluation, RunRecord

        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            record = RunRecord(
                profile_name="p", prompt="x", workflow="hybrid", final_output="done",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
                client.post("/api/runs", json={"prompt": "x", "profile": {"allowed_roots": [str(Path.cwd())]}})
            run_id = record.id
            self.assertEqual(client.delete(f"/api/runs/{run_id}").status_code, 200)
            self.assertEqual(client.get(f"/api/runs/{run_id}").status_code, 404)


class WorkerCancellationTests(unittest.TestCase):
    def test_worker_failure_cancels_siblings(self):
        async def scenario():
            from workbench.schemas import Profile, RunRequest
            from workbench.workflow import run_workflow

            started = {"count": 0}
            cancelled = {"count": 0}

            async def flaky_complete(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                content = messages[0]["content"]
                if "Break the user task" in content:
                    return '{"subtasks":[{"title":"A","prompt":"a"},{"title":"B","prompt":"b"},{"title":"C","prompt":"c"}]}'
                # Worker calls: first raises, the others should get cancelled while sleeping.
                started["count"] += 1
                if started["count"] == 1:
                    raise RuntimeError("worker boom")
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled["count"] += 1
                    raise
                return "late"

            profile = Profile(
                workflow="parallel_subtask",
                worker_models=["w1", "w2", "w3"],
                allowed_roots=[str(Path.cwd())],
            )
            request = RunRequest(prompt="do it", profile=profile)
            with self.assertRaises(RuntimeError):
                await run_workflow(request, complete_fn=flaky_complete)
            # At least one sibling was cancelled rather than left running.
            self.assertGreaterEqual(cancelled["count"], 1)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
