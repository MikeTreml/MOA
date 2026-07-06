import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch



class WorkflowTests(unittest.TestCase):
    def test_hybrid_runs_parallel_workers_then_synthesizes_and_evaluates(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into 2 to 4" in text:
                return '{"subtasks":[{"title":"Inspect","prompt":"Inspect the repo"},{"title":"Plan","prompt":"Plan the change"}]}'
            if "Worker subtask" in text:
                return f"worker:{model}"
            if "Synthesize" in text:
                return "draft answer"
            if "Evaluate" in text:
                return '{"status":"PASS","feedback":"good","score":0.92}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(
            name="test",
            worker_models=["worker-a", "worker-b"],
            aggregator_model="aggregator",
            evaluator_model="evaluator",
            allowed_roots=[str(Path.cwd())],
        )
        request = RunRequest(prompt="Build a thing", profile=profile)

        record = asyncio.run(run_workflow(request, complete_fn=fake_complete))

        self.assertEqual(record.final_output, "draft answer")
        self.assertEqual(record.evaluation.status, "PASS")
        self.assertEqual(
            [step.stage for step in record.trace],
            ["orchestrator", "worker", "worker", "synthesizer", "evaluator"],
        )

    def test_evaluator_stops_at_max_iterations_and_preserves_trace(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into 2 to 4" in text:
                return '{"subtasks":[{"title":"Only","prompt":"Do it"}]}'
            if "Worker subtask" in text:
                return "worker output"
            if "Synthesize" in text:
                return "draft v1"
            if "Revise the draft" in text:
                return "draft v2"
            if "Evaluate" in text:
                return '{"status":"REVISE","feedback":"tighten it","score":0.4}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(
            name="test",
            worker_models=["worker-a"],
            aggregator_model="aggregator",
            evaluator_model="evaluator",
            max_iterations=2,
            allowed_roots=[str(Path.cwd())],
        )
        request = RunRequest(prompt="Build a thing", profile=profile)

        record = asyncio.run(run_workflow(request, complete_fn=fake_complete))

        self.assertEqual(record.final_output, "draft v2")
        self.assertEqual(record.evaluation.status, "REVISE")
        self.assertEqual(
            [step.stage for step in record.trace],
            ["orchestrator", "worker", "synthesizer", "evaluator", "refiner", "evaluator"],
        )


    def test_on_step_emits_each_agent_step_live_and_in_order(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into 2 to 4" in text:
                return '{"subtasks":[{"title":"Inspect","prompt":"Inspect"},{"title":"Plan","prompt":"Plan"}]}'
            if "Worker subtask" in text:
                return f"worker:{model}"
            if "Synthesize" in text:
                return "draft answer"
            if "Evaluate" in text:
                return '{"status":"PASS","feedback":"good","score":0.92}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(
            name="test",
            worker_models=["worker-a", "worker-b"],
            aggregator_model="aggregator",
            evaluator_model="evaluator",
            allowed_roots=[str(Path.cwd())],
        )
        request = RunRequest(prompt="Build a thing", profile=profile)

        seen = []
        record = asyncio.run(run_workflow(request, complete_fn=fake_complete, on_step=seen.append))

        # Every step is surfaced live, in the same order, and the final record is unchanged.
        self.assertEqual(
            [step.stage for step in seen],
            ["orchestrator", "worker", "worker", "synthesizer", "evaluator"],
        )
        self.assertEqual([step.id for step in seen], [step.id for step in record.trace])

    def test_on_event_streams_stage_starts_and_tagged_tokens(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def streaming_complete(model, messages, json_mode=False, on_token=None, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into 2 to 4" in text:
                return '{"subtasks":[{"title":"A","prompt":"a"}]}'
            if "Evaluate" in text:
                return '{"status":"PASS","feedback":"ok","score":0.9}'
            if on_token is not None:  # worker / synthesize / refine stream their tokens
                for token in ["draft ", "answer"]:
                    on_token(token)
            return "draft answer"

        profile = Profile(
            name="t",
            worker_models=["w"],
            aggregator_model="a",
            evaluator_model="e",
            allowed_roots=[str(Path.cwd())],
        )
        request = RunRequest(prompt="x", profile=profile)

        events = []
        asyncio.run(
            run_workflow(request, complete_fn=streaming_complete, on_event=lambda kind, payload: events.append((kind, payload)))
        )

        kinds = [kind for kind, _ in events]
        self.assertIn("stage_start", kinds)
        self.assertIn("token", kinds)
        # Every token is tagged with a step id that announced itself via stage_start.
        start_ids = {payload["id"] for kind, payload in events if kind == "stage_start"}
        token_ids = {payload["id"] for kind, payload in events if kind == "token"}
        self.assertTrue(token_ids.issubset(start_ids))

    def test_on_step_is_optional_and_default_run_is_unchanged(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into 2 to 4" in text:
                return '{"subtasks":[{"title":"Only","prompt":"Do it"}]}'
            if "Worker subtask" in text:
                return "worker output"
            if "Synthesize" in text:
                return "draft"
            if "Evaluate" in text:
                return '{"status":"PASS","feedback":"ok","score":0.9}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(name="test", worker_models=["w"], aggregator_model="a", evaluator_model="e", allowed_roots=[str(Path.cwd())])
        request = RunRequest(prompt="x", profile=profile)

        record = asyncio.run(run_workflow(request, complete_fn=fake_complete))
        self.assertEqual(record.final_output, "draft")


class FileSafetyTests(unittest.TestCase):
    def test_rejects_writes_outside_allowed_roots(self):
        from workbench.file_safety import PathOutsideAllowedRoots, resolve_allowed_path

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            inside = Path(root) / "inside.txt"
            outside = Path(other) / "outside.txt"
            inside.write_text("inside", encoding="utf-8")
            outside.write_text("outside", encoding="utf-8")

            self.assertEqual(resolve_allowed_path(str(inside), [root]), inside.resolve())
            with self.assertRaises(PathOutsideAllowedRoots):
                resolve_allowed_path(str(outside), [root])


class RunRegistryTests(unittest.TestCase):
    def test_session_buffers_replays_and_fans_out_steps(self):
        from workbench.run_registry import RunRegistry
        from workbench.schemas import Evaluation, RunRecord, TraceStep

        async def scenario():
            registry = RunRegistry()
            session = registry.create("run_1")
            queue = session.subscribe()
            session.on_step(TraceStep(stage="orchestrator", title="t", output="o"))
            session.finish(
                RunRecord(
                    id="run_1",
                    profile_name="p",
                    prompt="x",
                    workflow="hybrid",
                    final_output="done",
                    evaluation=Evaluation(status="PASS", score=1),
                )
            )
            events = []
            while not queue.empty():
                events.append(await queue.get())
            return session, events

        session, events = asyncio.run(scenario())
        self.assertEqual([kind for kind, _ in events], ["step", "complete", "end"])
        self.assertEqual(session.events[0][1]["stage"], "orchestrator")
        self.assertEqual(session.status, "complete")

    def test_registry_trims_finished_sessions_but_keeps_running(self):
        from workbench.run_registry import RunRegistry

        registry = RunRegistry(max_sessions=2)
        first = registry.create("a")
        first.status = "complete"
        registry.create("b").status = "complete"
        registry.create("c")  # still running; exceeds the cap of 2

        self.assertIsNone(registry.get("a"))  # oldest finished session evicted
        self.assertIsNotNone(registry.get("c"))  # in-flight run preserved


class StorageTests(unittest.TestCase):
    def test_profiles_persist_in_local_app_data(self):
        from workbench.schemas import Profile
        from workbench.storage import WorkbenchStorage

        with tempfile.TemporaryDirectory() as appdata:
            with patch.dict(os.environ, {"LOCALAPPDATA": appdata}):
                storage = WorkbenchStorage()
                profile = Profile(
                    name="Saved",
                    worker_models=["worker"],
                    aggregator_model="agg",
                    evaluator_model="eval",
                    allowed_roots=[str(Path.cwd())],
                )

                storage.save_profile(profile)
                storage.set_active_profile("Saved")

                reloaded = WorkbenchStorage()
                self.assertEqual(reloaded.get_active_profile().name, "Saved")
                self.assertEqual(reloaded.get_profile("Saved").aggregator_model, "agg")


class CloudModelRoutingTests(unittest.TestCase):
    def _profile(self):
        from workbench.schemas import CloudModel, Profile

        return Profile(
            name="Mixed",
            provider="openai-compatible",
            base_url="http://local:1235/v1",
            worker_models=["qwen-3b", "gpt4o"],
            aggregator_model="gpt4o",
            evaluator_model="qwen-3b",
            cloud_models=[
                CloudModel(alias="gpt4o", provider="openai", model="gpt-4o-mini"),
                CloudModel(alias="claude", provider="omp", model="claude-3-5-sonnet", base_url="http://127.0.0.1:4141/v1"),
            ],
        )

    def test_alias_routes_to_cloud_provider(self):
        from workbench.workflow import resolve_model_route

        profile = self._profile()
        self.assertEqual(resolve_model_route(profile, "gpt4o"), ("openai", None, "gpt-4o-mini"))
        self.assertEqual(
            resolve_model_route(profile, "claude"),
            ("omp", "http://127.0.0.1:4141/v1", "claude-3-5-sonnet"),
        )
        # Non-alias falls through to the profile's default provider.
        self.assertEqual(
            resolve_model_route(profile, "qwen-3b"),
            ("openai-compatible", "http://local:1235/v1", "qwen-3b"),
        )

    def test_alias_with_empty_model_uses_alias_as_id(self):
        from workbench.schemas import CloudModel, Profile
        from workbench.workflow import resolve_model_route

        profile = Profile(name="X", cloud_models=[CloudModel(alias="gpt-4o-mini", provider="openai")])
        self.assertEqual(resolve_model_route(profile, "gpt-4o-mini"), ("openai", None, "gpt-4o-mini"))

    def test_provider_complete_sends_resolved_route(self):
        async def scenario():
            from unittest.mock import AsyncMock, patch

            from workbench.workflow import provider_complete

            profile = self._profile()
            fake = AsyncMock()
            fake.return_value = type(
                "R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]}
            )()
            with patch("workbench.workflow.generate_chat_completion_async", new=fake):
                out = await provider_complete("gpt4o", [{"role": "user", "content": "x"}], profile=profile, json_mode=True)
            self.assertEqual(out, "hi")
            kwargs = fake.call_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-4o-mini")
            self.assertEqual(kwargs["provider"], "openai")
            self.assertIsNone(kwargs["base_url"])

        asyncio.run(scenario())


class TraceTimingTests(unittest.TestCase):
    def test_stage_started_at_precedes_ended_at(self):
        async def scenario():
            import time

            from workbench.schemas import Profile, RunRequest
            from workbench.workflow import run_workflow

            async def slow_complete(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                content = messages[0]["content"]
                if "Break the user task" in content:
                    return '{"subtasks":[{"title":"A","prompt":"a"}]}'
                time.sleep(0.02)  # ensure a measurable stage duration
                return "answer"

            profile = Profile(
                workflow="parallel_subtask",
                worker_models=["w1"],
                allowed_roots=[str(Path.cwd())],
            )
            record = await run_workflow(RunRequest(prompt="hi", profile=profile), complete_fn=slow_complete)
            worker = next(step for step in record.trace if step.stage == "worker")
            # started_at was captured before the call, ended_at at construction after.
            self.assertLessEqual(worker.started_at, worker.ended_at)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
