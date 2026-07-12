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
            if "Break the user task into exactly" in text:
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
            if "Break the user task into exactly" in text:
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
            if "Break the user task into exactly" in text:
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
            if "Break the user task into exactly" in text:
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
            if "Break the user task into exactly" in text:
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


class WorkerFanoutTests(unittest.TestCase):
    """The worker roster drives the fan-out: N entries -> N workers, padded or
    trimmed against whatever subtask count the orchestrator returns."""

    def _run(self, worker_models, orchestrator_json, max_iterations=1):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into exactly" in text:
                return orchestrator_json
            if "Worker subtask" in text:
                return f"worker:{model}"
            if "Synthesize" in text:
                return "draft"
            if "Evaluate" in text:
                return '{"status":"PASS","feedback":"ok","score":0.9}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(
            name="t",
            worker_models=worker_models,
            aggregator_model="agg",
            evaluator_model="eval",
            max_iterations=max_iterations,
            allowed_roots=[str(Path.cwd())],
        )
        return asyncio.run(run_workflow(RunRequest(prompt="x", profile=profile), complete_fn=fake_complete))

    def test_roster_larger_than_subtasks_pads_to_roster_size(self):
        record = self._run(
            ["m", "m", "m", "m"],
            '{"subtasks":[{"title":"A","prompt":"a"},{"title":"B","prompt":"b"}]}',
        )
        workers = [step for step in record.trace if step.stage == "worker"]
        self.assertEqual(len(workers), 4)
        self.assertIn("(take 2)", workers[2].title)

    def test_roster_smaller_than_subtasks_trims_to_roster_size(self):
        record = self._run(
            ["m", "m"],
            '{"subtasks":[{"title":"A","prompt":"a"},{"title":"B","prompt":"b"},'
            '{"title":"C","prompt":"c"},{"title":"D","prompt":"d"}]}',
        )
        workers = [step for step in record.trace if step.stage == "worker"]
        self.assertEqual(len(workers), 2)

    def test_worker_titles_carry_instance_numbers(self):
        record = self._run(
            ["m", "m"],
            '{"subtasks":[{"title":"Same","prompt":"a"},{"title":"Same","prompt":"a"}]}',
        )
        titles = [step.title for step in record.trace if step.stage == "worker"]
        self.assertEqual(titles, ["Worker 1: Same", "Worker 2: Same"])

    def test_roster_is_capped_at_max_workers(self):
        from workbench.workflow import MAX_WORKERS

        record = self._run(
            ["m"] * (MAX_WORKERS + 4),
            '{"subtasks":[{"title":"A","prompt":"a"}]}',
        )
        workers = [step for step in record.trace if step.stage == "worker"]
        self.assertEqual(len(workers), MAX_WORKERS)

    def test_eval_and_refine_iterations_are_numbered(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into exactly" in text:
                return '{"subtasks":[{"title":"A","prompt":"a"}]}'
            if "Worker subtask" in text:
                return "w"
            if "Synthesize" in text:
                return "draft v1"
            if "Revise the draft" in text:
                return "draft v2"
            if "Evaluate" in text:
                return '{"status":"REVISE","feedback":"more","score":0.3}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(
            name="t", worker_models=["m"], aggregator_model="a", evaluator_model="e",
            max_iterations=2, allowed_roots=[str(Path.cwd())],
        )
        record = asyncio.run(run_workflow(RunRequest(prompt="x", profile=profile), complete_fn=fake_complete))
        loop_titles = [step.title for step in record.trace if step.stage in {"evaluator", "refiner"}]
        self.assertEqual(loop_titles, ["Evaluate draft", "Revise draft", "Evaluate draft (2)"])


class ReasoningStripTests(unittest.TestCase):
    def test_strip_reasoning_removes_complete_blocks(self):
        from workbench.workflow import strip_reasoning

        self.assertEqual(
            strip_reasoning("<think>I should say hi.</think>Hello there."),
            "Hello there.",
        )

    def test_strip_reasoning_without_think_is_untouched(self):
        from workbench.workflow import strip_reasoning

        self.assertEqual(strip_reasoning("plain answer\n"), "plain answer\n")

    def test_strip_reasoning_keeps_text_of_unclosed_tag(self):
        from workbench.workflow import strip_reasoning

        self.assertEqual(strip_reasoning("<think>truncated reasoning"), "truncated reasoning")

    def test_strip_reasoning_handles_closing_only_template_shape(self):
        # DeepSeek-R1-style templates pre-fill <think> in the prompt, so the
        # completion arrives as "reasoning…</think>answer" with no open tag.
        from workbench.workflow import strip_reasoning

        self.assertEqual(
            strip_reasoning("Some chain of thought reasoning here.</think>The real answer."),
            "The real answer.",
        )

    def test_strip_reasoning_leaves_midtext_tags_alone(self):
        # Think-tags inside the body are content (e.g. code that mentions the
        # tags), not reasoning — stripping them would corrupt file_changes.
        from workbench.workflow import strip_reasoning

        code = 'OPEN = "<think>"\nCLOSE = "</think>"\nreturn OPEN + CLOSE'
        self.assertEqual(strip_reasoning(code), code)

    def test_stream_filter_suppresses_leading_cot_split_across_deltas(self):
        from workbench.workflow import ThinkStreamFilter

        stream_filter = ThinkStreamFilter()
        visible = "".join(
            stream_filter.feed(delta)
            for delta in ["<thi", "nk>secret ", "plan</th", "ink>Hello world"]
        )
        visible += stream_filter.flush()
        self.assertEqual(visible, "Hello world")

    def test_stream_filter_passes_midtext_think_through(self):
        from workbench.workflow import ThinkStreamFilter

        stream_filter = ThinkStreamFilter()
        text = 'Use OPEN = "<think>" and CLOSE = "</think>" in the filter.'
        visible = stream_filter.feed(text) + stream_filter.flush()
        self.assertEqual(visible, text)

    def test_stream_filter_flush_emits_unclosed_thought(self):
        # Mirrors strip_reasoning: a truncated all-thinking output is emitted
        # at end-of-stream rather than leaving the feed empty.
        from workbench.workflow import ThinkStreamFilter

        stream_filter = ThinkStreamFilter()
        held = stream_filter.feed("<think>partial reasoning")
        self.assertEqual(held, "")
        self.assertEqual(stream_filter.flush(), "partial reasoning")

    def test_stream_filter_emits_partial_angle_text_at_flush(self):
        from workbench.workflow import ThinkStreamFilter

        stream_filter = ThinkStreamFilter()
        visible = stream_filter.feed("a < b and <thi")
        visible += stream_filter.flush()
        self.assertEqual(visible, "a < b and <thi")


class DegenerationGuardTests(unittest.TestCase):
    LOOP = (
        "I need to produce a final answer that is a short story. "
        'But instructions: "Synthesize the worker outputs into one final draft for the user." '
    ) * 8

    def test_looks_degenerate_flags_instruction_echo_loop(self):
        from workbench.workflow import looks_degenerate

        self.assertTrue(looks_degenerate(self.LOOP))

    def test_looks_degenerate_passes_normal_text(self):
        from workbench.workflow import looks_degenerate

        healthy = " ".join(f"Sentence number {i} says something new and different." for i in range(30))
        self.assertFalse(looks_degenerate(healthy))

    def test_looks_degenerate_catches_comma_separated_loop(self):
        from workbench.workflow import looks_degenerate

        self.assertTrue(looks_degenerate("the same phrase repeating forever and ever, " * 40))

    def test_looks_degenerate_tolerates_a_legitimate_refrain(self):
        # A 4x chorus is a refrain, not a runaway loop — it must not trigger a
        # retry that could replace a correct answer.
        from workbench.workflow import looks_degenerate

        verses = [
            f"Verse {i} tells a different part of the story with its own words and imagery here" for i in range(4)
        ]
        chorus = "And the chorus comes back around singing the same line again"
        song = "\n".join(verses[:2] + [chorus, chorus] + verses[2:] + [chorus, chorus])
        self.assertFalse(looks_degenerate(song))

    def test_provider_complete_retries_once_when_stream_degenerates(self):
        from workbench.schemas import Profile
        from workbench.workflow import DEGENERATE_RETRY_NOTE, provider_complete

        calls = {"n": 0}
        loop_text = self.LOOP

        def fake_stream(**kwargs):
            calls["n"] += 1
            text = loop_text if calls["n"] == 1 else "clean answer"

            async def gen():
                yield text

            return gen()

        tokens: list[str] = []
        profile = Profile(allowed_roots=[str(Path.cwd())])
        with patch("workbench.workflow.stream_chat_completion_async", new=fake_stream):
            out = asyncio.run(
                provider_complete(
                    "m", [{"role": "user", "content": "x"}], profile=profile, on_token=tokens.append
                )
            )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(out, "clean answer")
        self.assertIn(DEGENERATE_RETRY_NOTE, tokens)


class FlowEdgeTests(unittest.TestCase):
    """Every trace step records the step ids that fed it (metadata['deps']),
    so the pop-out can draw the actual agent flow."""

    def test_hybrid_trace_carries_upstream_step_ids(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into exactly" in text:
                return '{"subtasks":[{"title":"A","prompt":"a"},{"title":"B","prompt":"b"}]}'
            if "Worker subtask" in text:
                return "w"
            if "Synthesize" in text:
                return "draft v1"
            if "Revise the draft" in text:
                return "draft v2"
            if "Evaluate" in text:
                return '{"status":"REVISE","feedback":"more","score":0.3}'
            raise AssertionError(f"Unexpected prompt: {text}")

        profile = Profile(
            name="t", worker_models=["m", "m"], aggregator_model="a", evaluator_model="e",
            max_iterations=2, allowed_roots=[str(Path.cwd())],
        )
        record = asyncio.run(run_workflow(RunRequest(prompt="x", profile=profile), complete_fn=fake_complete))

        orchestrator = record.trace[0]
        workers = [s for s in record.trace if s.stage == "worker"]
        synth = next(s for s in record.trace if s.stage == "synthesizer")
        evals = [s for s in record.trace if s.stage == "evaluator"]
        refiner = next(s for s in record.trace if s.stage == "refiner")

        for worker in workers:
            self.assertEqual(worker.metadata["deps"], [orchestrator.id])
        self.assertEqual(sorted(synth.metadata["deps"]), sorted(w.id for w in workers))
        self.assertEqual(evals[0].metadata["deps"], [synth.id])
        self.assertEqual(refiner.metadata["deps"], [evals[0].id])
        self.assertEqual(evals[1].metadata["deps"], [refiner.id])

    def test_stage_start_events_carry_deps(self):
        from workbench.schemas import Profile, RunRequest
        from workbench.workflow import run_workflow

        async def fake_complete(model, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            if "Break the user task into exactly" in text:
                return '{"subtasks":[{"title":"A","prompt":"a"}]}'
            if "Evaluate" in text:
                return '{"status":"PASS","feedback":"ok","score":0.9}'
            return "out"

        profile = Profile(
            name="t", worker_models=["m"], aggregator_model="a", evaluator_model="e",
            allowed_roots=[str(Path.cwd())],
        )
        events = []
        asyncio.run(
            run_workflow(
                RunRequest(prompt="x", profile=profile),
                complete_fn=fake_complete,
                on_event=lambda kind, payload: events.append((kind, payload)),
            )
        )
        starts = {p["stage"]: p for kind, p in events if kind == "stage_start"}
        self.assertNotIn("deps", starts["orchestrator"])
        self.assertEqual(starts["worker"]["deps"], [starts["orchestrator"]["id"]])
        self.assertEqual(starts["synthesizer"]["deps"], [starts["worker"]["id"]])
        self.assertEqual(starts["evaluator"]["deps"], [starts["synthesizer"]["id"]])


class ModelListEnrichmentTests(unittest.TestCase):
    """list_models merges LM Studio /api/v0/models metadata (quant, state, ctx)
    into the /v1/models list and sizes every model — real bytes when the server
    reports them (llama.cpp meta.size), otherwise a name+quant estimate."""

    def _urlopen_by_url(self, responses: dict):
        import io

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake(url, timeout=0):
            for suffix, body in responses.items():
                if url.endswith(suffix):
                    return FakeResponse(body)
            raise OSError(f"unexpected url {url}")

        return fake

    def test_lmstudio_metadata_and_estimate_are_merged(self):
        from workbench.provider_models import list_models

        responses = {
            "/v1/models": b'{"data":[{"id":"meta/llama-3.3-70b"}]}',
            "/api/v0/models": (
                b'{"data":[{"id":"meta/llama-3.3-70b","type":"llm","quantization":"Q4_K_M",'
                b'"state":"loaded","max_context_length":131072}]}'
            ),
        }
        with patch("workbench.provider_models.urllib.request.urlopen", self._urlopen_by_url(responses)):
            models = list_models("http://127.0.0.1:1234/v1")
        self.assertEqual(len(models), 1)
        model = models[0]
        self.assertEqual(model["quantization"], "Q4_K_M")
        self.assertEqual(model["state"], "loaded")
        self.assertEqual(model["maxContextLength"], 131072)
        self.assertTrue(model["sizeIsEstimate"])
        # 70e9 params at ~0.58 bytes/param for Q4 — sanity range, not exactness.
        self.assertGreater(model["sizeBytes"], 30e9)
        self.assertLess(model["sizeBytes"], 50e9)

    def test_llamacpp_meta_size_is_reported_as_real(self):
        from workbench.provider_models import list_models

        responses = {
            "/v1/models": b'{"data":[{"id":"qwen-3b","meta":{"size":4000000000}}]}',
            "/api/v0/models": b"[]",  # not LM Studio: non-object answer ignored
        }
        with patch("workbench.provider_models.urllib.request.urlopen", self._urlopen_by_url(responses)):
            models = list_models("http://127.0.0.1:1235/v1")
        self.assertEqual(models[0]["sizeBytes"], 4000000000)
        self.assertFalse(models[0]["sizeIsEstimate"])

    def test_estimate_reads_total_params_not_moe_active_count(self):
        from workbench.provider_models import estimate_size_bytes

        moe = estimate_size_bytes("baidu/ernie-4.5-21b-a3b", "Q8_0")
        self.assertGreater(moe, 20e9)  # 21B total at ~1.07 B/param
        self.assertLess(moe, 25e9)
        self.assertEqual(estimate_size_bytes("ibm/granite-4-h-tiny", "Q4_K_M"), 0)


class SlotProbeTests(unittest.TestCase):
    """probe_total_slots reads llama.cpp's /props at the server root and must
    tolerate any non-llama.cpp answer without raising."""

    def _urlopen_returning(self, body: bytes, seen: dict):
        import io

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake(url, timeout=0):
            seen["url"] = url
            return FakeResponse(body)

        return fake

    def test_reads_total_slots_from_props_at_server_root(self):
        from workbench.provider_models import probe_total_slots

        seen: dict = {}
        with patch(
            "workbench.provider_models.urllib.request.urlopen",
            self._urlopen_returning(b'{"total_slots": 4}', seen),
        ):
            self.assertEqual(probe_total_slots("http://127.0.0.1:1235/v1"), 4)
        self.assertEqual(seen["url"], "http://127.0.0.1:1235/props")

    def test_non_object_json_from_a_proxy_yields_none(self):
        from workbench.provider_models import probe_total_slots

        for body in (b"[]", b'"ok"', b"null", b"3"):
            with patch(
                "workbench.provider_models.urllib.request.urlopen",
                self._urlopen_returning(body, {}),
            ):
                self.assertIsNone(probe_total_slots("http://127.0.0.1:1235/v1"))

    def test_unreachable_props_yields_none(self):
        from workbench.provider_models import probe_total_slots

        def boom(url, timeout=0):
            raise OSError("connection refused")

        with patch("workbench.provider_models.urllib.request.urlopen", boom):
            self.assertIsNone(probe_total_slots("http://127.0.0.1:1235/v1"))


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
