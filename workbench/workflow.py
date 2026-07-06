from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from providers import (
    generate_chat_completion_async,
    get_completion_text,
    stream_chat_completion_async,
)

from .file_safety import create_file_change, read_context_files
from .schemas import Evaluation, FileChange, Profile, RunRecord, RunRequest, Subtask, TraceStep
from .schemas import new_id, now_iso


CompleteFn = Callable[..., Awaitable[str]]
TokenFn = Callable[[str], None]
StepFn = Callable[[TraceStep], None]
EventFn = Callable[[str, dict[str, Any]], None]


class ActivityEmitter:
    """Surfaces transient agent activity (a stage starting, tokens streaming in)
    over a single ``on_event(kind, payload)`` callback. Distinct from completed
    trace steps, which are stored on the run record."""

    def __init__(self, on_event: EventFn | None):
        self._on_event = on_event

    def start(
        self,
        stage: str,
        title: str,
        model: str | None,
        lane: int | None = None,
        order: int | None = None,
    ) -> str:
        step_id = new_id("step")
        if self._on_event is not None:
            payload = {"id": step_id, "stage": stage, "title": title, "model": model}
            # Saved layout (from a custom graph) rides along so the renderer can
            # place the node deterministically instead of guessing.
            if lane is not None:
                payload["lane"] = lane
            if order is not None:
                payload["order"] = order
            self._on_event("stage_start", payload)
        return step_id

    def token_sink(self, step_id: str) -> TokenFn | None:
        if self._on_event is None:
            return None

        def sink(text: str) -> None:
            self._on_event("token", {"id": step_id, "text": text})

        return sink


def extract_json(text: str) -> Any | None:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fence.group(1) if fence else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


def resolve_model_route(profile: Profile, model: str) -> tuple[str, str | None, str]:
    """Map a model reference to (provider, base_url, real model id).

    If the reference matches a registered cloud-model alias, the call routes to
    that provider (base_url empty -> the provider's own default/env resolution);
    otherwise it goes to the profile's default provider. Cloud models run on
    someone else's hardware, so nothing local-size-related ever applies to them."""
    for cloud in profile.cloud_models:
        if cloud.alias == model:
            return cloud.provider, (cloud.base_url or None), (cloud.model or cloud.alias)
    return profile.provider, profile.base_url, model


async def provider_complete(
    model, messages, profile: Profile, json_mode=False, on_token: TokenFn | None = None, **kwargs
) -> str:
    provider, base_url, model_id = resolve_model_route(profile, model)
    response_format = {"type": "json_object"} if json_mode else None
    if on_token is not None and not json_mode:
        try:
            parts: list[str] = []
            async for delta in stream_chat_completion_async(
                model=model_id,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.2),
                provider=provider,
                base_url=base_url,
            ):
                if delta:
                    parts.append(delta)
                    on_token(delta)
            return "".join(parts)
        except Exception:
            # Streaming not supported / failed mid-flight: fall back to a single
            # blocking completion so the run still produces an answer.
            pass
    try:
        response = await generate_chat_completion_async(
            model=model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.2),
            provider=provider,
            base_url=base_url,
            response_format=response_format,
        )
    except TypeError:
        response = await generate_chat_completion_async(
            model=model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.2),
            provider=provider,
            base_url=base_url,
        )
    except Exception:
        if not json_mode:
            raise
        response = await generate_chat_completion_async(
            model=model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.2),
            provider=provider,
            base_url=base_url,
        )
    return get_completion_text(response)


def _select_model(profile: Profile, primary: str | None = None) -> str:
    """Pick the first configured model, preferring ``primary`` then the
    aggregator and workers. Raises a clear error instead of an IndexError when
    a saved profile has no models at all."""
    for candidate in (primary, profile.aggregator_model, *profile.worker_models, profile.evaluator_model):
        if candidate:
            return candidate
    raise ValueError("Profile has no models configured. Set at least one worker or aggregator model.")


def _trace(
    stage: str,
    title: str,
    model: str | None,
    input_text: str,
    output: str,
    step_id: str | None = None,
    started_at: str | None = None,
    **metadata,
):
    fields = {"id": step_id} if step_id else {}
    # started_at is captured before the model call so ended_at (defaulted at
    # construction, i.e. now) yields a real per-stage duration.
    if started_at is not None:
        fields["started_at"] = started_at
    return TraceStep(
        stage=stage,
        title=title,
        model=model,
        input=input_text,
        output=output,
        metadata=metadata,
        **fields,
    )


def _context_block(request: RunRequest) -> str:
    if not request.context_files:
        return ""
    contexts = read_context_files(request.context_files, request.profile.allowed_roots)
    parts = []
    for context in contexts:
        suffix = " (truncated)" if context["truncated"] else ""
        parts.append(f"### {context['path']}{suffix}\n{context['content']}")
    return "\n\nRelevant local files:\n" + "\n\n".join(parts)


def _fallback_subtasks(prompt: str) -> list[Subtask]:
    return [
        Subtask(title="Understand", prompt=f"Clarify the goal and constraints for: {prompt}"),
        Subtask(title="Solve", prompt=f"Produce a concrete solution for: {prompt}"),
        Subtask(title="Risks", prompt=f"Identify risks, gaps, and verification steps for: {prompt}"),
    ]


async def make_subtasks(
    request: RunRequest, complete_fn: CompleteFn, activity: ActivityEmitter | None = None
):
    profile = request.profile
    model = _select_model(profile)
    step_id = activity.start("orchestrator", "Break into subtasks", model) if activity else None
    started = now_iso()
    prompt = (
        "Break the user task into 2 to 4 independent subtasks for parallel LLM workers. "
        "Return only JSON in this shape: {\"subtasks\":[{\"title\":\"...\",\"prompt\":\"...\"}]}.\n\n"
        f"User task:\n{request.prompt}{_context_block(request)}"
    )
    output = await complete_fn(
        model,
        [{"role": "user", "content": prompt}],
        profile=profile,
        json_mode=True,
        max_tokens=900,
        temperature=0.1,
    )
    parsed = extract_json(output)
    subtasks = []
    if isinstance(parsed, dict):
        for item in parsed.get("subtasks", [])[:4]:
            if isinstance(item, dict) and item.get("title") and item.get("prompt"):
                subtasks.append(Subtask(title=str(item["title"]), prompt=str(item["prompt"])))
    if len(subtasks) < 1:
        subtasks = _fallback_subtasks(request.prompt)
    return subtasks[:4], _trace(
        "orchestrator", "Break into subtasks", model, prompt, output, step_id=step_id, started_at=started
    )


async def run_worker(
    profile: Profile,
    subtask: Subtask,
    model: str,
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
):
    step_id = activity.start("worker", subtask.title, model) if activity else None
    started = now_iso()
    prompt = (
        "Worker subtask: solve the assigned subtask independently. "
        "Be specific, concise, and include assumptions.\n\n"
        f"Subtask title: {subtask.title}\nSubtask prompt: {subtask.prompt}"
    )
    output = await complete_fn(
        model,
        [{"role": "user", "content": prompt}],
        profile=profile,
        max_tokens=1200,
        temperature=0.4,
        on_token=activity.token_sink(step_id) if activity and step_id else None,
    )
    return output, _trace("worker", subtask.title, model, prompt, output, step_id=step_id, started_at=started)


async def synthesize(
    request: RunRequest,
    worker_outputs: list[str],
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
):
    profile = request.profile
    model = _select_model(profile)
    step_id = activity.start("synthesizer", "Synthesize worker outputs", model) if activity else None
    started = now_iso()
    prompt = (
        "Synthesize the worker outputs into one final draft for the user. "
        "Resolve conflicts, remove duplication, and be actionable. If file edits are needed, "
        "include a fenced JSON block with {\"file_changes\":[{\"path\":\"...\",\"proposed_content\":\"...\"}]}.\n\n"
        f"User task:\n{request.prompt}\n\nWorker outputs:\n"
        + "\n\n".join(f"{index + 1}. {output}" for index, output in enumerate(worker_outputs))
    )
    output = await complete_fn(
        model,
        [{"role": "user", "content": prompt}],
        profile=profile,
        max_tokens=1800,
        temperature=0.25,
        on_token=activity.token_sink(step_id) if activity and step_id else None,
    )
    return output, _trace(
        "synthesizer", "Synthesize worker outputs", model, prompt, output, step_id=step_id, started_at=started
    )


async def evaluate(
    request: RunRequest, draft: str, complete_fn: CompleteFn, activity: ActivityEmitter | None = None
):
    profile = request.profile
    model = _select_model(profile, profile.evaluator_model)
    step_id = activity.start("evaluator", "Evaluate draft", model) if activity else None
    started = now_iso()
    prompt = (
        "Evaluate the draft against the user task. Return only JSON with "
        "{\"status\":\"PASS\"|\"REVISE\",\"feedback\":\"...\",\"score\":0.0}.\n\n"
        f"User task:\n{request.prompt}\n\nDraft:\n{draft}"
    )
    output = await complete_fn(
        model,
        [{"role": "user", "content": prompt}],
        profile=profile,
        json_mode=True,
        max_tokens=600,
        temperature=0,
    )
    parsed = extract_json(output) or {}
    status = str(parsed.get("status", "REVISE")).upper()
    score = float(parsed.get("score", 0) or 0)
    evaluation = Evaluation(
        status="PASS" if status == "PASS" and score >= 0.6 else "REVISE",
        feedback=str(parsed.get("feedback", output)),
        score=score,
    )
    return evaluation, _trace(
        "evaluator", "Evaluate draft", model, prompt, output, step_id=step_id, started_at=started
    )


async def refine(
    request: RunRequest,
    draft: str,
    evaluation: Evaluation,
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
):
    profile = request.profile
    model = _select_model(profile)
    step_id = activity.start("refiner", "Revise draft", model) if activity else None
    started = now_iso()
    prompt = (
        "Revise the draft using the evaluator feedback. Return only the improved answer.\n\n"
        f"User task:\n{request.prompt}\n\nEvaluator feedback:\n{evaluation.feedback}\n\nDraft:\n{draft}"
    )
    output = await complete_fn(
        model,
        [{"role": "user", "content": prompt}],
        profile=profile,
        max_tokens=1800,
        temperature=0.2,
        on_token=activity.token_sink(step_id) if activity and step_id else None,
    )
    return output, _trace("refiner", "Revise draft", model, prompt, output, step_id=step_id, started_at=started)


_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}|\{(input|item)\}")


def _label_matches(label: str, value: str) -> str | bool:
    """Whole-word (case-insensitive) match, so a branch label 'code' matches
    'code' but not 'decode'/'barcode'. Falls back to substring when the label
    isn't word-boundable (e.g. contains punctuation)."""
    label = label.strip().lower()
    if not label:
        return True
    text = value.lower()
    if re.fullmatch(r"[\w ]+", label):
        return re.search(r"\b" + re.escape(label) + r"\b", text) is not None
    return label in text


def _render_template(template: str, input_text: str, outputs: dict[str, str], item: str | None = None) -> str:
    if not template:
        return input_text

    def _sub(match: "re.Match") -> str:
        node_id = match.group(1)
        simple = match.group(2)
        if node_id is not None:
            # Unknown node -> leave the placeholder literal rather than blanking.
            return outputs.get(node_id.strip(), match.group(0))
        if simple == "input":
            return input_text
        if simple == "item":
            return item if item is not None else match.group(0)
        return match.group(0)

    # Single pass: substituted values are NOT re-scanned, so a node whose output
    # happens to contain "{{other}}" can't inject another node's output.
    return _PLACEHOLDER_RE.sub(_sub, template)


def _fanout_items(text: str, limit: int = 8) -> list[str]:
    """Turn a node's output into a list to fan over: a JSON array, the first list
    inside a JSON object, or non-empty lines. Capped so a bad upstream output
    can't spawn an unbounded number of parallel calls."""
    items: Any = None
    parsed_a_list = False
    # A bare JSON array (extract_json only finds {...} objects, so handle [...] here).
    try:
        direct = json.loads(text.strip())
        if isinstance(direct, list):
            items, parsed_a_list = direct, True
        elif isinstance(direct, dict):
            found = next((v for v in direct.values() if isinstance(v, list)), None)
            if found is not None:
                items, parsed_a_list = found, True
    except (json.JSONDecodeError, ValueError):
        pass
    if items is None:
        parsed = extract_json(text)
        if isinstance(parsed, list):
            items, parsed_a_list = parsed, True
        elif isinstance(parsed, dict):
            found = next((v for v in parsed.values() if isinstance(v, list)), None)
            if found is not None:
                items, parsed_a_list = found, True
    if items is None:
        items = [line for line in text.splitlines() if line.strip()]
    result: list[str] = []
    for item in items:
        if isinstance(item, dict):
            result.append(str(item.get("prompt") or item.get("title") or json.dumps(item)))
        else:
            result.append(str(item))
    result = result[:limit]
    # A parsed-but-empty list means "no items" (0 instances); only fall back to
    # the whole blob when we couldn't find a list at all and there's some text.
    if not result and not parsed_a_list and text.strip():
        return [text]
    return result


def _build_gate_prompt(node, outputs: dict[str, str]) -> str:
    terms = ", ".join(check.term for check in node.checks) or "quality"
    content = "\n\n".join(outputs.get(dep, "") for dep in node.depends_on) or outputs.get("", "")
    instruction = node.prompt.strip() + "\n\n" if node.prompt.strip() else ""
    schema = ", ".join(f'"{check.term}": {{"score": 0.0, "note": "..."}}' for check in node.checks)
    return (
        f"{instruction}Evaluate the content below on each criterion, scoring 0.0-1.0 with a short note. "
        f"Criteria: {terms}. Return ONLY JSON: {{{schema}}}.\n\nContent:\n{content}"
    )


def _parse_gate(raw: str, checks) -> tuple[bool, str, str]:
    """Return (passed, feedback, summary). A term passes when its score >= min;
    the gate passes when every term passes. Missing/garbled scores count as 0."""
    data = extract_json(raw)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}
    if isinstance(data, dict):
        for term, value in data.items():
            if isinstance(value, dict):
                try:
                    scores[term] = float(value.get("score", 0) or 0)
                except (TypeError, ValueError):
                    scores[term] = 0.0
                notes[term] = str(value.get("note", ""))
            else:
                try:
                    scores[term] = float(value)
                except (TypeError, ValueError):
                    scores[term] = 0.0
    passed = True
    failing = []
    for check in checks:
        score = scores.get(check.term, 0.0)
        if score < check.min:
            passed = False
            failing.append(f"{check.term} ({score:.2f} < {check.min:.2f}): {notes.get(check.term, '')}".strip())
    summary_parts = [f"{c.term}={scores.get(c.term, 0.0):.2f}" for c in checks]
    summary = ("PASS " if passed else "REVISE ") + ", ".join(summary_parts)
    if failing:
        summary += "\nNeeds work: " + "; ".join(failing)
    return passed, "; ".join(failing), summary


async def run_graph_workflow(
    request: RunRequest,
    complete_fn: CompleteFn,
    emit,
    activity: ActivityEmitter | None,
) -> str:
    """Execute a custom flow graph (llm / fanout / gate nodes) and return the
    output node's text. Gate nodes score the draft against their checks and loop
    back to `loop_to` (re-running the loop body) until every check passes or
    `max_loops` is hit. Reuses the same trace/streaming machinery (via ``emit``)
    and emits each node's saved lane/order so the pop-out renders the authored
    layout; loop iterations stack by incrementing `order`."""
    from .graph import execution_order, loop_body, validate_graph

    profile = request.profile
    graph = profile.graph
    if graph is None:
        raise ValueError("Profile.workflow is 'graph' but no graph is defined.")
    problems = validate_graph(graph)
    if problems:
        raise ValueError("Invalid graph: " + "; ".join(problems))

    input_text = request.prompt + _context_block(request)
    outputs: dict[str, str] = {}
    skipped: set[str] = set()

    def is_skipped(node) -> bool:
        """A node is skipped when its routing condition doesn't hold, or every
        dependency it needs was itself skipped (so an untaken branch's whole tail
        drops out, while a join that had at least one live input still runs)."""
        if node.when_node:
            if node.when_node in skipped:
                return True
            value = outputs.get(node.when_node, "")
            if node.when_equals and not _label_matches(node.when_equals, value):
                return True
        deps = list(node.depends_on)
        if node.kind == "fanout" and node.over:
            deps.append(node.over)
        if deps and all(dep in skipped for dep in deps):
            return True
        return False

    async def run_node(node, iteration: int) -> None:
        model = node.model or _select_model(profile)
        base_title = node.title or node.id
        title = base_title if iteration == 0 else f"{base_title} ({iteration + 1})"
        order = node.order + iteration
        if node.kind == "fanout":
            items = _fanout_items(outputs.get(node.over, ""))

            async def run_item(index: int, item: str):
                step_id = activity.start(node.id, f"{base_title} {index + 1}", model, node.lane, order + index) if activity else None
                started = now_iso()
                prompt = _render_template(node.prompt, input_text, outputs, item=item)
                out = await complete_fn(
                    model, [{"role": "user", "content": prompt}], profile=profile,
                    max_tokens=1200, temperature=0.4,
                    on_token=activity.token_sink(step_id) if activity and step_id else None,
                )
                return out, _trace(node.id, f"{base_title} {index + 1}", model, prompt, out, step_id=step_id, started_at=started)

            tasks = [asyncio.ensure_future(run_item(i, it)) for i, it in enumerate(items)]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            parts = []
            for out, step in results:
                parts.append(out)
                emit(step)
            outputs[node.id] = "\n\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))
        else:  # llm
            step_id = activity.start(node.id, title, model, node.lane, order) if activity else None
            started = now_iso()
            prompt = _render_template(node.prompt, input_text, outputs)
            output = await complete_fn(
                model, [{"role": "user", "content": prompt}], profile=profile,
                max_tokens=1400, temperature=0.3,
                on_token=activity.token_sink(step_id) if activity and step_id else None,
            )
            emit(_trace(node.id, title, model, prompt, output, step_id=step_id, started_at=started))
            outputs[node.id] = output

    async def run_gate(node, iteration: int) -> bool:
        model = node.model or _select_model(profile, profile.evaluator_model)
        base_title = node.title or node.id
        title = base_title if iteration == 0 else f"{base_title} ({iteration + 1})"
        step_id = activity.start(node.id, title, model, node.lane, node.order + iteration) if activity else None
        started = now_iso()
        prompt = _build_gate_prompt(node, outputs)
        raw = await complete_fn(
            model, [{"role": "user", "content": prompt}], profile=profile,
            json_mode=True, max_tokens=500, temperature=0,
        )
        passed, _feedback, summary = _parse_gate(raw, node.checks)
        outputs[node.id] = summary  # available to a refine node via {{gate_id}}
        emit(_trace(node.id, title, model, prompt, summary, step_id=step_id, started_at=started, passed=passed))
        return passed

    for node in execution_order(graph.nodes):
        if is_skipped(node):
            skipped.add(node.id)
            continue
        if node.kind == "gate":
            passed = await run_gate(node, 0)
            iteration = 0
            while not passed and iteration < node.max_loops:
                iteration += 1
                for body_node in loop_body(graph.nodes, node):
                    # Never resurrect a node an untaken branch already skipped.
                    # (Validation guarantees no nested gate in the loop body.)
                    if body_node.id in skipped:
                        continue
                    await run_node(body_node, iteration)
                passed = await run_gate(node, iteration)
            # On exhaustion, proceed best-effort with the latest draft.
        else:
            await run_node(node, 0)

    return outputs.get(graph.output, "")


def extract_file_changes(record: RunRecord, allowed_roots: list[str]) -> list[FileChange]:
    parsed = extract_json(record.final_output)
    if not isinstance(parsed, dict):
        return []
    changes = []
    for item in parsed.get("file_changes", []):
        if not isinstance(item, dict) or not item.get("path") or "proposed_content" not in item:
            continue
        try:
            changes.append(
                create_file_change(
                    run_id=record.id,
                    path=str(item["path"]),
                    proposed_content=str(item["proposed_content"]),
                    allowed_roots=allowed_roots,
                )
            )
        except Exception as exc:
            # Never let one bad proposal abort the run, but don't let it vanish
            # silently either — record why it was dropped.
            logger.warning(f"Dropped proposed file change for {item.get('path')!r}: {exc}")
            continue
    return changes


async def run_workflow(
    request: RunRequest,
    complete_fn: CompleteFn | None = None,
    on_step: StepFn | None = None,
    run_id: str | None = None,
    on_event: EventFn | None = None,
) -> RunRecord:
    complete = complete_fn or provider_complete
    profile = request.profile
    trace: list[TraceStep] = []
    activity = ActivityEmitter(on_event) if on_event is not None else None

    def emit(step: TraceStep) -> TraceStep:
        """Append a completed step to the trace and notify any live observer.

        ``on_step`` lets callers (e.g. the streaming endpoint) surface agent
        activity as it happens; when it is None this is a plain append, so the
        blocking workflow behaves exactly as before."""
        trace.append(step)
        if on_step is not None:
            on_step(step)
        return step

    if profile.workflow == "graph":
        draft = await run_graph_workflow(request, complete, emit, activity)
    elif profile.workflow == "iterative_evaluator":
        draft, synth_step = await synthesize(
            request, [request.prompt + _context_block(request)], complete, activity
        )
        emit(synth_step)
    else:
        subtasks, orchestrator_step = await make_subtasks(request, complete, activity)
        emit(orchestrator_step)
        worker_models = [model for model in profile.worker_models if model] or [_select_model(profile)]
        worker_tasks = [
            asyncio.ensure_future(
                run_worker(profile, subtask, worker_models[index % len(worker_models)], complete, activity)
            )
            for index, subtask in enumerate(subtasks)
        ]
        try:
            worker_results = await asyncio.gather(*worker_tasks)
        except BaseException:
            # One worker failed (or the run was cancelled): stop the siblings
            # instead of leaving them making LLM calls for an already-doomed run,
            # and retrieve their results so no exception goes unobserved.
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            raise
        worker_outputs = []
        for output, step in worker_results:
            worker_outputs.append(output)
            emit(step)
        draft, synth_step = await synthesize(request, worker_outputs, complete, activity)
        emit(synth_step)

    evaluation = Evaluation(status="PASS", feedback="Evaluation skipped.", score=1)
    if profile.workflow in {"hybrid", "iterative_evaluator"}:
        for iteration in range(max(profile.max_iterations, 1)):
            evaluation, eval_step = await evaluate(request, draft, complete, activity)
            emit(eval_step)
            if evaluation.status == "PASS" or iteration == profile.max_iterations - 1:
                break
            draft, refine_step = await refine(request, draft, evaluation, complete, activity)
            emit(refine_step)

    record = RunRecord(
        profile_name=profile.name,
        prompt=request.prompt,
        workflow=profile.workflow,
        trace=trace,
        final_output=draft,
        evaluation=evaluation,
        context_files=request.context_files,
    )
    if run_id is not None:
        record.id = run_id
    record.file_changes = extract_file_changes(record, profile.allowed_roots)
    return record
