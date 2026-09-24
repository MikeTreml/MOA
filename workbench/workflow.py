from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
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

# Hard ceiling on parallel worker instances, matching the fanout cap: the roster
# drives the fan-out, but a runaway roster can't spawn an unbounded swarm.
MAX_WORKERS = 8


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
        deps: list[str] | None = None,
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
            # Upstream step ids: which agents feed this one, so the pop-out can
            # draw the actual flow edges instead of generic column connectors.
            if deps:
                payload["deps"] = list(deps)
            self._on_event("stage_start", payload)
        return step_id

    def token_sink(self, step_id: str) -> TokenFn | None:
        if self._on_event is None:
            return None

        def sink(text: str) -> None:
            self._on_event("token", {"id": step_id, "text": text})

        return sink


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Drop the chain-of-thought a reasoning model emits before its answer.

    Two shapes are handled: a LEADING <think>…</think> block, and the template
    shape where the opening tag lives in the prompt (DeepSeek-R1 style) so the
    completion is "…reasoning…</think>answer". Think-tags later in the body are
    left alone — there they're content (e.g. code that mentions the tags), not
    reasoning, and stripping them would corrupt legitimate output. An unclosed
    leading <think> keeps its text: a truncated all-thinking output still beats
    an empty answer."""
    lower = text.lower()
    open_at = lower.find(_THINK_OPEN)
    close_at = lower.find(_THINK_CLOSE)
    if text.lstrip().lower().startswith(_THINK_OPEN):
        if close_at != -1:
            return text[close_at + len(_THINK_CLOSE) :].strip()
        return text[open_at + len(_THINK_OPEN) :].strip()
    if close_at != -1 and (open_at == -1 or open_at > close_at):
        # A close tag with no opener before it: the opener was in the prompt
        # template, so everything up to the close is reasoning.
        return text[close_at + len(_THINK_CLOSE) :].strip()
    return text


class ThinkStreamFilter:
    """Streaming counterpart of :func:`strip_reasoning`: suppresses a LEADING
    <think>…</think> block so chain-of-thought never reaches the live feed,
    then passes everything through verbatim (a think-tag later in the body is
    content, not reasoning). Thinking text is buffered rather than dropped, so
    an unclosed block is emitted at flush() — matching strip_reasoning's
    keep-the-text behavior. The closing-only template shape cannot be filtered
    live (its text has already streamed before the close tag arrives); the
    stored output is still cleaned by strip_reasoning."""

    def __init__(self) -> None:
        self._buffer = ""
        self._thought = ""
        self._state = "start"  # start | thinking | passthrough

    def feed(self, delta: str) -> str:
        if self._state == "passthrough":
            return delta
        self._buffer += delta
        if self._state == "start":
            lead = self._buffer.lstrip()
            if not lead:
                return ""
            probe = lead[: len(_THINK_OPEN)].lower()
            if _THINK_OPEN.startswith(probe):
                if len(lead) < len(_THINK_OPEN):
                    return ""  # could still become the open tag; hold it back
                self._buffer = lead[len(_THINK_OPEN) :]
                self._state = "thinking"
            else:
                # Not a leading think block: everything is real output.
                self._state = "passthrough"
                out, self._buffer = self._buffer, ""
                return out
        # thinking: swallow until the close tag, holding back a partial-tag tail.
        low = self._buffer.lower()
        index = low.find(_THINK_CLOSE)
        if index == -1:
            held = self._partial_close_suffix()
            keep_from = len(self._buffer) - held
            self._thought += self._buffer[:keep_from]
            self._buffer = self._buffer[keep_from:]
            return ""
        self._buffer = self._buffer[index + len(_THINK_CLOSE) :]
        self._state = "passthrough"
        self._thought = ""  # block closed: the reasoning is discarded
        out, self._buffer = self._buffer, ""
        return out

    def _partial_close_suffix(self) -> int:
        """Length of the longest buffer suffix that could still grow into the
        close tag."""
        low = self._buffer.lower()
        for size in range(min(len(low), len(_THINK_CLOSE) - 1), 0, -1):
            if _THINK_CLOSE.startswith(low[-size:]):
                return size
        return 0

    def flush(self) -> str:
        out = (self._thought + self._buffer) if self._state == "thinking" else self._buffer
        self._buffer = ""
        self._thought = ""
        return out


def looks_degenerate(text: str) -> bool:
    """Heuristic for instruction-echo loops (a small-local-model failure mode
    where the same sentence repeats until max_tokens): flag when one non-trivial
    segment both repeats many times and dominates the tail of the output.
    Thresholds are deliberately high — a legitimate refrain (a 4× chorus) must
    not trigger a retry that could replace a correct answer."""
    if len(text) < 400:
        return False
    tail = text[-2000:]
    segments = [s.strip() for s in re.split(r"[.!?\n,;]+", tail) if len(s.strip()) >= 20]
    if len(segments) < 8:
        return False
    top = Counter(segments).most_common(1)[0][1]
    return top >= 6 and top / len(segments) >= 0.5


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


async def _complete_once(
    model_id: str,
    messages,
    provider: str,
    base_url: str | None,
    json_mode: bool,
    on_token: TokenFn | None,
    max_tokens: int,
    temperature: float,
    frequency_penalty: float | None,
    presence_penalty: float | None,
) -> str:
    """One completion attempt: streaming when a token sink is present, with
    chain-of-thought filtered out of both the live feed and the returned text."""
    penalties: dict[str, float] = {}
    if frequency_penalty is not None:
        penalties["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        penalties["presence_penalty"] = presence_penalty
    if on_token is not None and not json_mode:
        try:
            think_filter = ThinkStreamFilter()
            parts: list[str] = []
            async for delta in stream_chat_completion_async(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=provider,
                base_url=base_url,
                **penalties,
            ):
                if delta:
                    parts.append(delta)
                    visible = think_filter.feed(delta)
                    if visible:
                        on_token(visible)
            tail = think_filter.flush()
            if tail:
                on_token(tail)
            return strip_reasoning("".join(parts))
        except Exception:
            # Streaming not supported / failed mid-flight: fall back to a single
            # blocking completion so the run still produces an answer.
            pass
    response_format = {"type": "json_object"} if json_mode else None
    try:
        response = await generate_chat_completion_async(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            provider=provider,
            base_url=base_url,
            response_format=response_format,
            **penalties,
        )
    except TypeError:
        response = await generate_chat_completion_async(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            provider=provider,
            base_url=base_url,
        )
    except Exception:
        # Optional extras (json_object response_format, sampling penalties) are
        # the usual rejection cause; retry bare once. Without extras there is
        # nothing to remove, so surface the real error.
        if response_format is None and not penalties:
            raise
        response = await generate_chat_completion_async(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            provider=provider,
            base_url=base_url,
        )
    return strip_reasoning(get_completion_text(response))


DEGENERATE_RETRY_NOTE = "\n\n[Repetition detected — retrying once with stronger anti-repetition sampling]\n\n"


async def provider_complete(
    model, messages, profile: Profile, json_mode=False, on_token: TokenFn | None = None, **kwargs
) -> str:
    provider, base_url, model_id = resolve_model_route(profile, model)
    max_tokens = kwargs.get("max_tokens", 1024)
    temperature = kwargs.get("temperature", 0.2)
    output = await _complete_once(
        model_id, messages, provider, base_url, json_mode, on_token,
        max_tokens, temperature,
        profile.frequency_penalty, profile.presence_penalty,
    )
    if json_mode or not looks_degenerate(output):
        return output
    # The model fell into an instruction-echo loop; one hotter retry with a
    # strong frequency penalty usually breaks it. The feed gets a marker so the
    # restart is visible instead of looking like more looping.
    if on_token is not None:
        on_token(DEGENERATE_RETRY_NOTE)
    retry = await _complete_once(
        model_id, messages, provider, base_url, json_mode, on_token,
        max_tokens, min(temperature + 0.35, 1.0),
        0.7, profile.presence_penalty,
    )
    if looks_degenerate(retry):
        # Both attempts looped; keep the shorter one (less noise downstream).
        return retry if len(retry) <= len(output) else output
    return retry


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


def _fit_subtasks(subtasks: list[Subtask], target: int) -> list[Subtask]:
    """Pad or trim the orchestrator's subtasks so exactly `target` workers run —
    the roster the user configured, not whatever count the model felt like
    returning. Padding cycles the parsed subtasks as independent alternate
    takes, which is real MoA ensembling rather than dead configuration."""
    if not subtasks or target < 1 or len(subtasks) == target:
        return subtasks[:target] if subtasks else subtasks
    if len(subtasks) > target:
        return subtasks[:target]
    fitted = list(subtasks)
    while len(fitted) < target:
        base = subtasks[(len(fitted) - len(subtasks)) % len(subtasks)]
        take = (len(fitted) // len(subtasks)) + 1
        fitted.append(
            Subtask(
                title=f"{base.title} (take {take})",
                prompt=base.prompt
                + "\n\nThis is an independent second opinion on the same subtask: "
                "approach it from a different angle than another worker might.",
            )
        )
    return fitted


async def make_subtasks(
    request: RunRequest, complete_fn: CompleteFn, activity: ActivityEmitter | None = None
):
    profile = request.profile
    model = _select_model(profile)
    # The worker roster IS the requested fan-out: N entries -> N workers (the
    # point of the ×N stepper). Empty roster falls back to 3; MAX_WORKERS keeps
    # a runaway roster from spawning a swarm.
    requested = len([m for m in profile.worker_models if m])
    target = min(requested, MAX_WORKERS) if requested else 3
    step_id = activity.start("orchestrator", "Break into subtasks", model) if activity else None
    started = now_iso()
    prompt = (
        f"Break the user task into exactly {target} independent subtask"
        f"{'s' if target != 1 else ''} for parallel LLM workers. "
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
        for item in parsed.get("subtasks", [])[:MAX_WORKERS]:
            if isinstance(item, dict) and item.get("title") and item.get("prompt"):
                subtasks.append(Subtask(title=str(item["title"]), prompt=str(item["prompt"])))
    if len(subtasks) < 1:
        subtasks = _fallback_subtasks(request.prompt)
    subtasks = _fit_subtasks(subtasks, target)
    return subtasks, _trace(
        "orchestrator", "Break into subtasks", model, prompt, output,
        step_id=step_id, started_at=started, requested_workers=requested, workers=len(subtasks),
    )


async def run_worker(
    profile: Profile,
    subtask: Subtask,
    model: str,
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
    index: int = 0,
    parent_ids: list[str] | None = None,
):
    # The instance number keeps N same-model workers tellable-apart everywhere
    # a title is shown (pipeline strip, pop-out nodes, trace cards).
    title = f"Worker {index + 1}: {subtask.title}"
    step_id = activity.start("worker", title, model, deps=parent_ids) if activity else None
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
    return output, _trace(
        "worker", title, model, prompt, output,
        step_id=step_id, started_at=started, worker_index=index + 1,
        **({"deps": list(parent_ids)} if parent_ids else {}),
    )


async def synthesize(
    request: RunRequest,
    worker_outputs: list[str],
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
    parent_ids: list[str] | None = None,
):
    profile = request.profile
    model = _select_model(profile)
    step_id = (
        activity.start("synthesizer", "Synthesize worker outputs", model, deps=parent_ids)
        if activity
        else None
    )
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
        "synthesizer", "Synthesize worker outputs", model, prompt, output,
        step_id=step_id, started_at=started,
        **({"deps": list(parent_ids)} if parent_ids else {}),
    )


async def evaluate(
    request: RunRequest,
    draft: str,
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
    iteration: int = 0,
    parent_ids: list[str] | None = None,
):
    profile = request.profile
    model = _select_model(profile, profile.evaluator_model)
    # Iteration suffix matches the graph engine's "(N)" convention so repeated
    # loop passes don't render as identical twins.
    title = "Evaluate draft" if iteration == 0 else f"Evaluate draft ({iteration + 1})"
    step_id = activity.start("evaluator", title, model, deps=parent_ids) if activity else None
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
        "evaluator", title, model, prompt, output, step_id=step_id, started_at=started,
        **({"deps": list(parent_ids)} if parent_ids else {}),
    )


async def refine(
    request: RunRequest,
    draft: str,
    evaluation: Evaluation,
    complete_fn: CompleteFn,
    activity: ActivityEmitter | None = None,
    iteration: int = 0,
    parent_ids: list[str] | None = None,
):
    profile = request.profile
    model = _select_model(profile)
    title = "Revise draft" if iteration == 0 else f"Revise draft ({iteration + 1})"
    step_id = activity.start("refiner", title, model, deps=parent_ids) if activity else None
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
    return output, _trace(
        "refiner", title, model, prompt, output, step_id=step_id, started_at=started,
        **({"deps": list(parent_ids)} if parent_ids else {}),
    )


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
    # Latest iteration's step ids per node, so downstream nodes can record which
    # concrete agent instances fed them (drawn as edges in the pop-out).
    steps_by_node: dict[str, list[str]] = {}

    def dep_step_ids(dep_nodes: list[str]) -> list[str]:
        ids: list[str] = []
        for dep in dep_nodes:
            ids.extend(steps_by_node.get(dep, []))
        return ids

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
            deps = dep_step_ids([node.over, *node.depends_on])

            async def run_item(index: int, item: str):
                step_id = activity.start(node.id, f"{base_title} {index + 1}", model, node.lane, order + index, deps=deps or None) if activity else None
                started = now_iso()
                prompt = _render_template(node.prompt, input_text, outputs, item=item)
                out = await complete_fn(
                    model, [{"role": "user", "content": prompt}], profile=profile,
                    max_tokens=1200, temperature=0.4,
                    on_token=activity.token_sink(step_id) if activity and step_id else None,
                )
                return out, _trace(
                    node.id, f"{base_title} {index + 1}", model, prompt, out,
                    step_id=step_id, started_at=started,
                    **({"deps": deps} if deps else {}),
                )

            tasks = [asyncio.ensure_future(run_item(i, it)) for i, it in enumerate(items)]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            parts = []
            instance_ids = []
            for out, step in results:
                parts.append(out)
                emit(step)
                instance_ids.append(step.id)
            steps_by_node[node.id] = instance_ids
            outputs[node.id] = "\n\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))
        else:  # llm
            deps = dep_step_ids(node.depends_on)
            step_id = activity.start(node.id, title, model, node.lane, order, deps=deps or None) if activity else None
            started = now_iso()
            prompt = _render_template(node.prompt, input_text, outputs)
            output = await complete_fn(
                model, [{"role": "user", "content": prompt}], profile=profile,
                max_tokens=1400, temperature=0.3,
                on_token=activity.token_sink(step_id) if activity and step_id else None,
            )
            step = _trace(
                node.id, title, model, prompt, output, step_id=step_id, started_at=started,
                **({"deps": deps} if deps else {}),
            )
            emit(step)
            steps_by_node[node.id] = [step.id]
            outputs[node.id] = output

    async def run_gate(node, iteration: int) -> bool:
        model = node.model or _select_model(profile, profile.evaluator_model)
        base_title = node.title or node.id
        title = base_title if iteration == 0 else f"{base_title} ({iteration + 1})"
        deps = dep_step_ids(node.depends_on)
        step_id = activity.start(node.id, title, model, node.lane, node.order + iteration, deps=deps or None) if activity else None
        started = now_iso()
        prompt = _build_gate_prompt(node, outputs)
        raw = await complete_fn(
            model, [{"role": "user", "content": prompt}], profile=profile,
            json_mode=True, max_tokens=500, temperature=0,
        )
        passed, _feedback, summary = _parse_gate(raw, node.checks)
        outputs[node.id] = summary  # available to a refine node via {{gate_id}}
        step = _trace(
            node.id, title, model, prompt, summary, step_id=step_id, started_at=started,
            passed=passed, **({"deps": deps} if deps else {}),
        )
        emit(step)
        steps_by_node[node.id] = [step.id]
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

    # The id of the step that produced the current draft — each stage's trace
    # step records its upstream ids so the pop-out can draw real flow edges.
    draft_step_id: str | None = None
    review_verified: int | None = None
    if profile.workflow == "bounded_review":
        from .review_agents import run_bounded_review
        draft, review_verified = await run_bounded_review(request, complete, emit, activity)
    elif profile.workflow == "graph":
        draft = await run_graph_workflow(request, complete, emit, activity)
    elif profile.workflow == "iterative_evaluator":
        draft, synth_step = await synthesize(
            request, [request.prompt + _context_block(request)], complete, activity
        )
        emit(synth_step)
        draft_step_id = synth_step.id
    else:
        subtasks, orchestrator_step = await make_subtasks(request, complete, activity)
        emit(orchestrator_step)
        worker_models = [model for model in profile.worker_models if model] or [_select_model(profile)]
        worker_tasks = [
            asyncio.ensure_future(
                run_worker(
                    profile, subtask, worker_models[index % len(worker_models)], complete, activity,
                    index=index, parent_ids=[orchestrator_step.id],
                )
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
        worker_step_ids = []
        for output, step in worker_results:
            worker_outputs.append(output)
            emit(step)
            worker_step_ids.append(step.id)
        draft, synth_step = await synthesize(
            request, worker_outputs, complete, activity, parent_ids=worker_step_ids
        )
        emit(synth_step)
        draft_step_id = synth_step.id

    if review_verified is not None:
        evaluation = Evaluation(status="PASS", feedback=f"Bounded review completed with {review_verified} verified finding(s).", score=1)
    else:
        evaluation = Evaluation(status="PASS", feedback="Evaluation skipped.", score=1)
    if profile.workflow in {"hybrid", "iterative_evaluator"}:
        for iteration in range(max(profile.max_iterations, 1)):
            evaluation, eval_step = await evaluate(
                request, draft, complete, activity, iteration=iteration,
                parent_ids=[draft_step_id] if draft_step_id else None,
            )
            emit(eval_step)
            if evaluation.status == "PASS" or iteration == profile.max_iterations - 1:
                break
            draft, refine_step = await refine(
                request, draft, evaluation, complete, activity, iteration=iteration,
                parent_ids=[eval_step.id],
            )
            emit(refine_step)
            draft_step_id = refine_step.id

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
    # A bounded review is read-only by definition, and its report embeds
    # agent-authored finding text verbatim — parsing that for file_changes
    # would let a prompt-injected fenced JSON block in reviewed source become
    # a real "proposed" file change. Reviews never propose edits.
    if profile.workflow != "bounded_review":
        record.file_changes = extract_file_changes(record, profile.allowed_roots)
    return record
