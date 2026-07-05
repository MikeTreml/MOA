from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from .schemas import RunRecord, TraceStep


# Upper bound on the events replay buffer per run. A pathological run streaming
# millions of token events can't grow this without bound; the oldest events are
# dropped for late-subscriber replay (they get an explicit notice), while the
# terminal complete/end events are always the last ones retained.
MAX_EVENTS = 100_000

# Upper bound on a single subscriber's pending queue. A viewer whose connection
# stalls (laptop asleep mid-stream) is dropped once this many events back up,
# rather than letting its queue grow until the process runs out of memory.
MAX_SUBSCRIBER_QUEUE = 20_000


class _Subscriber:
    """One SSE viewer's delivery queue plus an overflow flag."""

    __slots__ = ("queue", "overflowed")

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_SUBSCRIBER_QUEUE)
        self.overflowed = False


class RunSession:
    """Runtime state for one in-progress (or just-finished) run.

    A session buffers every event (stage starts, streamed tokens, completed
    steps, the terminal result) so late viewers can replay what they missed,
    and fans new events out to any number of subscriber queues. Because the run
    is driven by a background task and only *published* here, viewers are pure
    observers: subscribing, unsubscribing, or disconnecting can never affect the
    run itself.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: deque[tuple[str, Any]] = deque(maxlen=MAX_EVENTS)
        self.replay_truncated = False
        self.subscribers: list[_Subscriber] = []
        # Completed trace steps kept as objects so a stopped run can be
        # persisted with whatever work finished before cancellation.
        self.trace_steps: list[TraceStep] = []
        self.status: str = "running"  # running | complete | error | stopped
        self.record: dict[str, Any] | None = None
        self.error: dict[str, Any] | None = None
        self.task: asyncio.Task | None = None

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def _publish(self, kind: str, data: Any) -> None:
        if len(self.events) == self.events.maxlen:
            self.replay_truncated = True
        self.events.append((kind, data))
        dropped: list[_Subscriber] = []
        for subscriber in self.subscribers:
            try:
                subscriber.queue.put_nowait((kind, data))
            except asyncio.QueueFull:
                # Slow/stalled consumer: stop feeding it and let its stream close.
                subscriber.overflowed = True
                dropped.append(subscriber)
        for subscriber in dropped:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    def on_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Transient activity: ``stage_start`` and ``token``."""
        self._publish(kind, payload)

    def on_step(self, step: TraceStep) -> None:
        self.trace_steps.append(step)
        self._publish("step", step.model_dump(mode="json"))

    def finish(self, record: RunRecord) -> None:
        if self.finished:
            return
        self.status = "complete"
        self.record = record.model_dump(mode="json")
        self._publish("complete", self.record)
        self._publish("end", None)

    def fail(self, status_code: int, detail: str) -> None:
        if self.finished:
            return
        self.status = "error"
        self.error = {"status": status_code, "detail": detail}
        # Named "failed" (not "error") so the browser EventSource doesn't confuse
        # a server-sent failure with a transport-level connection error.
        self._publish("failed", self.error)
        self._publish("end", None)

    def stop(self) -> None:
        if self.finished:
            return
        self.status = "stopped"
        self._publish("stopped", {"detail": "Run stopped by user."})
        self._publish("end", None)

    def subscribe(self) -> asyncio.Queue:
        subscriber = _Subscriber()
        self.subscribers.append(subscriber)
        return subscriber.queue

    def attach(self) -> tuple[asyncio.Queue, list[tuple[str, Any]]]:
        """Register a subscriber and snapshot the replay buffer as one atomic
        step (no await between them, so under the single-threaded event loop an
        event can be neither missed nor duplicated across the handoff)."""
        subscriber = _Subscriber()
        self.subscribers.append(subscriber)
        return subscriber.queue, list(self.events)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        for subscriber in list(self.subscribers):
            if subscriber.queue is queue:
                self.subscribers.remove(subscriber)


class RunRegistry:
    """In-memory registry of run sessions, capped to recent runs.

    Sessions are runtime-only; completed runs are persisted to disk by the
    workflow, so trimming finished sessions here loses no durable data.
    """

    def __init__(self, max_sessions: int = 50):
        self.sessions: dict[str, RunSession] = {}
        self.order: deque[str] = deque()
        self.max_sessions = max_sessions

    def create(self, run_id: str) -> RunSession:
        session = RunSession(run_id)
        self.sessions[run_id] = session
        self.order.append(run_id)
        self._trim()
        return session

    def get(self, run_id: str) -> RunSession | None:
        return self.sessions.get(run_id)

    def _trim(self) -> None:
        excess = len(self.order) - self.max_sessions
        if excess <= 0:
            return
        # Evict oldest finished sessions only; never drop a run still in flight.
        removable = [
            run_id
            for run_id in self.order
            if (session := self.sessions.get(run_id)) is not None and session.status != "running"
        ][:excess]
        removable_set = set(removable)
        for run_id in removable:
            self.sessions.pop(run_id, None)
        if removable_set:
            self.order = deque(rid for rid in self.order if rid not in removable_set)
