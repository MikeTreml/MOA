from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def default_allowed_roots() -> list[str]:
    # parents[0] == workbench/, parents[1] == the repo root. Scope defaults to the
    # repo only — never its parent directory (which would expose sibling repos).
    return [str(Path(__file__).resolve().parents[1])]


class ModelInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: str = "llm"
    model_key: str = Field(alias="modelKey")
    display_name: str | None = Field(default=None, alias="displayName")
    size_bytes: int = Field(default=0, alias="sizeBytes")
    params_string: str | None = Field(default=None, alias="paramsString")
    architecture: str | None = None
    quantization: dict[str, Any] | None = None
    trained_for_tool_use: bool = Field(default=False, alias="trainedForToolUse")
    vision: bool = False
    max_context_length: int = Field(default=0, alias="maxContextLength")
    device_identifier: str | None = Field(default=None, alias="deviceIdentifier")


class ModelPlan(BaseModel):
    memory_cap_gb: float = 80
    worker_models: list[str] = Field(default_factory=list)
    aggregator_model: str = ""
    evaluator_model: str = ""
    total_size_bytes: int = 0
    selected_models: list[ModelInfo] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProviderPreset(BaseModel):
    id: str
    name: str
    provider: str
    base_url: str = ""
    worker_models: list[str] = Field(default_factory=list)
    aggregator_model: str = ""
    evaluator_model: str = ""
    env_keys: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    name: str = "Default LM Studio"
    provider: str = "lmstudio"
    base_url: str = "http://127.0.0.1:1234/v1"
    memory_cap_gb: float = 80
    workflow: Literal["hybrid", "parallel_subtask", "iterative_evaluator", "graph"] = "hybrid"
    worker_models: list[str] = Field(
        default_factory=lambda: ["llama-3.2-1b-instruct", "llama-3.2-1b-instruct"]
    )
    aggregator_model: str = "llama-3.2-1b-instruct"
    evaluator_model: str = "llama-3.2-1b-instruct"
    image_model: str = ""
    max_iterations: int = 2
    allowed_roots: list[str] = Field(default_factory=default_allowed_roots)
    graph: FlowGraph | None = None  # used when workflow == "graph"


class GraphCheck(BaseModel):
    """One gate criterion: the model scores `term` 0.0–1.0; the gate passes the
    term when the score is at least `min`. Terms are freeform (e.g. "scope",
    "direction", "best_practices", "factuality") — pick what fits the flow."""

    term: str
    min: float = 0.7


class GraphNode(BaseModel):
    """One step in a custom flow graph.

    `kind="llm"` runs a single completion; `kind="fanout"` runs one parallel
    instance per item of an upstream node's list output; `kind="gate"` scores
    the draft against `checks` and, if any fall short, loops back to `loop_to`
    (up to `max_loops` times) so an upstream refine node can revise. `lane`/`order`
    are the saved layout (column / row) — computed once at authoring time so the
    renderer just draws, never guesses. Prompt templates may reference `{input}`
    (the run prompt), `{{node_id}}` (an upstream node's output — including a
    gate's feedback), and `{item}` (the current item, fanout only)."""

    id: str
    title: str = ""
    kind: Literal["llm", "fanout", "gate"] = "llm"
    model: str = ""  # empty -> falls back to profile roles
    prompt: str = ""
    depends_on: list[str] = Field(default_factory=list)
    over: str = ""  # fanout: id of the node whose output is the list to fan over
    checks: list[GraphCheck] = Field(default_factory=list)  # gate
    loop_to: str = ""  # gate: node to re-run on failure (must be an ancestor)
    max_loops: int = 3  # gate: iteration counter cap
    # Conditional routing: run this node only if `when_node`'s output contains
    # `when_equals` (case-insensitive). Empty `when_equals` = run if `when_node`
    # itself ran. A node whose every dependency was skipped is skipped too.
    when_node: str = ""
    when_equals: str = ""
    lane: int = 0
    order: int = 0


class FlowGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    output: str = ""  # id of the node whose output is the flow's final answer
    renderer: str = "auto"  # escape hatch: a flow may name its own renderer


class Subtask(BaseModel):
    title: str
    prompt: str


class Evaluation(BaseModel):
    status: Literal["PASS", "REVISE"] = "REVISE"
    feedback: str = ""
    score: float = 0


class TraceStep(BaseModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    stage: str
    title: str
    model: str | None = None
    status: Literal["running", "complete", "error"] = "complete"
    input: str = ""
    output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: str = Field(default_factory=now_iso)
    ended_at: str = Field(default_factory=now_iso)


class FileChange(BaseModel):
    id: str = Field(default_factory=lambda: new_id("change"))
    run_id: str
    path: str
    original_content: str = ""
    proposed_content: str
    diff: str = ""
    status: Literal["proposed", "applied", "rejected"] = "proposed"
    allowed_roots: list[str] = Field(default_factory=default_allowed_roots)
    created_at: str = Field(default_factory=now_iso)
    applied_at: str | None = None


class RunRequest(BaseModel):
    prompt: str
    profile: Profile = Field(default_factory=Profile)
    context_files: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    created_at: str = Field(default_factory=now_iso)
    profile_name: str
    prompt: str
    workflow: str
    trace: list[TraceStep] = Field(default_factory=list)
    final_output: str = ""
    evaluation: Evaluation = Field(default_factory=Evaluation)
    file_changes: list[FileChange] = Field(default_factory=list)
    context_files: list[str] = Field(default_factory=list)
