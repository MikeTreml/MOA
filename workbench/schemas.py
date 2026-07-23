from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def default_allowed_roots() -> list[str]:
    # parents[0] == workbench/, parents[1] == the repo root. Scope defaults to the
    # repo only — never its parent directory (which would expose sibling repos).
    return [str(Path(__file__).resolve().parents[1])]


def default_base_url() -> str:
    # An OpenAI-compatible server (LM Studio by default — it can serve several
    # models at once, which a mixture needs); override with MOA_BASE_URL.
    import os

    return os.environ.get("MOA_BASE_URL", "http://127.0.0.1:1234/v1")


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


class CloudModel(BaseModel):
    """A cloud/API model registered under a local alias. Anywhere a model name
    is used (workers, aggregator, evaluator, graph nodes), the alias routes the
    call to this provider instead of the profile's default. API keys are never
    stored — they come from env (OPENAI_API_KEY, TOGETHER_API_KEY, MOA_API_KEY…).
    Cloud models have no local footprint, so nothing size-related applies."""

    alias: str
    provider: str = "openai"  # openai | together | omp | openai-compatible
    model: str = ""  # the provider's real model id (defaults to alias if empty)
    base_url: str = ""  # optional override; required for omp/openai-compatible


ReviewCategory = str


def default_review_categories() -> list[ReviewCategory]:
    """Every catalog domain is enabled by default; categories organize work only."""
    return [
        "performance_efficiency",
        "maintainability",
        "testing",
        "error_handling",
        "architecture",
        "dependencies",
        "concurrency",
        "edge_cases",
        "logic_correctness",
        "operations",
        "review_strategy",
        "data_refresh",
        "state_management",
        "ui_visual",
        "forms",
        "resource_management",
        "ui_interaction",
        "accessibility",
        "backend_data",
        "api_contract",
        "security",
        "observability",
    ]


class ReviewPolicy(BaseModel):
    """Exhaustive review policy: one atomic checklist skill per bounded agent."""

    findings_per_skill: Literal[1] = 1
    # Accepted only to migrate profiles saved by the earlier category-sampling
    # implementation. It is deliberately excluded and never limits coverage.
    findings_per_agent: Literal[1] = Field(default=1, exclude=True)
    max_findings_per_category: int | None = Field(default=None, ge=1, le=5, exclude=True)
    categories: list[ReviewCategory] = Field(default_factory=default_review_categories)
    excluded_skill_ids: list[str] = Field(default_factory=list)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: list[ReviewCategory]) -> list[ReviewCategory]:
        legacy = {
            "business_logic": ["logic_correctness", "edge_cases"],
            "test_coverage": ["testing"],
        }
        expanded = [replacement for item in value for replacement in legacy.get(item, [item])]
        return list(dict.fromkeys(expanded))

    @field_validator("excluded_skill_ids")
    @classmethod
    def validate_excluded_skill_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class ReviewSkillSource(BaseModel):
    ordinal: int = Field(ge=1)
    section: str
    text: str


class ReviewSkillDefinition(BaseModel):
    id: str
    category: ReviewCategory
    section: str
    question: str
    source_items: list[ReviewSkillSource]


class ReviewCategoryDefinition(BaseModel):
    id: ReviewCategory
    title: str
    mission: str
    scopes: list[str]
    exclusions: list[str]
    skill_count: int = Field(ge=1)


class ReviewCatalog(BaseModel):
    version: int = Field(ge=1)
    source: str
    source_item_count: int = Field(ge=1)
    atomic_skill_count: int = Field(ge=1)
    merged_duplicate_count: int = Field(ge=0)
    deduplication_rule: str
    findings_per_skill: Literal[1] = 1
    verifier_batch_size: int = Field(ge=1, le=100)
    categories: list[ReviewCategoryDefinition]
    skills: list[ReviewSkillDefinition]

class ReviewFinding(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    severity: Literal["critical", "high", "medium", "low"]
    file: str = Field(min_length=1, max_length=1000)
    location: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=8, max_length=4000)
    impact: str = Field(min_length=8, max_length=2000)
    verification: str = Field(min_length=8, max_length=2000)


class ReviewAgentResult(BaseModel):
    status: Literal["finding", "exhausted", "not_applicable", "blocked"]
    # Defaulted because the parser overwrites it on every path — an agent that
    # omits it must not be blocked on a technicality.
    skill_status: Literal["partial", "exhausted", "not_applicable", "blocked"] = "blocked"
    files_inspected: list[str] = Field(default_factory=list)
    finding: ReviewFinding | None = None
    blocked_reason: str = ""


class Profile(BaseModel):
    name: str = "LM Studio"
    provider: str = "openai-compatible"
    base_url: str = Field(default_factory=default_base_url)
    memory_cap_gb: float = 80
    workflow: Literal["hybrid", "parallel_subtask", "iterative_evaluator", "graph", "bounded_review"] = "hybrid"
    worker_models: list[str] = Field(default_factory=lambda: ["qwen-3b", "qwen-3b"])
    aggregator_model: str = "qwen-3b"
    evaluator_model: str = "qwen-3b"
    image_model: str = ""
    max_iterations: int = 2
    # Anti-repetition sampling forwarded to every completion (None = provider
    # default). Local models are prone to instruction-echo loops without these.
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    allowed_roots: list[str] = Field(default_factory=default_allowed_roots)
    graph: FlowGraph | None = None  # used when workflow == "graph"
    cloud_models: list[CloudModel] = Field(default_factory=list)
    review_policy: ReviewPolicy = Field(default_factory=ReviewPolicy)


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
