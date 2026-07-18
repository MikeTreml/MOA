export type Workflow = "hybrid" | "parallel_subtask" | "iterative_evaluator" | "graph" | "bounded_review";

export interface GraphCheck {
  term: string;
  min: number;
}

export interface GraphNode {
  id: string;
  title?: string;
  kind?: "llm" | "fanout" | "gate";
  model?: string;
  prompt?: string;
  depends_on?: string[];
  over?: string;
  checks?: GraphCheck[];
  loop_to?: string;
  max_loops?: number;
  when_node?: string;
  when_equals?: string;
  lane?: number;
  order?: number;
}

export interface FlowGraph {
  nodes: GraphNode[];
  output: string;
  renderer?: string;
}

export interface ReviewCategoryDefinition {
  id: string;
  title: string;
  mission: string;
  scopes: string[];
  exclusions: string[];
  skill_count: number;
}

export interface ReviewSkillSummary {
  id: string;
  category: string;
  section: string;
  question: string;
}

/** GET /api/review-agents — the server-side atomic-skill catalog. Skills are
 * trimmed summaries (no source_items); one bounded agent runs per enabled skill. */
export interface ReviewCatalog {
  version: number;
  source: string;
  source_item_count: number;
  atomic_skill_count: number;
  merged_duplicate_count: number;
  deduplication_rule: string;
  findings_per_skill: number;
  verifier_batch_size: number;
  categories: ReviewCategoryDefinition[];
  skills: ReviewSkillSummary[];
}

/** Policy trims the catalog only — no finding caps: enabled category ids plus
 * per-skill opt-outs. Always spread the existing object when patching so
 * excluded_skill_ids round-trips. */
export interface ReviewPolicy {
  findings_per_skill: number;
  categories: string[];
  excluded_skill_ids: string[];
}

export interface CloudModel {
  alias: string;
  provider: string;
  model: string;
  base_url: string;
}

export interface Profile {
  name: string;
  provider: string;
  base_url: string;
  memory_cap_gb: number;
  workflow: Workflow;
  worker_models: string[];
  aggregator_model: string;
  evaluator_model: string;
  image_model: string;
  max_iterations: number;
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
  allowed_roots: string[];
  graph?: FlowGraph | null;
  cloud_models: CloudModel[];
  review_policy: ReviewPolicy;
}

export interface GeneratedImage {
  b64_json: string | null;
  url: string | null;
}

/** One entry from the active provider's /v1/models (LM Studio, llama.cpp,
 * OpenAI, any compatible server), enriched with whatever the server exposes.
 * Snake-case aliases kept for older stored payloads. */
export interface ProviderModel {
  type?: string;
  modelKey?: string;
  model_key?: string;
  displayName?: string;
  display_name?: string;
  maxContextLength?: number;
  max_context_length?: number;
  /** Real file size when the server reports one; otherwise estimated from the
   * id's parameter count + quantization (sizeIsEstimate=true, shown with ~). */
  sizeBytes?: number;
  size_bytes?: number;
  sizeIsEstimate?: boolean;
  quantization?: string;
  /** LM Studio load state: "loaded" | "not-loaded". */
  state?: string;
}

export interface ProviderPreset {
  id: string;
  name: string;
  provider: string;
  base_url: string;
  worker_models: string[];
  aggregator_model: string;
  evaluator_model: string;
  env_keys: string[];
  notes: string[];
}

export interface TraceStep {
  id: string;
  stage: string;
  title: string;
  model?: string;
  output: string;
  metadata: Record<string, unknown>;
  started_at?: string;
  ended_at?: string;
}

export interface Evaluation {
  status: "PASS" | "REVISE";
  feedback: string;
  score: number;
}

export interface FileChange {
  id: string;
  run_id: string;
  path: string;
  diff: string;
  status: "proposed" | "applied" | "rejected";
}

export interface RunRecord {
  id: string;
  created_at: string;
  profile_name: string;
  prompt: string;
  workflow: string;
  trace: TraceStep[];
  final_output: string;
  evaluation: Evaluation;
  file_changes: FileChange[];
  context_files: string[];
}

export type RunStatus = "idle" | "running" | "complete" | "error" | "stopped";

export interface ActivityStage {
  id: string;
  stage: string;
  title: string;
  model?: string;
  status: "running" | "complete";
  tokens: string;
  output?: string;
  startedAt?: string;
  endedAt?: string;
  lane?: number;
  order?: number;
  /** Step ids of the agents that feed this one — drawn as flow edges. */
  deps?: string[];
}

export interface ProviderStatus {
  running: boolean;
  status: string;
  base_url: string;
  http_ok: boolean;
  models_count: number;
  /** llama.cpp parallel generation slots (from /props); null when unknown. */
  total_slots?: number | null;
}
