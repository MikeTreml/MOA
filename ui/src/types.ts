export type Workflow = "hybrid" | "parallel_subtask" | "iterative_evaluator" | "graph";

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
  allowed_roots: string[];
  graph?: FlowGraph | null;
}

export interface GeneratedImage {
  b64_json: string | null;
  url: string | null;
}

export interface LmModel {
  type?: string;
  modelKey?: string;
  model_key?: string;
  displayName?: string;
  display_name?: string;
  sizeBytes?: number;
  size_bytes?: number;
  paramsString?: string;
  params_string?: string;
  architecture?: string;
  trainedForToolUse?: boolean;
  trained_for_tool_use?: boolean;
  vision?: boolean;
  maxContextLength?: number;
  max_context_length?: number;
}

export interface ModelPlan {
  memory_cap_gb: number;
  worker_models: string[];
  aggregator_model: string;
  evaluator_model: string;
  total_size_bytes: number;
  selected_models: LmModel[];
  notes: string[];
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
}

export interface LmStudioStatus {
  running: boolean;
  status: string;
  base_url: string;
  http_ok: boolean;
  models_count: number;
  loaded_models: Array<Record<string, unknown>>;
}
