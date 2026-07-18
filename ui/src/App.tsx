import { useEffect, useMemo, useRef, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  Database,
  ExternalLink,
  FileText,
  FolderOpen,
  History,
  Image as ImageIcon,
  Layers3,
  Moon,
  Play,
  RefreshCw,
  Save,
  Server,
  Settings,
  ShieldCheck,
  Sparkles,
  Square,
  Sun,
  Trash2,
  UploadCloud,
  Users,
  XCircle
} from "lucide-react";
import { api, modelKey, modelName, modelSizeLabel } from "./api";
import { ActivityView } from "./ActivityView";
import { GraphBuilder } from "./GraphBuilder";
import { useRunActivity, type RunActivityState } from "./useRunActivity";
import type { ActivityStage, CloudModel, FileChange, GeneratedImage, Profile, ProviderModel, ProviderPreset, ProviderStatus, ReviewCatalog, RunRecord, Workflow } from "./types";

function recordToActivity(record: RunRecord | null): RunActivityState {
  if (!record) return { stages: [], record: null, status: "idle", error: null, reconnecting: false };
  return {
    stages: record.trace.map((step) => ({
      id: step.id,
      stage: step.stage,
      title: step.title,
      model: step.model,
      status: "complete" as const,
      tokens: "",
      output: step.output,
      startedAt: step.started_at,
      endedAt: step.ended_at,
      deps: Array.isArray(step.metadata?.deps)
        ? (step.metadata.deps as unknown[]).filter((d): d is string => typeof d === "string")
        : undefined
    })),
    record,
    status: "complete",
    error: null,
    reconnecting: false
  };
}

/** Parse a numeric input, keeping a fallback when the field is cleared (empty
 * string coerces to 0 otherwise) and enforcing a minimum. */
function numberOr(value: string, fallback: number, min: number): number {
  const parsed = Number(value);
  if (value.trim() === "" || Number.isNaN(parsed)) return fallback;
  return Math.max(min, parsed);
}

const tabs = [
  { id: "run", label: "Run", icon: Play },
  { id: "images", label: "Images", icon: ImageIcon },
  { id: "models", label: "Models", icon: Database },
  { id: "files", label: "Files", icon: FolderOpen },
  { id: "profiles", label: "Profiles", icon: Settings },
  { id: "history", label: "History", icon: History }
] as const;

type TabId = (typeof tabs)[number]["id"];

const workflowLabels: Record<Workflow, string> = {
  hybrid: "Hybrid",
  parallel_subtask: "Subtask Parallel",
  iterative_evaluator: "Iterative Eval",
  graph: "Custom Graph",
  bounded_review: "Bounded Review"
};

// One honest sentence per shape — the Run tab subtitle must describe the
// workflow that will actually execute, not always the hybrid one.
const workflowDescriptions: Record<Workflow, string> = {
  hybrid: "Decomposes into one subtask per worker, runs workers in parallel, synthesizes, then evaluates and refines.",
  parallel_subtask: "Decomposes into one subtask per worker, runs workers in parallel, and synthesizes — no evaluation loop.",
  iterative_evaluator: "Synthesizes directly from the prompt, then evaluates and refines — no workers.",
  graph: "Runs your custom flow graph exactly as authored in the Profiles tab.",
  bounded_review: "Runs one bounded agent per atomic checklist skill, then verifies candidates in batches and reports coverage honestly."
};

const SEED_GRAPH = JSON.stringify(
  {
    nodes: [
      { id: "plan", kind: "llm", title: "Plan", prompt: "Break this task into 2-4 angles as a JSON list: {input}" },
      { id: "work", kind: "fanout", title: "Research", over: "plan", prompt: "Research this angle in depth: {item}" },
      { id: "final", kind: "llm", title: "Synthesize", depends_on: ["work"], prompt: "Synthesize into one brief for: {input}\n\n{{work}}" }
    ],
    output: "final"
  },
  null,
  2
);

// A refine loop: a gate scores the draft on settable terms and loops back to the
// refine node until each term clears its threshold (or max_loops is hit).
const SEED_LOOP_GRAPH = JSON.stringify(
  {
    nodes: [
      { id: "draft", kind: "llm", title: "Draft", prompt: "Draft an answer for: {input}" },
      { id: "refine", kind: "llm", title: "Revise", depends_on: ["draft"], prompt: "Revise using this feedback: {{gate}}\n\nPrevious draft: {{refine}}\n\nTask: {input}" },
      {
        id: "gate", kind: "gate", title: "Check", depends_on: ["refine"], loop_to: "refine", max_loops: 3,
        checks: [
          { term: "scope", min: 0.7 },
          { term: "direction", min: 0.7 },
          { term: "best_practices", min: 0.8 }
        ]
      }
    ],
    output: "refine"
  },
  null,
  2
);

// Conditional routing: a router emits a label; only the matching branch runs,
// and a final node merges whichever branch was taken.
const SEED_BRANCH_GRAPH = JSON.stringify(
  {
    nodes: [
      { id: "router", kind: "llm", title: "Route", prompt: "Classify this task as exactly one word — code OR prose: {input}" },
      { id: "code", kind: "llm", title: "Code path", depends_on: ["router"], when_node: "router", when_equals: "code", prompt: "Answer as code for: {input}" },
      { id: "prose", kind: "llm", title: "Prose path", depends_on: ["router"], when_node: "router", when_equals: "prose", prompt: "Answer as prose for: {input}" },
      { id: "final", kind: "llm", title: "Deliver", depends_on: ["code", "prose"], prompt: "Deliver the result:\n{{code}}{{prose}}" }
    ],
    output: "final"
  },
  null,
  2
);

function uniqueLines(value: string): string[] {
  return value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function textFromList(values: string[]): string {
  return values.join("\n");
}

function StatusPill({ status }: { status?: string }) {
  const ok = status === "PASS" || status === "applied" || status === "complete" || status === "loaded";
  return <span className={`pill ${ok ? "pillOk" : "pillWarn"}`}>{status ?? "pending"}</span>;
}

/** Group a worker multiset for display: ["m","m","x"] -> [["m",2],["x",1]]. */
function groupWorkers(models: string[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const model of models) {
    if (!model) continue;
    counts.set(model, (counts.get(model) ?? 0) + 1);
  }
  return [...counts.entries()];
}

/** Where a role's model actually runs: cloud alias -> its own provider route,
 * anything else -> the profile's provider/base_url. Mirrors the backend's
 * resolve_model_route so "same model everywhere" is detected accurately. */
function resolvedRouteKey(profile: Profile, model: string): string {
  const cloud = (profile.cloud_models ?? []).find((c) => c.alias === model);
  if (cloud) return `${cloud.provider}|${cloud.base_url}|${cloud.model || cloud.alias}`;
  return `${profile.provider}|${profile.base_url}|${model}`;
}

function providerDisplayName(provider: string): string {
  const normalized = provider.trim().toLowerCase();
  if (normalized === "omp") return "OMP";
  if (normalized === "openai-compatible") return "OpenAI-compatible";
  if (normalized === "openai") return "OpenAI";
  if (normalized === "together") return "Together";
  if (normalized === "atomic") return "Atomic";
  return provider || "Provider";
}

function App() {
  const [tab, setTab] = useState<TabId>("run");
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [presets, setPresets] = useState<ProviderPreset[]>([]);
  const [models, setModels] = useState<ProviderModel[]>([]);
  const [reviewCatalog, setReviewCatalog] = useState<ReviewCatalog | null>(null);
  const [status, setStatus] = useState<ProviderStatus | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [latestRun, setLatestRun] = useState<RunRecord | null>(null);
  const [prompt, setPrompt] = useState("Analyze this project and propose the next safest implementation step.");
  const [contextInput, setContextInput] = useState("");
  const [contextFiles, setContextFiles] = useState<string[]>([]);
  const [filePreview, setFilePreview] = useState("");
  const [browser, setBrowser] = useState<{ path: string | null; parent: string | null; entries: Array<{ name: string; path: string; is_dir: boolean }> } | null>(null);
  const [imagePrompt, setImagePrompt] = useState("A watercolor fox reading a book, soft light");
  const [imageSize, setImageSize] = useState("1024x1024");
  const [imageCount, setImageCount] = useState(1);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [imageBusy, setImageBusy] = useState(false);
  const [graphText, setGraphText] = useState("");
  const [graphErrors, setGraphErrors] = useState<string[]>([]);
  const [graphMode, setGraphMode] = useState<"visual" | "json">("visual");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [bootError, setBootError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [autoPopAgents, setAutoPopAgents] = useState<boolean>(
    () => localStorage.getItem("moa-auto-pop-agents") === "1"
  );
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem("moa-theme") as "dark" | "light") || "dark"
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("moa-theme", theme);
  }, [theme]);

  const activity = useRunActivity(activeRunId, (record) => {
    setLatestRun(record);
    api.runs().then((payload) => setRuns(payload.runs)).catch(() => undefined);
  });
  const running = activity.status === "running";

  async function refreshAll() {
    const [profilePayload, presetPayload, modelPayload, statusPayload, runPayload] = await Promise.all([
      api.profiles(),
      api.providerPresets(),
      api.models(),
      api.status(),
      api.runs()
    ]);
    // The review catalog only decorates the bounded-review UI — a failure here
    // must not fail the whole boot with a misleading "backend unreachable"
    // error. Every consumer already guards on reviewCatalog being non-null.
    const reviewPayload = await api.reviewCatalog().catch(() => null);
    setProfiles(profilePayload.profiles);
    setProfile(profilePayload.profiles.find((item) => item.name === profilePayload.active) ?? profilePayload.profiles[0]);
    setPresets(presetPayload.presets);
    setModels(modelPayload.models);
    setReviewCatalog(reviewPayload);
    setStatus(statusPayload);
    setRuns(runPayload.runs);
    setLatestRun((prev) => prev ?? runPayload.runs[0] ?? null);
    setDirty(false);
    setBootError(null);
  }

  function handleRefresh() {
    refreshAll().catch((error) =>
      setMessage(error instanceof Error ? error.message : String(error))
    );
  }

  useEffect(() => {
    refreshAll().catch((error) =>
      // On first load a failure means the backend is unreachable — surface it as
      // a boot error with a retry rather than an eternal "Loading…" screen.
      setBootError(error instanceof Error ? error.message : String(error))
    );
  }, []);

  function onGraphTextChange(value: string) {
    // JSON mode writes through to profile.graph on every VALID parse, so the
    // visual view and validation never operate on stale content, and toggling
    // modes can't silently drop edits.
    setGraphText(value);
    try {
      updateProfile({ graph: JSON.parse(value) });
      setGraphErrors([]);
    } catch {
      /* keep typing; invalid JSON just isn't written through yet */
    }
  }

  function applySeed(seedJson: string) {
    setGraphText(seedJson);
    setGraphErrors([]);
    try {
      updateProfile({ graph: JSON.parse(seedJson) });
    } catch {
      /* seed is always valid JSON */
    }
  }

  async function validateCurrentGraph() {
    // Source of truth is profile.graph in visual mode; the textarea in JSON mode.
    let candidate = profile?.graph ?? null;
    if (graphMode === "json") {
      try {
        candidate = JSON.parse(graphText || "{}");
      } catch (error) {
        setGraphErrors(["Invalid JSON: " + (error instanceof Error ? error.message : String(error))]);
        return;
      }
    }
    if (!candidate) {
      setGraphErrors(["No graph yet — add a node or insert an example."]);
      return;
    }
    try {
      const result = await api.validateGraph(candidate);
      setGraphErrors(result.errors);
      if (!result.errors.length) {
        updateProfile({ graph: result.graph });
        setGraphText(JSON.stringify(result.graph, null, 2));
        setMessage("Graph valid — layout tidied. Save to persist.");
      }
    } catch (error) {
      setGraphErrors([error instanceof Error ? error.message : String(error)]);
    }
  }

  const llmModels = useMemo(() => models.filter((model) => (model.type ?? "llm") === "llm"), [models]);
  const proposedChanges: FileChange[] = latestRun?.file_changes ?? [];
  const activeProviderName = providerDisplayName(profile?.provider ?? "openai-compatible");

  // Honest parallelism and mixture checks for workflows that dispatch workers.
  // Hybrid/parallel use the roster as their fan-out (capped at MAX_WORKERS=8;
  // an empty roster falls back to 3× the first configured model); bounded
  // review uses it as concurrency (one fallback slot). Cloud aliases are remote.
  const runsWorkers = profile
    ? ["hybrid", "parallel_subtask", "bounded_review"].includes(profile.workflow)
    : false;
  const effectiveWorkers = useMemo(() => {
    if (!profile) return [] as string[];
    const roster = profile.worker_models.filter(Boolean);
    // Bounded review round-robins over the FULL roster (only concurrency is
    // capped at 8), so every entry is really used; hybrid/parallel cap the
    // fan-out itself at MAX_WORKERS=8.
    if (roster.length) return profile.workflow === "bounded_review" ? roster : roster.slice(0, 8);
    const fallback = profile.aggregator_model || profile.evaluator_model;
    return fallback ? Array(profile.workflow === "bounded_review" ? 1 : 3).fill(fallback) : [];
  }, [profile]);
  const workerFanout = runsWorkers ? effectiveWorkers.length : 0;
  // Actual bounded-review parallelism: at most 8 simultaneous slots regardless
  // of roster length (the label must show parallelism, not roster size).
  const boundedSlots = Math.min(effectiveWorkers.length || 1, 8);
  const localWorkerCount = useMemo(() => {
    if (!profile || !runsWorkers) return 0;
    return effectiveWorkers.filter((m) => !(profile.cloud_models ?? []).some((c) => c.alias === m)).length;
  }, [profile, runsWorkers, effectiveWorkers]);
  const selfEnsemble = useMemo(() => {
    if (!profile || !runsWorkers) return false;
    const roleModels = [...effectiveWorkers, profile.aggregator_model, profile.evaluator_model].filter(Boolean);
    if (roleModels.length < 2) return false;
    return new Set(roleModels.map((m) => resolvedRouteKey(profile, m))).size === 1;
  }, [profile, runsWorkers, effectiveWorkers]);
  const slotsShort =
    status?.total_slots != null && localWorkerCount > status.total_slots ? status.total_slots : null;

  // How many atomic skills a bounded review will actually run: skill_count over
  // enabled categories minus the excluded ids that belong to those categories.
  const enabledSkillCount = useMemo(() => {
    if (!profile || !reviewCatalog) return 0;
    const enabled = new Set(profile.review_policy.categories);
    const total = reviewCatalog.categories
      .filter((category) => enabled.has(category.id))
      .reduce((sum, category) => sum + category.skill_count, 0);
    const excluded = new Set(profile.review_policy.excluded_skill_ids);
    const excludedInEnabled = reviewCatalog.skills.filter(
      (skill) => excluded.has(skill.id) && enabled.has(skill.category)
    ).length;
    return total - excludedInEnabled;
  }, [profile, reviewCatalog]);

  function updateProfile(patch: Partial<Profile>) {
    if (!profile) return;
    setProfile({ ...profile, ...patch });
    setDirty(true);
  }

  function toggleReviewCategory(category: string) {
    if (!profile) return;
    const current = profile.review_policy.categories;
    const selected = current.includes(category);
    if (selected && current.length === 1) {
      setMessage("Bounded review needs at least one category.");
      return;
    }
    const categories = selected
      ? current.filter((item) => item !== category)
      : [...current, category];
    updateProfile({ review_policy: { ...profile.review_policy, categories } });
  }

  function selectProfile(next: Profile) {
    if (dirty && next.name !== profile?.name) {
      const ok = window.confirm(
        "You have unsaved changes to the current profile. Switch and discard them?"
      );
      if (!ok) return;
    }
    setProfile(next);
    setGraphText(next.graph ? JSON.stringify(next.graph, null, 2) : "");
    setGraphErrors([]);
    setDirty(false);
  }

  // Workers are a multiset: the same model can appear several times (N parallel
  // instances), so mutations are add-one / remove-one, never a toggle.
  function workerCount(model: string): number {
    return profile ? profile.worker_models.filter((m) => m === model).length : 0;
  }

  function addWorker(model: string) {
    if (!profile || !model) return;
    updateProfile({ worker_models: [...profile.worker_models.filter(Boolean), model] });
  }

  function removeWorker(model: string) {
    if (!profile) return;
    const index = profile.worker_models.indexOf(model);
    if (index === -1) return;
    updateProfile({ worker_models: profile.worker_models.filter((_, i) => i !== index) });
  }

  function removeWorkerAt(index: number) {
    if (!profile) return;
    updateProfile({ worker_models: profile.worker_models.filter((_, i) => i !== index) });
  }

  function addCloudModel() {
    if (!profile) return;
    const existing = new Set((profile.cloud_models ?? []).map((c) => c.alias));
    let n = 1;
    let alias = "cloud-model";
    while (existing.has(alias)) alias = `cloud-model-${++n}`;
    updateProfile({
      cloud_models: [...(profile.cloud_models ?? []), { alias, provider: "openai", model: "", base_url: "" }]
    });
  }

  function updateCloudModel(index: number, patch: Partial<CloudModel>) {
    if (!profile) return;
    const next = (profile.cloud_models ?? []).map((c, i) => (i === index ? { ...c, ...patch } : c));
    updateProfile({ cloud_models: next });
  }

  function removeCloudModel(index: number) {
    if (!profile) return;
    updateProfile({ cloud_models: (profile.cloud_models ?? []).filter((_, i) => i !== index) });
  }

  async function refreshModels() {
    // Re-fetch the model list + reachability from the ACTIVE provider's /v1/models.
    setBusy(true);
    setMessage("");
    try {
      const [modelPayload, statusPayload] = await Promise.all([api.models(), api.status()]);
      setModels(modelPayload.models);
      setStatus(statusPayload);
      setMessage(`${modelPayload.models.length} model(s) from ${activeProviderName}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function saveProfile() {
    if (!profile) return;
    setBusy(true);
    setMessage("");
    try {
      const saved = await api.saveProfile(profile);
      setProfile(saved);
      const payload = await api.profiles();
      setProfiles(payload.profiles);
      setDirty(false);
      setMessage("Profile saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  function applyPreset(preset: ProviderPreset) {
    updateProfile({
      name: preset.name,
      provider: preset.provider,
      base_url: preset.base_url,
      worker_models: preset.worker_models,
      aggregator_model: preset.aggregator_model,
      evaluator_model: preset.evaluator_model
    });
    setStatus(null);
    const envText = preset.env_keys.length ? ` Env: ${preset.env_keys.join(", ")}.` : "";
    setMessage(`${preset.name} preset applied.${envText}`);
  }

  async function renameProfile(name: string) {
    const next = window.prompt(`Rename profile "${name}" to:`, name);
    if (!next || next.trim() === name) return;
    setBusy(true);
    setMessage("");
    try {
      const saved = await api.renameProfile(name, next.trim());
      const payload = await api.profiles();
      setProfiles(payload.profiles);
      setProfile(saved);
      setDirty(false);
      setMessage(`Renamed to "${saved.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function duplicateProfile(source: Profile) {
    const base = `${source.name} copy`;
    const existing = new Set(profiles.map((item) => item.name));
    let name = base;
    let n = 2;
    while (existing.has(name)) name = `${base} ${n++}`;
    setBusy(true);
    setMessage("");
    try {
      const saved = await api.saveProfile({ ...source, name });
      const payload = await api.profiles();
      setProfiles(payload.profiles);
      setProfile(saved);
      setDirty(false);
      setMessage(`Duplicated as "${saved.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function deleteProfile(name: string) {
    if (!window.confirm(`Delete profile "${name}"? This cannot be undone.`)) return;
    setBusy(true);
    setMessage("");
    try {
      await api.deleteProfile(name);
      const payload = await api.profiles();
      setProfiles(payload.profiles);
      const active = payload.profiles.find((item) => item.name === payload.active) ?? payload.profiles[0] ?? null;
      setProfile(active);
      setDirty(false);
      setMessage(`Profile "${name}" deleted.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function runWorkflow() {
    if (!profile || !prompt.trim() || running) return;
    setMessage("");
    // Take context paths straight from the textarea so they're included even if
    // the user never clicked "Preview context".
    const paths = uniqueLines(contextInput);
    setContextFiles(paths);
    try {
      const { run_id } = await api.startRun(prompt, profile, paths);
      setActiveRunId(run_id);
      setTab("run");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function viewHistoryRun(run: RunRecord) {
    // Don't tear down a live run to browse history — that would orphan it (no
    // Stop button, tokens streaming to nobody). Ask the user to stop it first.
    if (running) {
      setMessage("A run is in progress. Stop it before browsing history.");
      setTab("run");
      return;
    }
    setActiveRunId(null);
    setLatestRun(run);
    setTab("run");
  }

  async function stopActiveRun() {
    if (!activeRunId) return;
    try {
      await api.stopRun(activeRunId);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function popOutActivity() {
    if (!activeRunId) return;
    window.open(
      `activity.html?run_id=${activeRunId}`,
      `moa-activity-${activeRunId}`,
      "width=560,height=780,resizable=yes"
    );
  }

  function agentWindowUrl(runId: string, stage: ActivityStage): string {
    const title = encodeURIComponent(stage.title || stage.stage);
    return `activity.html?run_id=${runId}&step_id=${encodeURIComponent(stage.id)}&title=${title}`;
  }

  function popOutAgent(stage: ActivityStage) {
    if (!activeRunId) return;
    const opened = window.open(
      agentWindowUrl(activeRunId, stage),
      `moa-agent-${stage.id}`,
      "width=460,height=640,resizable=yes"
    );
    if (!opened) setMessage("Popup blocked — allow popups for this site to open agent windows.");
  }

  function toggleAutoPop() {
    const next = !autoPopAgents;
    if (next) {
      // Only agents that START after enabling should pop — not history, and
      // not completed stages an SSE reconnect replays as briefly "running".
      for (const stage of activity.stages) autoPoppedRef.current.add(stage.id);
    }
    setAutoPopAgents(next);
    localStorage.setItem("moa-auto-pop-agents", next ? "1" : "0");
  }

  // Serena-style per-agent windows: when enabled, every agent that starts gets
  // its own popup bound to its step id (the SSE buffer replays, so a window
  // opened mid-stage still shows the full feed). Browsers may block popups not
  // born from a click — we detect that once and tell the user what to allow.
  const autoPoppedRef = useRef<Set<string>>(new Set());
  const popupBlockedRef = useRef(false);
  useEffect(() => {
    autoPoppedRef.current = new Set();
    popupBlockedRef.current = false;
  }, [activeRunId]);
  useEffect(() => {
    if (!autoPopAgents || !activeRunId || activity.status !== "running") return;
    for (const stage of activity.stages) {
      if (stage.status !== "running" || autoPoppedRef.current.has(stage.id)) continue;
      autoPoppedRef.current.add(stage.id);
      const opened = window.open(
        agentWindowUrl(activeRunId, stage),
        `moa-agent-${stage.id}`,
        "width=460,height=640,resizable=yes"
      );
      if (!opened && !popupBlockedRef.current) {
        popupBlockedRef.current = true;
        setMessage("Popup blocked — allow popups for this site so each agent can open its own window.");
      }
    }
  }, [autoPopAgents, activeRunId, activity.stages, activity.status]);

  async function previewFiles() {
    if (!profile) return;
    const paths = uniqueLines(contextInput);
    setContextFiles(paths);
    if (!paths.length) {
      setFilePreview("");
      return;
    }
    try {
      const payload = await api.readFiles(paths, profile.allowed_roots);
      setFilePreview(payload.files.map((file) => `# ${file.path}\n${file.content}`).join("\n\n"));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function generateImages() {
    if (!profile || !imagePrompt.trim() || imageBusy) return;
    setImageBusy(true);
    setMessage("");
    try {
      const result = await api.generateImage(imagePrompt, imageCount, imageSize, profile.image_model);
      setImages(result.images);
      if (!result.images.length) setMessage("No images returned by the provider.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setImageBusy(false);
    }
  }

  function imageSrc(image: GeneratedImage): string {
    if (image.b64_json) return `data:image/png;base64,${image.b64_json}`;
    const url = image.url ?? "";
    // Only allow safe schemes — a compromised provider could return a
    // javascript: URL that would run when the download anchor is clicked.
    return /^(https?:|data:image\/)/i.test(url) ? url : "";
  }

  async function downloadImage(image: GeneratedImage, index: number) {
    const src = imageSrc(image);
    if (!src) return;
    try {
      // Route through a Blob URL so a multi-MB base64 image doesn't blow the
      // browser's href length limit (a data: URI in an anchor can silently fail).
      const blob = await (await fetch(src)).blob();
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `moa-image-${index + 1}.png`;
      anchor.click();
      URL.revokeObjectURL(objectUrl);
    } catch {
      setMessage("Could not download the image.");
    }
  }

  async function browseTo(path: string | null) {
    try {
      const listing = await api.listFiles(path);
      setBrowser(listing);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function addContextPath(path: string) {
    const paths = uniqueLines(contextInput);
    if (paths.includes(path)) return;
    const next = [...paths, path];
    setContextInput(next.join("\n"));
    setContextFiles(next);
  }

  async function resolveChange(id: string, action: "apply" | "reject") {
    if (action === "apply") {
      const change = proposedChanges.find((item) => item.id === id);
      const target = change ? change.path : "this file";
      if (!window.confirm(`Apply overwrites ${target} entirely. Continue?`)) return;
    }
    setBusy(true);
    setMessage("");
    try {
      if (action === "apply") {
        await api.applyChange(id);
      } else {
        await api.rejectChange(id);
      }
      if (latestRun) {
        const refreshed = await api.runs();
        setRuns(refreshed.runs);
        setLatestRun(refreshed.runs.find((run) => run.id === latestRun.id) ?? latestRun);
      }
      setMessage(action === "apply" ? "File change applied." : "File change rejected.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function resolveAll(action: "apply" | "reject") {
    if (!latestRun) return;
    const proposed = proposedChanges.filter((change) => change.status === "proposed");
    if (!proposed.length) return;
    if (action === "apply" && !window.confirm(`Apply all ${proposed.length} proposed change(s)? Each overwrites its file.`)) {
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const result = action === "apply" ? await api.applyAll(latestRun.id) : await api.rejectAll(latestRun.id);
      const refreshed = await api.runs();
      setRuns(refreshed.runs);
      setLatestRun(refreshed.runs.find((run) => run.id === latestRun.id) ?? latestRun);
      const failed = result.results.filter((entry) => !entry.ok);
      setMessage(
        failed.length
          ? `${result.applied} ${action === "apply" ? "applied" : "rejected"}, ${failed.length} skipped (${failed[0].detail}).`
          : `${result.applied} change(s) ${action === "apply" ? "applied" : "rejected"}.`
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  const shownActivity = activeRunId ? activity : recordToActivity(latestRun);
  const proposedCount = proposedChanges.filter((change) => change.status === "proposed").length;

  if (!profile) {
    if (bootError) {
      return (
        <div className="boot bootError">
          <strong>Can't reach the Workbench backend.</strong>
          <p>{bootError}</p>
          <p className="bootHint">
            Start it with <code>uvicorn workbench.api:app --port 8008</code>, then retry.
          </p>
          <button className="primaryButton" onClick={handleRefresh}>
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      );
    }
    return <div className="boot">Loading MoA Workbench...</div>;
  }

  return (
    <div className="appShell">
      <aside className="navRail">
        <div className="brand">
          <Layers3 size={24} />
          <span>MoA Workbench</span>
        </div>
        <nav>
          {tabs.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}>
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{tabs.find((item) => item.id === tab)?.label}</h1>
            <p>{profile.name} - {workflowLabels[profile.workflow]}</p>
          </div>
          <div className="topActions">
            {/* Status probes the ACTIVE provider's /v1/models, so online/offline
                reflects the endpoint the run will actually hit. */}
            <span className={`serverState ${status?.http_ok ? "online" : "offline"}`}>
              <Server size={16} /> {activeProviderName} {status?.http_ok ? "online" : "offline"}
            </span>
            <button
              className="iconButton"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={theme === "dark" ? "Switch to light" : "Switch to dark"}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <button className="iconButton" onClick={handleRefresh} title="Refresh">
              <RefreshCw size={17} />
            </button>
          </div>
        </header>

        {message && <div className="notice">{message}</div>}

        {tab === "run" && (
          <div className="runGrid">
            <section className="panel primaryPanel">
              <div className="sectionHeader">
                <div>
                  <h2>Prompt</h2>
                  <p>{workflowDescriptions[profile.workflow]}</p>
                </div>
                <div className="runActions">
                  <label className="autoPopToggle" title="Open one popup window per agent as it starts (serena-style)">
                    <input type="checkbox" checked={autoPopAgents} onChange={toggleAutoPop} />
                    Pop out each agent
                  </label>
                  {running ? (
                    <button className="dangerButton" onClick={stopActiveRun}>
                      <Square size={15} /> Stop
                    </button>
                  ) : (
                    <button className="primaryButton" onClick={runWorkflow} disabled={busy}>
                      <Play size={17} /> Run
                    </button>
                  )}
                  {activeRunId && (
                    <button className="iconButton" onClick={popOutActivity} title="Pop out the flow graph (view only)">
                      <ExternalLink size={16} />
                    </button>
                  )}
                </div>
              </div>
              {selfEnsemble && (
                <div className="notice noticeWarn">
                  All roles resolve to one model on one endpoint — this run is a self-ensemble, not a mixture.
                  Add a second local or cloud model (Models tab) for real diversity.
                </div>
              )}
              {slotsShort != null && (
                <div className="notice noticeWarn">
                  Your server reports {slotsShort} parallel slot{slotsShort === 1 ? "" : "s"} but this profile sends{" "}
                  {localWorkerCount} concurrent local worker request{localWorkerCount === 1 ? "" : "s"} — they will
                  queue and look serial. Start llama.cpp with <code>--parallel {localWorkerCount}</code>.
                </div>
              )}
              <textarea
                className="promptBox"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                    event.preventDefault();
                    if (!running) runWorkflow();
                  }
                }}
                title="Ctrl+Enter to run"
              />
              <div className="controlRow">
                <label>
                  Workflow
                  <select value={profile.workflow} onChange={(event) => updateProfile({ workflow: event.target.value as Workflow })}>
                    <option value="hybrid">Hybrid</option>
                    <option value="parallel_subtask">Subtask Parallel</option>
                    <option value="iterative_evaluator">Iterative Eval</option>
                    <option value="graph">Custom Graph</option>
                    <option value="bounded_review">Bounded Review</option>
                  </select>
                </label>
                {profile.workflow !== "bounded_review" && (
                  <label>
                    Max iterations
                    <input
                      type="number"
                      min={1}
                      value={profile.max_iterations}
                      onChange={(event) => updateProfile({ max_iterations: numberOr(event.target.value, profile.max_iterations, 1) })}
                    />
                  </label>
                )}
              </div>
              {profile.workflow === "bounded_review" && reviewCatalog && (
                <div className="reviewPolicyInline">
                  <strong>
                    {enabledSkillCount} atomic skills will run — one bounded agent and at most one finding each
                  </strong>
                  <div className="reviewCategoryToggles">
                    {reviewCatalog.categories.map((category) => (
                      <label key={category.id}>
                        <input
                          type="checkbox"
                          checked={profile.review_policy.categories.includes(category.id)}
                          onChange={() => toggleReviewCategory(category.id)}
                        />
                        {category.title} ({category.skill_count})
                      </label>
                    ))}
                  </div>
                  <small>
                    Candidates are verified per category in batches of {reviewCatalog.verifier_batch_size}; rejected or
                    duplicate candidates never enter the report.
                  </small>
                </div>
              )}
            </section>

            <section className="panel inspectorPanel">
              <h2>Setup</h2>
              <div className="metric">
                <span>Provider</span>
                <strong>{activeProviderName}</strong>
              </div>
              <div className="roleList">
                <span>{profile.workflow === "bounded_review" ? `Concurrency slots (${boundedSlots})` : `Workers (${workerFanout})`}</span>
                {runsWorkers ? (
                  groupWorkers(effectiveWorkers).map(([model, count]) => (
                    <code key={model}>{model}{count > 1 ? ` ×${count}` : ""}</code>
                  ))
                ) : (
                  <code>{profile.workflow === "graph" ? "per graph nodes" : "none"}</code>
                )}
                <span title="Also runs the orchestrator and refiner stages — there is no separate model for those.">
                  {profile.workflow === "bounded_review" ? "Fallback model" : "Aggregator"}
                </span>
                {/* Bounded review resolves these server-side: workers fall back to
                    aggregator || evaluator; verifier = evaluator || aggregator || workers[0]. */}
                <code>
                  {profile.workflow === "bounded_review"
                    ? profile.aggregator_model || profile.evaluator_model || "none"
                    : profile.aggregator_model}
                </code>
                <span>{profile.workflow === "bounded_review" ? "Verifier" : "Evaluator"}</span>
                <code>
                  {profile.workflow === "bounded_review"
                    ? profile.evaluator_model || profile.aggregator_model || effectiveWorkers[0] || "none"
                    : profile.evaluator_model}
                </code>
              </div>
              <p className="providerHint">
                {profile.workflow === "bounded_review"
                  ? "The worker roster limits simultaneous specialist calls; the evaluator independently verifies every candidate."
                  : <>The aggregator model also runs the <em>orchestrator</em> and <em>refiner</em> stages you'll see
                    during a run; the evaluator scores drafts.</>}
              </p>
              <button className="secondaryButton" onClick={saveProfile} disabled={busy}>
                <Save size={16} /> Save setup
              </button>
            </section>

            <section className="panel activityPanel">
              <div className="sectionHeader">
                <h2>Activity</h2>
              </div>
              {!activeRunId && !latestRun ? (
                <p className="empty">No run yet. Enter a prompt and hit Run to watch the agents work.</p>
              ) : (
                <ActivityView activity={shownActivity} onPopOut={activeRunId ? popOutAgent : undefined} />
              )}
            </section>
          </div>
        )}

        {tab === "images" && (
          <div className="runGrid">
            <section className="panel primaryPanel">
              <div className="sectionHeader">
                <div>
                  <h2>Image Prompt</h2>
                  <p>Text-to-image via the active profile's OpenAI-compatible endpoint.</p>
                </div>
                <div className="runActions">
                  <button className="primaryButton" onClick={generateImages} disabled={imageBusy || !imagePrompt.trim()}>
                    <Sparkles size={17} /> {imageBusy ? "Generating…" : "Generate"}
                  </button>
                </div>
              </div>
              <textarea
                className="promptBox"
                value={imagePrompt}
                onChange={(event) => setImagePrompt(event.target.value)}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                    event.preventDefault();
                    generateImages();
                  }
                }}
                title="Ctrl+Enter to generate"
              />
              <div className="controlRow">
                <label>
                  Size
                  <select value={imageSize} onChange={(event) => setImageSize(event.target.value)}>
                    <option value="256x256">256 × 256</option>
                    <option value="512x512">512 × 512</option>
                    <option value="1024x1024">1024 × 1024</option>
                    <option value="1792x1024">1792 × 1024</option>
                    <option value="1024x1792">1024 × 1792</option>
                  </select>
                </label>
                <label>
                  Count
                  <input
                    type="number"
                    min={1}
                    max={4}
                    value={imageCount}
                    onChange={(event) => setImageCount(Math.min(4, Math.max(1, numberOr(event.target.value, imageCount, 1))))}
                  />
                </label>
                <label>
                  Image model
                  <input
                    value={profile.image_model}
                    placeholder="e.g. dall-e-3 / sdxl"
                    onChange={(event) => updateProfile({ image_model: event.target.value })}
                  />
                </label>
              </div>
              <p className="providerHint">
                Uses <code>{activeProviderName}</code>{profile.base_url ? <> at <code>{profile.base_url}</code></> : null}. Set an image model here or via <code>MOA_IMAGE_MODEL</code>. Save the profile to persist it.
              </p>
            </section>
            <section className="panel activityPanel">
              <div className="sectionHeader">
                <h2>Result</h2>
              </div>
              {!images.length ? (
                <p className="empty">{imageBusy ? "Generating…" : "No images yet. Enter a prompt and hit Generate."}</p>
              ) : (
                <div className="imageGrid">
                  {images.map((image, index) => {
                    const src = imageSrc(image);
                    return (
                      <figure className="imageCard" key={index}>
                        {src ? <img src={src} alt={`Generated ${index + 1}`} /> : <p className="empty">Empty image.</p>}
                        <figcaption>
                          <button className="secondaryButton" onClick={() => downloadImage(image, index)}>
                            <UploadCloud size={15} /> Download
                          </button>
                        </figcaption>
                      </figure>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        )}

        {tab === "models" && (
          <div className="twoColumn">
            <section className="panel">
              <div className="sectionHeader">
                <div>
                  <h2>{activeProviderName} Models</h2>
                  <p>
                    {llmModels.length
                      ? `${llmModels.length} model(s) from ${profile.base_url}. Click to assign a role.`
                      : `No models found at ${profile.base_url}. Check the server is up, then Refresh.`}
                    {status?.total_slots != null &&
                      ` Server has ${status.total_slots} parallel slot${status.total_slots === 1 ? "" : "s"}.`}
                  </p>
                </div>
                <button className="secondaryButton" onClick={refreshModels} disabled={busy}>
                  <RefreshCw size={16} /> Refresh
                </button>
              </div>
              <div className="modelTable">
                {llmModels.map((model, index) => {
                  const key = modelKey(model);
                  const count = workerCount(key);
                  const isAggregator = profile.aggregator_model === key;
                  const isEvaluator = profile.evaluator_model === key;
                  return (
                    <article className="modelRow" key={`${key}-${index}`}>
                      <div className="modelMeta">
                        <div className="modelHeader">
                          <span>{modelName(model)}</span>
                          {model.state === "loaded" && <span className="pill pillOk">loaded</span>}
                        </div>
                        <code>{key}</code>
                        <span className="modelFacts">
                          {modelSizeLabel(model) && (
                            <span title={model.sizeIsEstimate ? "Estimated from the model name's parameter count and its quantization — LM Studio's API reports no file size." : "Size reported by the server."}>
                              {modelSizeLabel(model)}
                            </span>
                          )}
                          {model.quantization && <span>{model.quantization}</span>}
                          {(model.maxContextLength ?? 0) > 0 && (
                            <span>{Math.round((model.maxContextLength ?? 0) / 1024)}k ctx</span>
                          )}
                        </span>
                      </div>
                      <div className="roleButtons">
                        <span className={`workerStepper ${count > 0 ? "roleActive" : ""}`} title="Number of parallel worker instances of this model">
                          <button type="button" onClick={() => removeWorker(key)} disabled={count === 0} aria-label={`Remove a ${key} worker`}>−</button>
                          <span className="stepperLabel"><Users size={14} /> Worker{count > 0 ? ` ×${count}` : ""}</span>
                          <button type="button" onClick={() => addWorker(key)} aria-label={`Add a ${key} worker`}>+</button>
                        </span>
                        <button type="button" className={isAggregator ? "roleActive" : ""} onClick={() => updateProfile({ aggregator_model: key })} aria-pressed={isAggregator}>
                          <BrainCircuit size={14} /> Aggregator
                        </button>
                        <button type="button" className={isEvaluator ? "roleActive" : ""} onClick={() => updateProfile({ evaluator_model: key })} aria-pressed={isEvaluator}>
                          <ShieldCheck size={14} /> Evaluator
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
            <section className="panel rolePanel">
              <h2>Cloud API Models</h2>
              <p className="providerHint">
                Hosted models (keys from env) — no local footprint, so size never applies. Assign them roles like any local model; each call routes to its own provider, so you can mix local workers with a cloud aggregator.
              </p>
              <div className="modelTable">
                {(profile.cloud_models ?? []).map((cloud, index) => {
                  const count = workerCount(cloud.alias);
                  const isAggregator = profile.aggregator_model === cloud.alias;
                  const isEvaluator = profile.evaluator_model === cloud.alias;
                  return (
                    <article className="modelRow" key={`${cloud.alias}-${index}`}>
                      <div className="modelMeta">
                        <div className="modelHeader">
                          <span>{cloud.alias}</span>
                          <span className="pill pillOk">API · {providerDisplayName(cloud.provider)}</span>
                        </div>
                        <code>{cloud.model || cloud.alias}</code>
                      </div>
                      <div className="roleButtons">
                        <span className={`workerStepper ${count > 0 ? "roleActive" : ""}`} title="Number of parallel worker instances of this model">
                          <button type="button" onClick={() => removeWorker(cloud.alias)} disabled={count === 0} aria-label={`Remove a ${cloud.alias} worker`}>−</button>
                          <span className="stepperLabel"><Users size={14} /> Worker{count > 0 ? ` ×${count}` : ""}</span>
                          <button type="button" onClick={() => addWorker(cloud.alias)} aria-label={`Add a ${cloud.alias} worker`}>+</button>
                        </span>
                        <button type="button" className={isAggregator ? "roleActive" : ""} onClick={() => updateProfile({ aggregator_model: cloud.alias })} aria-pressed={isAggregator}>
                          <BrainCircuit size={14} /> Aggregator
                        </button>
                        <button type="button" className={isEvaluator ? "roleActive" : ""} onClick={() => updateProfile({ evaluator_model: cloud.alias })} aria-pressed={isEvaluator}>
                          <ShieldCheck size={14} /> Evaluator
                        </button>
                      </div>
                    </article>
                  );
                })}
                {!(profile.cloud_models ?? []).length && (
                  <p className="empty">None yet — add cloud models in the Profiles tab.</p>
                )}
              </div>

              <h2 className="subsectionTitle">Role Profile</h2>
              <div className="roleSelectors">
                <label>
                  Aggregator
                  <select value={profile.aggregator_model} onChange={(event) => updateProfile({ aggregator_model: event.target.value })}>
                    {llmModels.map((model, index) => (
                      <option key={`${modelKey(model)}-${index}`} value={modelKey(model)}>
                        {modelName(model)}{modelSizeLabel(model) ? ` — ${modelSizeLabel(model)}` : ""}
                      </option>
                    ))}
                    {(profile.cloud_models ?? []).map((cloud, index) => <option key={`cloud-${cloud.alias}-${index}`} value={cloud.alias}>{cloud.alias} (API)</option>)}
                  </select>
                </label>
                <label>
                  Evaluator
                  <select value={profile.evaluator_model} onChange={(event) => updateProfile({ evaluator_model: event.target.value })}>
                    {llmModels.map((model, index) => (
                      <option key={`${modelKey(model)}-${index}`} value={modelKey(model)}>
                        {modelName(model)}{modelSizeLabel(model) ? ` — ${modelSizeLabel(model)}` : ""}
                      </option>
                    ))}
                    {(profile.cloud_models ?? []).map((cloud, index) => <option key={`cloud-${cloud.alias}-${index}`} value={cloud.alias}>{cloud.alias} (API)</option>)}
                  </select>
                </label>
              </div>
              <div className="workerChips">
                {profile.worker_models.length === 0 && <span className="empty">No workers — add some above.</span>}
                {profile.worker_models.map((model, index) => (
                  <button type="button" key={`${model}-${index}`} onClick={() => removeWorkerAt(index)} title="Remove this worker">
                    {model} ✕
                  </button>
                ))}
              </div>
              <div className="roleActionRow">
                <button className="secondaryButton" onClick={saveProfile} disabled={busy}>
                  <Save size={16} /> Save setup
                </button>
              </div>
              <p className="providerHint">
                Model names are the aliases your server reports at <code>/v1/models</code>. Different workers can be different models (slower if your server swaps one model at a time).
              </p>
            </section>
          </div>
        )}

        {tab === "files" && (
          <div className="twoColumn">
            <section className="panel">
              <h2>Context Files</h2>
              <label>
                Allowed roots
                <textarea
                  className="compactText"
                  value={textFromList(profile.allowed_roots)}
                  onChange={(event) => updateProfile({ allowed_roots: uniqueLines(event.target.value) })}
                />
              </label>
              <label>
                Files to include
                <textarea className="compactText" value={contextInput} onChange={(event) => setContextInput(event.target.value)} />
              </label>
              <div className="roleActionRow">
                <button className="secondaryButton" onClick={previewFiles}>
                  <FileText size={16} /> Preview context
                </button>
                <button className="secondaryButton" onClick={() => browseTo(browser ? browser.path : null)}>
                  <FolderOpen size={16} /> {browser ? "Refresh browser" : "Browse files"}
                </button>
                <button className="secondaryButton" onClick={saveProfile} disabled={busy} title="Persist allowed roots to the active profile">
                  <Save size={16} /> Save roots
                </button>
              </div>
              {browser && (
                <div className="fileBrowser">
                  <div className="browserPath">
                    <button
                      className="browserUp"
                      disabled={!browser.parent && browser.path === null}
                      onClick={() => browseTo(browser.path === null ? null : browser.parent)}
                      title="Up"
                    >
                      ← {browser.path ?? "Roots"}
                    </button>
                    <button className="browserClose" onClick={() => setBrowser(null)} title="Close browser">
                      <XCircle size={14} />
                    </button>
                  </div>
                  <div className="browserList">
                    {!browser.entries.length && <p className="empty">Empty.</p>}
                    {browser.entries.map((entry) => (
                      <div className="browserRow" key={entry.path}>
                        {entry.is_dir ? (
                          <button className="browserEntry dir" onClick={() => browseTo(entry.path)}>
                            <FolderOpen size={14} /> {entry.name}
                          </button>
                        ) : (
                          <span className="browserEntry file">
                            <FileText size={14} /> {entry.name}
                          </span>
                        )}
                        {!entry.is_dir && (
                          <button className="browserAdd" title="Add to context files" onClick={() => addContextPath(entry.path)}>
                            + Add
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <pre className="filePreview">{filePreview || "No context preview loaded."}</pre>
            </section>
            <section className="panel">
              <div className="sectionHeader">
                <h2>Proposed Diffs</h2>
                {proposedCount > 1 && (
                  <div className="runActions">
                    <button className="primaryButton" onClick={() => resolveAll("apply")} disabled={busy}>
                      <CheckCircle2 size={16} /> Apply all ({proposedCount})
                    </button>
                    <button className="secondaryButton" onClick={() => resolveAll("reject")} disabled={busy}>
                      <XCircle size={16} /> Reject all
                    </button>
                  </div>
                )}
              </div>
              {!proposedChanges.length && <p className="empty">Agents have not proposed file changes yet.</p>}
              {proposedChanges.map((change) => (
                <div className="diffBlock" key={change.id}>
                  <div className="diffHeader">
                    <span>{change.path}</span>
                    <StatusPill status={change.status} />
                  </div>
                  <pre>{change.diff}</pre>
                  {change.status === "proposed" && (
                    <div className="diffActions">
                      <button className="primaryButton" onClick={() => resolveChange(change.id, "apply")} disabled={busy}>
                        <CheckCircle2 size={16} /> Apply
                      </button>
                      <button className="secondaryButton" onClick={() => resolveChange(change.id, "reject")} disabled={busy}>
                        <XCircle size={16} /> Reject
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </section>
          </div>
        )}

        {tab === "profiles" && (
          <section className="panel formPanel">
            <div className="sectionHeader">
              <div>
                <h2>Saved Setup</h2>
                <p>Profiles persist in user app data and become active when saved.</p>
              </div>
              <div className="runActions">
                <button className="secondaryButton" onClick={() => duplicateProfile(profile)} disabled={busy}>
                  <FileText size={16} /> Duplicate
                </button>
                <button className="secondaryButton" onClick={() => renameProfile(profile.name)} disabled={busy}>
                  <Settings size={16} /> Rename
                </button>
                <button className="primaryButton" onClick={saveProfile} disabled={busy}>
                  <Save size={17} /> Save
                </button>
              </div>
            </div>
            <div className="presetGrid">
              {presets.map((preset) => (
                <button
                  type="button"
                  key={preset.id}
                  className={profile.provider === preset.provider && profile.base_url === preset.base_url ? "selected" : ""}
                  onClick={() => applyPreset(preset)}
                >
                  <strong>{preset.name}</strong>
                  <span>{providerDisplayName(preset.provider)} {preset.base_url ? `- ${preset.base_url}` : "- adapter"}</span>
                  <small>{preset.notes[0]}</small>
                </button>
              ))}
            </div>
            <div className="formGrid">
              <label>Name<input value={profile.name} onChange={(event) => updateProfile({ name: event.target.value })} /></label>
              <label>
                Provider
                <select value={profile.provider} onChange={(event) => updateProfile({ provider: event.target.value })}>
                  <option value="openai-compatible">OpenAI-compatible (LM Studio / llama.cpp)</option>
                  <option value="omp">OMP</option>
                  <option value="openai">OpenAI</option>
                  <option value="together">Together</option>
                  <option value="atomic">Atomic placeholder</option>
                </select>
              </label>
              <label>Base URL<input value={profile.base_url} onChange={(event) => updateProfile({ base_url: event.target.value })} /></label>
            </div>
            <p className="providerHint">
              API keys are read from environment variables, not saved in profiles. OpenAI uses <code>OPENAI_API_KEY</code>, OMP uses <code>OMP_API_KEY</code>, Together uses <code>TOGETHER_API_KEY</code>, and generic proxies use <code>MOA_API_KEY</code>.
            </p>
            <label>
              Worker models (one line per instance — the list length is the parallel fan-out)
              <textarea className="compactText" value={textFromList(profile.worker_models)} onChange={(event) => updateProfile({ worker_models: uniqueLines(event.target.value) })} />
            </label>
            <div className="formGrid">
              <label>
                Aggregator (also runs the orchestrator and refiner stages)
                <input value={profile.aggregator_model} onChange={(event) => updateProfile({ aggregator_model: event.target.value })} />
              </label>
              <label>Evaluator<input value={profile.evaluator_model} onChange={(event) => updateProfile({ evaluator_model: event.target.value })} /></label>
            </div>
            <div className="formGrid">
              <label>Image model<input value={profile.image_model} placeholder="e.g. dall-e-3 / sdxl (optional)" onChange={(event) => updateProfile({ image_model: event.target.value })} /></label>
              <label>
                Frequency penalty (anti-repetition, e.g. 0.3)
                <input
                  type="number"
                  step={0.1}
                  min={-2}
                  max={2}
                  value={profile.frequency_penalty ?? ""}
                  placeholder="provider default"
                  onChange={(event) =>
                    updateProfile({
                      frequency_penalty: event.target.value.trim() === "" ? null : Number(event.target.value)
                    })
                  }
                />
              </label>
              <label>
                Presence penalty
                <input
                  type="number"
                  step={0.1}
                  min={-2}
                  max={2}
                  value={profile.presence_penalty ?? ""}
                  placeholder="provider default"
                  onChange={(event) =>
                    updateProfile({
                      presence_penalty: event.target.value.trim() === "" ? null : Number(event.target.value)
                    })
                  }
                />
              </label>
            </div>

            {profile.workflow === "bounded_review" && reviewCatalog && (
              <section className="reviewPolicyPanel">
                <div className="sectionHeader">
                  <div>
                    <h2>Bounded Review Catalog</h2>
                    <p>
                      {reviewCatalog.atomic_skill_count} atomic skills distilled from {reviewCatalog.source_item_count} checklist
                      items ({reviewCatalog.merged_duplicate_count} duplicates merged). One bounded agent per enabled skill, at
                      most one finding each; the verifier checks candidates in batches of {reviewCatalog.verifier_batch_size}.
                    </p>
                  </div>
                </div>
                <p className="providerHint">
                  {enabledSkillCount} skills enabled across {profile.review_policy.categories.length} of{" "}
                  {reviewCatalog.categories.length} categories
                  {profile.review_policy.excluded_skill_ids.length
                    ? ` (${profile.review_policy.excluded_skill_ids.length} individual skills excluded)`
                    : ""}
                  . No finding caps — policy trims only by category and per-skill exclusion.
                </p>
                <div className="reviewAgentGrid">
                  {reviewCatalog.categories.map((category) => {
                    const enabled = profile.review_policy.categories.includes(category.id);
                    return (
                      <article className={enabled ? "reviewAgentCard selected" : "reviewAgentCard"} key={category.id}>
                        <label className="reviewAgentToggle">
                          <input type="checkbox" checked={enabled} onChange={() => toggleReviewCategory(category.id)} />
                          <span>
                            <strong>{category.title} ({category.skill_count})</strong>
                            <small>{category.mission}</small>
                          </span>
                        </label>
                      </article>
                    );
                  })}
                </div>
              </section>
            )}
            <div className="cloudModels">
              <div className="sectionHeader">
                <div>
                  <h2>Cloud API Models</h2>
                  <p>Hosted models under a local alias — assignable to any role, keys from env, no local size/hosting.</p>
                </div>
                <button className="secondaryButton" onClick={addCloudModel} disabled={busy}>
                  <Sparkles size={15} /> Add cloud model
                </button>
              </div>
              {(profile.cloud_models ?? []).map((cloud, index) => (
                <div className="cloudRow" key={index}>
                  <label>Alias<input value={cloud.alias} onChange={(event) => updateCloudModel(index, { alias: event.target.value })} /></label>
                  <label>
                    Provider
                    <select value={cloud.provider} onChange={(event) => updateCloudModel(index, { provider: event.target.value })}>
                      <option value="openai">OpenAI</option>
                      <option value="together">Together</option>
                      <option value="omp">OMP</option>
                      <option value="openai-compatible">OpenAI-compatible</option>
                    </select>
                  </label>
                  <label>Model id<input value={cloud.model} placeholder="e.g. gpt-4o-mini" onChange={(event) => updateCloudModel(index, { model: event.target.value })} /></label>
                  <label>Base URL<input value={cloud.base_url} placeholder="(provider default)" onChange={(event) => updateCloudModel(index, { base_url: event.target.value })} /></label>
                  <button className="profileDelete" title="Remove" onClick={() => removeCloudModel(index)}>
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
              {!(profile.cloud_models ?? []).length && <p className="empty">None yet. Add one, then assign it a role on the Models tab.</p>}
            </div>
            {profile.workflow === "graph" && (
              <div className="graphEditor">
                <div className="sectionHeader">
                  <div>
                    <h2>Flow Graph</h2>
                    <p>Nodes (llm / fanout / gate) + dependencies. Authored once, run as-is.</p>
                  </div>
                  <div className="runActions">
                    <div className="segmented">
                      <button className={graphMode === "visual" ? "active" : ""} onClick={() => setGraphMode("visual")}>Visual</button>
                      <button className={graphMode === "json" ? "active" : ""} onClick={() => { setGraphText(profile.graph ? JSON.stringify(profile.graph, null, 2) : graphText); setGraphMode("json"); }}>JSON</button>
                    </div>
                    <button className="secondaryButton" onClick={validateCurrentGraph}>
                      Validate &amp; tidy
                    </button>
                  </div>
                </div>
                <div className="seedRow">
                  <span>Start from:</span>
                  <button className="linklike" onClick={() => applySeed(SEED_GRAPH)}>fan-out</button>
                  <button className="linklike" onClick={() => applySeed(SEED_LOOP_GRAPH)}>refine loop</button>
                  <button className="linklike" onClick={() => applySeed(SEED_BRANCH_GRAPH)}>branch</button>
                </div>
                {graphMode === "visual" ? (
                  <GraphBuilder
                    graph={profile.graph ?? { nodes: [], output: "" }}
                    onChange={(g) => updateProfile({ graph: g })}
                    models={llmModels}
                  />
                ) : (
                  <textarea
                    className="graphText"
                    spellCheck={false}
                    value={graphText}
                    onChange={(event) => onGraphTextChange(event.target.value)}
                    placeholder='{ "nodes": [...], "output": "..." }'
                  />
                )}
                {graphErrors.length > 0 && (
                  <ul className="graphErrors">
                    {graphErrors.map((err, index) => (
                      <li key={index}>{err}</li>
                    ))}
                  </ul>
                )}
                {profile.graph && !graphErrors.length && (
                  <p className="providerHint">
                    {profile.graph.nodes.length} node(s), output <code>{profile.graph.output || "(unset)"}</code>. Pop out a run to watch this graph live.
                  </p>
                )}
              </div>
            )}
            <div className="flowCall">
              <span>Call this saved flow like an agent (server resolves the config by name):</span>
              <pre>{`POST /api/flows/${encodeURIComponent(profile.name)}/run\n{ "prompt": "..." }`}</pre>
              <button
                className="secondaryButton"
                onClick={() =>
                  navigator.clipboard
                    ?.writeText(
                      `curl -X POST http://127.0.0.1:8008/api/flows/${encodeURIComponent(profile.name)}/run -H "Content-Type: application/json" -d '{"prompt":"your task here"}'`
                    )
                    .then(() => setMessage("Flow call copied as curl."))
                    .catch(() => undefined)
                }
              >
                <FileText size={16} /> Copy curl
              </button>
            </div>
            <div className="savedProfiles">
              {profiles.map((item) => (
                <span key={item.name} className={`savedProfile ${item.name === profile.name ? "selected" : ""}`}>
                  <button onClick={() => selectProfile(item)}>{item.name}</button>
                  {profiles.length > 1 && (
                    <button
                      className="profileDelete"
                      title={`Delete ${item.name}`}
                      onClick={() => deleteProfile(item.name)}
                      disabled={busy}
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </span>
              ))}
            </div>
          </section>
        )}

        {tab === "history" && (
          <section className="panel">
            <h2>Run History</h2>
            <div className="historyList">
              {!runs.length && <p className="empty">No runs yet.</p>}
              {runs.map((run) => (
                <div className="historyRow" key={run.id}>
                  <button className="historyOpen" onClick={() => viewHistoryRun(run)}>
                    <span>{new Date(run.created_at).toLocaleString()}</span>
                    <strong>{run.prompt}</strong>
                    <StatusPill status={run.evaluation.status} />
                  </button>
                  <button
                    className="profileDelete"
                    title="Delete run"
                    disabled={busy}
                    onClick={async () => {
                      if (!window.confirm("Delete this run from history?")) return;
                      try {
                        await api.deleteRun(run.id);
                        const refreshed = await api.runs();
                        setRuns(refreshed.runs);
                        if (latestRun?.id === run.id) setLatestRun(null);
                      } catch (error) {
                        setMessage(error instanceof Error ? error.message : String(error));
                      }
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
