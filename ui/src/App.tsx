import { useEffect, useMemo, useState } from "react";
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
import { api, formatBytes, modelKey, modelName, modelSize } from "./api";
import { ActivityView } from "./ActivityView";
import { GraphBuilder } from "./GraphBuilder";
import { useRunActivity, type RunActivityState } from "./useRunActivity";
import type { FileChange, GeneratedImage, LmModel, LmStudioStatus, ModelPlan, Profile, ProviderPreset, RunRecord, Workflow } from "./types";

function recordToActivity(record: RunRecord | null): RunActivityState {
  if (!record) return { stages: [], record: null, status: "idle", error: null, reconnecting: false };
  return {
    stages: record.trace.map((step) => ({
      id: step.id,
      stage: step.stage,
      title: step.title,
      model: step.model,
      status: "complete",
      tokens: "",
      output: step.output,
      startedAt: step.started_at,
      endedAt: step.ended_at
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
  graph: "Custom Graph"
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

function loadedModelKey(record: Record<string, unknown>): string {
  const value = record.modelKey ?? record.model_key ?? record.identifier ?? record.id ?? record.model;
  return typeof value === "string" ? value : "";
}

function uniqueModelKeys(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)));
}

function providerDisplayName(provider: string): string {
  const normalized = provider.trim().toLowerCase();
  if (normalized === "lmstudio") return "LM Studio";
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
  const [models, setModels] = useState<LmModel[]>([]);
  const [status, setStatus] = useState<LmStudioStatus | null>(null);
  const [plan, setPlan] = useState<ModelPlan | null>(null);
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
  const [loadResults, setLoadResults] = useState<Array<{ model: string; ok: boolean; stdout: string; stderr: string }>>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
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
    setProfiles(profilePayload.profiles);
    setProfile(profilePayload.profiles.find((item) => item.name === profilePayload.active) ?? profilePayload.profiles[0]);
    setPresets(presetPayload.presets);
    setModels(modelPayload.models);
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
  const loadedModelKeys = useMemo(
    () => new Set((status?.loaded_models ?? []).map(loadedModelKey).filter(Boolean)),
    [status]
  );
  const selectedProfileModelKeys = useMemo(() => {
    if (!profile) return [];
    return uniqueModelKeys([...profile.worker_models, profile.aggregator_model, profile.evaluator_model]);
  }, [profile]);
  const selectedProfileSize = useMemo(() => {
    const sizeByKey = new Map(llmModels.map((model) => [modelKey(model), modelSize(model)]));
    return selectedProfileModelKeys.reduce((total, key) => total + (sizeByKey.get(key) ?? 0), 0);
  }, [llmModels, selectedProfileModelKeys]);
  const selectedPlanSize = plan ? formatBytes(plan.total_size_bytes) : formatBytes(selectedProfileSize);
  const activeProviderName = providerDisplayName(profile?.provider ?? "lmstudio");
  const isLmStudioProfile = (profile?.provider ?? "lmstudio").trim().toLowerCase() === "lmstudio";

  function updateProfile(patch: Partial<Profile>) {
    if (!profile) return;
    setProfile({ ...profile, ...patch });
    setDirty(true);
  }

  function selectProfile(next: Profile) {
    if (dirty && next.name !== profile?.name) {
      const ok = window.confirm(
        "You have unsaved changes to the current profile. Switch and discard them?"
      );
      if (!ok) return;
    }
    setProfile(next);
    setPlan(null);
    setLoadResults([]);
    setGraphText(next.graph ? JSON.stringify(next.graph, null, 2) : "");
    setGraphErrors([]);
    setDirty(false);
  }

  function toggleWorker(model: string) {
    if (!profile || !model) return;
    const current = profile.worker_models.filter(Boolean);
    const exists = current.includes(model);
    const next = exists ? current.filter((item) => item !== model) : [...current, model];
    updateProfile({ worker_models: next.length ? next : [model] });
  }

  async function loadProfileModels() {
    if (!profile) return;
    if (!isLmStudioProfile) {
      setMessage("Load selected uses LM Studio. Save this remote provider setup, then run it through its OpenAI-compatible base URL.");
      return;
    }
    const keys = uniqueModelKeys([...profile.worker_models, profile.aggregator_model, profile.evaluator_model]);
    setBusy(true);
    setMessage("");
    try {
      const result = await api.loadModels(keys);
      setLoadResults(result.results);
      const statusPayload = await api.status();
      setStatus(statusPayload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function suggestPlan() {
    if (!profile) return;
    if (!isLmStudioProfile) {
      setMessage("Suggest uses LM Studio model metadata. Use provider presets or profile fields for remote model IDs.");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const suggested = await api.modelPlan(profile.memory_cap_gb, models);
      setPlan(suggested);
      updateProfile({
        worker_models: suggested.worker_models,
        aggregator_model: suggested.aggregator_model,
        evaluator_model: suggested.evaluator_model
      });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function loadSelectedPlan() {
    if (!plan) return;
    if (!isLmStudioProfile) {
      setMessage("Load plan uses the LM Studio lms CLI. Remote providers run through their base URL and do not need local loading.");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const result = await api.loadPlan(plan);
      setLoadResults(result.results);
      const statusPayload = await api.status();
      setStatus(statusPayload);
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
    setPlan(null);
    setStatus(null);
    setLoadResults([]);
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
            {isLmStudioProfile ? (
              <span className={`serverState ${status?.http_ok ? "online" : "offline"}`}>
                <Server size={16} /> {status?.http_ok ? "LM Studio online" : "LM Studio offline"}
              </span>
            ) : (
              // The status probe only reflects LM Studio; for a remote provider we
              // can't assert online/offline, so show the provider name neutrally.
              <span className="serverState remote">
                <Server size={16} /> {activeProviderName} (remote)
              </span>
            )}
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
                  <p>Hybrid runs decompose, parallelize, synthesize, then evaluate.</p>
                </div>
                <div className="runActions">
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
                    <button className="iconButton" onClick={popOutActivity} title="Pop out activity (view only)">
                      <ExternalLink size={16} />
                    </button>
                  )}
                </div>
              </div>
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
                  </select>
                </label>
                <label>
                  Memory cap
                  <input
                    type="number"
                    min={1}
                    value={profile.memory_cap_gb}
                    onChange={(event) => updateProfile({ memory_cap_gb: numberOr(event.target.value, profile.memory_cap_gb, 1) })}
                  />
                </label>
                <label>
                  Max iterations
                  <input
                    type="number"
                    min={1}
                    value={profile.max_iterations}
                    onChange={(event) => updateProfile({ max_iterations: numberOr(event.target.value, profile.max_iterations, 1) })}
                  />
                </label>
              </div>
            </section>

            <section className="panel inspectorPanel">
              <h2>Setup</h2>
              <div className="metric">
                <span>Model plan</span>
                <strong>{selectedPlanSize}</strong>
              </div>
              <div className="roleList">
                <span>Workers</span>
                {profile.worker_models.map((model, index) => <code key={`${model}-${index}`}>{model}</code>)}
                <span>Aggregator</span>
                <code>{profile.aggregator_model}</code>
                <span>Evaluator</span>
                <code>{profile.evaluator_model}</code>
              </div>
              <button className="secondaryButton" onClick={suggestPlan} disabled={busy || !isLmStudioProfile}>
                <Sparkles size={16} /> Suggest under cap
              </button>
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
                <ActivityView activity={shownActivity} />
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
                          <a href={src} download={`moa-image-${index + 1}.png`} className="secondaryButton">
                            <UploadCloud size={15} /> Download
                          </a>
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
                  <h2>LM Studio Models</h2>
                  <p>
                    {isLmStudioProfile
                      ? `${llmModels.length} local or linked LLMs detected.`
                      : "Remote provider model IDs are edited in Profiles; this planner is for LM Studio."}
                  </p>
                </div>
                <button className="secondaryButton" onClick={suggestPlan} disabled={busy || !isLmStudioProfile}>
                  <Sparkles size={16} /> Suggest
                </button>
              </div>
              <div className="modelTable">
                {llmModels.map((model, index) => {
                  const key = modelKey(model);
                  const isWorker = profile.worker_models.includes(key);
                  const isAggregator = profile.aggregator_model === key;
                  const isEvaluator = profile.evaluator_model === key;
                  return (
                    <article className="modelRow" key={`${key}-${index}`}>
                      <div className="modelMeta">
                        <div className="modelHeader">
                          <span>{modelName(model)}</span>
                          <StatusPill status={loadedModelKeys.has(key) ? "loaded" : "idle"} />
                        </div>
                        <code>{key}</code>
                        <small>{formatBytes(modelSize(model))} - {model.architecture ?? "unknown"}</small>
                      </div>
                      <div className="roleButtons">
                        <button
                          type="button"
                          className={isWorker ? "roleActive" : ""}
                          onClick={() => toggleWorker(key)}
                          aria-pressed={isWorker}
                        >
                          <Users size={14} /> Worker
                        </button>
                        <button
                          type="button"
                          className={isAggregator ? "roleActive" : ""}
                          onClick={() => updateProfile({ aggregator_model: key })}
                          aria-pressed={isAggregator}
                        >
                          <BrainCircuit size={14} /> Aggregator
                        </button>
                        <button
                          type="button"
                          className={isEvaluator ? "roleActive" : ""}
                          onClick={() => updateProfile({ evaluator_model: key })}
                          aria-pressed={isEvaluator}
                        >
                          <ShieldCheck size={14} /> Evaluator
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
            <section className="panel rolePanel">
              <h2>Role Profile</h2>
              <div className="metric">
                <span>Selected footprint</span>
                <strong>{formatBytes(selectedProfileSize)}</strong>
              </div>
              <div className="roleSelectors">
                <label>
                  Aggregator
                  <select value={profile.aggregator_model} onChange={(event) => updateProfile({ aggregator_model: event.target.value })}>
                    {llmModels.map((model, index) => <option key={`${modelKey(model)}-${index}`} value={modelKey(model)}>{modelName(model)}</option>)}
                  </select>
                </label>
                <label>
                  Evaluator
                  <select value={profile.evaluator_model} onChange={(event) => updateProfile({ evaluator_model: event.target.value })}>
                    {llmModels.map((model, index) => <option key={`${modelKey(model)}-${index}`} value={modelKey(model)}>{modelName(model)}</option>)}
                  </select>
                </label>
              </div>
              <div className="workerChips">
                {profile.worker_models.map((model, index) => (
                  <button type="button" key={`${model}-${index}`} onClick={() => toggleWorker(model)} title="Remove worker">
                    {model}
                  </button>
                ))}
              </div>
              <div className="roleActionRow">
                <button className="secondaryButton" onClick={saveProfile} disabled={busy}>
                  <Save size={16} /> Save setup
                </button>
                <button className="primaryButton" onClick={loadProfileModels} disabled={busy || !isLmStudioProfile}>
                  <UploadCloud size={17} /> Load selected
                </button>
              </div>

              <h2 className="subsectionTitle">Suggested Load Plan</h2>
              {plan ? (
                <>
                  <div className="metric">
                    <span>Total selected</span>
                    <strong>{formatBytes(plan.total_size_bytes)}</strong>
                  </div>
                  <div className="roleList">
                    <span>Workers</span>
                    {plan.worker_models.map((model, index) => <code key={`${model}-${index}`}>{model}</code>)}
                    <span>Aggregator</span>
                    <code>{plan.aggregator_model}</code>
                    <span>Evaluator</span>
                    <code>{plan.evaluator_model}</code>
                  </div>
                  <button className="primaryButton wide" onClick={loadSelectedPlan} disabled={busy || !isLmStudioProfile}>
                    <UploadCloud size={17} /> Load plan
                  </button>
                </>
              ) : (
                <p className="empty">Suggest a plan to select workers and evaluator under the cap.</p>
              )}
              {loadResults.map((result, index) => (
                <div className="loadResult" key={`${result.model}-${index}`}>
                  <StatusPill status={result.ok ? "complete" : "error"} />
                  <span>{result.model}</span>
                </div>
              ))}
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
                  <option value="lmstudio">LM Studio</option>
                  <option value="omp">OMP</option>
                  <option value="openai-compatible">OpenAI-compatible</option>
                  <option value="openai">OpenAI</option>
                  <option value="together">Together</option>
                  <option value="atomic">Atomic placeholder</option>
                </select>
              </label>
              <label>Base URL<input value={profile.base_url} onChange={(event) => updateProfile({ base_url: event.target.value })} /></label>
              <label>Memory cap<input type="number" min={1} value={profile.memory_cap_gb} onChange={(event) => updateProfile({ memory_cap_gb: numberOr(event.target.value, profile.memory_cap_gb, 1) })} /></label>
            </div>
            <p className="providerHint">
              API keys are read from environment variables, not saved in profiles. OMP uses <code>OMP_API_KEY</code>, Together uses <code>TOGETHER_API_KEY</code>, and generic proxies use <code>MOA_API_KEY</code>.
            </p>
            <label>Worker models<textarea className="compactText" value={textFromList(profile.worker_models)} onChange={(event) => updateProfile({ worker_models: uniqueLines(event.target.value) })} /></label>
            <div className="formGrid">
              <label>Aggregator<input value={profile.aggregator_model} onChange={(event) => updateProfile({ aggregator_model: event.target.value })} /></label>
              <label>Evaluator<input value={profile.evaluator_model} onChange={(event) => updateProfile({ evaluator_model: event.target.value })} /></label>
            </div>
            <div className="formGrid">
              <label>Image model<input value={profile.image_model} placeholder="e.g. dall-e-3 / sdxl (optional)" onChange={(event) => updateProfile({ image_model: event.target.value })} /></label>
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
