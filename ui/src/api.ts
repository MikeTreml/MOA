import type { FlowGraph, GeneratedImage, Profile, ProviderModel, ProviderPreset, ProviderStatus, RunRecord } from "./types";

const TOKEN_KEY = "moa-api-token";

export function getApiToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function setApiToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

function authHeaders(): Record<string, string> {
  const token = getApiToken();
  return token ? { "X-MoA-Token": token } : {};
}

async function request<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(init?.headers ?? {}) },
    ...init
  });
  if (response.status === 401 && retry) {
    // The backend has MOA_WORKBENCH_TOKEN set. Ask once, store, and retry.
    const entered = typeof window !== "undefined" ? window.prompt("This Workbench requires an API token:") : null;
    if (entered) {
      setApiToken(entered.trim());
      return request<T>(path, init, false);
    }
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  status: () => request<ProviderStatus>("/api/status"),
  models: () => request<{ models: ProviderModel[] }>("/api/models"),
  providerPresets: () => request<{ presets: ProviderPreset[] }>("/api/provider-presets"),
  profiles: () => request<{ profiles: Profile[]; active: string }>("/api/profiles"),
  saveProfile: (profile: Profile) =>
    request<Profile>("/api/profiles", { method: "POST", body: JSON.stringify(profile) }),
  deleteProfile: (name: string) =>
    request<{ status: string; name: string }>(`/api/profiles/${encodeURIComponent(name)}`, { method: "DELETE" }),
  renameProfile: (name: string, newName: string) =>
    request<Profile>(`/api/profiles/${encodeURIComponent(name)}/rename`, {
      method: "POST",
      body: JSON.stringify({ new_name: newName })
    }),
  listFiles: (path: string | null) =>
    request<{ path: string | null; parent: string | null; entries: Array<{ name: string; path: string; is_dir: boolean }> }>(
      "/api/files/list",
      { method: "POST", body: JSON.stringify({ path }) }
    ),
  validateGraph: (graph: FlowGraph) =>
    request<{ errors: string[]; graph: FlowGraph }>("/api/graph/validate", {
      method: "POST",
      body: JSON.stringify({ graph })
    }),
  runs: () => request<{ runs: RunRecord[] }>("/api/runs"),
  deleteRun: (runId: string) => request(`/api/runs/${runId}`, { method: "DELETE" }),
  createRun: (prompt: string, profile: Profile, contextFiles: string[]) =>
    request<RunRecord>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ prompt, profile, context_files: contextFiles })
    }),
  startRun: (prompt: string, profile: Profile, contextFiles: string[]) =>
    request<{ run_id: string; status: string }>("/api/runs/start", {
      method: "POST",
      body: JSON.stringify({ prompt, profile, context_files: contextFiles })
    }),
  stopRun: (runId: string) => request(`/api/runs/${runId}/stop`, { method: "POST" }),
  getRun: (runId: string) => request<RunRecord>(`/api/runs/${runId}`),
  applyChange: (id: string) =>
    request(`/api/file-changes/${id}/apply`, { method: "POST" }),
  rejectChange: (id: string) =>
    request(`/api/file-changes/${id}/reject`, { method: "POST" }),
  applyAll: (runId: string) =>
    request<{ results: Array<{ id: string; path: string; ok: boolean; detail: string }>; applied: number }>(
      `/api/runs/${runId}/file-changes/apply-all`,
      { method: "POST" }
    ),
  rejectAll: (runId: string) =>
    request<{ results: Array<{ id: string; path: string; ok: boolean; detail: string }>; applied: number }>(
      `/api/runs/${runId}/file-changes/reject-all`,
      { method: "POST" }
    ),
  generateImage: (prompt: string, n: number, size: string, model?: string) =>
    request<{ images: GeneratedImage[]; model: string | null }>("/api/images", {
      method: "POST",
      body: JSON.stringify({ prompt, n, size, model: model || null })
    }),
  readFiles: (paths: string[], allowedRoots: string[]) =>
    request<{ files: Array<{ path: string; content: string; truncated: boolean }> }>("/api/files/read", {
      method: "POST",
      body: JSON.stringify({ paths, allowed_roots: allowedRoots })
    })
};

export function runEventsUrl(runId: string): string {
  // EventSource can't set headers, so the optional API token rides as a query
  // param (only used when one is configured).
  const token = getApiToken();
  const suffix = token ? `?token=${encodeURIComponent(token)}` : "";
  return `/api/runs/${runId}/events${suffix}`;
}

export function modelKey(model: ProviderModel): string {
  return model.modelKey ?? model.model_key ?? "";
}

export function modelName(model: ProviderModel): string {
  return model.displayName ?? model.display_name ?? modelKey(model);
}

export function modelSize(model: ProviderModel): number {
  return model.sizeBytes ?? model.size_bytes ?? 0;
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 GB";
  return `${(bytes / 1024 ** 3).toFixed(bytes > 10 * 1024 ** 3 ? 1 : 2)} GB`;
}

/** "~4.2 GB" for estimates, "4.2 GB" for server-reported sizes, "" if unknown. */
export function modelSizeLabel(model: ProviderModel): string {
  const size = modelSize(model);
  if (!size) return "";
  return `${model.sizeIsEstimate ? "~" : ""}${formatBytes(size)}`;
}
