import { useEffect, useRef, useState } from "react";
import { api, runEventsUrl } from "./api";
import type { ActivityStage, RunRecord, RunStatus } from "./types";

export interface RunActivityState {
  stages: ActivityStage[];
  record: RunRecord | null;
  status: RunStatus;
  error: { status: number; detail: string } | null;
  reconnecting: boolean;
}

const IDLE: RunActivityState = {
  stages: [],
  record: null,
  status: "idle",
  error: null,
  reconnecting: false
};

// After this many consecutive failed reconnects we stop trying and fall back to
// the persisted record, instead of spinning forever against a dead backend.
const MAX_RECONNECTS = 6;

function safeParse<T>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    // A truncated/malformed event must not throw inside a listener (which would
    // freeze the stage it was updating). Drop it and keep streaming.
    return null;
  }
}

/**
 * Subscribe to a run's server-sent activity. Purely an observer: mounting,
 * unmounting, or losing the connection never affects the run on the server.
 *
 * The backend replays the whole event buffer on every (re)connection, so a
 * transient network drop self-heals: we reset the stage list on each open and
 * let the replay rebuild it deterministically — no duplicated tokens. Only
 * after MAX_RECONNECTS consecutive failures do we fall back to the persisted
 * record (or surface a lost-connection error if it isn't there yet).
 */
export function useRunActivity(
  runId: string | null,
  onComplete?: (record: RunRecord) => void
): RunActivityState {
  const [state, setState] = useState<RunActivityState>(IDLE);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (!runId) {
      setState(IDLE);
      return;
    }
    setState({ stages: [], record: null, status: "running", error: null, reconnecting: false });

    let ended = false;
    let reconnects = 0;
    let source: EventSource | null = null;

    const upsert = (id: string, patch: Partial<ActivityStage>) =>
      setState((prev) => {
        const index = prev.stages.findIndex((stage) => stage.id === id);
        const stages = [...prev.stages];
        if (index === -1) {
          stages.push({ id, stage: "", title: "", status: "running", tokens: "", ...patch });
        } else {
          stages[index] = { ...stages[index], ...patch };
        }
        return { ...prev, stages };
      });

    const fallbackToPersisted = () => {
      if (ended) return;
      ended = true;
      source?.close();
      api
        .getRun(runId)
        .then((record) =>
          setState((prev) => ({
            ...prev,
            record,
            reconnecting: false,
            status: prev.status === "running" ? "complete" : prev.status
          }))
        )
        .catch(() =>
          setState((prev) =>
            prev.status === "running"
              ? {
                  ...prev,
                  reconnecting: false,
                  status: "error",
                  error: { status: 0, detail: "Lost connection to the run." }
                }
              : { ...prev, reconnecting: false }
          )
        );
    };

    const connect = () => {
      source = new EventSource(runEventsUrl(runId));

      source.addEventListener("open", () => {
        reconnects = 0;
        // A fresh connection replays from the start, so clear the stage list and
        // let the replay rebuild it — prevents double-appended tokens.
        setState((prev) =>
          prev.status === "running"
            ? { ...prev, stages: [], reconnecting: false }
            : { ...prev, reconnecting: false }
        );
      });

      source.addEventListener("stage_start", (event) => {
        const data = safeParse<any>((event as MessageEvent).data);
        if (!data) return;
        // Stamp the start client-side: the server's started_at only arrives with
        // the terminal step event, but a live timer needs a start now.
        upsert(data.id, {
          stage: data.stage,
          title: data.title,
          model: data.model,
          status: "running",
          tokens: "",
          startedAt: new Date().toISOString(),
          // Present only for custom-graph flows — the authored layout.
          lane: data.lane,
          order: data.order,
          ...(Array.isArray(data.deps) ? { deps: data.deps as string[] } : {})
        });
      });

      source.addEventListener("token", (event) => {
        const data = safeParse<any>((event as MessageEvent).data);
        if (!data) return;
        setState((prev) => {
          const index = prev.stages.findIndex((stage) => stage.id === data.id);
          if (index === -1) return prev;
          const stages = [...prev.stages];
          stages[index] = { ...stages[index], tokens: stages[index].tokens + data.text };
          return { ...prev, stages };
        });
      });

      source.addEventListener("step", (event) => {
        const data = safeParse<any>((event as MessageEvent).data);
        if (!data) return;
        // Prefer the server's authoritative started_at/ended_at for a completed
        // step: they're stable, so a reconnect that replays this event restores
        // the real duration instead of the client re-stamping it to "now". The
        // client stage_start stamp only drove the live timer before completion.
        upsert(data.id, {
          stage: data.stage,
          title: data.title,
          model: data.model,
          status: "complete",
          output: data.output,
          ...(data.started_at ? { startedAt: data.started_at } : {}),
          ...(Array.isArray(data.metadata?.deps) ? { deps: data.metadata.deps as string[] } : {}),
          endedAt: data.ended_at ?? new Date().toISOString()
        });
      });

      source.addEventListener("complete", (event) => {
        const record = safeParse<RunRecord>((event as MessageEvent).data);
        if (!record) return;
        setState((prev) => ({ ...prev, record, status: "complete", reconnecting: false }));
        onCompleteRef.current?.(record);
      });

      source.addEventListener("failed", (event) => {
        const data = safeParse<{ status: number; detail: string }>((event as MessageEvent).data);
        setState((prev) => ({ ...prev, status: "error", error: data, reconnecting: false }));
      });

      source.addEventListener("stopped", () => {
        setState((prev) => ({ ...prev, status: "stopped", reconnecting: false }));
      });

      source.addEventListener("end", () => {
        ended = true;
        source?.close();
      });

      // Transport error. EventSource sets readyState to CONNECTING while it plans
      // to retry, or CLOSED when it has given up. We drive reconnection manually
      // so we can cap attempts and surface a "reconnecting" state to the user.
      source.addEventListener("error", () => {
        if (ended) return;
        source?.close();
        reconnects += 1;
        if (reconnects > MAX_RECONNECTS) {
          fallbackToPersisted();
          return;
        }
        setState((prev) => (prev.status === "running" ? { ...prev, reconnecting: true } : prev));
        const delay = Math.min(1000 * reconnects, 5000);
        window.setTimeout(() => {
          if (!ended) connect();
        }, delay);
      });
    };

    connect();

    return () => {
      ended = true;
      source?.close();
    };
  }, [runId]);

  return state;
}
