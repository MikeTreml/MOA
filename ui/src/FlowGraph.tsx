import { useEffect, useMemo, useState } from "react";
import { Cpu, Layers3, Network, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import type { ActivityStage } from "./types";
import type { RunActivityState } from "./useRunActivity";

const STAGE_ICON: Record<string, typeof Cpu> = {
  orchestrator: Network,
  worker: Cpu,
  synthesizer: Layers3,
  evaluator: ShieldCheck,
  refiner: RefreshCw
};

// Left-to-right column each stage lives in; workers stack in one column.
const STAGE_COLUMN: Record<string, number> = {
  orchestrator: 0,
  worker: 1,
  synthesizer: 2,
  evaluator: 3,
  refiner: 4
};

function columnOf(stage: string): number {
  return STAGE_COLUMN[stage] ?? 1;
}

/** Re-render once a second while `active`, so running timers tick. */
function useTick(active: boolean): number {
  const [, setN] = useState(0);
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(() => setN((n) => n + 1), 1000);
    return () => window.clearInterval(t);
  }, [active]);
  return Date.now();
}

function elapsedMs(stage: ActivityStage, now: number): number | null {
  if (!stage.startedAt) return null;
  const start = new Date(stage.startedAt).getTime();
  const end = stage.endedAt ? new Date(stage.endedAt).getTime() : stage.status === "running" ? now : null;
  if (end == null || !Number.isFinite(start)) return null;
  return Math.max(0, end - start);
}

function formatMs(ms: number | null): string {
  if (ms == null) return "–";
  if (ms >= 60000) {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

// Streamed/known text -> rough token estimate (~4 chars/token) for a live stat.
function approxTokens(stage: ActivityStage): number {
  const text = stage.tokens || stage.output || "";
  return text ? Math.max(1, Math.round(text.length / 4)) : 0;
}

const STATUS_LABEL: Record<string, string> = {
  idle: "Idle",
  running: "Running",
  complete: "Complete",
  error: "Failed",
  stopped: "Stopped"
};

export function FlowGraph({ activity }: { activity: RunActivityState }) {
  const { stages, status, record, error, reconnecting } = activity;
  const running = status === "running";
  const now = useTick(running);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const activeCount = stages.filter((s) => s.status === "running").length;

  // Overall wall-clock: earliest start -> latest end (or now while running).
  const starts = stages.map((s) => (s.startedAt ? new Date(s.startedAt).getTime() : NaN)).filter(Number.isFinite);
  const ends = stages.map((s) => (s.endedAt ? new Date(s.endedAt).getTime() : NaN)).filter(Number.isFinite);
  const overall =
    starts.length > 0 ? (running ? now : Math.max(...ends, ...starts)) - Math.min(...starts) : null;

  // A custom-graph flow carries its own saved layout (lane/order); use it per
  // node when present so rendering is deterministic, falling back to the
  // known-shape column map for any node lacking a lane. Memoized on `stages` so
  // the once-a-second timer tick re-renders without re-bucketing/sorting.
  const hasRefiner = stages.some((s) => typeof s.lane !== "number" && s.stage === "refiner");
  const columns = useMemo(() => {
    const byColumn = new Map<number, ActivityStage[]>();
    stages.forEach((s) => {
      const c = typeof s.lane === "number" ? s.lane : columnOf(s.stage);
      if (!byColumn.has(c)) byColumn.set(c, []);
      byColumn.get(c)!.push(s);
    });
    return [...byColumn.keys()]
      .sort((a, b) => a - b)
      .map((c) => ({
        c,
        nodes: byColumn.get(c)!.slice().sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
      }));
  }, [stages]);

  const selected = stages.find((s) => s.id === selectedId) ?? null;

  return (
    <div className="flowGraph">
      <div className="graphStats">
        <span className={`statusTag ${status}`}>
          {running && <span className="liveDot" aria-hidden="true" />}
          {STATUS_LABEL[status] ?? status}
        </span>
        {reconnecting && <span className="reconnecting">Reconnecting…</span>}
        <span className="graphStat">⏱ {formatMs(overall)}</span>
        <span className="graphStat">◉ {activeCount} active</span>
        <span className="graphStat">{stages.length} agents</span>
      </div>

      {stages.length === 0 ? (
        <p className="empty">{running ? "Waiting for the first agent…" : "No activity."}</p>
      ) : (
        <div className="graphCanvas">
          {columns.map(({ c, nodes }, colIndex) => (
            <div className="graphColumn" key={c}>
              <div className="graphNodes">
                {nodes.map((stage) => {
                  const Icon = STAGE_ICON[stage.stage] ?? Sparkles;
                  const ms = elapsedMs(stage, now);
                  const tokens = approxTokens(stage);
                  return (
                    <button
                      key={stage.id}
                      className={`graphNode ${stage.status} ${selectedId === stage.id ? "selected" : ""}`}
                      onClick={() => setSelectedId(selectedId === stage.id ? null : stage.id)}
                      title="Click to view this agent's feed"
                    >
                      {stage.status === "running" && <span className="nodePulse" aria-hidden="true" />}
                      <span className="nodeIcon">
                        <Icon size={15} />
                      </span>
                      <span className="nodeTitle">{stage.title || stage.stage}</span>
                      {stage.model && <span className="nodeModel">{stage.model}</span>}
                      <span className="nodeStats">
                        <span>⏱ {formatMs(ms)}</span>
                        {tokens > 0 && <span title="approximate (chars ÷ 4)">≈{tokens.toLocaleString()} tok</span>}
                      </span>
                    </button>
                  );
                })}
              </div>
              {colIndex < columns.length - 1 && <div className="graphConnector" aria-hidden="true" />}
            </div>
          ))}
          {hasRefiner && <div className="graphLoopHint">↺ refiner loops back to evaluator</div>}
        </div>
      )}

      {selected && (
        <div className="nodeFeed">
          <div className="nodeFeedHead">
            <strong>{selected.title || selected.stage}</strong>
            <span className={`statusTag ${selected.status === "running" ? "running" : "complete"}`}>
              {selected.status === "running" ? "Active" : "Complete"}
            </span>
          </div>
          <pre>{selected.tokens || selected.output || "No output yet."}</pre>
        </div>
      )}

      {error && (
        <div className="activityError">
          <strong>Run failed.</strong> {error.detail}
        </div>
      )}

      {record && (
        <div className="finalOutput">
          <div className="finalHead">
            <strong>Final output</strong>
            <span className={`statusTag ${record.evaluation.status === "PASS" ? "complete" : "stopped"}`}>
              {record.evaluation.status} · {record.evaluation.score.toFixed(2)}
            </span>
          </div>
          <pre>{record.final_output}</pre>
        </div>
      )}
    </div>
  );
}
