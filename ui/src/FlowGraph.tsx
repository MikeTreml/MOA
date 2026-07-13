import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Cpu, ExternalLink, Layers3, Network, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import type { ActivityStage } from "./types";
import type { RunActivityState } from "./useRunActivity";

interface EdgePath {
  key: string;
  d: string;
  active: boolean;
}

export function agentPopoutUrl(runId: string, stage: ActivityStage): string {
  const title = encodeURIComponent(stage.title || stage.stage);
  return `activity.html?run_id=${encodeURIComponent(runId)}&step_id=${encodeURIComponent(stage.id)}&title=${title}`;
}

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

// Stages that stream tokens live. Only for these does "running with no output
// yet" mean the request is waiting (queued on the server, or a reasoning model
// still thinking) — orchestrator/evaluator/gates run in JSON mode and never
// stream, so token absence there is normal.
const STREAMING_STAGES = new Set(["worker", "synthesizer", "refiner"]);

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

export function FlowGraph({
  activity,
  runId
}: {
  activity: RunActivityState;
  /** When set, each node offers "open this agent in its own window". */
  runId?: string | null;
}) {
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

  // Real flow edges: each stage carries the step ids that feed it (emitted by
  // the backend), so we can draw who-feeds-whom instead of generic connectors.
  const edgePairs = useMemo(() => {
    const known = new Set(stages.map((s) => s.id));
    const pairs: Array<{ from: string; to: string; active: boolean }> = [];
    for (const stage of stages) {
      for (const dep of stage.deps ?? []) {
        if (known.has(dep)) pairs.push({ from: dep, to: stage.id, active: stage.status === "running" });
      }
    }
    return pairs;
  }, [stages]);

  const canvasRef = useRef<HTMLDivElement | null>(null);
  const nodeRefs = useRef(new Map<string, HTMLElement>());
  const [edges, setEdges] = useState<EdgePath[]>([]);
  const [, bumpLayout] = useState(0);

  useEffect(() => {
    const onResize = () => bumpLayout((n) => n + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Measure after every render (node heights shift as token counts stream in);
  // setEdges keeps the previous array identity when nothing moved, so this
  // cannot loop.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || edgePairs.length === 0) {
      setEdges((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    const cbox = canvas.getBoundingClientRect();
    const next: EdgePath[] = [];
    for (const pair of edgePairs) {
      const from = nodeRefs.current.get(pair.from);
      const to = nodeRefs.current.get(pair.to);
      if (!from || !to) continue;
      const fbox = from.getBoundingClientRect();
      const tbox = to.getBoundingClientRect();
      const sx = fbox.right - cbox.left + canvas.scrollLeft;
      const sy = fbox.top + fbox.height / 2 - cbox.top + canvas.scrollTop;
      const tx = tbox.left - cbox.left + canvas.scrollLeft;
      const ty = tbox.top + tbox.height / 2 - cbox.top + canvas.scrollTop;
      const bend = Math.max(24, (tx - sx) / 2);
      next.push({
        key: `${pair.from}->${pair.to}`,
        d: `M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`,
        active: pair.active
      });
    }
    setEdges((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next));
  });

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
        <span className="graphStat">{stages.length} steps</span>
      </div>

      {stages.length === 0 ? (
        <p className="empty">{running ? "Waiting for the first agent…" : "No activity."}</p>
      ) : (
        <div className="graphCanvas" ref={canvasRef}>
          {edges.length > 0 && (
            <svg className="graphEdges" aria-hidden="true">
              <defs>
                <marker id="edgeArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M0,0.5 L7.5,4 L0,7.5 Z" className="edgeArrowHead" />
                </marker>
                <marker id="edgeArrowActive" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M0,0.5 L7.5,4 L0,7.5 Z" className="edgeArrowHeadActive" />
                </marker>
              </defs>
              {edges.map((edge) => (
                <path
                  key={edge.key}
                  d={edge.d}
                  className={edge.active ? "edgeActive" : ""}
                  markerEnd={`url(#${edge.active ? "edgeArrowActive" : "edgeArrow"})`}
                />
              ))}
            </svg>
          )}
          {columns.map(({ c, nodes }, colIndex) => (
            <div className="graphColumn" key={c}>
              <div className="graphNodes">
                {nodes.map((stage) => {
                  const Icon = STAGE_ICON[stage.stage] ?? Sparkles;
                  const ms = elapsedMs(stage, now);
                  const tokens = approxTokens(stage);
                  const waiting =
                    stage.status === "running" &&
                    STREAMING_STAGES.has(stage.stage) &&
                    !stage.tokens &&
                    !stage.output;
                  return (
                    <button
                      key={stage.id}
                      ref={(el) => {
                        if (el) nodeRefs.current.set(stage.id, el);
                        else nodeRefs.current.delete(stage.id);
                      }}
                      className={`graphNode ${stage.status} ${waiting ? "waiting" : ""} ${selectedId === stage.id ? "selected" : ""}`}
                      onClick={() => setSelectedId(selectedId === stage.id ? null : stage.id)}
                      title="Click to view this agent's feed"
                    >
                      {stage.status === "running" && !waiting && <span className="nodePulse" aria-hidden="true" />}
                      <span className="nodeIcon">
                        <Icon size={15} />
                      </span>
                      <span className="nodeTitle">{stage.title || stage.stage}</span>
                      {stage.model && <span className="nodeModel">{stage.model}</span>}
                      <span className="nodeStats">
                        <span>⏱ {formatMs(ms)}</span>
                        {waiting && <span className="nodeWaiting">waiting…</span>}
                        {tokens > 0 && <span title="approximate (chars ÷ 4)">≈{tokens.toLocaleString()} tok</span>}
                        {runId && (
                          <span
                            role="button"
                            tabIndex={0}
                            className="nodePop"
                            title="Open this agent in its own window"
                            onClick={(event) => {
                              event.stopPropagation();
                              window.open(agentPopoutUrl(runId, stage), `moa-agent-${stage.id}`, "width=460,height=640,resizable=yes");
                            }}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.stopPropagation();
                                window.open(agentPopoutUrl(runId, stage), `moa-agent-${stage.id}`, "width=460,height=640,resizable=yes");
                              }
                            }}
                          >
                            <ExternalLink size={12} />
                          </span>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
              {colIndex < columns.length - 1 && edgePairs.length === 0 && (
                <div className="graphConnector" aria-hidden="true" />
              )}
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
