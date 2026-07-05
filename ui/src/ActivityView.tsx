import type { ReactNode } from "react";
import { Copy, Cpu, Layers3, Network, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import type { RunActivityState } from "./useRunActivity";

const STAGE_ICON: Record<string, typeof Cpu> = {
  orchestrator: Network,
  worker: Cpu,
  synthesizer: Layers3,
  evaluator: ShieldCheck,
  refiner: RefreshCw
};

function StageIcon({ stage }: { stage: string }) {
  const Icon = STAGE_ICON[stage] ?? Sparkles;
  return <Icon size={15} />;
}

const STATUS_LABEL: Record<string, string> = {
  idle: "Idle",
  running: "Running",
  complete: "Complete",
  error: "Failed",
  stopped: "Stopped"
};

function formatDuration(startedAt?: string, endedAt?: string): string | null {
  if (!startedAt || !endedAt) return null;
  const ms = new Date(endedAt).getTime() - new Date(startedAt).getTime();
  if (!Number.isFinite(ms) || ms < 0) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export function ActivityView({
  activity,
  actions
}: {
  activity: RunActivityState;
  actions?: ReactNode;
}) {
  const { stages, record, status, error, reconnecting } = activity;

  return (
    <div className="activity">
      <div className="activityHead">
        <span className={`statusTag ${status}`}>
          {status === "running" && <span className="liveDot" aria-hidden="true" />}
          {STATUS_LABEL[status] ?? status}
        </span>
        {reconnecting && <span className="reconnecting">Reconnecting…</span>}
        {actions}
      </div>

      {stages.length > 0 && (
        <div className="pipeline">
          {stages.map((stage) => (
            <div className={`pipeNode ${stage.status}`} key={`pipe-${stage.id}`}>
              <span className="pipeIcon">
                <StageIcon stage={stage.stage} />
              </span>
              <span className="stageLabel">{stage.stage}</span>
            </div>
          ))}
        </div>
      )}

      {stages.map((stage) => {
        const body = stage.output ?? stage.tokens;
        const duration = formatDuration(stage.startedAt, stage.endedAt);
        const chars = body ? body.length : 0;
        return (
          <div className={`stageCard ${stage.status}`} key={stage.id}>
            <div className="stageHead">
              <span className="stageLabel">{stage.stage}</span>
              <strong>{stage.title}</strong>
              {stage.model && <code>{stage.model}</code>}
              {stage.status === "running" && <span className="liveDot" aria-hidden="true" />}
              {(duration || chars > 0) && (
                <span className="stageMeta">
                  {duration && <span>{duration}</span>}
                  {chars > 0 && <span>{chars.toLocaleString()} chars</span>}
                </span>
              )}
            </div>
            {body && <pre className="tokenStream">{body}</pre>}
          </div>
        );
      })}

      {stages.length === 0 && status === "running" && (
        <p className="empty">Waiting for the first agent to report…</p>
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
            {record.final_output && (
              <button
                className="iconButton"
                title="Copy final output"
                onClick={() => navigator.clipboard?.writeText(record.final_output).catch(() => undefined)}
              >
                <Copy size={15} />
              </button>
            )}
          </div>
          <pre>{record.final_output}</pre>
          {record.evaluation.feedback && <p className="feedback">{record.evaluation.feedback}</p>}
        </div>
      )}
    </div>
  );
}
