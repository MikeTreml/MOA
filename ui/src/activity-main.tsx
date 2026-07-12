import React, { useEffect } from "react";
import { createRoot } from "react-dom/client";
import { FlowGraph } from "./FlowGraph";
import { useRunActivity } from "./useRunActivity";
import "./styles.css";

document.documentElement.dataset.theme = localStorage.getItem("moa-theme") ?? "dark";

const STATUS_LABEL: Record<string, string> = {
  idle: "Idle",
  running: "Running",
  complete: "Complete",
  error: "Failed",
  stopped: "Stopped"
};

/** One agent's live feed in its own window (serena-style): bound to a step id,
 * titled after the agent, replaying history then tailing live. */
function AgentPopout({ runId, stepId, fallbackTitle }: { runId: string; stepId: string; fallbackTitle: string }) {
  const activity = useRunActivity(runId);
  const stage =
    activity.stages.find((s) => s.id === stepId) ??
    activity.record?.trace
      .filter((step) => step.id === stepId)
      .map((step) => ({
        id: step.id,
        stage: step.stage,
        title: step.title,
        model: step.model,
        status: "complete" as const,
        tokens: "",
        output: step.output
      }))[0] ??
    null;

  const title = stage?.title || fallbackTitle || "Agent";
  useEffect(() => {
    document.title = `${title} — MoA agent`;
  }, [title]);

  const body = stage ? stage.output ?? stage.tokens : "";
  return (
    <div className="popoutShell">
      <header className="popoutBar">
        <span className="brand">
          <span className="brandGlyph" aria-hidden="true" />
          {title}
        </span>
        {stage?.model && <code className="agentModel">{stage.model}</code>}
        <span className={`statusTag ${stage?.status ?? activity.status}`}>
          {stage
            ? stage.status === "running"
              ? "Active"
              : "Complete"
            : STATUS_LABEL[activity.status] ?? activity.status}
        </span>
      </header>
      <main className="popoutBody agentFeedBody">
        {!stage && activity.status === "running" && <p className="empty">Waiting for this agent to start…</p>}
        {!stage && activity.status !== "running" && <p className="empty">No feed recorded for this agent.</p>}
        {stage && <pre className="tokenStream">{body || "No output yet."}</pre>}
        {activity.error && (
          <div className="activityError">
            <strong>Run failed.</strong> {activity.error.detail}
          </div>
        )}
      </main>
    </div>
  );
}

function PopoutActivity() {
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run_id");
  const stepId = params.get("step_id");
  const fallbackTitle = params.get("title") ?? "";
  const activity = useRunActivity(stepId ? null : runId);

  if (runId && stepId) {
    return <AgentPopout runId={runId} stepId={stepId} fallbackTitle={fallbackTitle} />;
  }

  return (
    <div className="popoutShell">
      <header className="popoutBar">
        <span className="brand">
          <span className="brandGlyph" aria-hidden="true" />
          Flow activity
        </span>
        <span className="popoutHint">Stays open until you close it — closing never affects the run.</span>
      </header>
      <main className="popoutBody">
        {runId ? (
          <FlowGraph activity={activity} runId={runId} />
        ) : (
          <p className="empty">No run id provided.</p>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <PopoutActivity />
  </React.StrictMode>
);
