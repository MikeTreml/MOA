import React from "react";
import { createRoot } from "react-dom/client";
import { FlowGraph } from "./FlowGraph";
import { useRunActivity } from "./useRunActivity";
import "./styles.css";

document.documentElement.dataset.theme = localStorage.getItem("moa-theme") ?? "dark";

function PopoutActivity() {
  const runId = new URLSearchParams(window.location.search).get("run_id");
  const activity = useRunActivity(runId);

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
          <FlowGraph activity={activity} />
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
