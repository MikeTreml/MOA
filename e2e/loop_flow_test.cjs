// End-to-end: a custom graph with a GATE that loops. The gate scores the draft
// on settable terms; the fake fails twice then passes, so the pop-out shows the
// refine node and gate re-running across iterations before completing.
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/loop_flow_test.cjs
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

const LOOP_PROFILE = {
  name: "LoopFlow",
  provider: "openai-compatible",
  base_url: "http://127.0.0.1:8773/v1",
  memory_cap_gb: 80,
  workflow: "graph",
  worker_models: ["m"],
  aggregator_model: "m",
  evaluator_model: "m",
  image_model: "",
  max_iterations: 1,
  allowed_roots: ["."],
  graph: {
    nodes: [
      { id: "draft", kind: "llm", title: "Draft", prompt: "Draft an answer for: {input}", lane: 0, order: 0 },
      { id: "refine", kind: "llm", title: "Revise", depends_on: ["draft"], prompt: "revise using {{gate}}: {{draft}}", lane: 1, order: 0 },
      {
        id: "gate", kind: "gate", title: "Check", depends_on: ["refine"], loop_to: "refine", max_loops: 4,
        checks: [{ term: "scope", min: 0.7 }, { term: "direction", min: 0.7 }], lane: 2, order: 0
      }
    ],
    output: "refine"
  }
};

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await (await browser.newContext()).newPage();

  await page.request.post(`${BASE}/api/profiles`, { data: LOOP_PROFILE });
  const resp = await page.request.post(`${BASE}/api/flows/LoopFlow/run-stream`, { data: { prompt: "explain X" } });
  const { run_id } = await resp.json();

  await page.goto(`${BASE}/activity.html?run_id=${run_id}`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".graphNode", { timeout: 15000 });
  await page.waitForFunction(
    () => {
      const el = document.querySelector(".finalOutput pre");
      return el && el.textContent;
    },
    { timeout: 25000 }
  );

  const titles = await page.locator(".nodeTitle").allTextContents();
  await browser.close();

  const revises = titles.filter((t) => /revise/i.test(t)).length;
  const checks = titles.filter((t) => /check/i.test(t)).length;
  console.log("node titles :", titles.join(" | "));
  console.log("revise nodes:", revises, "| check nodes:", checks);

  // Fails twice then passes -> refine and gate each appear across >= 2 iterations.
  const ok = revises >= 2 && checks >= 2;
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
