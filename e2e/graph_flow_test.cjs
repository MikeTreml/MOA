// End-to-end: a CUSTOM graph flow executes server-side and renders in the
// pop-out from its saved layout (lanes), with fan-out expanding to real nodes.
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/graph_flow_test.cjs
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

const GRAPH_PROFILE = {
  name: "GraphFlow",
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
      { id: "plan", kind: "llm", title: "Plan", prompt: "Break into angles as a JSON list: {input}", lane: 0, order: 0 },
      { id: "work", kind: "fanout", title: "Research", over: "plan", prompt: "Research: {item}", lane: 1, order: 0 },
      { id: "final", kind: "llm", title: "Synthesize", depends_on: ["work"], prompt: "Synthesize {{work}}", lane: 2, order: 0 }
    ],
    output: "final"
  }
};

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await (await browser.newContext()).newPage();

  // Save the graph flow, then run it (streamed) and grab the run id.
  await page.request.post(`${BASE}/api/profiles`, { data: GRAPH_PROFILE });
  const resp = await page.request.post(`${BASE}/api/flows/GraphFlow/run-stream`, { data: { prompt: "study bikes" } });
  const { run_id } = await resp.json();

  await page.goto(`${BASE}/activity.html?run_id=${run_id}`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".graphNode", { timeout: 15000 });

  // Wait for completion.
  await page.waitForFunction(
    () => {
      const el = document.querySelector(".finalOutput pre");
      return el && el.textContent;
    },
    { timeout: 25000 }
  );

  const nodeCount = await page.locator(".graphNode").count();
  const columns = await page.locator(".graphColumn").count();
  // Titles rendered on nodes (plan / research x3 / synthesize).
  const titles = await page.locator(".nodeTitle").allTextContents();
  await browser.close();

  console.log("graph nodes :", nodeCount);
  console.log("columns     :", columns);
  console.log("node titles :", titles.join(" | "));

  const hasPlan = titles.some((t) => /plan/i.test(t));
  const research = titles.filter((t) => /research/i.test(t)).length;
  const hasFinal = titles.some((t) => /synth/i.test(t));
  // plan (lane 0) + 3 research (lane 1, fanned out) + synthesize (lane 2)
  const ok = columns >= 3 && nodeCount >= 5 && hasPlan && research >= 2 && hasFinal;
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
