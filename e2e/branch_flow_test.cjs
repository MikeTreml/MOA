// End-to-end: a custom graph with CONDITIONAL routing. A router node emits a
// label; only the matching branch runs, and a join merges whichever ran.
// (Labels are chosen to match the fake's streamed output so one branch is taken
// and the other is skipped.)
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/branch_flow_test.cjs
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

const BRANCH_PROFILE = {
  name: "BranchFlow",
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
      { id: "router", kind: "llm", title: "Route", prompt: "Classify: {input}", lane: 0, order: 0 },
      // The fake streams "...fake upstream." so the "fake" branch is taken...
      { id: "taken", kind: "llm", title: "Taken", depends_on: ["router"], when_node: "router", when_equals: "fake", prompt: "Handle: {input}", lane: 1, order: 0 },
      // ...and the "zzz" branch is skipped.
      { id: "untaken", kind: "llm", title: "Untaken", depends_on: ["router"], when_node: "router", when_equals: "zzz", prompt: "Other: {input}", lane: 1, order: 1 },
      { id: "final", kind: "llm", title: "Deliver", depends_on: ["taken", "untaken"], prompt: "Deliver {{taken}}{{untaken}}", lane: 2, order: 0 }
    ],
    output: "final"
  }
};

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await (await browser.newContext()).newPage();

  await page.request.post(`${BASE}/api/profiles`, { data: BRANCH_PROFILE });
  const resp = await page.request.post(`${BASE}/api/flows/BranchFlow/run-stream`, { data: { prompt: "do a thing" } });
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

  console.log("node titles :", titles.join(" | "));
  const hasRouter = titles.some((t) => /route/i.test(t));
  const hasTaken = titles.some((t) => /taken/i.test(t) && !/untaken/i.test(t));
  const hasUntaken = titles.some((t) => /untaken/i.test(t));
  const hasFinal = titles.some((t) => /deliver/i.test(t));

  // Router + taken branch + final render; the untaken branch is skipped.
  const ok = hasRouter && hasTaken && hasFinal && !hasUntaken;
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
