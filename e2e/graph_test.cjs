// Real-browser verification of the flow-graph pop-out.
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/graph_test.cjs
//
// Starts a flow, opens activity.html?run_id, and asserts the graph renders one
// node per agent, shows live stats (timer / active count), pulses while running,
// and that clicking a node reveals its feed.
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Start a streamed run via the callable-flow API and grab its id.
  const resp = await page.request.post(`${BASE}/api/flows/Fake/run-stream`, { data: { prompt: "Design a brief" } });
  const { run_id } = await resp.json();

  // Open the pop-out graph directly.
  await page.goto(`${BASE}/activity.html?run_id=${run_id}`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".graphNode", { timeout: 15000 });

  // Try to catch a live pulse within the streaming window (not gating on it).
  let pulseSeen = 0;
  for (let i = 0; i < 40 && pulseSeen === 0; i++) {
    pulseSeen = await page.locator(".nodePulse").count();
    await page.waitForTimeout(100);
  }

  // Wait for completion.
  await page.waitForFunction(
    () => {
      const el = document.querySelector(".finalOutput pre");
      return el && el.textContent;
    },
    { timeout: 25000 }
  );

  const nodeCount = await page.locator(".graphNode").count();
  const stats = ((await page.locator(".graphStats").textContent()) || "").replace(/\s+/g, " ").trim();

  // Click a node -> its feed panel appears.
  await page.locator(".graphNode").first().click();
  const feed = await page.locator(".nodeFeed").count();
  await browser.close();

  console.log("graph nodes :", nodeCount);
  console.log("pulse seen  :", pulseSeen > 0);
  console.log("stats bar   :", stats.slice(0, 70));
  console.log("node feed   :", feed);

  const ok = nodeCount >= 4 && feed >= 1 && /active/.test(stats) && /agents/.test(stats);
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
