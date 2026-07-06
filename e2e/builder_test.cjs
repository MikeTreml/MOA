// Real-browser verification of the visual graph builder.
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/builder_test.cjs
//
// Seeds a graph flow (so it's the active profile), opens the Profiles tab, and
// checks the visual builder renders one draggable node per graph node, that
// clicking a node opens its editor, and that "Add node" adds one.
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

const GRAPH_PROFILE = {
  name: "BuilderFlow",
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
      { id: "plan", kind: "llm", title: "Plan", prompt: "list: {input}", lane: 0, order: 0 },
      { id: "work", kind: "fanout", title: "Work", over: "plan", prompt: "do {item}", lane: 1, order: 0 },
      { id: "final", kind: "llm", title: "Final", depends_on: ["work"], prompt: "join {{work}}", lane: 2, order: 0 }
    ],
    output: "final"
  }
};

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await (await browser.newContext()).newPage();

  // Save it so it becomes the active profile.
  await page.request.post(`${BASE}/api/profiles`, { data: GRAPH_PROFILE });

  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".promptBox", { timeout: 15000 });
  await page.getByRole("button", { name: "Profiles" }).click();

  await page.waitForSelector(".builderNode", { timeout: 8000 });
  const initial = await page.locator(".builderNode").count();

  // Click a node -> editor opens.
  await page.locator(".builderNode").first().click();
  const editorOpened = (await page.locator(".nodeEditor").count()) > 0;

  // Add a node -> count grows.
  await page.getByRole("button", { name: "Add node" }).click();
  const afterAdd = await page.locator(".builderNode").count();

  // Special edges (fanout "over") render as dashed lines.
  const overEdges = await page.locator(".builderEdges .edge.over").count();

  // Drag a node and confirm it actually repositions (lane/order snap).
  const planNode = page.locator(".builderNode:has(.bnId:text-is('plan'))");
  const before = await planNode.evaluate((el) => el.style.left);
  const box = await planNode.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 430, box.y + box.height / 2 + 130, { steps: 6 });
  await page.mouse.up();
  const after = await planNode.evaluate((el) => el.style.left);
  const dragMoved = before !== after;

  // Rename the fanout source 'plan' -> 'planX'; the 'over' edge must survive
  // (cascade), i.e. references are NOT orphaned.
  await planNode.click();
  const idInput = page.locator(".nodeEditor input").first();
  await idInput.fill("planX");
  await idInput.press("Enter");
  await page.waitForTimeout(150);
  const overEdgesAfterRename = await page.locator(".builderEdges .edge.over").count();
  const renamedNode = await page.locator(".builderNode:has(.bnId:text-is('planX'))").count();

  await browser.close();

  console.log("initial nodes :", initial);
  console.log("editor opened :", editorOpened);
  console.log("after add     :", afterAdd);
  console.log("over edges    :", overEdges);
  console.log("drag moved    :", dragMoved);
  console.log("renamed node  :", renamedNode, "| over edge kept:", overEdgesAfterRename);

  const ok =
    initial === 3 && editorOpened && afterAdd === 4 && overEdges >= 1 &&
    dragMoved && renamedNode === 1 && overEdgesAfterRename >= 1;
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
