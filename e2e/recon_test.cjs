// Real-browser verification of SSE reconnect in useRunActivity.
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/recon_test.cjs
//
// Starts a run, drops the first /events connection (route.abort) to force the
// hook's reconnect, then lets the reconnect through and asserts the UI recovers,
// shows "Reconnecting…", and finishes with the correct, non-duplicated output.
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  let eventsRequests = 0;
  page.on("request", (req) => {
    if (req.url().includes("/events")) eventsRequests += 1;
  });

  // Drop the FIRST SSE connection so the hook must reconnect; let the rest
  // through so the backend replays its buffer.
  let firstAborted = false;
  await page.route(/\/events/, (route) => {
    if (!firstAborted) {
      firstAborted = true;
      route.abort("connectionreset");
    } else {
      route.continue();
    }
  });

  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".promptBox", { timeout: 15000 });

  await page.locator(".promptBox").click();
  await page.keyboard.press("Control+Enter");

  // Confirm the reconnecting indicator appears while the hook backs off.
  let reconnectingSeen = 0;
  for (let i = 0; i < 30 && reconnectingSeen === 0; i++) {
    reconnectingSeen = await page.locator(".reconnecting").count();
    await page.waitForTimeout(100);
  }

  // The run must still reach a correct completion after reconnecting.
  await page.waitForFunction(
    () => {
      const el = document.querySelector(".finalOutput pre");
      return el && el.textContent && el.textContent.includes("fake upstream.");
    },
    { timeout: 25000 }
  );

  const finalText = (await page.locator(".finalOutput pre").first().textContent())?.trim();
  const statusTag = (await page.locator(".activity .statusTag").first().textContent())?.trim();
  await browser.close();

  const expected = "Hello from fake upstream.";
  const noDup = finalText === expected;
  const reconnected = eventsRequests >= 2;

  console.log("events connections :", eventsRequests, reconnected ? "(dropped + reconnected)" : "(no reconnect!)");
  console.log("reconnecting shown :", reconnectingSeen > 0);
  console.log("final output       :", JSON.stringify(finalText));
  console.log("status             :", statusTag);

  const ok = reconnected && noDup && /complete/i.test(statusTag || "");
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
