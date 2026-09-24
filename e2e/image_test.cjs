// Real-browser verification of the Images tab (text-to-image).
//
// Prereqs: `cd ui && npm run build`, then `python e2e/serve_stack.py` running.
// Run:     NODE_PATH=ui/node_modules node e2e/image_test.cjs
//
// Opens the Images tab, generates 2 images against the fake upstream, and
// asserts real image elements render with data-URI sources and working download
// buttons. Downloads are intentionally created through Blob URLs at click time
// so large data URIs do not exceed browser href limits.
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await (await browser.newContext()).newPage();

  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".promptBox", { timeout: 15000 });

  // Go to the Images tab.
  await page.getByRole("button", { name: "Images" }).click();
  await page.waitForSelector('textarea.promptBox', { timeout: 5000 });

  // Ask for 2 images and generate.
  await page.locator('.controlRow input[type="number"]').fill("2");
  await page.getByRole("button", { name: /Generate/ }).click();

  // Wait for rendered images.
  await page.waitForSelector(".imageCard img", { timeout: 20000 });
  const count = await page.locator(".imageCard img").count();
  const firstSrc = await page.locator(".imageCard img").first().getAttribute("src");
  const downloadButtons = page.getByRole("button", { name: /Download/ });
  const downloads = await downloadButtons.count();
  const download = page.waitForEvent("download");
  await downloadButtons.first().click();
  const suggestedFilename = (await download).suggestedFilename();
  await browser.close();

  console.log("images rendered :", count);
  console.log("first src prefix:", (firstSrc || "").slice(0, 24));
  console.log("download buttons:", downloads);
  console.log("download filename:", suggestedFilename);

  const ok = count === 2 && (firstSrc || "").startsWith("data:image/png;base64,")
    && downloads === 2 && suggestedFilename === "moa-image-1.png";
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("ERROR:", e.message);
  process.exit(1);
});
