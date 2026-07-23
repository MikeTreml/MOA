// Browser smoke test for every primary navigation control and tab scroll state.
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8099";
const TABS = ["Run", "Images", "Models", "Files", "Profiles", "History"];

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await (await browser.newContext({ viewport: { width: 390, height: 844 } })).newPage();
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".promptBox", { timeout: 15000 });

  const reached = [];
  const navigation = page.getByRole("navigation");
  for (const tab of TABS) {
    await navigation.getByRole("button", { name: tab, exact: true }).click();
    await page.getByRole("heading", { name: tab, level: 1 }).waitFor();
    reached.push(tab);
  }

  // Profiles is long on a phone-sized viewport. A tab switch must not retain
  // its scroll offset and strand the next screen halfway down the document.
  await navigation.getByRole("button", { name: "Profiles", exact: true }).click();
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  const before = await page.evaluate(() => window.scrollY);
  await navigation.getByRole("button", { name: "Run", exact: true }).click();
  await page.waitForFunction(() => window.scrollY === 0);
  const after = await page.evaluate(() => window.scrollY);

  await browser.close();
  console.log("tabs reached :", reached.join(" | "));
  console.log("scroll reset :", before, "->", after);
  console.log("console errors:", errors.length);
  const ok = reached.length === TABS.length && before > 0 && after === 0 && errors.length === 0;
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((error) => {
  console.error("ERROR:", error.message);
  process.exit(1);
});
