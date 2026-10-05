// UI smoke test: open every page of the running app, wait until its data has loaded,
// fail on any browser error, and save a screenshot of each page (plus figure mode).
//
// Usage (with `xag ui` running):  npm run smoke  [-- --base http://localhost:8000]
// Screenshots go to web/screenshots/ (git-ignored).
import { chromium } from "playwright";
import { mkdirSync, readFileSync } from "node:fs";

const base = process.argv.includes("--base") ? process.argv[process.argv.indexOf("--base") + 1] : "http://localhost:8000";
const pages = [
  ["overview", "/"],
  ["runs", "/runs"],
  ["compare", "/compare"],
  ["calibration", "/calibration"],
  ["signals", "/signals"],
  ["folds", "/folds"],
  ["data", "/data"],
  ["graph", "/graph"],
];
mkdirSync("screenshots/exports", { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(`${page.url()}: ${m.text()}`));
page.on("pageerror", (e) => errors.push(`${page.url()}: ${e.message}`));

async function settle() {
  await page.waitForLoadState("networkidle");
  // Wait until no "Loading …" placeholders remain (data arrived) or 20 s pass.
  await page.waitForFunction(() => !/Loading [a-z]+…/.test(document.body.innerText), null, { timeout: 20000 });
  await page.waitForTimeout(600); // chart animations
}

for (const [name, path] of pages) {
  await page.goto(base + path);
  await settle();
  await page.screenshot({ path: `screenshots/${name}.png`, fullPage: true });
  console.log(`ok  ${path}`);
}
// Figure mode on the compare page, then back.
await page.goto(base + "/compare");
await settle();
await page.getByRole("button", { name: "Figure mode" }).click();
await page.waitForTimeout(600);
await page.screenshot({ path: "screenshots/compare-figure-mode.png", fullPage: true });
await page.getByRole("button", { name: "Figure mode" }).click();
console.log("ok  figure mode");

// Figure export, started from the dark console: the files must be real figures in the
// white print palette, and the page must stay dark afterwards.
function fail(msg) {
  errors.push(`export: ${msg}`);
}
for (const kind of ["SVG", "PNG"]) {
  const [dl] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: new RegExp(`^Export per-fold-.* as ${kind}$`) }).click(),
  ]);
  const file = `screenshots/exports/${dl.suggestedFilename()}`;
  await dl.saveAs(file);
  const buf = readFileSync(file);
  if (kind === "SVG") {
    const text = buf.toString("utf8");
    if (!text.startsWith("<svg")) fail("SVG does not start with <svg");
    if (!/width="800"/.test(text)) fail("SVG is not 800 px wide");
    if (!/fill="(#ffffff|#fff|white|rgb\(255,\s*255,\s*255\))"/i.test(text)) fail("SVG has no white background");
  } else {
    if (buf.readUInt32BE(0) !== 0x89504e47) fail("PNG signature missing");
    const width = buf.readUInt32BE(16);
    if (width !== 2400) fail(`PNG is ${width} px wide, expected 2400`);
  }
  console.log(`ok  export ${kind}: ${dl.suggestedFilename()} (${buf.length.toLocaleString("en")} bytes)`);
}
if (await page.evaluate(() => document.documentElement.classList.contains("fig"))) fail("page left in figure mode after export");

await browser.close();
if (errors.length) {
  console.error(`\n${errors.length} browser error(s):\n` + errors.join("\n"));
  process.exit(1);
}
console.log("\nall pages loaded without browser errors");
