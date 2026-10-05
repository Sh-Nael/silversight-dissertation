// The figure pack: every chart the report may use, exported from the running app
// in the white print palette as SVG (vector) and PNG (2400 px wide), with an index.
//
// Usage (with `xag ui` running and a current build):  npm run figures
//   [-- --base http://localhost:8000] [-- --out ../experiments/figures] [-- --only F07,F08]
//
// Each entry of FIGURES names a page, the models to show (the `?models=` filter), the
// controls to set, and the chart to export. The script drives the same buttons a person
// would use, so a figure here is exactly what the app shows. Files get stable names
// (no date), so a re-run replaces the pack in place.
import { chromium } from "playwright";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const arg = (name, fallback) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : fallback);
const here = dirname(fileURLToPath(import.meta.url));
const base = arg("--base", "http://localhost:8000");
const out = resolve(arg("--out", resolve(here, "../../experiments/figures")));
const only = arg("--only", "").split(",").filter(Boolean);

const LADDER = "ar_garch,linear,gbm,lstm,gru,static_gnn,aimdg";
const NEURAL = "linear,gbm,lstm,gru,static_gnn,aimdg";
const PAIR = "static_gnn,aimdg";
const CORE = "linear,gbm,static_gnn,aimdg";
const ENS = "linear,gbm,static_gnn,ens_lin_gbm,ens_lin_gbm_sgnn";
const TRADED = "linear,gbm,static_gnn,aimdg,ens_lin_gbm_sgnn";
const ABL = "static_gnn,aimdg,aimdg_A1,aimdg_A2";

// Control steps: ["seg", group label, button label] clicks a segmented switch (the n-th
// group with that label when a 4th item is given); ["select", element id, option value or
// label]; ["month", "YYYY-MM"] picks the first selection month on or after that month.
const h = (d) => ["seg", "Horizon", d === 1 ? "1 day" : "5 days"];
const loss = (n, l) => ["seg", "Loss", l, n]; // Compare: 0 significance, 1 per fold, 2 cumulative
const ref = (m) => ["select", "reference", m];

const FIGURES = [
  { id: "F01", name: "silver-price-regimes", page: "/", chart: "silver-price-regimes", chapter: "1, 3",
    title: "Silver weekly close with the four regime windows and the start of each test fold" },
  { id: "F02", name: "walk-forward-calendar", page: "/folds", chart: "walk-forward-timeline", chapter: "3",
    title: "The frozen walk-forward calendar: training spans, test blocks and tuning points" },
  { id: "F03", name: "gold-silver-ect", page: "/data", chart: "gold-silver-ect", chapter: "3",
    title: "Gold–silver error-correction term (z-score of the log ratio)" },

  { id: "F04", name: "per-fold-qlike-1d-vs-static-gnn", page: "/compare", models: NEURAL, chapter: "4",
    steps: [ref("static_gnn"), h(1), loss(1, "QLIKE")], chart: /^per-fold-qlike-h1-vs-static_gnn$/,
    title: "QLIKE per fold, 1 day: each model minus the static graph (below zero: better)" },
  { id: "F05", name: "cumulative-qlike-1d-vs-static-gnn", page: "/compare", models: NEURAL, chapter: "4",
    steps: [ref("static_gnn"), h(1), loss(2, "QLIKE")], chart: /^cumulative-qlike-h1-vs-static_gnn$/,
    title: "Cumulative QLIKE difference against the static graph, 1 day" },
  { id: "F06", name: "per-fold-qlike-1d-aimdg-vs-static-gnn", page: "/compare", models: PAIR, chapter: "4",
    steps: [ref("static_gnn"), h(1), loss(1, "QLIKE")], chart: /^per-fold-qlike-h1-vs-static_gnn$/,
    title: "AIM-DG minus the static graph, QLIKE per fold, 1 day" },
  { id: "F07", name: "cumulative-qlike-1d-aimdg-vs-static-gnn", page: "/compare", models: PAIR, chapter: "4",
    steps: [ref("static_gnn"), h(1), loss(2, "QLIKE")], chart: /^cumulative-qlike-h1-vs-static_gnn$/,
    title: "AIM-DG minus the static graph, cumulative QLIKE difference, 1 day" },
  { id: "F08", name: "cumulative-qlike-5d-aimdg-vs-static-gnn", page: "/compare", models: PAIR, chapter: "4",
    steps: [ref("static_gnn"), h(5), loss(2, "QLIKE")], chart: /^cumulative-qlike-h5-vs-static_gnn$/,
    title: "AIM-DG minus the static graph, cumulative QLIKE difference, 5 days" },
  { id: "F09", name: "per-fold-brier-1d-vs-climatology", page: "/compare", models: `climatology,${LADDER}`, chapter: "4",
    steps: [ref("climatology"), h(1), loss(1, "Brier")], chart: /^per-fold-brier-h1-vs-climatology$/,
    title: "Brier score per fold, 1 day: each model minus climatology" },
  { id: "F10", name: "cumulative-brier-1d-vs-climatology", page: "/compare", models: `climatology,${LADDER}`, chapter: "4",
    steps: [ref("climatology"), h(1), loss(2, "Brier")], chart: /^cumulative-brier-h1-vs-climatology$/,
    title: "Cumulative Brier difference against climatology, 1 day" },
  { id: "F11", name: "cumulative-brier-5d-vs-climatology", page: "/compare", models: `climatology,${LADDER}`, chapter: "4",
    steps: [ref("climatology"), h(5), loss(2, "Brier")], chart: /^cumulative-brier-h5-vs-climatology$/,
    title: "Cumulative Brier difference against climatology, 5 days" },
  { id: "F12", name: "cumulative-qlike-1d-ensembles-vs-gbm", page: "/compare", models: ENS, chapter: "4",
    steps: [ref("gbm"), h(1), loss(2, "QLIKE")], chart: /^cumulative-qlike-h1-vs-gbm$/,
    title: "Ensembles and their members, cumulative QLIKE difference against gradient boosting, 1 day" },
  { id: "F13", name: "cumulative-qlike-1d-ablations-vs-aimdg", page: "/compare", models: ABL, chapter: "4",
    steps: [ref("aimdg"), h(1), loss(2, "QLIKE")], chart: /^cumulative-qlike-h1-vs-aimdg$/,
    title: "Ablations A1 and A2 and the static graph, cumulative QLIKE difference against AIM-DG, 1 day" },

  { id: "F14", name: "reliability-1d", page: "/calibration", models: CORE, chapter: "4",
    steps: [h(1)], chart: "reliability-h1", title: "Reliability diagram, 1-day direction" },
  { id: "F15", name: "reliability-5d", page: "/calibration", models: CORE, chapter: "4",
    steps: [h(5)], chart: "reliability-h5", title: "Reliability diagram, 5-day direction" },
  { id: "F16", name: "volatility-static-gnn-1d", page: "/calibration", models: CORE, chapter: "4",
    steps: [h(1), ["select", "vol-model", "static_gnn"]], chart: "volatility-static_gnn-h1",
    title: "Static graph: forecast against realised volatility, 1 day (annualised, weekly means)" },
  { id: "F17", name: "volatility-aimdg-1d", page: "/calibration", models: CORE, chapter: "4",
    steps: [h(1), ["select", "vol-model", "aimdg"]], chart: "volatility-aimdg-h1",
    title: "AIM-DG: forecast against realised volatility, 1 day (annualised, weekly means)" },
  { id: "F18", name: "coverage-1d-both", page: "/calibration", models: LADDER, chapter: "4",
    steps: [h(1), ["seg", "Selection", "causal"], ["seg", "Calls", "both"]], chart: "coverage-causal-h1",
    title: "Hit rate against coverage, 1 day, all calls (causal confidence gate)" },
  { id: "F19", name: "coverage-1d-buy", page: "/calibration", models: LADDER, chapter: "4",
    steps: [h(1), ["seg", "Selection", "causal"], ["seg", "Calls", "up (buy)"]], chart: "coverage-causal-h1",
    title: "Share of up (buy) calls that were right against coverage, 1 day" },
  { id: "F20", name: "coverage-1d-sell", page: "/calibration", models: LADDER, chapter: "4",
    steps: [h(1), ["seg", "Selection", "causal"], ["seg", "Calls", "down (sell)"]], chart: "coverage-causal-h1",
    title: "Share of down (sell) calls that were right against coverage, 1 day" },
  { id: "F21", name: "coverage-5d-both", page: "/calibration", models: LADDER, chapter: "4",
    steps: [h(5), ["seg", "Selection", "causal"], ["seg", "Calls", "both"]], chart: "coverage-causal-h5",
    title: "Hit rate against coverage, 5 days, all calls (causal confidence gate)" },

  ...["both", "buy", "sell"].map((side, i) => ({
    id: `F${22 + i}`, name: `equity-1d-top10-${side}`, page: "/signals", models: TRADED, chapter: "4",
    steps: [h(1), ["select", "coverage", "0.1"], ["seg", "Trades", side]], chart: `equity-${side}-h1-c0.1`,
    title: `Account equity of the signal and risk layer, 1-day ${side === "both" ? "buy and sell" : side} trades on the 10% most confident days`,
  })),
  { id: "F25", name: "equity-1d-all-days", page: "/signals", models: TRADED, chapter: "4",
    steps: [h(1), ["select", "coverage", "1"], ["seg", "Trades", "both"]], chart: "equity-both-h1-c1",
    title: "Account equity when trading every day, 1-day trades (the gate switched off)" },

  { id: "F26", name: "edge-timeline-active", page: "/graph", chapter: "5",
    steps: [["seg", "Show", "active"]], chart: "edge-timeline-on",
    title: "Which of the 15 edges AIM-DG kept at each monthly selection, with the regime windows" },
  { id: "F27", name: "edge-timeline-reliability", page: "/graph", chapter: "5",
    steps: [["seg", "Show", "reliability"]], chart: "edge-timeline-reliability",
    title: "Reliability of each edge at each monthly selection" },
  { id: "F28", name: "cluster-activity-stress", page: "/graph", chapter: "5", chart: "cluster-activity-stress",
    title: "Active share of each cluster's edges in stress months against the rest" },
  ...[["2011-04", "the 2011 silver spike"], ["2013-06", "the 2013 taper tantrum"], ["2020-03", "the 2020 COVID shock"],
      ["2022-09", "the 2022 rate shock"], ["2026-06", "the last selection of the test period"]].map(([m, what], i) => ({
    id: `F${29 + i}`, name: `driver-graph-${m}`, page: "/graph", chapter: "5",
    steps: [["month", m]], chart: /^driver-graph-/, title: `The driver graph AIM-DG selected in ${what} (${m})`,
  })),
];

mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(`${page.url()}: ${m.text()}`));
page.on("pageerror", (e) => errors.push(`${page.url()}: ${e.message}`));

async function settle() {
  await page.waitForLoadState("networkidle");
  // Pages keep showing the previous data while new data loads, so "no Loading text" is not
  // enough: wait until the app reports no loads in flight (lib/api.ts, trackLoad), twice,
  // with a pause between, so that a load started by the last click is also seen.
  for (let i = 0; i < 2; i++) {
    await page.waitForTimeout(300);
    await page.waitForFunction(() => (window.__silversightLoads ?? 0) === 0 && !/Loading [a-z ]+…/.test(document.body.innerText),
      null, { timeout: 60000 });
  }
  await page.waitForTimeout(500);
}

async function apply(step) {
  const [kind, a, b, nth = 0] = step;
  if (kind === "seg") {
    const button = page.getByRole("group", { name: a, exact: true }).nth(nth).getByRole("button", { name: b, exact: true });
    if ((await button.getAttribute("aria-pressed")) !== "true") await button.click();
  } else if (kind === "select") {
    await page.locator(`#${a}`).selectOption(b);
  } else if (kind === "month") {
    const options = await page.locator("#edge-month option").evaluateAll((os) => os.map((o) => [o.value, o.textContent]));
    const hit = options.find(([, label]) => label >= a) ?? options[options.length - 1];
    await page.locator("#edge-month").selectOption(hit[0]);
  }
  await settle();
}

const rows = [];
let url = "";
for (const f of FIGURES) {
  if (only.length && !only.includes(f.id)) continue;
  const target = base + f.page + (f.models ? `?models=${f.models}` : "");
  if (target !== url) {
    await page.goto(target);
    url = target;
  }
  await settle();
  try {
    for (const s of f.steps ?? []) await apply(s);
    const files = {};
    let shown = "";
    for (const kind of ["SVG", "PNG"]) {
      const label = typeof f.chart === "string" ? `Export ${f.chart} as ${kind}` : new RegExp(`^Export ${f.chart.source.replace(/^\^|\$$/g, "")}.* as ${kind}$`);
      const button = page.getByRole("button", { name: label, exact: typeof f.chart === "string" });
      const [dl] = await Promise.all([page.waitForEvent("download"), button.click()]);
      const file = `${f.id}-${f.name}.${kind.toLowerCase()}`;
      await dl.saveAs(resolve(out, file));
      shown = dl.suggestedFilename().replace(/^silversight-|-\d{4}-\d{2}-\d{2}\.\w+$/g, "");
      const buf = readFileSync(resolve(out, file));
      if (kind === "SVG" && !buf.toString("utf8").startsWith("<svg")) throw new Error("not an SVG file");
      if (kind === "PNG" && (buf.readUInt32BE(0) !== 0x89504e47 || buf.readUInt32BE(16) !== 2400)) throw new Error("PNG is not 2400 px wide");
      files[kind] = { file, bytes: buf.length };
    }
    rows.push({ ...f, files, shown });
    console.log(`ok  ${f.id}  ${f.name}  (chart ${shown})`);
  } catch (e) {
    errors.push(`${f.id} ${f.name}: ${e.message.split("\n")[0]}`);
    console.log(`FAIL ${f.id}  ${f.name}`);
    url = ""; // reload the page for the next figure
  }
}
await browser.close();

if (!only.length) {
  const status = await (await fetch(`${base}/api/health`)).json().catch(() => ({}));
  const lines = [
    "# Figure pack", "",
    "Exported by `npm run figures` (web/scripts/figures.mjs) from the running app, in the print palette.",
    "SVG is the vector file (Word, Inkscape, LaTeX); PNG is 2400 px wide (300 dpi at 20 cm).",
    "Figure numbers are working labels; renumber when placing a figure in the report. The captions are",
    "descriptions of what the chart shows.", "",
    `Exported ${new Date().toISOString().slice(0, 10)}${status.commit ? `, code commit ${status.commit}` : ""}.`, "",
    "| Figure | Shows | Suggested chapter | Models | App page and chart | Files |", "|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.id} | ${r.title} | ${r.chapter} | ${(r.models ?? "–").replaceAll(",", ", ")} | \`${r.page}\` · ${r.shown} | [SVG](${r.files.SVG.file}) · [PNG](${r.files.PNG.file}) |`),
  ];
  writeFileSync(resolve(out, "index.md"), lines.join("\n") + "\n");
}
console.log(`\n${rows.length} figures -> ${out}`);
if (errors.length) {
  console.error(`\n${errors.length} error(s):\n` + errors.join("\n"));
  process.exit(1);
}
