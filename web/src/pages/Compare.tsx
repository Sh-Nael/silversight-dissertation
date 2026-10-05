// Compare: scoreboard, significance, per-fold and cumulative loss differences, development
// scores. Everything is computed by POST /api/compare (the library's report functions).
import { useCallback, useMemo, useState } from "react";
import Chart from "../components/Chart";
import { ModelName, Seg, Select } from "../components/controls";
import { ErrorNote, Loading, NUM, PageHeader, Panel, Pill, TD, TH, TXT, Table } from "../components/ui";
import { MODEL_ORDER, latestPerModel, postJSON, useApi, useAsync, type CompareResponse, type RunSummary } from "../lib/api";
import { LOWER_IS_BETTER, num, pct } from "../lib/format";
import { axis, base, modelColor, xy, type ChartTheme } from "../lib/theme";

type Loss = "brier" | "logloss" | "qlike";
const LOSS_LABEL: Record<Loss, string> = { brier: "Brier", logloss: "Log-loss", qlike: "QLIKE" };

const SCORE_COLS: { key: string; label: string; fmt: (v: unknown) => string }[] = [
  { key: "brier", label: "Brier", fmt: (v) => num(v, 4) },
  { key: "logloss", label: "Log-loss", fmt: (v) => num(v, 4) },
  { key: "accuracy", label: "Hit rate", fmt: (v) => pct(v) },
  { key: "balanced_accuracy", label: "Balanced acc.", fmt: (v) => pct(v) },
  { key: "up_precision", label: "Up calls right", fmt: (v) => pct(v) },
  { key: "down_precision", label: "Down calls right", fmt: (v) => pct(v) },
  { key: "auc", label: "AUC", fmt: (v) => num(v, 4) },
  { key: "vol_qlike", label: "QLIKE", fmt: (v) => num(v, 3) },
  { key: "vol_rmse_vol", label: "Vol RMSE", fmt: (v) => num(v, 4) },
  { key: "vol_mz_r2", label: "MZ R²", fmt: (v) => num(v, 4) },
];

function PValue({ p, diff }: { p: unknown; diff: unknown }) {
  if (typeof p !== "number") return <Pill tone="neutral">–</Pill>;
  const sig = p < 0.05;
  const tone = sig ? ((diff as number) < 0 ? "good" : "bad") : "neutral";
  return <Pill tone={tone}>{p < 0.0001 ? "<0.0001" : p.toFixed(4)}</Pill>;
}

export default function Compare() {
  const runs = useApi<RunSummary[]>("/api/runs");
  const [choice, setChoice] = useState<Record<string, string>>({}); // model -> run id
  const [reference, setReference] = useState("climatology");
  const [h, setH] = useState<1 | 5>(1);
  const [sigLoss, setSigLoss] = useState<Loss>("brier");
  const [pfLoss, setPfLoss] = useState<Loss>("brier");
  const [cumLoss, setCumLoss] = useState<Loss>("brier");

  const byModel = useMemo(() => {
    const m = new Map<string, RunSummary[]>();
    for (const r of runs.data ?? []) m.set(r.model, [...(m.get(r.model) ?? []), r]);
    return m;
  }, [runs.data]);
  const selected = useMemo(
    () => latestPerModel(runs.data ?? []).map((r) => choice[r.model] ?? r.id),
    [runs.data, choice],
  );
  const models = latestPerModel(runs.data ?? []).map((r) => r.model);
  const ref = models.includes(reference) ? reference : models[0];
  const key = selected.length >= 2 && ref ? JSON.stringify({ selected, ref }) : null;
  const cmp = useAsync<CompareResponse>(() => postJSON("/api/compare", { runs: selected, reference: ref }), key);
  const d = cmp.data;

  const board = (d?.scoreboard ?? []).filter((r) => r.h === h);
  const bestOf = (k: string) => {
    const vals = board.map((r) => r[k]).filter((v): v is number => typeof v === "number");
    return LOWER_IS_BETTER.has(k) ? Math.min(...vals) : Math.max(...vals);
  };
  const sig = (d?.significance ?? []).filter((r) => r.h === h && r.loss === sigLoss);
  // Ladder order (climatology first); unknown models last.
  const order = (m: string) => (MODEL_ORDER.includes(m) ? MODEL_ORDER.indexOf(m) : 99);

  const perFold = useCallback(
    (t: ChartTheme) => {
      const series = d!.per_fold[`${pfLoss}_h${h}`] ?? {};
      const ms = Object.keys(series).sort((a, b) => order(a) - order(b));
      return base(t, {
        tooltip: { trigger: "axis", backgroundColor: t.panel, borderColor: t.line, textStyle: { color: t.text, fontSize: 12 },
          valueFormatter: (v) => (typeof v === "number" ? v.toFixed(pfLoss === "qlike" ? 3 : 5) : "–") },
        grid: { left: 78, right: 12, top: 30, bottom: 28 },
        xAxis: axis(t, "category", { data: [0, 1, 2, 3, 4, 5, 6, 7].map((k) => `F${k}`) }),
        yAxis: axis(t, "value", { scale: true, name: `mean ${LOSS_LABEL[pfLoss]} − ${d!.reference} (below 0: better)`,
          nameLocation: "middle", nameGap: 62, nameTextStyle: { color: t.muted, fontSize: 11 } }),
        series: ms.map((m) => ({ name: m, type: "bar" as const, barGap: "15%", color: modelColor(m, t.fig), itemStyle: { color: modelColor(m, t.fig), borderRadius: [2, 2, 0, 0] }, data: series[m].map((v) => v ?? "-") })),
      });
    },
    [d, pfLoss, h],
  );

  const cumulative = useCallback(
    (t: ChartTheme) => {
      const cols = d!.cumulative[`${cumLoss}_h${h}`];
      const ms = Object.keys(cols).filter((k) => k !== "date").sort((a, b) => order(a) - order(b));
      const dates = cols.date as string[];
      return base(t, {
        grid: { left: 72, right: 14, top: 30, bottom: 28 },
        tooltip: { trigger: "axis", backgroundColor: t.panel, borderColor: t.line, textStyle: { color: t.text, fontSize: 12 },
          valueFormatter: (v) => (typeof v === "number" ? v.toFixed(3) : "–") },
        xAxis: axis(t, "time"),
        yAxis: axis(t, "value", { scale: true, name: `cumulative ${LOSS_LABEL[cumLoss]} − ${d!.reference} (falling: better)`,
          nameLocation: "middle", nameGap: 52, nameTextStyle: { color: t.muted, fontSize: 11 } }),
        series: ms.map((m, i) => ({
          name: m,
          type: "line" as const,
          showSymbol: false,
          color: modelColor(m, t.fig),
          lineStyle: { color: modelColor(m, t.fig), width: 1.5 },
          data: xy(dates, cols[m]),
          ...(i === 0 ? { markLine: { silent: true, symbol: "none", label: { show: false }, lineStyle: { color: t.faint, type: "dashed" as const }, data: [{ yAxis: 0 }] } } : {}),
        })),
      });
    },
    [d, cumLoss, h],
  );

  const lossOpts = (["brier", "logloss", "qlike"] as Loss[]).map((l) => ({ value: l, label: LOSS_LABEL[l] }));

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <PageHeader eyebrow="Compare" title="Scoreboard and significance">
          Lower is better for Brier, log-loss and QLIKE. Significance: Diebold–Mariano (primary), Wilcoxon over the 8 folds,
          block permutation. All runs are scored on the same 4,141 test days.
        </PageHeader>
        <Seg label="Horizon" value={h} onChange={setH} options={[{ value: 1, label: "1 day" }, { value: 5, label: "5 days" }]} />
      </div>

      <Panel title="Runs in this comparison" sub="The latest run of each model is used by default; pick an older run to compare superseded versions.">
        {runs.error && <ErrorNote error={runs.error} />}
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          {latestPerModel(runs.data ?? []).map((r) => (
            <Select
              key={r.model}
              id={`run-${r.model}`}
              label={r.model}
              value={choice[r.model] ?? r.id}
              onChange={(v) => setChoice((c) => ({ ...c, [r.model]: v }))}
              options={(byModel.get(r.model) ?? []).map((x) => ({ value: x.id, label: x.id.replace(`${r.model}-`, "") }))}
            />
          ))}
          {models.length > 0 && (
            <Select id="reference" label="reference" value={ref} onChange={setReference} options={models.map((m) => ({ value: m, label: m }))} />
          )}
        </div>
      </Panel>

      {cmp.error && <ErrorNote error={cmp.error} />}
      {!d && !cmp.error && <Loading what="comparison" />}
      {d && (
        <>
          <Panel title={`Scoreboard · ${h === 1 ? "1-day" : "5-day"} horizon`} sub="Pooled over all test days. Best value per column highlighted.">
            <Table head={<><th className={TH}>Model</th>{SCORE_COLS.map((c) => <th key={c.key} className={TH}>{c.label}</th>)}</>}>
              {[...board].sort((a, b) => order(a.model as string) - order(b.model as string)).map((r) => (
                <tr key={r.model as string} className="hover:bg-panel-2">
                  <td className={TD}><ModelName model={r.model as string} /></td>
                  {SCORE_COLS.map((c) => (
                    <td key={c.key} className={`${TD} ${NUM} ${r[c.key] === bestOf(c.key) ? "font-semibold text-signal" : ""}`}>{c.fmt(r[c.key])}</td>
                  ))}
                </tr>
              ))}
            </Table>
          </Panel>

          <Panel
            title={`Significance vs ${d.reference}`}
            sub="Mean difference below zero: the model's loss is lower. Green: significantly better; red: significantly worse (p < 0.05); grey: not significant."
            tools={<Seg label="Loss" value={sigLoss} onChange={setSigLoss} options={lossOpts} />}
          >
            <Table head={<><th className={TH}>Model</th><th className={TH}>Mean diff</th><th className={TH}>DM stat</th><th className={TH}>DM p</th><th className={TH}>Wilcoxon p</th><th className={TH}>Permutation p</th><th className={TH}>Folds better</th></>}>
              {[...sig].sort((a, b) => order(a.model as string) - order(b.model as string)).map((r) => (
                <tr key={r.model as string} className="hover:bg-panel-2">
                  <td className={TD}><ModelName model={r.model as string} /></td>
                  <td className={`${TD} ${NUM}`}>{num(r.mean_diff, 5)}</td>
                  <td className={`${TD} ${NUM}`}>{num(r.dm_stat, 2)}</td>
                  <td className={TD}><PValue p={r.dm_p} diff={r.mean_diff} /></td>
                  <td className={TD}><PValue p={r.wilcoxon_p} diff={r.mean_diff} /></td>
                  <td className={TD}><PValue p={r.perm_p} diff={r.mean_diff} /></td>
                  <td className={`${TD} ${NUM}`}>{r.folds_better as string}</td>
                </tr>
              ))}
            </Table>
          </Panel>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <Panel
              title={`Per fold · loss minus ${d.reference}`}
              sub="Mean loss in each ~2-year test fold, minus the reference's. Below zero: the model won that fold."
              tools={<Seg label="Loss" value={pfLoss} onChange={setPfLoss} options={lossOpts} />}
            >
              <Chart build={perFold} name={`per-fold-${pfLoss}-h${h}-vs-${d.reference}`} height={320} />
            </Panel>
            <Panel
              title="Cumulative loss difference over time"
              sub={`Running sum of (model − ${d.reference}) over test days. Trending down: the model keeps winning; flat: no edge.`}
              tools={<Seg label="Loss" value={cumLoss} onChange={setCumLoss} options={lossOpts} />}
            >
              <Chart build={cumulative} name={`cumulative-${cumLoss}-h${h}-vs-${d.reference}`} height={320} />
            </Panel>
          </div>

          {d.dev_scoreboard.length > 0 && (
            <Panel title="Development scores" sub="Out-of-fold losses at the yearly tuning points (mean). Used for design decisions only; never reported as performance.">
              <Table head={<><th className={TH}>Model</th><th className={`${TH} ${TXT}`}>Task</th><th className={TH}>Mean out-of-fold loss</th><th className={TH}>Tuning points</th></>}>
                {d.dev_scoreboard.map((r) => (
                  <tr key={`${r.model}-${r.task}`} className="hover:bg-panel-2">
                    <td className={TD}><ModelName model={r.model as string} /></td>
                    <td className={`${TD} ${NUM} ${TXT}`}>{(r.task as string).replace("_h", " h")}</td>
                    <td className={`${TD} ${NUM}`}>{num(r.mean, (r.task as string).startsWith("vol") ? 3 : 4)}</td>
                    <td className={`${TD} ${NUM}`}>{r.count as number}</td>
                  </tr>
                ))}
              </Table>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
