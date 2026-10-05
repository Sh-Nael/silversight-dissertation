// Calibration: do the direction probabilities mean what they say, how accurate is each model on
// its most confident days, and does the volatility forecast track realised volatility?
// Uses the latest run of each model.
import { useCallback, useState } from "react";
import Chart from "../components/Chart";
import { Seg, Select } from "../components/controls";
import { ErrorNote, Loading, PageHeader, Panel } from "../components/ui";
import { getJSON, latestPerModel, useApi, useAsync, type Columns, type Row, type RunSummary } from "../lib/api";
import { axis, base, modelColor, tooltip, xy, type ChartTheme } from "../lib/theme";

export default function Calibration() {
  const runs = useApi<RunSummary[]>("/api/runs");
  const latest = latestPerModel(runs.data ?? []);
  const [h, setH] = useState<1 | 5>(1);
  const [volModel, setVolModel] = useState("linear");

  const calKey = latest.length ? `${h}:${latest.map((r) => r.id).join(",")}` : null;
  const cal = useAsync<{ model: string; bins: Row[] }[]>(
    () => Promise.all(latest.map(async (r) => ({ model: r.model, bins: await getJSON<Row[]>(`/api/runs/${r.id}/calibration?h=${h}&bins=10`) }))),
    calKey,
  );
  const [mode, setMode] = useState<"causal" | "ranked">("causal");
  const [side, setSide] = useState<"both" | "up" | "down">("both");
  const sel = useAsync<{ model: string; rows: Row[] }[]>(
    () => Promise.all(latest.map(async (r) => ({ model: r.model, rows: await getJSON<Row[]>(`/api/runs/${r.id}/selective?h=${h}&mode=${mode}`) }))),
    latest.length ? `${h}:${mode}:${latest.map((r) => r.id).join(",")}` : null,
  );
  const volRun = latest.find((r) => r.model === volModel) ?? latest.find((r) => r.headline.qlike1 !== null);
  const vol = useApi<Columns>(volRun ? `/api/runs/${volRun.id}/volatility?h=${h}` : null);

  const reliability = useCallback(
    (t: ChartTheme) => {
      const all = cal.data!;
      const ps = all.flatMap((m) => m.bins.map((b) => [b.p_mean as number, b.observed as number])).flat();
      const lo = Math.floor(Math.min(...ps) * 20) / 20;
      const hi = Math.ceil(Math.max(...ps) * 20) / 20;
      return base(t, {
        tooltip: tooltip(t, {
          trigger: "item",
          formatter: (p: { seriesName: string; value: number[] }) =>
            `${p.seriesName}<br>forecast ${p.value[0].toFixed(3)} → observed ${p.value[1].toFixed(3)}<br>${p.value[2]} days`,
        }),
        grid: { left: 58, right: 16, top: 30, bottom: 46 },
        xAxis: axis(t, "value", { min: lo, max: hi, name: "forecast P(up)", nameLocation: "middle", nameGap: 28 }),
        yAxis: axis(t, "value", { min: lo, max: hi, name: "observed up-rate", nameLocation: "middle", nameGap: 42 }),
        series: [
          { name: "perfect", type: "line", color: t.faint, data: [[lo, lo], [hi, hi]], showSymbol: false, silent: true, lineStyle: { color: t.faint, type: "dashed", width: 1 } },
          ...all.map((m) => ({
            name: m.model,
            type: "line" as const,
            data: m.bins.map((b) => [b.p_mean as number, b.observed as number, b.n as number]),
            symbolSize: (v: number[]) => Math.max(5, Math.sqrt(v[2]) / 2.2),
            color: modelColor(m.model, t.fig),
            lineStyle: { color: modelColor(m.model, t.fig), width: 1.4 },
            itemStyle: { color: modelColor(m.model, t.fig) },
          })),
        ],
      });
    },
    [cal.data],
  );

  const coverageChart = useCallback(
    (t: ChartTheme) => {
      // A model whose probability (almost) never moves has no "confident days"; leave it out.
      // Per side (D-32): hit rate of that side's calls; the reference is the share of selected
      // days that moved that way, and the test is against the stricter of 50% and that share.
      const f = side === "both"
        ? { acc: "accuracy", n: "n", base: "always_up", p: "p_edge", baseLabel: "always-up on the same days", pLabel: "edge p" }
        : { acc: `${side}_right`, n: `${side}_n`, base: `${side}_base`, p: `p_${side}`, baseLabel: `share of those days that went ${side}`, pLabel: `${side}-side p` };
      const all = sel.data!
        .map((m) => ({ ...m, rows: m.rows.filter((r) => (r[f.n] as number) > 0) }))
        .filter((m) => m.rows.some((r) => (r.coverage_target as number) < 1));
      const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
      return base(t, {
        tooltip: tooltip(t, {
          trigger: "item",
          formatter: (p: { seriesName: string; value: number[] }) =>
            `${p.seriesName}: ${(p.value[1] * 100).toFixed(1)}% right on ${p.value[2]} ${side === "both" ? "" : side + " "}calls (${pct(p.value[0])} of days selected)` +
            `<br>${f.baseLabel} ${(p.value[3] * 100).toFixed(1)}% · ${f.pLabel} = ${Number.isFinite(p.value[4]) ? p.value[4].toFixed(3) : "–"}`,
        }),
        grid: { left: 58, right: 16, top: 30, bottom: 46 },
        xAxis: axis(t, "value", { min: 0, max: 1, name: "coverage (share of days with a call)", nameLocation: "middle", nameGap: 28, axisLabel: { color: t.muted, fontSize: 11, formatter: pct } }),
        yAxis: axis(t, "value", { min: 0.4, max: 0.7, name: "accuracy", nameLocation: "middle", nameGap: 42, axisLabel: { color: t.muted, fontSize: 11, formatter: pct } }),
        series: [
          { name: "coin (50%)", type: "line", color: t.faint, data: [[0, 0.5], [1, 0.5]], showSymbol: false, silent: true, lineStyle: { color: t.faint, type: "dotted", width: 1 } },
          { name: "goal (65%)", type: "line", color: t.muted, data: [[0, 0.65], [1, 0.65]], showSymbol: false, silent: true, lineStyle: { color: t.muted, type: "dashed", width: 1 } },
          ...all.map((m) => ({
            name: m.model,
            type: "line" as const,
            data: m.rows.map((r) => [r.coverage as number, r[f.acc] as number, r[f.n] as number, r[f.base] as number, (r[f.p] as number | null) ?? NaN]),
            symbolSize: 7,
            color: modelColor(m.model, t.fig),
            lineStyle: { color: modelColor(m.model, t.fig), width: 1.6 },
            itemStyle: { color: modelColor(m.model, t.fig) },
          })),
        ],
      });
    },
    [sel.data, side],
  );

  const volChart = useCallback(
    (t: ChartTheme) => {
      const c = vol.data!;
      const dates = c.date as string[];
      return base(t, {
        tooltip: tooltip(t, { valueFormatter: (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "–") }),
        grid: { left: 52, right: 14, top: 30, bottom: 28 },
        xAxis: axis(t, "time"),
        yAxis: axis(t, "value", { axisLabel: { color: t.muted, fontSize: 11, formatter: (v: number) => `${(v * 100).toFixed(0)}%` } }),
        series: [
          { name: "realised", type: "line", color: t.faint, showSymbol: false, data: xy(dates, c.realised), lineStyle: { color: t.faint, width: 1 } },
          { name: `forecast (${volRun?.model})`, type: "line", color: modelColor(volRun?.model ?? "", t.fig), showSymbol: false, data: xy(dates, c.forecast), lineStyle: { color: modelColor(volRun?.model ?? "", t.fig), width: 1.8 } },
        ],
      });
    },
    [vol.data, volRun?.model],
  );

  const volModels = latest.filter((r) => r.headline.qlike1 !== null).map((r) => r.model);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <PageHeader eyebrow="Calibration" title="Do the probabilities mean what they say?">
          A calibrated model's 55% days go up 55% of the time. The app's direction gate acts on the probability itself, so
          calibration matters as much as accuracy, and it is what makes "only call the confident days" work.
        </PageHeader>
        <Seg label="Horizon" value={h} onChange={setH} options={[{ value: 1, label: "1 day" }, { value: 5, label: "5 days" }]} />
      </div>
      {runs.error && <ErrorNote error={runs.error} />}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <Panel title={`Reliability diagram · ${h === 1 ? "1 day" : "5 days"}`} sub="Test days in 10 equal-count bins of forecast probability; dot size = days. Diagonal = perfect calibration. Click a legend entry to hide a model.">
          {cal.error && <ErrorNote error={cal.error} />}
          {cal.data ? <Chart build={reliability} name={`reliability-h${h}`} height={400} /> : !cal.error && <Loading what="calibration" />}
        </Panel>
        <Panel
          title={`Volatility forecast vs realised · ${h === 1 ? "1 day" : "5 days"}`}
          sub="Annualised, weekly averages. Forecasts should track the level; realised volatility is noisier by nature."
          tools={volModels.length > 0 && <Select id="vol-model" label="model" value={volRun?.model ?? ""} onChange={setVolModel} options={volModels.map((m) => ({ value: m, label: m }))} />}
        >
          {vol.error && <ErrorNote error={vol.error} />}
          {vol.data ? <Chart build={volChart} name={`volatility-${volRun?.model ?? "model"}-h${h}`} height={400} /> : !vol.error && <Loading what="volatility" />}
        </Panel>
      </div>
      <Panel
        title={`Accuracy vs coverage · ${side === "both" ? "all calls" : side + " calls"} · ${h === 1 ? "1 day" : "5 days"}`}
        sub={mode === "causal"
          ? "Causal: a call counts as confident when it is in the top share of the model's own last 252 forecasts, as a live app would decide. Coverage levels 10/20/30/50/100% were fixed in advance (D-31). Silver is traded both ways: switch to up (buy) or down (sell) calls to see each side's hit rate (D-32). The goal line is 65% on both sides."
          : "Ranked (look-ahead): each day ranked against the whole test period. Describes the model; no one could have traded it."}
        tools={
          <div className="flex flex-wrap gap-2">
            <Seg label="Calls" value={side} onChange={setSide} options={[{ value: "both", label: "both" }, { value: "up", label: "up (buy)" }, { value: "down", label: "down (sell)" }]} />
            <Seg label="Selection" value={mode} onChange={setMode} options={[{ value: "causal", label: "causal" }, { value: "ranked", label: "ranked" }]} />
          </div>
        }
      >
        {sel.error && <ErrorNote error={sel.error} />}
        {sel.data ? <Chart build={coverageChart} name={`coverage-${mode}-h${h}`} height={380} /> : !sel.error && <Loading what="accuracy vs coverage" />}
      </Panel>
    </div>
  );
}
