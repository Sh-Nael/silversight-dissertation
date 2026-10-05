// Signals: the signal and risk layer. BUY / SELL on confident days with a 1σ stop and
// a 1.5σ target, backtested on daily bars; buy and sell trades separately; the account's equity.
import { useCallback, useState } from "react";
import Chart from "../components/Chart";
import { Seg, Select } from "../components/controls";
import { ErrorNote, Loading, NUM, PageHeader, Panel, Pill, TD, TH, TXT, Table } from "../components/ui";
import { getJSON, latestPerModel, useApi, useAsync, type Columns, type Row, type RunSummary } from "../lib/api";
import { num, pct } from "../lib/format";
import { axis, base, modelColor, tooltip, xy, type ChartTheme } from "../lib/theme";

const COVERAGES = [0.1, 0.2, 0.3, 0.5, 1.0];

export default function Signals() {
  const runs = useApi<RunSummary[]>("/api/runs");
  const latest = latestPerModel(runs.data ?? []).filter((r) => r.headline.qlike1 !== null && r.model !== "climatology" && r.model !== "ewma");
  const [h, setH] = useState<1 | 5>(1);
  const [coverage, setCoverage] = useState(0.3);
  const [side, setSide] = useState<"both" | "buy" | "sell">("both");

  const key = latest.length ? `${h}:${latest.map((r) => r.id).join(",")}` : null;
  const table = useAsync<{ model: string; rows: Row[] }[]>(
    () => Promise.all(latest.map(async (r) => ({ model: r.model, rows: await getJSON<Row[]>(`/api/runs/${r.id}/signals?h=${h}`) }))),
    key,
  );
  const equity = useAsync<{ model: string; curve: Columns }[]>(
    () => Promise.all(latest.map(async (r) => ({ model: r.model, curve: await getJSON<Columns>(`/api/runs/${r.id}/equity?h=${h}&coverage=${coverage}&side=${side}`) }))),
    key ? `${key}:${coverage}:${side}` : null,
  );

  const equityChart = useCallback(
    (t: ChartTheme) => {
      const all = equity.data!;
      return base(t, {
        tooltip: tooltip(t, { valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "–") }),
        grid: { left: 52, right: 14, top: 30, bottom: 28 },
        xAxis: axis(t, "time"),
        yAxis: axis(t, "value", { name: "equity (start = 1)", nameLocation: "middle", nameGap: 38, scale: true }),
        series: [
          { name: "start", type: "line", color: t.faint, data: [[all[0].curve.date[0] as string, 1], [all[0].curve.date[all[0].curve.date.length - 1] as string, 1]], showSymbol: false, silent: true, lineStyle: { color: t.faint, type: "dashed", width: 1 } },
          ...all.map((m) => ({
            name: m.model,
            type: "line" as const,
            showSymbol: false,
            color: modelColor(m.model, t.fig),
            lineStyle: { color: modelColor(m.model, t.fig), width: 1.6 },
            data: xy(m.curve.date as string[], m.curve.equity),
          })),
        ],
      });
    },
    [equity.data],
  );

  const rowsFor = (m: { model: string; rows: Row[] }) =>
    m.rows.filter((r) => r.coverage_target === coverage && (r.n as number) > 0);
  const tone = (p: unknown, pf: unknown) => (typeof p === "number" && p < 0.05 ? ((pf as number) > 1 ? "good" : "bad") : "neutral");

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <PageHeader eyebrow="Signals" title="Can the forecasts be traded?">
          A signal fires only on confident days (the causal gate: today's confidence against the model's own last 252 forecasts).
          Every trade has a stop at 1× and a target at 1.5× the forecast move, 10 bps round-trip cost, 1% of capital at risk,
          one position at a time. A day that touches both levels counts as the stop. Buy and sell trades are judged separately.
        </PageHeader>
        <div className="flex flex-wrap gap-2">
          <Seg label="Horizon" value={h} onChange={setH} options={[{ value: 1, label: "1 day" }, { value: 5, label: "5 days" }]} />
          <Select id="coverage" label="confident share" value={String(coverage)} onChange={(v) => setCoverage(Number(v))} options={COVERAGES.map((c) => ({ value: String(c), label: pct(c) }))} />
          <Seg label="Trades" value={side} onChange={setSide} options={[{ value: "both", label: "both" }, { value: "buy", label: "buy" }, { value: "sell", label: "sell" }]} />
        </div>
      </div>
      {runs.error && <ErrorNote error={runs.error} />}
      <Panel title={`Trades · ${h === 1 ? "1 day" : "5 days"} · signals on the ${pct(coverage)} most confident days`} sub="Per model and side: independent trades after costs. Profit factor = gross wins / gross losses; p = one-sided block-permutation test that the mean net return is positive. Account columns run one position at a time.">
        {table.error && <ErrorNote error={table.error} />}
        {!table.data && !table.error && <Loading what="signals" />}
        {table.data && (
          <Table head={<><th className={TH}>Model</th><th className={TH}>Side</th><th className={TH}>Trades</th><th className={TH}>Win rate</th><th className={TH}>Profit factor</th><th className={TH}>Net / trade</th><th className={TH}>p (mean &gt; 0)</th><th className={TH}>Target hit</th><th className={TH}>Stopped</th><th className={TH}>Account return</th><th className={TH}>Max drawdown</th></>}>
            {table.data.flatMap((m) => rowsFor(m).map((r) => (
              <tr key={`${m.model}-${r.side}`} className="hover:bg-panel-2">
                <td className={`${TD} ${TXT}`}>{m.model}</td>
                <td className={`${TD} ${TXT}`}>{r.side as string}</td>
                <td className={`${TD} ${NUM}`}>{r.n as number}</td>
                <td className={`${TD} ${NUM}`}>{pct(r.win_rate)}</td>
                <td className={`${TD} ${NUM}`}>{num(r.profit_factor, 2)}</td>
                <td className={`${TD} ${NUM}`}>{num(r.mean_net_bps, 1)} bps</td>
                <td className={`${TD} ${NUM}`}><Pill tone={tone(r.p_positive, r.profit_factor)}>{num(r.p_positive, 3)}</Pill></td>
                <td className={`${TD} ${NUM}`}>{pct(r.target_rate)}</td>
                <td className={`${TD} ${NUM}`}>{pct(r.stop_rate)}</td>
                <td className={`${TD} ${NUM}`}>{pct(r.total_return)}</td>
                <td className={`${TD} ${NUM}`}>{pct(r.max_drawdown)}</td>
              </tr>
            )))}
          </Table>
        )}
      </Panel>
      <Panel title={`Account equity · ${side} trades · ${h === 1 ? "1 day" : "5 days"} · ${pct(coverage)} coverage`} sub="Units of starting capital, one position at a time, marked at trade exits. Flat lines are periods without a signal.">
        {equity.error && <ErrorNote error={equity.error} />}
        {equity.data ? <Chart build={equityChart} name={`equity-${side}-h${h}-c${coverage}`} height={380} /> : !equity.error && <Loading what="equity curves" />}
      </Panel>
    </div>
  );
}
