// Overview: system status, headline standing of the latest run of each model, silver's
// history with regime windows and fold boundaries, latest runs.
import { useCallback } from "react";
import { Link } from "react-router";
import Chart from "../components/Chart";
import { ModelName } from "../components/controls";
import { ErrorNote, Kpi, Loading, NUM, PageHeader, Panel, Pill, TD, TH, Table } from "../components/ui";
import { latestPerModel, useApi, type Columns, type FoldsResponse, type Regime, type RunSummary, type Status } from "../lib/api";
import { num, pct, secs } from "../lib/format";
import { axis, base, xy, type ChartTheme } from "../lib/theme";

function best(runs: RunSummary[], key: keyof RunSummary["headline"], lower: boolean) {
  const ok = runs.filter((r) => r.headline[key] !== null);
  if (!ok.length) return undefined;
  return ok.reduce((a, b) => ((lower ? b.headline[key]! < a.headline[key]! : b.headline[key]! > a.headline[key]!) ? b : a));
}

function PriceChart() {
  const price = useApi<Columns>("/api/data/panel?node=silver&freq=W-FRI");
  const regimes = useApi<Regime[]>("/api/data/regimes");
  const folds = useApi<FoldsResponse>("/api/folds");
  const build = useCallback(
    (t: ChartTheme) => {
      const p = price.data!;
      return base(t, {
        tooltip: { trigger: "axis", backgroundColor: t.panel, borderColor: t.line, textStyle: { color: t.text, fontSize: 12 },
          valueFormatter: (v) => (typeof v === "number" ? `$${v.toFixed(2)}` : "–") },
        grid: { left: 52, right: 16, top: 16, bottom: 30 },
        xAxis: axis(t, "time"),
        yAxis: axis(t, "log", { min: 5, max: 150, axisLabel: { color: t.muted, fontSize: 11, formatter: (v: number) => `$${v}` } }),
        series: [
          {
            name: "Silver",
            type: "line",
            showSymbol: false,
            lineStyle: { width: 1.6, color: t.silver },
            areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: t.signalDim }, { offset: 1, color: "rgba(0,0,0,0)" }] } },
            data: xy(p.date, p.close),
            markArea: { silent: true, itemStyle: { color: t.band }, label: { color: t.muted, fontSize: 10, position: "insideTop" },
              data: (regimes.data ?? []).map((r) => [{ name: r.label, xAxis: r.start }, { xAxis: r.end }]) as never },
            markLine: { silent: true, symbol: "none", label: { show: false }, lineStyle: { type: "dashed", color: t.line, width: 1 },
              data: (folds.data?.calendar.folds ?? []).map((f) => ({ xAxis: f.test_start })) },
          },
        ],
      });
    },
    [price.data, regimes.data, folds.data],
  );
  if (price.error) return <ErrorNote error={price.error} />;
  if (!price.data) return <Loading what="prices" />;
  return <Chart build={build} name="silver-price-regimes" height={360} />;
}

export default function Overview() {
  const status = useApi<Status>("/api/status");
  const runs = useApi<RunSummary[]>("/api/runs");
  const latest = runs.data ? latestPerModel(runs.data) : [];
  const vol1 = best(latest, "qlike1", true);
  const acc1 = best(latest, "accuracy1", false);
  const brier1 = best(latest, "brier1", true);
  const clim = latest.find((r) => r.model === "climatology");

  return (
    <div className="flex flex-col gap-5">
      <PageHeader eyebrow="Overview" title="Where the evidence stands">
        The latest run of each model through the frozen walk-forward protocol: 8 folds, 200 monthly refits, 4,141 test
        days (2010–2026). Every number comes from the same code as the command line.
      </PageHeader>

      {status.error && <ErrorNote error={status.error} />}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
        <Kpi label="Data integrity" value={status.data ? (status.data.ok ? "verified" : "FAILED") : "…"} note="SHA-256 recomputed on load" />
        <Kpi label="Models" value={latest.length || "…"} note={`${runs.data?.length ?? "…"} runs on disk`} />
        <Kpi label="Best 1d volatility" value={num(vol1?.headline.qlike1, 3)} note={vol1 ? `${vol1.model} · QLIKE, lower is better` : undefined} />
        <Kpi label="Best 1d Brier" value={num(brier1?.headline.brier1, 4)} note={brier1 ? `${brier1.model} · climatology ${num(clim?.headline.brier1, 4)}` : undefined} />
        <Kpi label="Best 1d hit rate" value={pct(acc1?.headline.accuracy1)} note={acc1 ? `${acc1.model} · always-up ${pct(clim?.headline.accuracy1)}` : undefined} />
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[2fr_1fr]">
        <Panel title="Silver, weekly close (log scale)" sub="Shaded: the four regime windows the adaptive model must handle. Dashed: the start of each test fold.">
          <PriceChart />
        </Panel>
        <Panel title="Latest runs" sub={<>From <code className="font-mono">experiments/runs/</code>. Details on the <Link className="text-signal" to="/runs">Runs</Link> page.</>}>
          {runs.error && <ErrorNote error={runs.error} />}
          {runs.data && (
            <Table head={<><th className={TH}>Model</th><th className={TH}>Brier 1d</th><th className={TH}>QLIKE 1d</th><th className={TH}>Wall</th></>}>
              {latest.map((r) => (
                <tr key={r.id} className="hover:bg-panel-2">
                  <td className={TD}><ModelName model={r.model} /></td>
                  <td className={`${TD} ${NUM}`}>{num(r.headline.brier1, 4)}</td>
                  <td className={`${TD} ${NUM}`}>{num(r.headline.qlike1, 3)}</td>
                  <td className={`${TD} ${NUM}`}>{secs(r.wall_seconds)}</td>
                </tr>
              ))}
            </Table>
          )}
          <p className="m-0 rounded-md border border-dashed border-line px-3 py-2 text-[12.5px] text-muted">
            <b className="text-text">Reading.</b> Flat use of the drivers forecasts volatility well and direction barely. That is the bar AIM-DG must clear.
          </p>
        </Panel>
      </div>

      {status.data && (
        <Panel title="Frozen snapshots" sub="Checksums are recomputed each time the page loads. A failure means the research data was modified.">
          <Table head={<><th className={TH}>Snapshot</th><th className={TH}>Files</th><th className={TH}>Data until</th><th className={TH}>Integrity</th></>}>
            {status.data.snapshots.map((s) => (
              <tr key={s.name}>
                <td className={`${TD} ${NUM}`}>{s.name}</td>
                <td className={`${TD} ${NUM}`}>{s.files}</td>
                <td className={`${TD} ${NUM}`}>{s.snapshot_end}</td>
                <td className={TD}><Pill tone={s.verified ? "good" : "bad"}>{s.verified ? "verified" : "checksum mismatch"}</Pill></td>
              </tr>
            ))}
          </Table>
        </Panel>
      )}
    </div>
  );
}
