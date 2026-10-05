// Runs: every run on disk; select one to see its manifest, fit times and tuning log.
import { useCallback, useState } from "react";
import Chart from "../components/Chart";
import { ModelName } from "../components/controls";
import { ErrorNote, Loading, NUM, PageHeader, Panel, Pill, TD, TH, TXT, Table } from "../components/ui";
import { useApi, type RunDetail, type RunSummary } from "../lib/api";
import { num, secs, when } from "../lib/format";
import { axis, base, modelColor, type ChartTheme } from "../lib/theme";

function Manifest({ d }: { d: RunDetail }) {
  const m = d.manifest;
  const p = m.protocol as Record<string, number>;
  const tuning = m.tuning as { n_points: number; wall_seconds: number } | null;
  const rows: [string, string][] = [
    ["run_id", d.id],
    ["model", `${m.model} (${m.model_class as string})`],
    ["started", when(m.started_utc)],
    ["wall time", secs(m.wall_seconds)],
    ["protocol", `${p.n_refits} refits · every ${p.refit_every} days · purge ${p.purge}`],
    ["tuning", tuning ? `${tuning.n_points} points · ${secs(tuning.wall_seconds)} wall` : "none (untuned model)"],
    ["code", `${m.code.commit.slice(0, 10)} ${m.code.dirty ? "(uncommitted changes!)" : "(clean)"}`],
    ["data", Object.keys(m.data as object).join(" · ") + " (sha-256 recorded)"],
    ["fold fingerprint", String(m.fold_fingerprint)],
    ["environment", `Python ${m.env.python} · torch ${m.env.torch ?? "–"} · sklearn ${m.env["scikit-learn"] ?? "–"}${m.env.gpu ? ` · ${m.env.gpu}` : ""}`],
  ];
  return (
    <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-[13px]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="font-mono text-[11.5px] text-muted">{k}</dt>
          <dd className="m-0 font-mono break-all">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Detail({ id }: { id: string }) {
  const d = useApi<RunDetail>(`/api/runs/${encodeURIComponent(id)}`);
  const build = useCallback(
    (t: ChartTheme) => {
      const rows = d.data!.timings;
      const c = modelColor(d.data!.manifest.model, t.fig);
      return base(t, {
        grid: { left: 44, right: 12, top: 12, bottom: 28 },
        legend: { show: false },
        tooltip: { trigger: "axis", backgroundColor: t.panel, borderColor: t.line, textStyle: { color: t.text, fontSize: 12 },
          valueFormatter: (v) => (typeof v === "number" ? `${v.toFixed(2)} s` : "–") },
        xAxis: axis(t, "category", { data: rows.map((r) => String(r.origin)), axisLabel: { color: t.muted, fontSize: 10, interval: 24 } }),
        yAxis: axis(t, "value", { name: "s" }),
        series: [{ name: "fit", type: "line", showSymbol: false, data: rows.map((r) => r.fit_s as number), lineStyle: { color: c, width: 1.4 }, areaStyle: { color: c, opacity: 0.12 } }],
      });
    },
    [d.data],
  );
  if (d.error) return <ErrorNote error={d.error} />;
  if (!d.data) return <Loading what="run" />;
  const tuning = d.data.tuning;
  const tasks = tuning ? Object.keys(tuning[0].results) : [];
  return (
    <>
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <Panel title={`Manifest · ${d.data.manifest.model}`} sub="What produced these numbers: code commit, data checksums, protocol, environment.">
          <Manifest d={d.data} />
        </Panel>
        <Panel title="Fit time per refit" sub="Seconds for each monthly refit. Measured runtimes feed Chapter 3.">
          <Chart build={build} name={`fit-time-${d.data.manifest.model}`} height={220} />
        </Panel>
      </div>
      {tuning && (
        <Panel title="Tuning log" sub="At each yearly tuning point: the out-of-fold loss of the chosen settings (development score; log-loss for direction, QLIKE for volatility).">
          <Table head={<><th className={TH}>Point</th><th className={TH}>Origin</th><th className={TH}>Trained to</th>{tasks.map((k) => <th key={k} className={TH}>{k.replace("_h", " h")}</th>)}<th className={TH}>Seconds</th></>}>
            {tuning.map((tp) => (
              <tr key={tp.tuning_point}>
                <td className={`${TD} ${NUM}`}>{tp.tuning_point}</td>
                <td className={`${TD} ${NUM}`}>{tp.origin}</td>
                <td className={`${TD} ${NUM}`}>{tp.train_end}</td>
                {tasks.map((k) => <td key={k} className={`${TD} ${NUM}`} title={JSON.stringify(tp.results[k].params)}>{num(tp.results[k].cv_loss, k.startsWith("vol") ? 3 : 4)}</td>)}
                <td className={`${TD} ${NUM}`}>{secs(tp.seconds)}</td>
              </tr>
            ))}
          </Table>
          <p className="m-0 text-xs text-muted">Hover a value to see the settings chosen at that point.</p>
        </Panel>
      )}
    </>
  );
}

export default function Runs() {
  const runs = useApi<RunSummary[]>("/api/runs");
  const [sel, setSel] = useState<string | null>(null);
  const current = sel ?? runs.data?.[0]?.id ?? null;
  return (
    <div className="flex flex-col gap-5">
      <PageHeader eyebrow="Runs" title="Every run, traceable to code and data">
        Each run folder holds the forecasts, timings, tuning log, diagnostics and a manifest with the exact commit and data
        checksums. Superseded runs are kept as evidence. Select a run to inspect it.
      </PageHeader>
      <Panel title="All runs" sub="Newest first. Headline scores pooled over the 4,141 test days.">
        {runs.error && <ErrorNote error={runs.error} />}
        {!runs.data && !runs.error && <Loading what="runs" />}
        {runs.data && (
          <Table head={<><th className={TH}>Run ID</th><th className={`${TH} ${TXT}`}>Model</th><th className={TH}>Started</th><th className={TH}>Brier 1d</th><th className={TH}>QLIKE 1d</th><th className={TH}>Wall</th><th className={TH}>Tuning</th><th className={TH}>Commit</th><th className={TH}>Code</th></>}>
            {runs.data.map((r) => (
              <tr
                key={r.id}
                onClick={() => setSel(r.id)}
                onKeyDown={(e) => e.key === "Enter" && setSel(r.id)}
                tabIndex={0}
                aria-selected={r.id === current}
                className={`cursor-pointer ${r.id === current ? "bg-signal-dim" : "hover:bg-panel-2"}`}
              >
                <td className={`${TD} ${NUM}`}>{r.id}</td>
                <td className={`${TD} ${TXT}`}><ModelName model={r.model} /></td>
                <td className={`${TD} ${NUM}`}>{when(r.started_utc)}</td>
                <td className={`${TD} ${NUM}`}>{num(r.headline.brier1, 4)}</td>
                <td className={`${TD} ${NUM}`}>{num(r.headline.qlike1, 3)}</td>
                <td className={`${TD} ${NUM}`}>{secs(r.wall_seconds)}</td>
                <td className={`${TD} ${NUM}`}>{r.tuning_points ? `${r.tuning_points} pts` : "–"}</td>
                <td className={`${TD} ${NUM}`}>{r.commit}</td>
                <td className={TD}><Pill tone={r.dirty ? "bad" : "good"}>{r.dirty ? "dirty" : "clean"}</Pill></td>
              </tr>
            ))}
          </Table>
        )}
      </Panel>
      {current && <Detail id={current} />}
    </div>
  );
}
