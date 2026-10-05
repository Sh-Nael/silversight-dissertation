// Data explorer: markets with regime windows and the macro event calendar, any feature
// of any node, the gold-silver error-correction term, the data-quality audit, and a
// snapshot of every feature on a chosen day.
import { useCallback, useMemo, useState } from "react";
import Chart from "../components/Chart";
import { Seg, Select } from "../components/controls";
import { ErrorNote, Loading, NUM, PageHeader, Panel, Pill, TD, TH, TXT, Table } from "../components/ui";
import { useApi, type Columns, type EventRow, type NodeInfo, type Regime, type Row } from "../lib/api";
import { num } from "../lib/format";
import { axis, base, tooltip, xy, type ChartTheme } from "../lib/theme";

type Freq = "D" | "W-FRI" | "ME";
type EventType = "fomc" | "cpi" | "nfp";
const FREQS: { value: Freq; label: string }[] = [
  { value: "D", label: "Daily" },
  { value: "W-FRI", label: "Weekly" },
  { value: "ME", label: "Monthly" },
];
const EVENT_LABEL: Record<EventType, string> = { fomc: "FOMC", cpi: "CPI", nfp: "NFP" };

function eventColor(t: ChartTheme, e: EventType, scheduled: boolean): string {
  if (!scheduled) return t.bad;
  return e === "fomc" ? t.warn : e === "cpi" ? t.signal : t.fig ? "#6F5CC2" : "#9D8CE0";
}

const zoom = (t: ChartTheme) => [
  { type: "inside" as const, filterMode: "none" as const },
  {
    type: "slider" as const,
    height: 18,
    bottom: 6,
    borderColor: t.line,
    fillerColor: t.signalDim,
    dataBackground: { lineStyle: { color: t.faint }, areaStyle: { color: t.grid } },
    textStyle: { color: t.muted, fontSize: 10 },
    handleStyle: { color: t.panel, borderColor: t.signal },
  },
];

function stats(values: unknown[]) {
  const v = values.filter((x): x is number => typeof x === "number");
  if (!v.length) return null;
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(v.length - 1, 1));
  return { n: v.length, missing: values.length - v.length, mean, sd, min: Math.min(...v), max: Math.max(...v) };
}

// ---------------------------------------------------------------- market chart
function MarketPanel({ nodes, regimes, events }: { nodes: NodeInfo[]; regimes: Regime[]; events: EventRow[] }) {
  const [node, setNode] = useState("silver");
  const [freq, setFreq] = useState<Freq>("W-FRI");
  const [show, setShow] = useState<Record<EventType, boolean>>({ fomc: true, cpi: false, nfp: false });
  const info = nodes.find((n) => n.key === node);
  const panel = useApi<Columns>(`/api/data/panel?node=${node}&freq=${freq}`);

  const build = useCallback(
    (t: ChartTheme) => {
      const c = panel.data!;
      const dates = c.date as string[];
      const first = dates[0];
      const last = dates[dates.length - 1];
      const marks = events
        .filter((e) => show[e.event] && e.date >= first && e.date <= last)
        .map((e) => ({ xAxis: e.date, lineStyle: { color: eventColor(t, e.event, e.scheduled), width: 1, type: "solid" as const, opacity: e.scheduled ? 0.3 : 0.8 } }));
      const isRate = node === "real_yield_10y";
      return base(t, {
        legend: { show: false },
        tooltip: tooltip(t, { valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(isRate ? 2 : 3) : "–") }),
        grid: { left: 58, right: 18, top: 16, bottom: 64 },
        xAxis: axis(t, "time"),
        // Log axes ignore `scale`; fit the range to the data explicitly.
        yAxis: axis(t, isRate ? "value" : "log", isRate
          ? { scale: true }
          : { min: (v: { min: number }) => +(v.min * 0.9).toPrecision(2), max: (v: { max: number }) => +(v.max * 1.1).toPrecision(2) }),
        dataZoom: zoom(t),
        series: [
          {
            name: info?.label ?? node,
            type: "line",
            showSymbol: false,
            color: node === "gold" ? t.gold : t.silver,
            lineStyle: { width: 1.4, color: node === "gold" ? t.gold : t.silver },
            data: xy(dates, c.close),
            markArea: { silent: true, itemStyle: { color: t.band }, label: { color: t.muted, fontSize: 10, position: "insideTop" },
              data: regimes.map((r) => [{ name: r.label, xAxis: r.start }, { xAxis: r.end }]) as never },
            markLine: { silent: true, symbol: "none", label: { show: false }, data: marks },
          },
        ],
      });
    },
    [panel.data, events, show, regimes, node, info?.label],
  );

  const eventCounts = useMemo(() => {
    const out: Record<EventType, number> = { fomc: 0, cpi: 0, nfp: 0 };
    for (const e of events) if (e.date >= "2006-01-01" && e.date <= "2026-06-30") out[e.event] += 1;
    return out;
  }, [events]);

  return (
    <Panel
      title={info?.label ?? node}
      sub={<>Source <code className="font-mono">{info?.source}</code> · cluster {info?.cluster}. Shaded: regime windows. Vertical lines: macro events (red = unscheduled FOMC action). Drag the slider or scroll on the chart to zoom.</>}
      tools={
        <>
          <Select id="market-node" label="market" value={node} onChange={setNode} options={nodes.map((n) => ({ value: n.key, label: n.key }))} />
          <Seg label="Frequency" value={freq} onChange={setFreq} options={FREQS} />
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        Events:
        {(Object.keys(EVENT_LABEL) as EventType[]).map((e) => (
          <label key={e} className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-line px-2 py-0.5">
            <input type="checkbox" checked={show[e]} onChange={() => setShow((s) => ({ ...s, [e]: !s[e] }))} />
            {EVENT_LABEL[e]} <span className="font-mono text-faint">{eventCounts[e]}</span>
          </label>
        ))}
      </div>
      {panel.error && <ErrorNote error={panel.error} />}
      {panel.data ? <Chart build={build} name={`market-${node}-${freq.toLowerCase()}`} height={400} /> : !panel.error && <Loading what="prices" />}
    </Panel>
  );
}

// ---------------------------------------------------------------- feature browser
function FeaturePanel({ nodes }: { nodes: NodeInfo[] }) {
  const [node, setNode] = useState("silver");
  const eventFeatures = ["cpi_days_to", "cpi_days_since", "nfp_days_to", "nfp_days_since", "fomc_days_to", "fomc_days_since"];
  const featureList = node === "events" ? eventFeatures : (nodes.find((n) => n.key === node)?.features ?? []);
  const [feature, setFeature] = useState("r21");
  const current = featureList.includes(feature) ? feature : featureList[0];
  const [freq, setFreq] = useState<Freq>("D");
  const data = useApi<Columns>(current ? `/api/data/features?node=${node}&names=${current}&freq=${freq}` : null);
  const st = data.data ? stats(data.data[current] ?? []) : null;

  const build = useCallback(
    (t: ChartTheme) => {
      const c = data.data!;
      return base(t, {
        legend: { show: false },
        tooltip: tooltip(t, { valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(4) : "–") }),
        grid: { left: 58, right: 18, top: 16, bottom: 64 },
        xAxis: axis(t, "time"),
        yAxis: axis(t, "value", { scale: true }),
        dataZoom: zoom(t),
        series: [{ name: `${node}.${current}`, type: "line", showSymbol: false, color: t.signal, lineStyle: { width: 1.1, color: t.signal }, data: xy(c.date, c[current]) }],
      });
    },
    [data.data, node, current],
  );

  return (
    <Panel
      title="Feature browser"
      sub="Any feature of any node, exactly as the models see it (before scaling)."
      tools={
        <>
          <Select id="feat-node" label="node" value={node} onChange={setNode} options={[...nodes.map((n) => ({ value: n.key, label: n.key })), { value: "events", label: "events" }]} />
          <Select id="feat-name" label="feature" value={current ?? ""} onChange={setFeature} options={featureList.map((f) => ({ value: f, label: f }))} />
          <Seg label="Frequency" value={freq} onChange={setFreq} options={FREQS} />
        </>
      }
    >
      {data.error && <ErrorNote error={data.error} />}
      {data.data ? <Chart build={build} name={`feature-${node}-${current}`} height={320} /> : !data.error && <Loading what="feature" />}
      {st && (
        <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs text-muted">
          <span>n <b className="text-text">{st.n.toLocaleString("en")}</b></span>
          <span>missing <b className="text-text">{st.missing}</b></span>
          <span>mean <b className="text-text">{num(st.mean, 4)}</b></span>
          <span>sd <b className="text-text">{num(st.sd, 4)}</b></span>
          <span>min <b className="text-text">{num(st.min, 4)}</b></span>
          <span>max <b className="text-text">{num(st.max, 4)}</b></span>
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- gold-silver ECT
function GsrPanel({ regimes }: { regimes: Regime[] }) {
  const d = useApi<Columns>("/api/data/features?node=silver&names=gsr_ect&freq=W-FRI");
  const build = useCallback(
    (t: ChartTheme) => {
      const c = d.data!;
      return base(t, {
        legend: { show: false },
        tooltip: tooltip(t, { valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "–") }),
        grid: { left: 44, right: 16, top: 16, bottom: 30 },
        xAxis: axis(t, "time"),
        yAxis: axis(t, "value"),
        series: [
          {
            name: "gsr_ect",
            type: "line",
            showSymbol: false,
            color: t.gold,
            lineStyle: { width: 1.3, color: t.gold },
            data: xy(c.date, c.gsr_ect),
            markArea: { silent: true, itemStyle: { color: t.band }, data: regimes.map((r) => [{ xAxis: r.start }, { xAxis: r.end }]) as never },
            markLine: { silent: true, symbol: "none", label: { color: t.muted, fontSize: 10, formatter: (p: unknown) => { const v = (p as { value: number }).value; return (v > 0 ? "+" : "") + v; } }, lineStyle: { color: t.faint, type: "dashed" }, data: [{ yAxis: 2 }, { yAxis: -2 }] },
          },
        ],
      });
    },
    [d.data, regimes],
  );
  return (
    <Panel title="Gold–silver error-correction term" sub="z-score of ln(gold/silver) against its trailing year. Above +2: silver unusually cheap relative to gold (mean-reversion pressure).">
      {d.error && <ErrorNote error={d.error} />}
      {d.data ? <Chart build={build} name="gold-silver-ect" height={280} /> : !d.error && <Loading what="feature" />}
    </Panel>
  );
}

// ---------------------------------------------------------------- quality + snapshot
function QualityPanel() {
  const q = useApi<Row[]>("/api/data/quality");
  return (
    <Panel title="Data quality (audit of the frozen snapshot)" sub="Repaired: close/open outside the day's range (Yahoo, mostly 2006–2011). Flat: no range information. Filled: carried forward ≤ 5 days.">
      {q.error && <ErrorNote error={q.error} />}
      {q.data && (
        <Table head={<><th className={TH}>Node</th><th className={TH}>Raw rows</th><th className={TH}>Repaired</th><th className={TH}>Flat bars</th><th className={TH}>Filled</th><th className={TH}>Missing</th><th className={TH}>Volume</th></>}>
          {q.data.map((r) => (
            <tr key={r.node as string} className="hover:bg-panel-2">
              <td className={`${TD} ${NUM}`}>{r.node as string}</td>
              <td className={`${TD} ${NUM}`}>{r.rows_raw as number}</td>
              <td className={`${TD} ${NUM}`}>{r.repaired_range as number}</td>
              <td className={`${TD} ${NUM}`}>{r.flat_bars as number}</td>
              <td className={`${TD} ${NUM}`}>{r.filled as number}</td>
              <td className={`${TD} ${NUM}`}>{r.missing_after_fill as number}</td>
              <td className={TD}><Pill tone={r.has_volume ? "good" : "neutral"}>{r.has_volume ? "yes" : "none"}</Pill></td>
            </tr>
          ))}
        </Table>
      )}
    </Panel>
  );
}

function SnapshotPanel({ nodes }: { nodes: NodeInfo[] }) {
  const [node, setNode] = useState("silver");
  const [date, setDate] = useState("2020-03-16");
  const f = useApi<Columns>(`/api/data/features?node=${node}`);
  const row = useMemo(() => {
    if (!f.data) return null;
    const dates = f.data.date as string[];
    // Latest trading day on or before the chosen date.
    let i = dates.length - 1;
    while (i >= 0 && dates[i] > date) i -= 1;
    if (i < 0) return null;
    return { day: dates[i], values: Object.keys(f.data).filter((k) => k !== "date").map((k) => [k, f.data![k][i]] as const) };
  }, [f.data, date]);
  return (
    <Panel
      title="Features on one day"
      sub="Everything a model saw at the close of the chosen day (the latest trading day on or before it)."
      tools={
        <>
          <Select id="snap-node" label="node" value={node} onChange={setNode} options={nodes.map((n) => ({ value: n.key, label: n.key }))} />
          <label htmlFor="snap-date" className="inline-flex items-center gap-2 text-xs text-muted">
            date
            <input id="snap-date" type="date" min="2006-01-03" max="2026-06-30" value={date} onChange={(e) => e.target.value && setDate(e.target.value)}
              className="rounded-md border border-line bg-panel-2 px-2 py-0.5 font-mono text-xs text-text" />
          </label>
        </>
      }
    >
      {f.error && <ErrorNote error={f.error} />}
      {!f.data && !f.error && <Loading what="features" />}
      {row && (
        <>
          <p className="m-0 font-mono text-xs text-muted">trading day {row.day}</p>
          <Table head={<><th className={TH}>Feature</th><th className={TH}>Value</th><th className={`${TH} ${TXT}`}>Feature</th><th className={TH}>Value</th></>}>
            {Array.from({ length: Math.ceil(row.values.length / 2) }, (_, i) => {
              const a = row.values[i];
              const b = row.values[i + Math.ceil(row.values.length / 2)];
              return (
                <tr key={a[0]}>
                  <td className={`${TD} ${NUM}`}>{a[0]}</td>
                  <td className={`${TD} ${NUM}`}>{num(a[1], 4)}</td>
                  <td className={`${TD} ${NUM} ${TXT}`}>{b?.[0] ?? ""}</td>
                  <td className={`${TD} ${NUM}`}>{b ? num(b[1], 4) : ""}</td>
                </tr>
              );
            })}
          </Table>
        </>
      )}
    </Panel>
  );
}

export default function Data() {
  const nodes = useApi<NodeInfo[]>("/api/data/nodes");
  const regimes = useApi<Regime[]>("/api/data/regimes");
  const events = useApi<EventRow[]>("/api/data/events");
  const ready = nodes.data && regimes.data && events.data;
  const err = nodes.error ?? regimes.error ?? events.error;
  return (
    <div className="flex flex-col gap-5">
      <PageHeader eyebrow="Data" title="Markets, features and data quality">
        Every market aligned onto silver's 5,154 trading days (gaps filled at most 5 days, FRED values usable the next day,
        bad bars repaired), with the macro event calendar, every feature the models see, and the data audit.
      </PageHeader>
      {err && <ErrorNote error={err} />}
      {!ready && !err && <Loading what="data" />}
      {ready && (
        <>
          <MarketPanel nodes={nodes.data!} regimes={regimes.data!} events={events.data!} />
          <FeaturePanel nodes={nodes.data!} />
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <GsrPanel regimes={regimes.data!} />
            <QualityPanel />
          </div>
          <SnapshotPanel nodes={nodes.data!} />
        </>
      )}
    </div>
  );
}
