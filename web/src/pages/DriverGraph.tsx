// Driver graph: which driver edges AIM-DG kept, when, and whether that follows market regimes
// (Phase 6.3, D-40). Timeline heat map, stress-versus-rest activity with its tests, and the
// graph of any chosen month.
import { useCallback, useMemo, useState } from "react";
import Chart from "../components/Chart";
import { Seg, Select } from "../components/controls";
import { ErrorNote, Kpi, Loading, NUM, PageHeader, Panel, Pill, TD, TH, TXT, Table } from "../components/ui";
import { useApi, type RunSummary } from "../lib/api";
import { num, pct } from "../lib/format";
import { axis, base, tooltip, type ChartTheme } from "../lib/theme";

interface Persist { edge: string; cluster: string; share_active: number; switches: number; mean_spell_months: number }
interface TestRow { cluster: string; measure: string; regime: string; hypothesis: string; inside: number | null; outside: number | null; diff: number | null; p: number | null; n_inside: number; primary: boolean }
interface Edges {
  edges: string[];
  clusters: Record<string, string>;
  dates: string[];
  on: number[][];
  reliability: number[][] | null;
  n_edges: number[];
  activity: Record<string, number[]>;
  persistence: Persist[];
  overall: { mean_jaccard: number; mean_edges_changed: number; months_unchanged: number; months: number; mean_edges: number };
  tests: TestRow[];
  regimes: { label: string; start: string; end: string }[];
}

const CLUSTER_ORDER = ["monetary", "industrial", "risk"];
const DRIVER_LABEL: Record<string, string> = { gold: "Gold", dxy: "US dollar", real_yield_10y: "Real yield", copper: "Copper", vix: "VIX" };
const clusterColor = (t: ChartTheme, c: string) => (c === "monetary" ? t.gold : c === "industrial" ? t.warn : t.signal);

export default function DriverGraph() {
  const runs = useApi<RunSummary[]>("/api/runs");
  const candidates = useMemo(() => (runs.data ?? []).filter((r) => r.model.startsWith("aimdg")), [runs.data]);
  const [picked, setPicked] = useState<string | null>(null);
  const runId = picked ?? candidates.find((r) => r.model === "aimdg")?.id ?? candidates[0]?.id ?? null;
  const d = useApi<Edges>(runId ? `/api/runs/${runId}/edges` : null);
  const [view, setView] = useState<"on" | "reliability">("on");
  const [month, setMonth] = useState<number | null>(null);
  const m = d.data ? Math.min(month ?? d.data.dates.length - 1, d.data.dates.length - 1) : 0;

  // edges ordered by cluster, then driver, then lag
  const order = useMemo(() => {
    if (!d.data) return [];
    return d.data.edges
      .map((e, i) => ({ e, i, c: d.data!.clusters[e] }))
      .sort((a, b) => CLUSTER_ORDER.indexOf(a.c) - CLUSTER_ORDER.indexOf(b.c) || a.i - b.i);
  }, [d.data]);

  const heat = useCallback(
    (t: ChartTheme) => {
      const x = d.data!;
      const showR = view === "reliability" && x.reliability !== null;
      const cells: (number | string)[][] = [];
      x.dates.forEach((date, j) => order.forEach((o, k) => cells.push([date, k, showR ? x.reliability![j][o.i] : x.on[j][o.i]])));
      return base(t, {
        tooltip: tooltip(t, {
          trigger: "item",
          formatter: (p: { value: (number | string)[] }) => {
            const o = order[p.value[1] as number];
            return `${o.e} · ${o.c}<br>${p.value[0]}: ${showR ? `reliability ${(p.value[2] as number).toFixed(3)}` : p.value[2] ? "active" : "off"}`;
          },
        }),
        legend: { show: false },
        grid: { left: 118, right: 16, top: 26, bottom: 44 },
        // a label at the first re-selection of every second year
        xAxis: axis(t, "category", {
          data: x.dates,
          axisTick: { show: false },
          axisLabel: {
            color: t.muted,
            fontSize: 11,
            interval: (i: number, v: string) => Number(v.slice(0, 4)) % 2 === 0 && (i === 0 || x.dates[i - 1].slice(0, 4) !== v.slice(0, 4)),
            formatter: (v: string) => v.slice(0, 4),
          },
          splitLine: { show: false },
        }),
        yAxis: axis(t, "category", { data: order.map((o) => o.e), inverse: true, splitLine: { show: false }, axisLabel: { color: t.muted, fontSize: 11 } }),
        visualMap: showR
          ? { min: 0, max: 0.6, calculable: false, orient: "horizontal", left: "center", bottom: 0, itemHeight: 120, textStyle: { color: t.muted, fontSize: 11 }, inRange: { color: [t.panel, t.signalDim, t.signal] } }
          : { type: "piecewise", show: false, pieces: [{ value: 0, color: t.grid }, { value: 1, color: t.signal }] },
        series: [
          {
            type: "heatmap",
            data: cells,
            progressive: 0,
            itemStyle: { borderWidth: 0 },
            markArea: {
              silent: true,
              itemStyle: { color: "transparent", borderColor: t.warn, borderWidth: 1, borderType: "dashed" },
              label: { color: t.muted, fontSize: 10, position: "insideTop", distance: -14 },
              data: x.regimes
                .map((r) => {
                  const inside = x.dates.filter((v) => v >= r.start && v <= r.end);
                  return inside.length ? [{ name: r.label, xAxis: inside[0] }, { xAxis: inside[inside.length - 1] }] : null;
                })
                .filter(Boolean) as never,
            },
          },
        ] as never,
      });
    },
    [d.data, order, view],
  );

  const bars = useCallback(
    (t: ChartTheme) => {
      const rows = d.data!.tests.filter((r) => r.primary && r.measure === "activity");
      const by = (c: string) => rows.find((r) => r.cluster === c);
      return base(t, {
        tooltip: tooltip(t, { valueFormatter: (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "–") }),
        grid: { left: 64, right: 14, top: 30, bottom: 28 },
        xAxis: axis(t, "category", { data: CLUSTER_ORDER, splitLine: { show: false } }),
        yAxis: axis(t, "value", { name: "share of the cluster's edges active", nameLocation: "middle", nameGap: 44,
          nameTextStyle: { color: t.muted, fontSize: 11 },
          axisLabel: { color: t.muted, fontSize: 11, formatter: (v: number) => `${(v * 100).toFixed(0)}%` } }),
        series: [
          { name: "stress months", type: "bar", barGap: "10%", color: t.warn, itemStyle: { color: t.warn }, data: CLUSTER_ORDER.map((c) => by(c)?.inside ?? null) },
          { name: "other months", type: "bar", color: t.faint, itemStyle: { color: t.faint }, data: CLUSTER_ORDER.map((c) => by(c)?.outside ?? null) },
        ],
      });
    },
    [d.data],
  );

  const graph = useCallback(
    (t: ChartTheme) => {
      const x = d.data!;
      const drivers = Object.keys(DRIVER_LABEL).filter((k) => x.edges.some((e) => e.startsWith(`${k}@`)));
      const lagsOn = (k: string) => x.edges.map((e, i) => ({ e, i })).filter((o) => o.e.startsWith(`${k}@`) && x.on[m][o.i] === 1).map((o) => o.e.split("@")[1]);
      const nodes = [
        { name: "Silver", x: 0, y: 0, symbolSize: 58, itemStyle: { color: t.silver }, label: { show: true, color: t.panel, fontWeight: 600 } },
        ...drivers.map((k, j) => {
          const a = (2 * Math.PI * j) / drivers.length - Math.PI / 2;
          const on = lagsOn(k).length > 0;
          const c = clusterColor(t, x.clusters[`${k}@1`]);
          return { name: DRIVER_LABEL[k], x: 220 * Math.cos(a), y: 220 * Math.sin(a), symbolSize: on ? 46 : 34, itemStyle: { color: on ? c : t.grid, borderColor: c, borderWidth: 1.5 }, label: { show: true, color: on ? t.panel : t.muted, fontSize: 11 } };
        }),
      ];
      const links = drivers
        .filter((k) => lagsOn(k).length > 0)
        .map((k) => ({ source: DRIVER_LABEL[k], target: "Silver", lineStyle: { width: 1.5 + 2 * lagsOn(k).length, color: clusterColor(t, x.clusters[`${k}@1`]), opacity: 0.85 }, label: { show: true, formatter: `lag ${lagsOn(k).join(", ")}`, color: t.text, fontSize: 11 } }));
      return base(t, {
        tooltip: { show: false },
        legend: { show: false },
        series: [{ type: "graph", layout: "none", roam: false, data: nodes, links, edgeSymbol: ["none", "arrow"], edgeSymbolSize: 9, left: 60, right: 60, top: 40, bottom: 40 }] as never,
      });
    },
    [d.data, m],
  );

  const primary = d.data?.tests.filter((r) => r.primary) ?? [];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <PageHeader eyebrow="Driver graph" title="Which drivers move silver, and when">
          Every month AIM-DG re-selects which of its 15 driver edges (5 drivers × lags of 1, 5 and 21 days) silver listens to.
          The regimes, measures and tests below were fixed before any by-date result was looked at (D-40).
        </PageHeader>
        {candidates.length > 1 && runId && (
          <Select id="edge-run" label="run" value={runId} onChange={setPicked} options={candidates.map((r) => ({ value: r.id, label: r.id }))} />
        )}
      </div>
      {runs.error && <ErrorNote error={runs.error} />}
      {d.error && <ErrorNote error={d.error} />}
      {!d.data && !d.error && <Loading what="edge dynamics" />}
      {d.data && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Kpi label="Edges active" value={num(d.data.overall.mean_edges, 2)} note="of 15, on average" />
            <Kpi label="Changed per month" value={num(d.data.overall.mean_edges_changed, 2)} note={`${d.data.overall.months_unchanged} of ${d.data.overall.months - 1} months unchanged`} />
            <Kpi label="Month-to-month overlap" value={pct(d.data.overall.mean_jaccard)} note="Jaccard similarity of the active sets" />
            <Kpi label="Most persistent edge" value={[...d.data.persistence].sort((a, b) => b.share_active - a.share_active)[0].edge} note={`active ${pct([...d.data.persistence].sort((a, b) => b.share_active - a.share_active)[0].share_active)} of months`} />
          </div>
          <Panel
            title={view === "on" ? "Edge timeline · active or off" : "Edge timeline · reliability"}
            sub="One column per monthly re-selection, one row per edge (monetary, industrial, risk). Dashed boxes mark the regimes."
            tools={<Seg label="Show" value={view} onChange={setView} options={[{ value: "on", label: "active" }, { value: "reliability", label: "reliability" }]} />}
          >
            <Chart build={heat} name={`edge-timeline-${view}`} height={440} />
          </Panel>
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <Panel title="Active share of each cluster's edges · stress months against the rest" sub="Stress = 2011 silver spike, 2013 taper tantrum, 2020 COVID, 2022 rate shock. Hypotheses: monetary and risk higher in stress, industrial lower.">
              <Chart build={bars} name="cluster-activity-stress" height={300} />
              <Table head={<><th className={TH}>Cluster</th><th className={TH}>Measure</th><th className={TH}>Stress</th><th className={TH}>Other</th><th className={TH}>Hypothesis</th><th className={TH}>p (one-sided)</th></>}>
                {primary.map((r) => (
                  <tr key={`${r.cluster}-${r.measure}`} className="hover:bg-panel-2">
                    <td className={`${TD} ${TXT}`}>{r.cluster}</td>
                    <td className={`${TD} ${TXT}`}>{r.measure}</td>
                    <td className={`${TD} ${NUM}`}>{r.measure === "activity" ? pct(r.inside) : num(r.inside, 3)}</td>
                    <td className={`${TD} ${NUM}`}>{r.measure === "activity" ? pct(r.outside) : num(r.outside, 3)}</td>
                    <td className={`${TD} ${TXT}`}>{r.hypothesis === "greater" ? "higher in stress" : "lower in stress"}</td>
                    <td className={`${TD} ${NUM}`}><Pill tone={typeof r.p === "number" && r.p < 0.05 ? "good" : "neutral"}>{num(r.p, 3)}</Pill></td>
                  </tr>
                ))}
              </Table>
            </Panel>
            <Panel
              title={`The driver graph · ${d.data.dates[m]}`}
              sub="Active drivers are filled; the line is thicker with more active lags. Colours: monetary, industrial, risk."
              tools={<Select id="edge-month" label="month" value={String(m)} onChange={(v) => setMonth(Number(v))} options={d.data.dates.map((x, j) => ({ value: String(j), label: x.slice(0, 7) }))} />}
            >
              <Chart build={graph} name={`driver-graph-${d.data.dates[m]}`} height={420} />
            </Panel>
          </div>
          <Panel title="Persistence per edge" sub="Share of months active, number of switches and the mean length of an active spell.">
            <Table head={<><th className={TH}>Edge</th><th className={TH}>Cluster</th><th className={TH}>Active</th><th className={TH}>Switches</th><th className={TH}>Mean spell (months)</th></>}>
              {order.map((o) => {
                const r = d.data!.persistence.find((p) => p.edge === o.e)!;
                return (
                  <tr key={o.e} className="hover:bg-panel-2">
                    <td className={`${TD} ${TXT} ${NUM}`}>{r.edge}</td>
                    <td className={`${TD} ${TXT}`}>{r.cluster}</td>
                    <td className={`${TD} ${NUM}`}>{pct(r.share_active)}</td>
                    <td className={`${TD} ${NUM}`}>{r.switches}</td>
                    <td className={`${TD} ${NUM}`}>{num(r.mean_spell_months, 1)}</td>
                  </tr>
                );
              })}
            </Table>
          </Panel>
        </>
      )}
    </div>
  );
}
