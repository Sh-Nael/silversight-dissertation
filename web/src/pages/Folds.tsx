// Folds: the frozen walk-forward calendar, refits and tuning points on one timeline.
import { useCallback } from "react";
import Chart from "../components/Chart";
import { ErrorNote, Kpi, Loading, NUM, PageHeader, Panel, TD, TH, Table } from "../components/ui";
import { useApi, type FoldsResponse } from "../lib/api";
import { axis, base, type ChartTheme } from "../lib/theme";

const ts = (s: string) => Date.parse(`${s}T00:00:00Z`);

export default function Folds() {
  const f = useApi<FoldsResponse>("/api/folds");

  const timeline = useCallback(
    (t: ChartTheme) => {
      const c = f.data!.calendar;
      const cats = c.folds.map((x) => `F${x.k}`).concat(["tuning"]);
      // [row, start, end, kind]: kind 0 = training span, 1 = test block
      const bars = c.folds.flatMap((x, i) => [
        [i, ts(c.train_start), ts(x.first_train_end), 0],
        [i, ts(x.test_start), ts(x.test_end), 1],
      ]);
      return base(t, {
        tooltip: { show: false },
        legend: { show: false },
        grid: { left: 60, right: 16, top: 12, bottom: 30 },
        xAxis: axis(t, "time", { min: ts("2006-06-01"), max: ts("2026-12-31") }),
        yAxis: axis(t, "category", { data: cats, inverse: true, splitLine: { show: false } }),
        series: [
          {
            type: "custom",
            encode: { x: [1, 2], y: 0 },
            data: bars,
            renderItem: (_params: unknown, api: { value: (i: number) => number; coord: (v: number[]) => number[]; size: (v: number[]) => number[] }) => {
              const y = api.value(0);
              const s = api.coord([api.value(1), y]);
              const e = api.coord([api.value(2), y]);
              const hgt = api.size([0, 1])[1] * 0.55;
              return {
                type: "rect",
                shape: { x: s[0], y: s[1] - hgt / 2, width: Math.max(1, e[0] - s[0]), height: hgt, r: 2 },
                style: { fill: api.value(3) === 1 ? t.signal : t.line },
              };
            },
          },
          { type: "scatter", symbol: "diamond", symbolSize: 9, itemStyle: { color: t.warn }, data: f.data!.tuning_points.map((d) => [d, "tuning"]) },
        ] as never,
      });
    },
    [f.data],
  );

  return (
    <div className="flex flex-col gap-5">
      <PageHeader eyebrow="Folds" title="The frozen walk-forward calendar">
        Frozen on 29 Jul 2026 (<code className="font-mono">experiments/folds_v1.json</code>) before any model was run. Each
        fold trains on everything before it; models are refitted every 21 trading days and re-tuned at yearly tuning points.
      </PageHeader>
      {f.error && <ErrorNote error={f.error} />}
      {!f.data && !f.error && <Loading what="calendar" />}
      {f.data && (
        <>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
            <Kpi label="Folds" value={f.data.calendar.folds.length} note="test blocks, ~2.1 years each" />
            <Kpi label="Test days" value={f.data.calendar.folds.reduce((a, x) => a + x.n_test, 0).toLocaleString("en")} note={`${f.data.calendar.folds[0].test_start} → ${f.data.calendar.folds.at(-1)!.test_end}`} />
            <Kpi label="Refits" value={f.data.refits.length} note={`every ${f.data.calendar.refit_every} trading days`} />
            <Kpi label="Tuning points" value={f.data.tuning_points.length} note="yearly hyperparameter search" />
            <Kpi label="Purge" value={`${f.data.calendar.purge} days`} note="longest label horizon" />
          </div>
          <Panel title="Timeline" sub="Grey: expanding training span. Cyan: test block. Diamonds: tuning points.">
            <Chart build={timeline} name="walk-forward-timeline" height={330} />
          </Panel>
          <Panel title="Folds" sub="Training always starts on the first day all features exist; each fold's last training day is 5 trading days (the purge) before its test block.">
            <Table head={<><th className={TH}>Fold</th><th className={TH}>Test start</th><th className={TH}>Test end</th><th className={TH}>Test days</th><th className={TH}>Refits</th><th className={TH}>Last training day</th></>}>
              {f.data.calendar.folds.map((x) => (
                <tr key={x.k} className="hover:bg-panel-2">
                  <td className={`${TD} ${NUM}`}>F{x.k}</td>
                  <td className={`${TD} ${NUM}`}>{x.test_start}</td>
                  <td className={`${TD} ${NUM}`}>{x.test_end}</td>
                  <td className={`${TD} ${NUM}`}>{x.n_test}</td>
                  <td className={`${TD} ${NUM}`}>{x.n_refits}</td>
                  <td className={`${TD} ${NUM}`}>{x.first_train_end}</td>
                </tr>
              ))}
            </Table>
          </Panel>
        </>
      )}
    </div>
  );
}
