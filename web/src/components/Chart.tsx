// One ECharts chart, drawn as SVG, with figure export.
//
// `build` receives the current chart theme (colours read from the design tokens) and
// returns the ECharts option. The chart is rebuilt when `build` changes or when
// figure mode flips, and it resizes with its container. `name` names exported files
// (silversight-<name>-<date>.svg / .png); see lib/exportChart.ts.
import { useContext, useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { exportChart, type ExportKind } from "../lib/exportChart";
import { FigureContext, chartTheme, type ChartTheme } from "../lib/theme";

export default function Chart({
  build,
  name,
  height = 320,
}: {
  build: (t: ChartTheme) => EChartsOption;
  name: string;
  height?: number;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const fig = useContext(FigureContext);

  useEffect(() => {
    const node = el.current!;
    chart.current = echarts.init(node, null, { renderer: "svg" });
    const ro = new ResizeObserver(() => chart.current?.resize());
    ro.observe(node);
    return () => {
      ro.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(build(chartTheme()), true);
  }, [build, fig]);

  const save = (kind: ExportKind) => {
    const w = el.current?.clientWidth || 800;
    exportChart(build, name, kind, height / w);
  };

  return (
    <div className="flex flex-col gap-1">
      <div ref={el} className="w-full" style={{ height }} data-chart={name} />
      <div className="flex items-center justify-end gap-1.5 text-[11px] text-faint">
        <span>Export figure</span>
        {(["svg", "png"] as ExportKind[]).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => save(k)}
            aria-label={`Export ${name} as ${k.toUpperCase()}`}
            title={k === "svg" ? "Vector (Word, LaTeX, Inkscape)" : "PNG at 3× (300 dpi at 20 cm)"}
            className="cursor-pointer rounded border border-line bg-panel-2 px-2 py-0.5 font-mono text-[11px] text-muted uppercase hover:border-signal hover:text-text"
          >
            {k}
          </button>
        ))}
      </div>
    </div>
  );
}
