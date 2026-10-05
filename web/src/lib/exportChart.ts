// Figure export for the dissertation (U1.6).
//
// A chart is re-rendered off-screen in the white figure palette at a fixed print width
// and downloaded as SVG (vector) or PNG (3x pixel density). The palette switch, render
// and switch-back happen synchronously in one task, so the screen never repaints in
// between and the user sees no flicker.
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { chartTheme, type ChartTheme } from "./theme";

export type ExportKind = "svg" | "png";

/** Logical width of an exported figure in CSS pixels; PNG is rendered at 3x this. */
export const EXPORT_WIDTH = 800;
export const PNG_PIXEL_RATIO = 3; // 2400 px wide = 300 dpi at 20.3 cm

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [head, body] = dataUrl.split(",");
  const mime = head.match(/data:([^;]+)/)?.[1] ?? "application/octet-stream";
  const bytes = atob(body);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

/** Render `build` in the figure palette at print size; return the file as a Blob. */
export function renderFigure(build: (t: ChartTheme) => EChartsOption, kind: ExportKind, aspect: number): Blob {
  const root = document.documentElement;
  const wasFig = root.classList.contains("fig");
  const height = Math.round(Math.min(Math.max(EXPORT_WIDTH * aspect, 280), 900));
  const host = document.createElement("div");
  host.style.cssText = `position:fixed;left:-10000px;top:0;width:${EXPORT_WIDTH}px;height:${height}px`;
  document.body.appendChild(host);
  root.classList.add("fig");
  try {
    const option = { ...build(chartTheme()), animation: false, backgroundColor: "#ffffff" } as EChartsOption;
    // Figures carry no interactive zoom slider.
    if (option.dataZoom) option.dataZoom = (option.dataZoom as object[]).filter((z) => (z as { type?: string }).type !== "slider");
    if (option.grid && !Array.isArray(option.grid)) option.grid = { ...option.grid, bottom: Math.min((option.grid.bottom as number) ?? 36, 40) };
    const chart = echarts.init(host, null, { renderer: kind === "svg" ? "svg" : "canvas", width: EXPORT_WIDTH, height });
    chart.setOption(option);
    const blob =
      kind === "svg"
        ? new Blob([chart.renderToSVGString()], { type: "image/svg+xml" })
        : dataUrlToBlob(chart.getDataURL({ type: "png", pixelRatio: PNG_PIXEL_RATIO, backgroundColor: "#ffffff" }));
    chart.dispose();
    return blob;
  } finally {
    if (!wasFig) root.classList.remove("fig");
    host.remove();
  }
}

export function exportChart(build: (t: ChartTheme) => EChartsOption, name: string, kind: ExportKind, aspect: number) {
  const blob = renderFigure(build, kind, aspect);
  const stamp = new Date().toISOString().slice(0, 10);
  download(blob, `silversight-${name}-${stamp}.${kind}`);
}
