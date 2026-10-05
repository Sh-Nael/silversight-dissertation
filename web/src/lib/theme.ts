// Chart theme: reads the design tokens (CSS variables) so charts follow figure mode,
// plus fixed per-model colours and the shared axis/tooltip styling from the U1.1 design.
import { createContext } from "react";
import type { EChartsOption } from "echarts";

/** True when figure mode is on. Charts re-render when it changes. */
export const FigureContext = createContext(false);

export interface ChartTheme {
  fig: boolean;
  text: string;
  muted: string;
  faint: string;
  line: string;
  grid: string;
  band: string;
  panel: string;
  silver: string;
  signal: string;
  signalDim: string;
  good: string;
  bad: string;
  warn: string;
  gold: string;
}

export function chartTheme(): ChartTheme {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    fig: document.documentElement.classList.contains("fig"),
    text: v("--text"),
    muted: v("--muted"),
    faint: v("--faint"),
    line: v("--line"),
    grid: v("--grid"),
    band: v("--band"),
    panel: v("--panel"),
    silver: v("--silver"),
    signal: v("--signal"),
    signalDim: v("--signal-dim"),
    good: v("--good"),
    bad: v("--bad"),
    warn: v("--warn"),
    gold: v("--goldc"),
  };
}

// One colour per model everywhere (dark console / figure mode).
const MODEL_COLORS: Record<string, [string, string]> = {
  climatology: ["#6E7A87", "#5B6570"],
  ewma: ["#9D8CE0", "#6F5CC2"],
  ar_garch: ["#E0A458", "#7F5539"],
  linear: ["#4CC2D6", "#1C7F92"],
  gbm: ["#E8739A", "#C2456F"],
  lstm: ["#7FD17F", "#3E8E3E"],
  gru: ["#D08BE0", "#9B2FAE"],
  static_gnn: ["#6FA8FF", "#2F63C0"],
  aimdg: ["#F2D06B", "#A8860F"],
  ens_lin_gbm: ["#F28F6B", "#B5481F"],
  ens_lin_gbm_sgnn: ["#C79BF2", "#7A3FB8"],
  aimdg_replay: ["#D9C48A", "#7D6A2A"],
  aimdg_A1: ["#5FD6B4", "#1F8A6C"],
  aimdg_A2: ["#F29BD0", "#B03A84"],
};

export function modelColor(model: string, fig: boolean): string {
  const c = MODEL_COLORS[model];
  return c ? c[fig ? 1 : 0] : fig ? "#444" : "#bbb";
}

export const MONO = "IBM Plex Mono, ui-monospace, monospace";
export const SANS = "IBM Plex Sans, system-ui, sans-serif";

export function tooltip(t: ChartTheme, extra: Record<string, unknown> = {}): EChartsOption["tooltip"] {
  return {
    trigger: "axis",
    backgroundColor: t.panel,
    borderColor: t.line,
    textStyle: { color: t.text, fontSize: 12, fontFamily: SANS },
    ...extra,
  };
}

export function base(t: ChartTheme, extra: EChartsOption = {}): EChartsOption {
  return {
    backgroundColor: "transparent",
    animationDuration: 400,
    textStyle: { fontFamily: SANS, color: t.text },
    grid: { left: 56, right: 18, top: 30, bottom: 36 },
    tooltip: tooltip(t),
    legend: { top: 0, right: 0, textStyle: { color: t.muted, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    ...extra,
  };
}

type AxisType = "value" | "category" | "time" | "log";

export function axis(t: ChartTheme, type: AxisType, extra: Record<string, unknown> = {}) {
  return {
    type,
    axisLine: { lineStyle: { color: t.line } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 11, fontFamily: MONO },
    splitLine: { lineStyle: { color: t.grid } },
    nameTextStyle: { color: t.muted, fontSize: 11 },
    ...extra,
  };
}

/** Chart points from a date column and a value column; missing values become gaps ("-"). */
export function xy(x: unknown[], y: unknown[]): [string, number | string][] {
  return x.map((d, i) => [String(d), typeof y[i] === "number" ? (y[i] as number) : "-"]);
}
