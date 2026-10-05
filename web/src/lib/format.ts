// Number and date formatting used across pages (tabular, fixed decimals).
export function num(v: unknown, digits = 4): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "–";
}

export function pct(v: unknown, digits = 1): string {
  return typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : "–";
}

export function secs(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  return v >= 100 ? `${v.toFixed(0)} s` : `${v.toFixed(1)} s`;
}

export function when(iso: string): string {
  return iso.replace("T", " ").replace("+00:00", " UTC");
}

/** Losses where lower is better (everything else in our tables: higher is better). */
export const LOWER_IS_BETTER = new Set(["brier", "logloss", "vol_qlike", "vol_rmse_vol"]);
