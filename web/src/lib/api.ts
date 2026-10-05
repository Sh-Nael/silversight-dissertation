// Typed access to the SilverSight API. Types mirror the pydantic models in
// xaglab/api/routes/*.py; keep them in sync when an endpoint changes
// (reference: http://localhost:8000/docs).
import { useEffect, useState } from "react";

export type Cell = number | string | boolean | null;
/** Columnar data: {"date": [...], "<column>": [...]}, NaN sent as null. */
export type Columns = Record<string, Cell[]>;
export type Row = Record<string, Cell>;

export interface SnapshotStatus {
  name: string;
  files: number;
  snapshot_end: string;
  verified: boolean;
  error: string | null;
}

export interface Status {
  ok: boolean;
  code: { commit: string; dirty: boolean; version: string };
  snapshots: SnapshotStatus[];
  runs: number;
}

export interface RunSummary {
  id: string;
  model: string;
  tag: string | null;
  started_utc: string;
  wall_seconds: number;
  fit_seconds_total: number;
  n_refits: number;
  tuning_points: number | null;
  commit: string;
  dirty: boolean;
  headline: Record<"brier1" | "brier5" | "logloss1" | "accuracy1" | "qlike1" | "qlike5", number | null>;
}

export interface TuningPoint {
  tuning_point: number;
  fold: number;
  refit: number;
  origin: string;
  train_end: string;
  seconds: number;
  results: Record<string, { params: Record<string, number>; cv_loss: number; n_trials: number; seconds: number }>;
}

export interface RunDetail {
  id: string;
  manifest: Record<string, unknown> & {
    model: string;
    started_utc: string;
    wall_seconds: number;
    protocol: Record<string, unknown>;
    code: { commit: string; dirty: boolean };
    env: Record<string, string>;
    model_config: Record<string, unknown>;
  };
  timings: Row[];
  tuning: TuningPoint[] | null;
  diagnostics: Row[];
}

export interface CompareResponse {
  models: string[];
  reference: string;
  scoreboard: Row[];
  dev_scoreboard: Row[];
  significance: Row[];
  per_fold: Record<string, Record<string, (number | null)[]>>;
  cumulative: Record<string, Columns>;
}

export interface FoldsResponse {
  calendar: {
    train_start: string;
    purge: number;
    embargo: number;
    refit_every: number;
    folds: { k: number; test_start: string; test_end: string; n_test: number; n_refits: number; first_train_end: string }[];
  };
  protocol: Record<string, number>;
  refits: { fold: number; j: number; origin: string; train_end: string; predict_end: string }[];
  tuning_points: string[];
}

export interface Regime {
  label: string;
  start: string;
  end: string;
}

async function parse<T>(res: Response, path: string): Promise<T> {
  if (!res.ok) {
    let detail = "";
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? "";
    } catch {
      /* not JSON */
    }
    throw new Error(`${path} → HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  return (await res.json()) as T;
}

export async function getJSON<T>(path: string): Promise<T> {
  return parse<T>(await fetch(path, { headers: { Accept: "application/json" } }), path);
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  return parse<T>(res, path);
}

export interface AsyncState<T> {
  data?: T;
  error?: string;
  loading: boolean;
}

/** Run an async loader whenever `key` changes; ignore results of stale requests. */
/** Loads in flight across the app. While new data loads, a page keeps showing the previous
 *  data (no "Loading" text), so the figure pack (scripts/figures.mjs) waits for this count to
 *  reach zero before it exports: otherwise it could save the previous chart under the new name. */
function trackLoad(delta: number) {
  const w = window as unknown as { __silversightLoads?: number };
  w.__silversightLoads = Math.max(0, (w.__silversightLoads ?? 0) + delta);
}

export function useAsync<T>(load: () => Promise<T>, key: string | null): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ loading: key !== null });
  useEffect(() => {
    if (key === null) return;
    let alive = true;
    trackLoad(1);
    setState((s) => ({ data: s.data, loading: true }));
    load()
      .then((data) => alive && setState({ data, loading: false }))
      .catch((e: unknown) => alive && setState({ error: e instanceof Error ? e.message : String(e), loading: false }))
      .finally(() => trackLoad(-1));
    return () => {
      alive = false;
    };
    // `load` is recreated each render; `key` captures everything it depends on.
  }, [key]);
  return state;
}

/** GET an endpoint (re-fetched when the path changes). */
export function useApi<T>(path: string | null): AsyncState<T> {
  return useAsync<T>(() => getJSON<T>(path as string), path);
}

/** Models named in the page address (`?models=linear,gbm`), or null when all are shown.
 *  Used by the figure pack (scripts/figures.mjs) so that each figure shows only the models
 *  it is about; it also works by hand in the browser. */
export function modelFilter(): string[] | null {
  const raw = new URLSearchParams(window.location.search).get("models");
  const names = (raw ?? "").split(",").map((m) => m.trim()).filter(Boolean);
  return names.length ? names : null;
}

/** The latest run of each model (the runs list arrives newest first), restricted to the
 *  models in the page address if it names any (see modelFilter). */
export function latestPerModel(runs: RunSummary[]): RunSummary[] {
  const only = modelFilter();
  const seen = new Map<string, RunSummary>();
  for (const r of runs) if (!seen.has(r.model) && (!only || only.includes(r.model))) seen.set(r.model, r);
  return MODEL_ORDER.filter((m) => seen.has(m))
    .map((m) => seen.get(m)!)
    .concat([...seen.values()].filter((r) => !MODEL_ORDER.includes(r.model)));
}

/** Display order: the baseline ladder, bottom rung first. */
export const MODEL_ORDER = ["climatology", "ewma", "ar_garch", "linear", "gbm", "lstm", "gru", "static_gnn", "aimdg"];

export interface NodeInfo {
  key: string;
  label: string;
  role: string;
  cluster: string;
  source: string;
  has_volume: boolean;
  features: string[];
}

export interface EventRow {
  date: string;
  event: "cpi" | "nfp" | "fomc";
  scheduled: boolean;
  source: string;
}
