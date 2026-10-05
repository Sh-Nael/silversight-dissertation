"""Comparison tables across runs: the numbers that go into Chapter 4.

Given several runs on the same fold calendar, produce
  * a scoreboard (pooled over all 4,141 test origins) per model and horizon;
  * per-fold losses (the 8 points the Wilcoxon test uses);
  * significance of every model against a reference model: Diebold-Mariano on
    per-origin losses, Wilcoxon on fold means, block permutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaglab.eval.harness import RunResult
from xaglab.eval.metrics import brier, logloss, qlike, summarise
from xaglab.eval.stats import diebold_mariano, permutation_test, wilcoxon_folds

LOSSES = {
    "brier": lambda d, h: brier(d[f"p_up{h}"].to_numpy(float), d[f"up{h}"].to_numpy(float)),
    "logloss": lambda d, h: logloss(d[f"p_up{h}"].to_numpy(float), d[f"up{h}"].to_numpy(float)),
    "qlike": lambda d, h: qlike(d[f"var{h}"].to_numpy(float), d[f"rv{h}"].to_numpy(float)),
}


def _aligned(runs: list[RunResult]) -> None:
    idx = runs[0].predictions.index
    for r in runs[1:]:
        if not r.predictions.index.equals(idx):
            raise ValueError(f"{r.run_id} was evaluated on different origins than {runs[0].run_id}")


def scoreboard(runs: list[RunResult]) -> pd.DataFrame:
    _aligned(runs)
    rows = []
    for r in runs:
        s = summarise(r.predictions)
        s.insert(0, "model", r.manifest["model"])
        rows.append(s.drop(columns="group"))
    return pd.concat(rows, ignore_index=True)


def fold_losses(run: RunResult, loss: str, h: int) -> pd.Series:
    d = run.predictions
    return pd.Series(LOSSES[loss](d, h), index=d.index).groupby(d["fold"].to_numpy()).mean()


def significance(runs: list[RunResult], reference: str, loss: str, h: int) -> pd.DataFrame:
    """Every model vs the reference. mean_diff < 0 means the model beats the reference."""
    _aligned(runs)
    by_name = {r.manifest["model"]: r for r in runs}
    ref = by_name[reference]
    la = LOSSES[loss](ref.predictions, h)
    rows = []
    for name, r in by_name.items():
        if name == reference:
            continue
        lb = LOSSES[loss](r.predictions, h)
        if np.all(np.isnan(lb)):
            continue
        dm = diebold_mariano(lb, la, h=h)
        wx = wilcoxon_folds(fold_losses(r, loss, h).to_numpy(), fold_losses(ref, loss, h).to_numpy())
        pm = permutation_test(lb, la, block=21)
        folds_better = int((fold_losses(r, loss, h) < fold_losses(ref, loss, h)).sum())
        rows.append({"model": name, "vs": reference, "loss": loss, "h": h,
                     "mean_diff": dm.mean_diff, "dm_stat": dm.stat, "dm_p": dm.p_value,
                     "wilcoxon_p": wx.p_value, "perm_p": pm.p_value,
                     "folds_better": f"{folds_better}/8"})
    return pd.DataFrame(rows)


def _dev_rows(runs: list) -> pd.DataFrame:
    """One row per (model, tuning point, task) with the development score.

    Accepts full runs and development runs (anything with ``manifest`` and ``tuning``).
    Rows are labelled ``manifest["label"]`` if set (to tell two runs of one model apart),
    else by the model name.
    Models tuned per task (linear, gbm) get a derived ``joint`` row, the same joint NLL
    the neural models are tuned on: logloss1 + logloss5 + 0.5 (QLIKE1 + QLIKE5) (D-28).
    """
    rows = []
    for r in runs:
        for t in r.tuning or []:
            res = {k: v["cv_loss"] for k, v in t["results"].items()}
            parts = [f"direction_h{h}" for h in (1, 5)] + [f"volatility_h{h}" for h in (1, 5)]
            if "joint" not in res and all(p in res for p in parts):
                res["joint"] = sum(res[p] * (1.0 if p.startswith("direction") else 0.5) for p in parts)
            rows += [{"model": r.manifest.get("label", r.manifest["model"]), "origin": t["origin"], "task": k,
                      "cv_loss": v}
                     for k, v in res.items()]
    return pd.DataFrame(rows)


def dev_scoreboard(runs: list) -> pd.DataFrame:
    """Development scores: out-of-fold losses at the tuning points, averaged (D-26).

    Only tuned models have them. Used to compare designs; never reported as test
    performance (they are the best of the searched trials, hence mildly optimistic).
    """
    df = _dev_rows(runs)
    if df.empty:
        return df
    return df.groupby(["model", "task"])["cv_loss"].agg(["mean", "count"]).reset_index()


def dev_paired(runs: list, reference: str) -> pd.DataFrame:
    """Each model against a reference on the tuning points they share, task by task.

    Columns: mean difference (model - reference; negative = better), points won, points
    compared. The points are yearly and their validation blocks do not overlap much, so
    "won 12 of 17" is a fair, if informal, read of consistency.
    """
    df = _dev_rows(runs)
    if df.empty:
        return df
    wide = df.pivot_table(index=["task", "origin"], columns="model", values="cv_loss")
    out = []
    for task, g in wide.groupby(level="task"):
        for m in g.columns:
            if m == reference or reference not in g.columns:
                continue
            d = (g[m] - g[reference]).dropna()
            if len(d):
                out.append({"task": task, "model": m, "mean_diff": float(d.mean()),
                            "won": int((d < 0).sum()), "points": len(d)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ chart series
# Library functions behind the web app's charts (and the dissertation's figures).

def per_fold(runs: list[RunResult], reference: str, loss: str, h: int) -> pd.DataFrame:
    """Per-fold mean loss of every model minus the reference's (rows: fold 0-7).

    Negative = the model beat the reference in that fold.
    """
    _aligned(runs)
    by_name = {r.manifest["model"]: r for r in runs}
    ref = fold_losses(by_name[reference], loss, h)
    out = {}
    for name, r in by_name.items():
        if name == reference or np.all(np.isnan(LOSSES[loss](r.predictions, h))):
            continue
        out[name] = fold_losses(r, loss, h) - ref
    return pd.DataFrame(out)


def cumulative_difference(runs: list[RunResult], reference: str, loss: str, h: int,
                          freq: str = "W-FRI") -> pd.DataFrame:
    """Running sum over test days of (model loss - reference loss), sampled at `freq`.

    A line trending down is a model that keeps beating the reference; flat means no edge.
    """
    _aligned(runs)
    by_name = {r.manifest["model"]: r for r in runs}
    ref = LOSSES[loss](by_name[reference].predictions, h)
    idx = by_name[reference].predictions.index
    out = {}
    for name, r in by_name.items():
        if name == reference:
            continue
        d = LOSSES[loss](r.predictions, h) - ref
        if np.all(np.isnan(d)):
            continue
        out[name] = pd.Series(np.nan_to_num(d), index=idx).cumsum()
    frame = pd.DataFrame(out, index=idx)
    return frame.resample(freq).last() if freq else frame


def calibration_bins(run: RunResult, h: int, bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram points: test days grouped into equal-count bins of the forecast
    probability; per bin the mean forecast, the observed up-rate and the day count."""
    d = run.predictions[[f"p_up{h}", f"up{h}"]].dropna()
    p = d[f"p_up{h}"]
    if p.nunique() < bins:
        groups = pd.Series(0, index=d.index)
    else:
        groups = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = d.groupby(groups)
    return pd.DataFrame({"p_mean": g[f"p_up{h}"].mean(), "observed": g[f"up{h}"].mean(),
                         "n": g.size()}).reset_index(drop=True)


def volatility_path(run: RunResult, h: int, freq: str = "W-FRI") -> pd.DataFrame:
    """Forecast vs realised volatility over h days, annualised (sqrt(var * 252 / h)),
    averaged over `freq` periods."""
    d = run.predictions[[f"var{h}", f"rv{h}"]]
    ann = np.sqrt(d * 252 / h).rename(columns={f"var{h}": "forecast", f"rv{h}": "realised"})
    return ann.resample(freq).mean() if freq else ann


# ------------------------------------------------------------------ selective prediction
# "On the days the model is most confident, how often is it right?" (D-31)

#: Coverage levels, fixed before any result was looked at (D-31).
COVERAGES = (0.1, 0.2, 0.3, 0.5, 1.0)


def causal_selection(conf: np.ndarray, coverage: float, window: int = 252, min_history: int = 63) -> np.ndarray:
    """Which forecasts count as "confident", using only earlier forecasts.

    Forecast t is selected when its confidence is *above* the (1 - coverage) quantile of
    the model's previous ``window`` confidences (strictly before t). Strictly above: a model
    whose probability barely moves (climatology changes it once a month) would otherwise
    tie with the threshold on most days and "select" far more than the target share
    (amendment to D-31, found on the first real numbers; continuous forecasts unaffected). Only confidences are
    used, never outcomes, so no label is needed and nothing from the future enters. The
    first ``min_history`` forecasts have no reference and are not selected. This is what
    a deployed app can do: it cannot rank today against days that have not happened yet.
    """
    conf = np.asarray(conf, float)
    if coverage >= 1.0:
        return np.ones(len(conf), bool)
    out = np.zeros(len(conf), bool)
    for t in range(min_history, len(conf)):
        past = conf[max(0, t - window): t]
        out[t] = conf[t] > np.quantile(past, 1.0 - coverage)
    return out


def ranked_selection(conf: np.ndarray, coverage: float) -> np.ndarray:
    """The textbook risk-coverage selection: forecasts whose confidence is above the
    (1 - coverage) quantile of *all* test forecasts. It ranks each day against the whole test
    period, i.e. with look-ahead, so it describes a model but is not a strategy anyone could
    have run. Strictly above, as in :func:`causal_selection`, so ties select nothing."""
    conf = np.asarray(conf, float)
    if coverage >= 1.0:
        return np.ones(len(conf), bool)
    return conf > np.quantile(conf, 1.0 - coverage)


def _one_sided(res) -> float:
    """One-sided p for "loss_a < loss_b" from the symmetric (sign-flip) two-sided test."""
    return res.p_value / 2 if res.stat < 0 else 1 - res.p_value / 2


def _side(name: str, calls: np.ndarray, happened: np.ndarray, sel: np.ndarray, h: int) -> dict:
    """One side of the two-sided scorecard: count, hit rate, base rate on the selected days,
    and a one-sided binomial test against the stricter of 50% and that base rate."""
    from scipy.stats import binomtest

    n = int(calls.sum())
    base = float(happened[sel].mean()) if sel.any() else float("nan")
    if n == 0:
        return {f"{name}_n": 0, f"{name}_right": np.nan, f"{name}_base": base, f"p_{name}": np.nan}
    right = float(happened[calls].mean())
    n_eff = max(1, n // h)
    p = binomtest(round(right * n_eff), n_eff, max(0.5, base), alternative="greater").pvalue
    return {f"{name}_n": n, f"{name}_right": right, f"{name}_base": base, f"p_{name}": float(p)}


def selective(run: RunResult, h: int, coverages: tuple[float, ...] = COVERAGES, mode: str = "causal",
              window: int = 252, min_history: int = 63) -> pd.DataFrame:
    """Accuracy against coverage for one run's direction forecasts at horizon h.

    Confidence is |p - 0.5|; the call is "up" when p >= 0.5. Per coverage level:
      n, coverage     selected days and their share of all test days
      accuracy        share of correct calls on the selected days
      up_calls        share of those calls that were "up"
      always_up       accuracy of always saying "up" on the *same* days (the base rate there)
      edge            accuracy - always_up
      p_edge          one-sided block permutation test (21-day blocks, in date order) of the
                      model's hits against always-up's on the selected days: is the edge
                      positive? (A model with no information usually *loses* to always-up,
                      because silver rises on ~52% of days; that is not what we ask.)
      p_coin          one-sided binomial test of accuracy > 50% with n / h effective
                      trials (h-day outcomes overlap)
    Two-sided scorecard (D-32), because silver is traded both ways:
      up_n, up_right      "up" calls among the selected days and the share that went up
      up_base             share of selected days that went up (what always-up would get)
      p_up                one-sided binomial test of up_right > max(50%, up_base)
      down_n, down_right  "down" calls and the share that went down
      down_base           share of selected days that went down
      p_down              one-sided binomial test of down_right > max(50%, down_base)
    The null rate is the stricter of a coin and that side's base rate, so a side can't look
    skilful just by riding the drift (up) or by beating an easy coin in a falling spell (down).
    """
    from scipy.stats import binomtest

    d = run.predictions[[f"p_up{h}", f"up{h}"]].dropna()
    p, y = d[f"p_up{h}"].to_numpy(float), d[f"up{h}"].to_numpy(float)
    conf = np.abs(p - 0.5)
    call = (p >= 0.5).astype(float)
    rows = []
    for c in coverages:
        sel = causal_selection(conf, c, window, min_history) if mode == "causal" else ranked_selection(conf, c)
        n = int(sel.sum())
        if n == 0:
            rows.append({"coverage_target": c, "n": 0})
            continue
        hit = (call[sel] == y[sel]).astype(float)
        base = y[sel]
        n_eff = max(1, n // h)
        rows.append({
            "coverage_target": c, "n": n, "coverage": n / len(p),
            "accuracy": float(hit.mean()), "up_calls": float(call[sel].mean()),
            "always_up": float(base.mean()), "edge": float(hit.mean() - base.mean()),
            "p_edge": _one_sided(permutation_test(1 - hit, 1 - base)) if n >= 42 else np.nan,
            "p_coin": float(binomtest(round(hit.mean() * n_eff), n_eff, 0.5, alternative="greater").pvalue),
            **_side("up", sel & (call == 1), y == 1, sel, h),
            **_side("down", sel & (call == 0), y == 0, sel, h),
        })
    return pd.DataFrame(rows)
