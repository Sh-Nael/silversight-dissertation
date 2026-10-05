"""The table pack for the report: report files turned into tables that paste into Word.

Nothing is computed here. Every table is a selection and a formatting of numbers that
an earlier, committed report already holds (``experiments/reports/<name>/*.csv``), so a
table can't differ from the evidence in the diary. Each table records its source files.

Outputs, per table: ``<key>.csv`` (formatted cells), ``<key>.md``; for the pack:
``tables.docx`` (real Word tables with captions and notes) and ``index.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: The model ladder, in the order the report presents it.
LADDER = ("climatology", "ewma", "ar_garch", "linear", "gbm", "lstm", "gru", "static_gnn", "aimdg")
LOSS_LABEL = {"brier": "Brier", "logloss": "Log-loss", "qlike": "QLIKE"}
ALPHA = 0.05
N_TEST_DAYS = 4141


@dataclass
class Table:
    key: str
    title: str
    frame: pd.DataFrame  # formatted cells (strings), one header row
    note: str = ""
    sources: tuple[str, ...] = ()
    chapter: str = "4"
    bold: set[tuple[int, str]] = field(default_factory=set)  # (row position, column) to embolden


# ------------------------------------------------------------------ cell formats

def num(v, digits: int = 3) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def pct(v, digits: int = 1) -> str:
    return "–" if v is None or pd.isna(v) else f"{100 * v:.{digits}f}%"


def count(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{int(v):,}"


def pval(v) -> str:
    """p-value with three decimals; below 0.001 as "<0.001"; significant ones carry a star."""
    if v is None or pd.isna(v):
        return "–"
    text = "<0.001" if v < 0.001 else f"{v:.3f}"
    return text + ("*" if v < ALPHA else "")


def _best(raw: pd.DataFrame, cols: dict[str, str], lower: set[str],
          digits: dict[str, int] | None = None) -> set[tuple[int, str]]:
    """Positions of the best value per column; ``cols`` maps raw column -> shown column.

    Compared at the precision shown (``digits`` per raw column, as a fraction's decimals),
    so values that print the same are all bold: a reader can't tell them apart either."""
    out = set()
    for c, shown in cols.items():
        v = raw[c].to_numpy(float)
        if np.all(np.isnan(v)):
            continue
        if digits and c in digits:
            v = np.round(v, digits[c])
        target = np.nanmin(v) if c in lower else np.nanmax(v)
        out |= {(i, shown) for i in np.flatnonzero(np.isclose(v, target, rtol=0, atol=1e-12))}
    return out


def _ordered(df: pd.DataFrame, models: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Rows in ladder order (then the order of appearance); optionally only ``models``."""
    if models is not None:
        df = df[df["model"].isin(models)]
    known = list(models or LADDER)
    rank = {m: known.index(m) if m in known else len(known) + i
            for i, m in enumerate(dict.fromkeys(df["model"]))}
    return df.assign(_r=df["model"].map(rank)).sort_values("_r", kind="stable").drop(columns="_r")


# ------------------------------------------------------------------ table builders

def volatility_board(board: pd.DataFrame, models=None) -> tuple[pd.DataFrame, set]:
    """One row per model: QLIKE, volatility RMSE and Mincer-Zarnowitz R² at 1 and 5 days."""
    wide = _ordered(board, models).pivot_table(index="model", columns="h", sort=False,
                                               values=["vol_qlike", "vol_rmse_vol", "vol_mz_r2"])
    raw = pd.DataFrame({"model": wide.index}).reset_index(drop=True)
    shown = raw.copy()
    names = {"vol_qlike": ("QLIKE", 3), "vol_rmse_vol": ("RMSE", 4), "vol_mz_r2": ("MZ R²", 3)}
    cols = {}
    for k, (label, digits) in names.items():
        for h in (1, 5):
            raw[f"{k}{h}"] = wide[(k, h)].to_numpy(float)
            shown[f"{label} {h}d"] = [num(v, digits) for v in raw[f"{k}{h}"]]
            cols[f"{k}{h}"] = f"{label} {h}d"
    lower = {c for c in cols if not c.startswith("vol_mz")}
    shown_digits = {c: names[c.rstrip("15")][1] for c in cols}
    return shown.rename(columns={"model": "Model"}), _best(raw, cols, lower, shown_digits)


def direction_board(board: pd.DataFrame, h: int, models=None) -> tuple[pd.DataFrame, set]:
    """One row per model at horizon h: probability scores, hit rate, and each side's calls."""
    raw = _ordered(board[board["h"] == h], models).reset_index(drop=True)
    spec = [("brier", "Brier", lambda v: num(v, 4)), ("logloss", "Log-loss", lambda v: num(v, 4)),
            ("auc", "AUC", num), ("accuracy", "Hit rate", pct), ("balanced_accuracy", "Balanced", pct),
            ("up_calls", "Up calls", pct), ("up_precision", "Up right", pct),
            ("down_precision", "Down right", pct)]
    shown = pd.DataFrame({"Model": raw["model"]})
    for k, label, f in spec:
        shown[label] = [f(v) for v in raw[k]]
    cols = {k: label for k, label, _ in spec if k != "up_calls"}
    shown_digits = {"brier": 4, "logloss": 4, "auc": 3}  # rates: one decimal of a percentage
    shown_digits |= {k: 3 for k in cols if k not in shown_digits}
    return shown, _best(raw, cols, {"brier", "logloss"}, shown_digits)


def significance_table(sig: pd.DataFrame, loss: str, models=None) -> pd.DataFrame:
    """One row per model, 1-day and 5-day side by side: difference in mean loss against the
    reference (negative: the model is better), the three p-values, folds won."""
    s = sig[sig["loss"] == loss]
    out = pd.DataFrame({"Model": _ordered(s[s["h"] == 1], models)["model"].to_numpy()})
    digits = 3 if loss == "qlike" else 5
    for h in (1, 5):
        part = s[s["h"] == h].set_index("model").reindex(out["Model"])
        out[f"Diff {h}d"] = [("–" if pd.isna(v) else f"{v:+.{digits}f}") for v in part["mean_diff"]]
        out[f"DM p {h}d"] = [pval(v) for v in part["dm_p"]]
        out[f"Wilcoxon p {h}d"] = [pval(v) for v in part["wilcoxon_p"]]
        out[f"Perm. p {h}d"] = [pval(v) for v in part["perm_p"]]
        out[f"Folds {h}d"] = part["folds_better"].fillna("–").to_numpy()
    return out


def one_model_tests(sig: pd.DataFrame, model: str) -> pd.DataFrame:
    """All losses and horizons for one model against the reference (rows: loss and horizon)."""
    s = sig[sig["model"] == model]
    rows = []
    for loss in ("qlike", "brier", "logloss"):
        for h in (1, 5):
            r = s[(s["loss"] == loss) & (s["h"] == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"Loss": LOSS_LABEL[loss], "Horizon": f"{h} day" + ("s" if h > 1 else ""),
                         "Difference": f"{r.mean_diff:+.{3 if loss == 'qlike' else 5}f}",
                         "DM statistic": num(r.dm_stat, 2), "DM p": pval(r.dm_p),
                         "Wilcoxon p": pval(r.wilcoxon_p), "Permutation p": pval(r.perm_p),
                         "Folds better": r.folds_better})
    return pd.DataFrame(rows)


def coverage_table(sel: pd.DataFrame, h: int, models=None) -> pd.DataFrame:
    """Hit rate on the most confident days (causal gate, D-31): one row per model, one column
    per coverage *target*, as "hit rate (edge over always-up)"; a star marks p_edge < 0.05.
    The share of days actually selected differs from the target (see :func:`realised`)."""
    s = sel[(sel["h"] == h) & (sel["mode"] == "causal")]
    out = pd.DataFrame({"Model": _ordered(s.drop_duplicates("model"), models)["model"].to_numpy()})
    for c in sorted(s["coverage_target"].unique()):
        part = s[s["coverage_target"] == c].set_index("model").reindex(out["Model"])
        out[f"Target {c:.0%}" if c < 1 else "All days"] = [
            "–" if pd.isna(a) else f"{100 * a:.1f}% ({100 * e:+.1f}){'*' if p < ALPHA else ''}"
            for a, e, p in zip(part["accuracy"], part["edge"], part["p_edge"].fillna(1.0), strict=True)]
    return out


def realised(shares: pd.Series) -> str:
    """"x–y%" range of the share of test days actually selected at a coverage target."""
    return f"{100 * shares.min():.1f}–{100 * shares.max():.1f}%"


def two_sided_table(sel: pd.DataFrame, coverage: float, models=None) -> pd.DataFrame:
    """The two-sided scorecard (D-32) at one coverage level, both horizons: calls, the share
    right and the one-sided p-value against max(50%, base rate), per side."""
    s = sel[(sel["mode"] == "causal") & np.isclose(sel["coverage_target"], coverage)]
    rows = []
    for h in (1, 5):
        for _, r in _ordered(s[s["h"] == h], models).iterrows():
            rows.append({"Model": r["model"], "Horizon": f"{h}d", "Days selected": pct(r.coverage),
                         "Up calls": count(r.up_n), "Up right": pct(r.up_right), "Up p": pval(r.p_up),
                         "Down calls": count(r.down_n), "Down right": pct(r.down_right),
                         "Down p": pval(r.p_down)})
    return pd.DataFrame(rows)


def signals_table(sig: pd.DataFrame, h: int, coverage: float, models=None) -> pd.DataFrame:
    """The signal and risk layer at one horizon and coverage: a row per model and side."""
    s = sig[(sig["h"] == h) & np.isclose(sig["coverage_target"], coverage)]
    side = {"both": 0, "buy": 1, "sell": 2}
    s = _ordered(s.assign(_s=s["side"].map(side)).sort_values("_s", kind="stable"), models)
    return pd.DataFrame({
        "Model": s["model"].to_numpy(), "Side": s["side"].to_numpy(),
        "Trades": [count(v) for v in s["n"]], "Win rate": [pct(v) for v in s["win_rate"]],
        "Profit factor": [num(v, 2) for v in s["profit_factor"]],
        "Mean net (bps)": [num(v, 1) for v in s["mean_net_bps"]],
        "p (mean > 0)": [pval(v) for v in s["p_positive"]],
        "Total return": [pct(v) for v in s["total_return"]],
        "Max drawdown": [pct(v) for v in s["max_drawdown"]]})


def lab_table(lab: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Setting": lab["setting"], "Joint loss": [num(v, 4) for v in lab["cv_loss"]],
        "Diff. vs default": [f"{v:+.4f}" for v in lab["diff_vs_ref"]],
        "Points better": [f"{int(a)}/{int(b)}" for a, b in zip(lab["points_better"], lab["points"], strict=True)],
        "Wilcoxon p": [pval(v) if s != "default" else "–"
                       for v, s in zip(lab["p_wilcoxon"], lab["setting"], strict=True)],
        "Edges on": [num(v, 2) for v in lab["edges"]],
        "Edges moved": [num(v, 2) for v in lab["edges_moved_vs_ref"]]})


def walkforward_table(wf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Setting": wf["setting"], "Cadence (days)": wf["cadence"], "Refits": wf["refits"],
        "Joint loss": [num(v, 4) for v in wf["joint_loss"]],
        "Diff. vs default": [f"{v:+.4f}" for v in wf["diff_vs_default"]],
        "Edges on": [num(v, 2) for v in wf["edges"]],
        "Edges changed per refit": [num(v, 2) for v in wf["edges_changed_per_refit"]],
        "Clock resets": wf["clock_resets"], "Seconds": [num(v, 0) for v in wf["seconds"]]})


def regime_table(tests: pd.DataFrame, primary: bool) -> pd.DataFrame:
    t = tests[tests["primary"] == primary]
    out = pd.DataFrame({
        "Cluster": t["cluster"], "Measure": t["measure"], "Hypothesis": t["hypothesis"],
        "Inside": [num(v) for v in t["inside"]], "Outside": [num(v) for v in t["outside"]],
        "Difference": [f"{v:+.3f}" for v in t["diff"]], "p": [pval(v) for v in t["p"]],
        "Months inside": t["n_inside"]})
    if not primary:
        out.insert(2, "Regime", t["regime"].to_numpy())
    return out.reset_index(drop=True)


def persistence_table(per: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Edge": per["edge"], "Cluster": per["cluster"],
        "Months active": [pct(v) for v in per["share_active"]], "Switches": per["switches"],
        "Mean active spell (months)": [num(v, 1) for v in per["mean_spell_months"]]})


def dev_table(dev: pd.DataFrame) -> pd.DataFrame:
    wide = dev.pivot_table(index="model", columns="task", values="mean").reset_index()
    wide = _ordered(wide)
    labels = {"joint": "Joint", "direction_h1": "Log-loss 1d", "direction_h5": "Log-loss 5d",
              "volatility_h1": "QLIKE 1d", "volatility_h5": "QLIKE 5d"}
    out = pd.DataFrame({"Model": wide["model"].to_numpy()})
    for k, label in labels.items():
        if k in wide:
            out[label] = [num(v, 4) for v in wide[k]]
    return out


# ------------------------------------------------------------------ the pack

_SIG_NOTE = ("Difference: mean loss of the model minus the reference's; negative means the model is "
             "better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 "
             "fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which "
             "the model beat the reference.")
_TEST_NOTE = "Test period 4 Jan 2010 – 23 Jun 2026, 4,141 days, 8 folds, monthly refits."


def build(reports: Path) -> list[Table]:
    """All tables of the pack, from the committed report folders under ``reports``."""
    def read(rel: str) -> pd.DataFrame:
        return pd.read_csv(reports / rel)

    p4, vs_sg, vs_gbm = "phase4-aimdg", "phase4-aimdg-vs-static_gnn", "phase4-aimdg-vs-gbm"
    board, sig, sel = read(f"{p4}/scoreboard.csv"), read(f"{p4}/significance.csv"), read(f"{p4}/selective.csv")
    sig_sg, sig_gbm = read(f"{vs_sg}/significance.csv"), read(f"{vs_gbm}/significance.csv")
    ens_board, ens_sig = read("ensemble-6.2/scoreboard.csv"), read("ensemble-6.2/significance.csv")
    abl_board = read("phase5-test-ablations/scoreboard.csv")
    abl_sig = read("phase5-test-ablations-vs-aimdg/significance.csv")
    abl_sig_sg = read("phase5-test-ablations/significance.csv")
    signals = pd.concat([read("signals-6.2/signals.csv"), read("signals-ensemble/signals.csv")])
    keys = ["model", "h", "coverage_target", "side"]
    # linear and gbm are in both reports: the two copies must agree before one is dropped.
    if (signals.round(10).drop_duplicates().duplicated(keys)).any():
        raise ValueError("signals-6.2 and signals-ensemble disagree on a model they share")
    signals = signals.drop_duplicates(keys)
    ens = ("linear", "gbm", "static_gnn", "ens_lin_gbm", "ens_lin_gbm_sgnn")
    abl = ("static_gnn", "aimdg", "aimdg_replay", "aimdg_A1", "aimdg_A2")
    traded = ("linear", "gbm", "lstm", "static_gnn", "aimdg", "ens_lin_gbm", "ens_lin_gbm_sgnn")
    sig_src = ("signals-6.2/signals.csv", "signals-ensemble/signals.csv")

    tables: list[Table] = []

    def add(key, title, frame, note="", sources=(), chapter="4", bold=None):
        tables.append(Table(key, title, frame.reset_index(drop=True), note, tuple(sources), chapter, bold or set()))

    f, b = volatility_board(board, LADDER)
    add("T01-volatility-scoreboard", "Volatility forecasts of the nine models", f,
        f"{_TEST_NOTE} QLIKE and RMSE: lower is better; MZ R²: higher is better. Best value per column in bold.",
        [f"{p4}/scoreboard.csv"], bold=b)
    for k, h in (("T02", 1), ("T03", 5)):
        f, b = direction_board(board, h, LADDER)
        add(f"{k}-direction-scoreboard-{h}d", f"Direction forecasts of the nine models, {h}-day horizon", f,
            f"{_TEST_NOTE} Up calls: share of days called up. Up right / Down right: share of each side's "
            "calls that were correct. Best value per column in bold.", [f"{p4}/scoreboard.csv"], bold=b)
    for k, loss in (("T04", "qlike"), ("T05", "brier"), ("T06", "logloss")):
        add(f"{k}-significance-{loss}-vs-climatology", f"{LOSS_LABEL[loss]}: every model against climatology",
            significance_table(sig, loss, LADDER), _SIG_NOTE, [f"{p4}/significance.csv"],
            chapter="4" if loss != "logloss" else "Appendix")
    add("T07-aimdg-vs-static-gnn", "AIM-DG against the static graph (the main hypothesis test)",
        one_model_tests(sig_sg, "aimdg"), _SIG_NOTE.replace("the model", "AIM-DG").replace(
            "the reference", "the static graph"), [f"{vs_sg}/significance.csv"])
    add("T08-qlike-vs-static-gnn", "QLIKE: every model against the static graph",
        significance_table(sig_sg, "qlike", LADDER), _SIG_NOTE, [f"{vs_sg}/significance.csv"])
    add("T09-brier-vs-gbm", "Brier: every model against gradient boosting",
        significance_table(sig_gbm, "brier", LADDER), _SIG_NOTE, [f"{vs_gbm}/significance.csv"])

    f, b = volatility_board(ens_board, ens)
    add("T10-ensembles-volatility", "Equal-weight ensembles and their members: volatility", f,
        f"{_TEST_NOTE} Members were chosen on development scores. Best value per column in bold.",
        ["ensemble-6.2/scoreboard.csv"], bold=b)
    f, b = direction_board(ens_board, 1, ens)
    add("T11-ensembles-direction-1d", "Equal-weight ensembles and their members: 1-day direction", f,
        "Best value per column in bold.", ["ensemble-6.2/scoreboard.csv"], bold=b)
    add("T12-ensembles-qlike-vs-gbm", "QLIKE: ensembles and members against gradient boosting",
        significance_table(ens_sig, "qlike", ens), _SIG_NOTE, ["ensemble-6.2/significance.csv"])

    f, b = volatility_board(abl_board, abl)
    add("T13-ablations-volatility", "Ablations on the test folds: volatility", f,
        "aimdg_replay: the official run reproduced from cached networks. A1: no evolution (top 3 edges "
        "by reliability). A2: no variance penalty in the reliability. Both variants were declared before "
        "the replays. Best value per column in bold.", ["phase5-test-ablations/scoreboard.csv"], bold=b)
    add("T14-ablations-qlike-vs-aimdg", "QLIKE: ablations against AIM-DG",
        significance_table(abl_sig, "qlike", abl),
        _SIG_NOTE + " Read A2 with care: its final selection differs from AIM-DG's in 3 of the 200 months and "
        "its forecasts on 266 of the 4,141 days, so its p-values rest on few months.",
        ["phase5-test-ablations-vs-aimdg/significance.csv"])
    add("T15-ablations-qlike-vs-static-gnn", "QLIKE: AIM-DG and its ablations against the static graph",
        significance_table(abl_sig_sg, "qlike", abl), _SIG_NOTE, ["phase5-test-ablations/significance.csv"])

    gate = ("Confident days: the forecast's distance from 50% is strictly above the matching quantile of the "
            "model's previous 252 forecasts (causal gate, D-31). Because forecast confidence drifts over time, "
            "a gate set on the past selects more days than its target")
    causal = sel[sel["mode"] == "causal"]
    learned = causal[~causal["model"].isin(["climatology", "ewma"])]

    def share(h, c):
        return realised(learned[(learned["h"] == h) & np.isclose(learned["coverage_target"], c)]["coverage"])

    for k, h in (("T16", 1), ("T17", 5)):
        add(f"{k}-accuracy-coverage-{h}d", f"Hit rate on the most confident days, {h}-day horizon",
            coverage_table(sel, h, LADDER),
            f"{gate}: the 10% target selected {share(h, 0.1)} of days for the learned models, 20% "
            f"{share(h, 0.2)}, 30% {share(h, 0.3)}, 50% {share(h, 0.5)}. Cells: hit rate (edge over always-up on "
            "the same days, in points); * one-sided block permutation p < 0.05 for a positive edge. Climatology "
            "and EWMA always call up.",
            [f"{p4}/selective.csv"])
    add("T18-two-sided-top10", "Two-sided scorecard at the 10% coverage target",
        two_sided_table(sel, 0.1, LADDER),
        f"{gate} (Days selected). p: one-sided binomial test of the share right against the stricter of 50% and that side's "
        "base rate on the selected days, with n/h effective trials. * p < 0.05.", [f"{p4}/selective.csv"])
    rules = ("Rules fixed beforehand: entry at the close on confident days, stop 1σ, target 1.5σ of the "
             "forecast move, a day hitting both counts as the stop, 10 bps round trip, 1% risk per trade, "
             "one position at a time. p: one-sided test that the mean net return is positive. * p < 0.05.")
    for k, h, c in (("T19", 1, 0.1), ("T20", 5, 0.1), ("T21", 1, 1.0)):
        label = "at the 10% coverage target" if c < 1 else "on every day"
        both = signals[(signals["h"] == h) & np.isclose(signals["coverage_target"], c) & (signals["side"] == "both")]
        note = rules if c >= 1 else (f"{rules} Signals: the causal gate of D-31, strictly above the threshold; the "
                                     f"10% target gave signals on {realised(both['n'] / N_TEST_DAYS)} of the "
                                     f"{N_TEST_DAYS:,} test days (Trades, side 'both').")
        add(f"{k}-signals-{h}d-{'top10' if c < 1 else 'all'}",
            f"Signal and risk layer, {h}-day trades {label}", signals_table(signals, h, c, traded), note, sig_src)

    add("T22-sensitivity-selection-lab", "Ablations and sensitivity of the selection settings (development data)",
        lab_table(read("phase5-lab/lab_summary.csv")),
        "Selection lab on cached networks at the 17 yearly tuning points (D-38). Joint loss: "
        "logloss1 + logloss5 + ½(QLIKE1 + QLIKE5), out of fold. Edges moved: edges that differ from the "
        "default's selection, per point. Development scores, not test performance. * p < 0.05.",
        ["phase5-lab/lab_summary.csv"], chapter="4 (F1)")
    add("T23-sensitivity-walkforward", "Sensitivity of the sequential settings (development walk-forward)",
        walkforward_table(read("phase5-walkforward/walkforward.csv")),
        "One development year replayed with each setting (D-38). Seconds: selection time for the year.",
        ["phase5-walkforward/walkforward.csv"], chapter="4 (F1)")
    tests = read("edges-6.3/regime_tests.csv")
    shift = ("One-sided circular-shift permutation test; hypotheses and regimes fixed before the results "
             "by date were computed (D-40). Inside / outside: mean over months inside and outside the regime.")
    add("T24-edge-regime-tests", "Edge clusters in stress months against the rest (primary tests)",
        regime_table(tests, True), shift + " * p < 0.05.", ["edges-6.3/regime_tests.csv"], chapter="5")
    add("T25-edge-regime-tests-per-regime", "Edge clusters per regime (descriptive)",
        regime_table(tests, False), shift + " Not corrected for the 24 comparisons: descriptive only.",
        ["edges-6.3/regime_tests.csv"], chapter="Appendix")
    add("T26-edge-persistence", "Persistence of the 15 edges over the 200 monthly selections",
        persistence_table(read("edges-6.3/persistence.csv")), "", ["edges-6.3/persistence.csv"], chapter="5")
    add("T27-development-scores", "Development scores of the tuned models",
        dev_table(read(f"{p4}/dev_scoreboard.csv")),
        "Out-of-fold losses at the 17 yearly tuning points, mean. Used for design decisions only; they are "
        "the best of the searched trials and so optimistic (about 0.01 in the joint loss).",
        [f"{p4}/dev_scoreboard.csv"], chapter="Appendix")
    return tables


# ------------------------------------------------------------------ writers

def to_markdown(t: Table) -> str:
    cols = list(t.frame.columns)
    def cell(i, c):
        v = str(t.frame.iloc[i][c]).replace("|", "\\|")
        return f"**{v}**" if (i, c) in t.bold else v
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(i, c) for c in cols) + " |" for i in range(len(t.frame))]
    return "\n".join(lines)


def write_docx(tables: list[Table], path: Path) -> None:
    """One Word file with every table: caption above, note and source below."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.size = Pt(9)
    doc.add_heading("SilverSight table pack", level=1)
    doc.add_paragraph("Generated by `xag tables` from the committed report files. Table numbers are "
                      "working labels; renumber when placing a table in the report.")
    for t in tables:
        cap = doc.add_paragraph()
        run = cap.add_run(f"{t.key.split('-')[0]}. {t.title}")
        run.bold = True
        cap.paragraph_format.keep_with_next = True
        cols = list(t.frame.columns)
        tb = doc.add_table(rows=1, cols=len(cols))
        tb.style = "Light Grid Accent 1"
        for j, c in enumerate(cols):
            cell = tb.rows[0].cells[j]
            cell.text = str(c)
            cell.paragraphs[0].runs[0].bold = True
        for i in range(len(t.frame)):
            row = tb.add_row().cells
            for j, c in enumerate(cols):
                row[j].text = str(t.frame.iloc[i][c])
                p = row[j].paragraphs[0]
                if j > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                p.runs[0].bold = (i, c) in t.bold
        note = doc.add_paragraph()
        r = note.add_run((t.note + " " if t.note else "") + "Source: " + ", ".join(t.sources) + ".")
        r.italic = True
        r.font.size = Pt(8)
    doc.save(path)


def write_pack(tables: list[Table], out: Path, docx: bool = True) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    intro = ("Generated by `xag tables`. Every table is a formatting of committed report files "
             "(`experiments/reports/`); nothing is recomputed. Table numbers are working labels.")
    index = ["# Table pack", "", intro, "",
             "| Table | Title | Suggested chapter | Rows | Source |", "|---|---|---|---|---|"]
    for t in tables:
        t.frame.to_csv(out / f"{t.key}.csv", index=False)
        body = f"**{t.key.split('-')[0]}. {t.title}**\n\n{to_markdown(t)}\n"
        if t.note:
            body += f"\n*{t.note}*\n"
        body += f"\nSource: {', '.join(f'`{s}`' for s in t.sources)}\n"
        (out / f"{t.key}.md").write_text(body, encoding="utf-8")
        written += [out / f"{t.key}.csv", out / f"{t.key}.md"]
        index.append(f"| [{t.key.split('-')[0]}]({t.key}.md) | {t.title} | {t.chapter} | {len(t.frame)} | "
                     + ", ".join(f"`{s}`" for s in t.sources) + " |")
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    written.append(out / "index.md")
    if docx:
        write_docx(tables, out / "tables.docx")
        written.append(out / "tables.docx")
    return written
