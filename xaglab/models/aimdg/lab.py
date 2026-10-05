"""The selection lab (Phase 5, D-38): ablations and the sensitivity sweep on fixed networks.

Nearly all of an AIM-DG run's cost is training the supernets; everything Phase 5 varies
(reliability weights, the search's settings, the fitness weights, "no evolution", "no
variance penalty") happens after they are trained and takes seconds. So the lab trains
the 5 supernets **once** per tuning point and development block, with the network settings
the AIM-DG development run chose there, and runs every setting through the **same**
networks: reliability → selection → out-of-fold forecasts → the development score.

That isolates the selection stage exactly. Step 4.3 found retraining noise as large as the
differences between masks, so a sweep that retrained per setting would mostly measure noise.

Inside one fit there is no history, so the warm start, the stability term λ2 and the
market-time clock are inactive here, as they are in tuning trials; those are measured by the
sequential development walk-forward (``walkforward.py``).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from xaglab.eval.folds import Refit
from xaglab.features.build import FeatureSet
from xaglab.models.aimdg.evolution import (
    GENERATIONS,
    POP_SIZE,
    RATE_MAX,
    MaskScorer,
    evolve,
    exhaustive,
)
from xaglab.models.aimdg.reliability import ALPHA, BETA, SUBWINDOW, contributions, reliability
from xaglab.models.base import FitView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.forecaster import joint_task
from xaglab.models.neural.learner import NeuralLearner, _targets_for
from xaglab.models.tuning import JointTask, cv_blocks

LAB_SEED = 17  # = view.seed * 7919 + 17 for seed 0: the seed the development run's trials used


@dataclass(frozen=True)
class Setting:
    """One way of choosing the active edges from trained supernets."""

    name: str
    mode: str = "evolve"           # evolve | topk | all_on | exhaustive
    alpha: float = ALPHA
    beta: float = BETA
    subwindow: int = SUBWINDOW
    pop_size: int = POP_SIZE
    generations: int = GENERATIONS
    rate: float = RATE_MAX         # the mutation rate; no clock history inside one fit
    lam3: float | None = None      # None = the tuning point's own tuned value
    lam2: float | None = None      # inactive without a previous mask (kept for the walk-forward)
    k: int = 3                     # top-k for mode "topk"


DEFAULT = Setting("default")

ABLATIONS = (
    Setting("A1_no_evolution_top3", mode="topk", k=3),
    Setting("A2_no_variance_penalty", beta=0.0),
    Setting("A3_no_sparsity", lam3=0.0, lam2=0.0),
    Setting("all_edges_on", mode="all_on"),
    Setting("exhaustive_optimum", mode="exhaustive"),
)

SWEEP = (
    Setting("pop_20", pop_size=20), Setting("pop_100", pop_size=100),
    Setting("gens_30", generations=30), Setting("gens_300", generations=300),
    Setting("rate_0.15", rate=0.15), Setting("rate_0.5", rate=0.5),
    Setting("alpha_beta_1", alpha=1.0, beta=1.0), Setting("alpha_beta_0.25", alpha=0.25, beta=0.25),
    Setting("subwindow_10", subwindow=10), Setting("subwindow_42", subwindow=42),
    Setting("lam3_0.002", lam3=0.002), Setting("lam3_0.01", lam3=0.01), Setting("lam3_0.02", lam3=0.02),
    Setting("top5_by_reliability", mode="topk", k=5), Setting("top8_by_reliability", mode="topk", k=8),
)

ALL_SETTINGS = (DEFAULT, *ABLATIONS, *SWEEP)


# ------------------------------------------------------------------ networks per block

@dataclass
class BlockNets:
    stage1: list[torch.nn.Module]   # early-stopped: have not seen the tail (selection)
    final: list[torch.nn.Module]    # all-rows refits (forecasts)
    scaler: Any
    missing_cols: list[str]
    window: int
    params: dict[str, Any]
    seconds: float


def learner_like_the_dev_run(n_seeds: int = 5) -> NeuralLearner:
    """The supernet learner with the same settings as the AIM-DG development run."""
    lrn = NeuralLearner("supernet", "lstm", 60, n_seeds, 150, 12, True, (20, 60), 1e-1, 0.5)
    lrn.keep_stage1 = True
    return lrn


def train_block(X: pd.DataFrame, task: JointTask, block, params: dict, seed: int = LAB_SEED,
                n_seeds: int = 5) -> BlockNets:
    t0 = time.perf_counter()
    lrn = learner_like_the_dev_run(n_seeds)
    fitted = lrn.fit_rows(X, block.train, task.target(block.train), params, seed)
    return BlockNets(fitted.stage1, fitted.nets, fitted.scaler, fitted.missing_cols, fitted.window,
                     dict(params), time.perf_counter() - t0)


def save_block(nets: BlockNets, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage1": [n.cpu() for n in nets.stage1], "final": [n.cpu() for n in nets.final],
                "scaler": nets.scaler, "missing_cols": nets.missing_cols, "window": nets.window,
                "params": nets.params, "seconds": nets.seconds}, path)


def load_block(path: Path, device: torch.device) -> BlockNets:
    d = torch.load(path, weights_only=False, map_location=device)
    return BlockNets([n.to(device) for n in d["stage1"]], [n.to(device) for n in d["final"]], d["scaler"],
                     d["missing_cols"], d["window"], d["params"], d["seconds"])


# ------------------------------------------------------------------ one block, many settings

@dataclass
class BlockContext:
    """Everything the settings share for one block: scaled inputs, targets, the tail, the raw
    per-row edge contributions, and a mask scorer whose cache serves every setting."""

    nets: BlockNets
    prep: Any
    tgt: dict[str, torch.Tensor]
    tail: np.ndarray
    contrib: np.ndarray
    scorer: MaskScorer
    val_rows: np.ndarray


def block_context(X: pd.DataFrame, task: JointTask, block, nets: BlockNets) -> BlockContext:
    dev = trainer.device()
    rows = rows_array(block.train)
    y = task.target(block.train)
    lookback = nets.window + 20  # the graph's lag-21 states (NeuralLearner._lookback for graph nets)
    prep = prepare(X, rows, lookback, dev, nets.scaler, nets.missing_cols)
    tgt = _targets_for(y, rows, len(X), dev)
    _, tail = trainer.early_stop_split(rows)
    contrib = contributions(nets.stage1, prep, tgt, tail)
    scorer = MaskScorer(nets.stage1, prep, tgt, tail)
    return BlockContext(nets, prep, tgt, tail, contrib, scorer, rows_array(block.val))


def select_mask(setting: Setting, ctx: BlockContext, lam2_default: float, lam3_default: float,
                prev: np.ndarray | None = None, seed: int = LAB_SEED) -> tuple[np.ndarray, dict[str, Any]]:
    """The active edges under ``setting``, from the block's cached networks."""
    n_edges = ctx.scorer.n_edges
    lam2 = lam2_default if setting.lam2 is None else setting.lam2
    lam3 = lam3_default if setting.lam3 is None else setting.lam3
    t0 = time.perf_counter()
    if setting.mode == "all_on":
        mask, info = np.ones(n_edges, bool), {}
    else:
        rel = reliability(ctx.contrib, size=setting.subwindow, alpha=setting.alpha, beta=setting.beta)
        if setting.mode == "topk":
            mask = np.zeros(n_edges, bool)
            mask[np.argsort(-rel.r, kind="stable")[: setting.k]] = True
            info = {}
        elif setting.mode == "exhaustive":
            mask, f = exhaustive(ctx.scorer, n_edges, prev, lam2, lam3)
            info = {"fitness": f}
        else:
            sel = evolve(ctx.scorer, rel.r, prev=prev, rate=setting.rate, lam2=lam2, lam3=lam3,
                         pop_size=setting.pop_size, generations=setting.generations, seed=seed)
            mask, info = sel.mask, {"fitness": sel.fitness, "evaluations": sel.evaluations}
    info["seconds"] = round(time.perf_counter() - t0, 3)
    return mask, info


def block_predictions(ctx: BlockContext, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Out-of-fold forecasts on the block's validation rows with ``mask`` fixed on the final nets."""
    for net in ctx.nets.final:
        net.set_mask(torch.as_tensor(mask))
    try:
        return trainer.predict(ctx.nets.final, ctx.prep, ctx.val_rows)
    finally:
        for net in ctx.nets.final:
            net.set_mask(None)


# ------------------------------------------------------------------ one tuning point

def tuning_refit(fs: FeatureSet, cal, point: int) -> Refit:
    from xaglab.models.tuning import tuning_points

    refits = cal.all_refits(None)
    idx = sorted(set(tuning_points([r.origin for r in refits], 252)))
    return refits[idx[point]]


def evaluate_point(fs: FeatureSet, cal, point: int, params: dict, settings: tuple[Setting, ...] = ALL_SETTINGS,
                   cache_dir: Path | None = None, seed: int = LAB_SEED) -> dict[str, Any]:
    """Every setting's development score at one tuning point, through the same networks."""
    refit = tuning_refit(fs, cal, point)
    view = FitView.build(fs, refit, 0)
    X, task = view.data.pooled(), joint_task(view.data)
    blocks = cv_blocks(refit.train_start, refit.train_end, purge=refit.origin - refit.train_end)
    lam2_default, lam3_default = float(params.get("lam2", 0.0)), float(params.get("lam3", 0.0))
    preds: dict[str, list[dict[str, np.ndarray]]] = {s.name: [] for s in settings}
    masks: dict[str, list[list[int]]] = {s.name: [] for s in settings}
    infos: dict[str, list[dict]] = {s.name: [] for s in settings}
    rows_all, train_seconds = [], 0.0
    for b_i, block in enumerate(blocks):
        path = None if cache_dir is None else cache_dir / f"point{point:02d}_block{b_i}.pt"
        if path is not None and path.exists():
            nets = load_block(path, trainer.device())
        else:
            nets = train_block(X, task, block, params, seed)
            if path is not None:
                save_block(nets, path)
                nets = load_block(path, trainer.device())
        train_seconds += nets.seconds
        ctx = block_context(X, task, block, nets)
        for s in settings:
            mask, info = select_mask(s, ctx, lam2_default, lam3_default, seed=seed)
            preds[s.name].append(block_predictions(ctx, mask))
            masks[s.name].append([int(v) for v in mask])
            infos[s.name].append(info)
        rows_all.append(ctx.val_rows)
    rows = np.concatenate(rows_all)
    out: dict[str, Any] = {"point": point, "origin": str(fs.index[refit.origin].date()),
                           "params": params, "train_seconds": round(train_seconds, 1), "settings": {}}
    for s in settings:
        pred = {k: np.concatenate([p[k] for p in preds[s.name]]) for k in preds[s.name][0]}
        out["settings"][s.name] = {
            "cv_loss": task.score(pred, rows), "components": task.components(pred, rows),
            "masks": masks[s.name], "n_edges": float(np.mean([sum(m) for m in masks[s.name]])),
            "seconds": float(np.sum([i.get("seconds", 0.0) for i in infos[s.name]])),
            "evaluations": float(np.mean([i.get("evaluations", np.nan) for i in infos[s.name]])),
            "setting": asdict(s),
        }
    return out


# ------------------------------------------------------------------ across points

def summarise(points: list[dict[str, Any]], reference: str = "default") -> pd.DataFrame:
    """Per setting: mean development score, difference to the reference, points won, the
    5-day direction part, edges kept, and how far the masks moved from the reference's."""
    rows = []
    names = list(points[0]["settings"])
    for name in names:
        diffs, d5, edges, moved, losses = [], [], [], [], []
        for p in points:
            s, r = p["settings"][name], p["settings"][reference]
            losses.append(s["cv_loss"])
            diffs.append(s["cv_loss"] - r["cv_loss"])
            d5.append(s["components"]["direction_h5"] - r["components"]["direction_h5"])
            edges.append(s["n_edges"])
            moved.append(np.mean([np.sum(np.array(a) != np.array(b)) for a, b in zip(s["masks"], r["masks"], strict=True)]))
        rows.append({"setting": name, "cv_loss": float(np.mean(losses)), "diff_vs_ref": float(np.mean(diffs)),
                     "points_better": int(np.sum(np.array(diffs) < 0)), "points": len(points),
                     "p_wilcoxon": _wilcoxon(np.array(diffs)),
                     "diff_direction_h5": float(np.mean(d5)), "edges": float(np.mean(edges)),
                     "edges_moved_vs_ref": float(np.mean(moved))})
    return pd.DataFrame(rows)


def _wilcoxon(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p over the tuning points (1.0 for an identical setting)."""
    from scipy.stats import wilcoxon

    d = diffs[np.abs(diffs) > 1e-12]
    return 1.0 if len(d) < 5 else float(wilcoxon(d).pvalue)


def with_name(setting: Setting, name: str) -> Setting:
    return replace(setting, name=name)
