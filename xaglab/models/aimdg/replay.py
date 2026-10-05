"""Replaying AIM-DG's monthly pipeline on cached networks (Phase 5, D-38).

A refit's expensive part is training its networks: the 3 out-of-fold blocks and the final
fit, 5 supernets each. Everything sequential in AIM-DG (last month's winner, the stability
term, the market-time clock) and every selection setting acts on those networks afterwards.
So a **bundle** per refit (its 4 sets of trained networks) can be trained once, in parallel
across refits, and any selection setting replayed through the bundles in order, in seconds:

  * the development walk-forward: one year of monthly refits inside the last training span,
    to measure the warm start, λ2, the clock's k and T, and the re-selection cadence;
  * the test-fold ablations A1/A2: the 200 test refits' bundles, replayed with "no
    evolution" and "no variance penalty", like for like with AIM-DG's own replay.

A replay reproduces `TunedNeural.fit` / `AimDG` exactly except for the networks being
shared across settings: out-of-fold selection and forecasts → Platt calibration and the
variance scale → the final selection (clock ticked once per refit) → forecasts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from xaglab.eval.folds import Refit
from xaglab.features.build import FeatureSet
from xaglab.models.aimdg.evolution import RATE_MAX, MarketClock
from xaglab.models.aimdg.lab import (
    LAB_SEED,
    BlockNets,
    Setting,
    block_context,
    block_predictions,
    load_block,
    save_block,
    select_mask,
    train_block,
)
from xaglab.models.aimdg.reliability import reliability
from xaglab.models.base import Calibrator, FitView, PredictView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.forecaster import HORIZONS, joint_task
from xaglab.models.tuning import cv_blocks, ewma_variance


@dataclass
class Bundle:
    """One refit's trained networks: three out-of-fold blocks and the final fit."""

    refit: Refit
    params: dict[str, Any]
    blocks: list[BlockNets]
    final: BlockNets

    @property
    def seconds(self) -> float:
        return sum(b.seconds for b in self.blocks) + self.final.seconds


def train_bundle(fs: FeatureSet, refit: Refit, params: dict, seed: int = 0, n_seeds: int = 5,
                 path: Path | None = None) -> Bundle:
    """Train (or load from ``path``) the refit's four sets of supernets."""
    if path is not None and path.exists():
        return load_bundle(path, refit, params)
    view = FitView.build(fs, refit, seed)
    X, task = view.data.pooled(), joint_task(view.data)
    blocks = cv_blocks(refit.train_start, refit.train_end, purge=refit.origin - refit.train_end)
    nets = [train_block(X, task, b, params, seed, n_seeds) for b in blocks]
    final = train_block(X, task, refit, params, seed, n_seeds)
    bundle = Bundle(refit, dict(params), nets, final)
    if path is not None:
        save_bundle(bundle, path)
        bundle = load_bundle(path, refit, params)
    return bundle


def save_bundle(bundle: Bundle, path: Path) -> None:
    for i, b in enumerate(bundle.blocks):
        save_block(b, path / f"block{i}.pt")
    save_block(bundle.final, path / "final.pt")


def load_bundle(path: Path, refit: Refit, params: dict) -> Bundle:
    dev = trainer.device()
    blocks = [load_block(path / f"block{i}.pt", dev) for i in range(3)]
    return Bundle(refit, dict(params), blocks, load_block(path / "final.pt", dev))


@dataclass(frozen=True)
class Sequential:
    """The settings that act across refits."""

    name: str
    warm_start: bool = True
    clock_k: float = 3.0
    clock_T: int = 6
    clock_eps: float = 0.10
    lam2: float | None = None      # None = the refit's tuned value
    base: Setting = field(default_factory=lambda: Setting("default"))


DEFAULT_SEQ = Sequential("default")


@dataclass
class Chain:
    prev: np.ndarray | None = None
    clock: MarketClock = field(default_factory=MarketClock)


def replay(fs: FeatureSet, bundles: list[Bundle | Callable[[], Bundle]], seq: Sequential = DEFAULT_SEQ,
           seed: int = LAB_SEED) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Forecasts for every bundle's origins, in order, under one sequential setting.

    ``bundles`` are given in refit order; an element may be a loader (callable) so that the
    200 test bundles need not sit on the GPU at once. Returns the forecast frame
    (``p_up1, p_up5, var1, var5`` by date) and one diagnostics record per refit."""
    chain = Chain(clock=MarketClock(T=seq.clock_T, eps=seq.clock_eps, k=seq.clock_k))
    frames, diags = [], []
    for item in bundles:
        bundle = item() if callable(item) else item
        refit = bundle.refit
        view = FitView.build(fs, refit, 0)
        X, task = view.data.pooled(), joint_task(view.data)
        lam2 = float(bundle.params.get("lam2", 0.0)) if seq.lam2 is None else seq.lam2
        lam3 = float(bundle.params.get("lam3", 0.0))
        prev = chain.prev if seq.warm_start else None
        # out-of-fold: the current rate, the clock not ticked (as AimDG._before_oof)
        blocks = cv_blocks(refit.train_start, refit.train_end, purge=refit.origin - refit.train_end)
        setting = replace(seq.base, rate=chain.clock.rate())
        oof, rows = [], []
        for block, nets in zip(blocks, bundle.blocks, strict=True):
            ctx = block_context(X, task, block, nets)
            mask, _ = select_mask(setting, ctx, lam2, lam3, prev=prev, seed=seed)
            oof.append(block_predictions(ctx, mask))
            rows.append(ctx.val_rows)
        oof = {k: np.concatenate([p[k] for p in oof]) for k in oof[0]}
        rows = np.concatenate(rows)
        calib = {h: Calibrator.fit(oof[f"up{h}"], task.direction[h].y[rows]) for h in HORIZONS}
        scale = {h: task.volatility[h].scale(oof[f"vol{h}"], rows) for h in HORIZONS}
        # final: tick the clock on this refit's reliability, then select
        ctx = block_context(X, task, refit, bundle.final)
        rel = reliability(ctx.contrib, size=seq.base.subwindow, alpha=seq.base.alpha, beta=seq.base.beta)
        rate = chain.clock.tick(rel.r) if seq.warm_start else RATE_MAX
        mask, info = select_mask(replace(seq.base, rate=rate), ctx, lam2, lam3, prev=prev, seed=seed)
        chain.prev = mask
        # forecasts for the refit's origins
        pview = PredictView.build(fs, refit)
        Xp = pview.data.pooled()[X.columns]
        prep = prepare(Xp, rows_array(refit.train), bundle.final.window + 20, trainer.device(),
                       bundle.final.scaler, bundle.final.missing_cols)
        for net in bundle.final.final:
            net.set_mask(torch.as_tensor(mask))
        pred = trainer.predict(bundle.final.final, prep, rows_array(pview.rows))
        for net in bundle.final.final:
            net.set_mask(None)
        s2 = ewma_variance(pview.data.nodes["silver"]["r1"])[pview.rows]
        out = pd.DataFrame(index=pview.dates)
        for h in HORIZONS:
            out[f"p_up{h}"] = calib[h](pred[f"up{h}"])
            out[f"var{h}"] = h * s2 * np.exp(pred[f"vol{h}"]) * scale[h]
        frames.append(out)
        edges = bundle.final.final[0].edges
        diags.append({"origin": str(pview.dates[0].date()), "fold": refit.fold, "refit": refit.j,
                      "active": [f"{d}@{lag}" for (d, lag), on in zip(edges, mask, strict=True) if on],
                      "n_edges": int(mask.sum()), "rate": rate, "tau": chain.clock.tau,
                      "shift": None if not np.isfinite(chain.clock.last_shift) else chain.clock.last_shift,
                      "warm_start": prev is not None, **{k: v for k, v in info.items() if k != "seconds"}})
    return pd.concat(frames).sort_index(), diags


def joint_loss(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Per-day joint loss of calibrated forecasts: logloss(1d) + logloss(5d) + ½(QLIKE 1d + 5d)."""
    from xaglab.eval.metrics import logloss, qlike

    y = labels.loc[pred.index]
    total = np.zeros(len(pred))
    for h in HORIZONS:
        total = total + logloss(pred[f"p_up{h}"].to_numpy(float), y[f"up{h}"].to_numpy(float))
        total = total + 0.5 * qlike(pred[f"var{h}"].to_numpy(float), y[f"rv{h}"].to_numpy(float))
    return pd.Series(total, index=pred.index)
