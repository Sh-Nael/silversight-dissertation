"""The AIM-DG forecaster (step 4.6): the monthly pipeline of the design note, as a walk-forward
model in the same tuning framework as every other neural model.

At every fit (each out-of-fold block, and the final fit on the whole training span):
  1. train the 5-network edge-dropout supernet; keep the early-stopped networks, which have
     not seen the held-out tail, and the all-rows refits (D-30), which will forecast;
  2. reliability r_e of the 15 edges on the tail, from the early-stopped networks  (4.2);
  3. evolutionary search on the tail with the ensemble's joint loss               (4.4),
     warm-started from last month's winner, at the market-time clock's mutation rate;
  4. the winner is fixed on the all-rows networks (finalists dropped, D-36).

State carried from refit to refit (so AIM-DG runs its refits in calendar order):
  * last month's winning mask (warm start and the stability term);
  * the market-time clock (ticked once per refit, by the final fit's reliability scores).
The out-of-fold fits of a refit use the same previous winner and the clock's current rate
without ticking it. Tuning trials have no history: no warm start, the exploring rate.

Attribution (4.6): per refit, the mask, reliability, the fitness terms and the search
statistics; per forecast day, how much each active edge moves the calibrated probability of
an up move (the probability with all active edges minus without that edge).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from xaglab.models.aimdg.evolution import (
    GENERATIONS,
    POP_SIZE,
    RATE_MAX,
    MarketClock,
    MaskScorer,
    evolve,
    exhaustive,
)
from xaglab.models.aimdg.reliability import contributions, reliability
from xaglab.models.base import PredictView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.forecaster import HORIZONS, TunedNeural
from xaglab.models.neural.learner import Fitted, NeuralLearner, _targets_for

LAM_MAX = 0.005  # upper bound of the searched stability (λ2) and sparsity (λ3) weights, nats


class AimDGLearner(NeuralLearner):
    """A supernet learner whose every fit ends with edge selection (a sequential learner)."""

    keep_stage1 = True

    def __init__(self, *args, lam_max: float = LAM_MAX, pop_size: int = POP_SIZE,
                 generations: int = GENERATIONS, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_max, self.pop_size, self.generations = lam_max, pop_size, generations
        self.prev: np.ndarray | None = None      # last deployed winner (set by the forecaster)
        self.rate_fn = None                      # reliability -> mutation rate; None = exploring
        self.exhaustive_check = False            # only at deployed selections (costly)

    def space(self, trial) -> dict[str, Any]:
        out = super().space(trial)
        out["lam2"] = trial.suggest_float("lam2", 0.0, self.lam_max)
        out["lam3"] = trial.suggest_float("lam3", 0.0, self.lam_max)
        return out

    def fit_rows(self, X: pd.DataFrame, rows: slice, y: dict[str, np.ndarray], params: dict, seed: int,
                 n_seeds: int | None = None) -> Fitted:
        fitted = super().fit_rows(X, rows, y, params, seed, n_seeds)
        dev = trainer.device()
        r = rows_array(rows)
        prep = prepare(X, r, self._lookback(fitted.window), dev, fitted.scaler, fitted.missing_cols)
        tgt = _targets_for(y, r, len(X), dev)
        _, tail = trainer.early_stop_split(r)
        screen = fitted.stage1
        rel = reliability(contributions(screen, prep, tgt, tail))
        rate = self.rate_fn(rel.r) if self.rate_fn is not None else RATE_MAX
        lam2, lam3 = float(params.get("lam2", 0.0)), float(params.get("lam3", 0.0))
        scorer = MaskScorer(screen, prep, tgt, tail)
        sel = evolve(scorer, rel.r, prev=self.prev, rate=rate, lam2=lam2, lam3=lam3,
                     pop_size=self.pop_size, generations=self.generations, seed=seed)
        extra: dict[str, Any] = {"mask": sel.mask, "r": rel.r, "phi": rel.phi.mean(axis=1), "rate": rate,
                                 "fitness": sel.fitness, "delta_loss": sel.delta_loss, "tail_loss": sel.loss,
                                 "evaluations": sel.evaluations, "search_seconds": round(sel.seconds, 2),
                                 "lam2": lam2, "lam3": lam3, "warm_start": self.prev is not None}
        if self.prev is not None:  # last month's winner, scored on this month's tail (performance drift)
            extra["prev_tail_loss"] = float(scorer(np.asarray(self.prev, bool)[None])[0])
        if self.exhaustive_check:
            best, f_best = exhaustive(scorer, len(rel.r), self.prev, lam2, lam3)
            extra["exhaustive"] = {"same": bool(np.array_equal(best, sel.mask)),
                                   "gap": float(sel.fitness - f_best)}
        for net in fitted.nets:
            net.set_mask(torch.as_tensor(sel.mask))
        fitted.stage1 = None  # free the screening networks
        fitted.extra = extra
        return fitted


class AimDG(TunedNeural):
    """AIM-DG: the static GNN's network with a monthly, reversible choice of active edges."""

    stateless = False  # last month's winner and the clock carry over: refits run in order

    def __init__(self, cell: str = "lstm", lam_max: float = LAM_MAX, pop_size: int = POP_SIZE,
                 generations: int = GENERATIONS, clock_T: int = 6, clock_eps: float = 0.10,
                 clock_k: float = 3.0, exhaustive_check: bool = True, **kwargs):
        super().__init__(cell=cell, net="supernet", **kwargs)
        self.name = "aimdg" if self.half_life is None else "aimdg_rw"
        self.lam_max, self.pop_size, self.generations = lam_max, pop_size, generations
        self.clock_T, self.clock_eps, self.clock_k = clock_T, clock_eps, clock_k
        self.exhaustive_check = exhaustive_check
        self.clock = MarketClock(T=clock_T, eps=clock_eps, k=clock_k)
        self.prev_mask: np.ndarray | None = None
        self._prev_tail_loss: float | None = None

    def config(self) -> dict:
        return {**super().config(), "selection": "PG-GFE transfer (D-34): reliability + DEAP evolution",
                "pop_size": self.pop_size, "generations": self.generations, "lam_max": self.lam_max,
                "clock_T": self.clock_T, "clock_eps": self.clock_eps, "clock_k": self.clock_k,
                "clock_rule": "shift > max(k x median of last 12 shifts, 0.01); eps until 3 shifts (D-37)",
                "finalists": "none (D-36)",
                "edge_dropout": "keep-rate q ~ U(0.2, 1)", "exhaustive_check": self.exhaustive_check}

    def _learner(self, n_seeds: int) -> AimDGLearner:
        return AimDGLearner(self.net, self.cell, self.window, n_seeds, self.max_epochs, self.patience,
                            self.refit_all, self.windows, self.max_weight_decay, self.max_dropout,
                            half_life=self.half_life, lam_max=self.lam_max, pop_size=self.pop_size,
                            generations=self.generations)

    # ---- state across refits (hooks called by TunedNeural.fit)
    def _before_oof(self, learner: AimDGLearner) -> None:
        learner.prev = self.prev_mask
        rate = self.clock.rate()                    # the current rate (0.3 before the first tick)
        learner.rate_fn = lambda r: rate            # ... the clock is not ticked here
        learner.exhaustive_check = False

    def _before_final(self, learner: AimDGLearner) -> None:
        learner.prev = self.prev_mask
        learner.rate_fn = self.clock.tick           # one tick per refit, on this refit's reliability
        learner.exhaustive_check = self.exhaustive_check

    def _after_final(self) -> None:
        x = self.fitted.extra
        # performance drift (PG-GFE Algorithm 1's second convergence signal, ε_p): how much worse
        # last month's winner does on this month's tail than it did on last month's
        x["drift"] = (x["prev_tail_loss"] - self._prev_tail_loss) if "prev_tail_loss" in x else None
        self._prev_tail_loss = x["tail_loss"]
        self.prev_mask = x["mask"]

    # ---- forecasts and attribution
    def predict(self, view: PredictView) -> pd.DataFrame:
        out = super().predict(view)
        self._attribution = self._attribute(view, out)
        return out

    def _attribute(self, view: PredictView, out: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
        """Per day and active edge: calibrated P(up) with all active edges minus without that edge."""
        X = view.data.pooled()[self.cols]
        mask = self.fitted.extra["mask"]
        edges = self.fitted.nets[0].edges
        res: dict[str, dict[str, list[float]]] = {str(d.date()): {} for d in view.dates}
        for e in np.flatnonzero(mask):
            without = mask.copy()
            without[e] = False
            for net in self.fitted.nets:
                net.set_mask(torch.as_tensor(without))
            pred = self.learner.predict_rows(self.fitted, X, view.rows)
            name = f"{edges[e][0]}@{edges[e][1]}"
            for h in HORIZONS:
                delta = out[f"p_up{h}"].to_numpy() - self.calib[h](pred[f"up{h}"])
                for d, v in zip(view.dates, delta, strict=True):
                    res[str(d.date())].setdefault(name, []).append(round(float(v), 5))
        for net in self.fitted.nets:
            net.set_mask(torch.as_tensor(mask))
        return res

    def diagnostics(self) -> dict:
        x = self.fitted.extra
        edges = [f"{d}@{lag}" for d, lag in self.fitted.nets[0].edges]
        sel = {"active": [e for e, on in zip(edges, x["mask"], strict=True) if on],
               "reliability": {e: round(float(v), 4) for e, v in zip(edges, x["r"], strict=True)},
               "contribution_mnats": {e: round(float(v) * 1000, 3) for e, v in zip(edges, x["phi"], strict=True)},
               "clock": {"tau": self.clock.tau, "rate": x["rate"], "threshold": self.clock.threshold(),
                         "shift": self.clock.last_shift if np.isfinite(self.clock.last_shift) else None},
               **{k: x[k] for k in ("fitness", "delta_loss", "tail_loss", "evaluations", "search_seconds",
                                    "lam2", "lam3", "warm_start")},
               "drift": x.get("drift")}
        if "exhaustive" in x:
            sel["exhaustive"] = x["exhaustive"]
        return {**super().diagnostics(), "selection": sel,
                "attribution": getattr(self, "_attribution", {})}

