"""Evolutionary edge selection: PG-GFE's elimination scheduling (Eqs. 4–7, 11) re-aimed at
edges (step 4.4, D-34).

* A mask m ∈ {0,1}^15 marks the **active** edges (PG-GFE's mask marks *eliminated* features;
  here 1 = on, so an edge is never eliminated for good: masks are re-chosen every refit).
* Initialisation is biased by reliability: an edge with reliability r_e starts **off** with
  probability 1 − r_e (Eq. 5). With a previous winner, part of the population is that winner
  and its one-bit neighbours (the warm start, PG-GFE's staged inheritance).
* Mutation flips each bit with a rate from the **market-time clock** (Eq. 6): high right after
  the reliability scores shift (Eq. 11's stability measure crosses ε_c), low as a regime ages.
* Fitness (minimised; Eq. 7):  F(m) = ΔL(m) + λ2·|m − m_prev|/E + λ3·|m|/E, where ΔL is the
  ensemble's joint loss on the held-out tail with mask m minus with all edges on.
* Two-point crossover, tournament selection, and elitism (the best mask always survives).

Because the encoders don't depend on the mask, :class:`MaskScorer` encodes the tail once and
scores thousands of masks cheaply; that also makes an **exhaustive** check over all 2^15
masks affordable, which tells us whether the search finds the fitness optimum.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from deap import base, tools
from torch import nn

from xaglab.models.neural.data import Prepared
from xaglab.models.neural.losses import qlike_rel

POP_SIZE = 50
GENERATIONS = 100
RATE_MIN, RATE_MAX = 0.1, 0.3
CX_PROB = 0.5
TOURNAMENT = 3
N_WARM = 10

LossFn = Callable[[np.ndarray], np.ndarray]  # masks [M, E] bool -> tail loss [M]


# ------------------------------------------------------------------ scoring masks

def mask_key(m: np.ndarray) -> int:
    return int(np.dot(np.asarray(m, np.int64), 1 << np.arange(len(m), dtype=np.int64)))


def all_masks(n_edges: int) -> np.ndarray:
    """Every mask over ``n_edges`` edges, [2^E, E] bool, row i = the bits of i."""
    return ((np.arange(2 ** n_edges)[:, None] >> np.arange(n_edges)) & 1).astype(bool)


class MaskScorer:
    """Joint loss (D-28) of the **ensemble** forecast on held-out rows, for any batch of masks.

    The ensemble averages probabilities and variance ratios over the networks, exactly as
    deployed (``trainer.predict``). Encodings are computed once; results are cached per mask.
    """

    def __init__(self, nets: list[nn.Module], prep: Prepared, y: dict[str, torch.Tensor], rows: np.ndarray,
                 budget: int = 2 ** 26):
        self.nets = nets
        idx = torch.as_tensor(rows, device=prep.X.device)
        self.y = {k: v[idx] for k, v in y.items()}
        self.enc = []
        with torch.no_grad():
            for net in nets:
                net.eval()
                self.enc.append(net.encode(prep.windows(idx)))
        sources = self.enc[0][1]
        self.n_rows, self.n_edges = len(rows), sources.shape[1]
        self.chunk = max(1, budget // (self.n_rows * self.n_edges * sources.shape[2]))
        self.cache: dict[int, float] = {}
        self.evaluations = 0

    @torch.no_grad()
    def _score(self, masks: np.ndarray) -> np.ndarray:
        m = torch.as_tensor(masks, device=self.y["up1"].device)
        M, R = len(m), self.n_rows
        rep = m.repeat_interleave(R, dim=0)                         # [M·R, E]
        prob = {1: 0.0, 5: 0.0}
        ratio = {1: 0.0, 5: 0.0}
        for net, (silver, sources) in zip(self.nets, self.enc, strict=True):
            out = net.propagate(silver.repeat(M, 1), sources.repeat(M, 1, 1), rep)
            for h in (1, 5):
                prob[h] = prob[h] + torch.sigmoid(out[f"logit{h}"])
                ratio[h] = ratio[h] + torch.exp(out[f"z{h}"])
        n = len(self.nets)
        loss = 0.0
        for h in (1, 5):
            y = self.y[f"up{h}"].repeat(M)
            p = (prob[h] / n).clamp(1e-6, 1 - 1e-6)
            loss = loss + F.binary_cross_entropy(p, y, reduction="none")
            loss = loss + 0.5 * qlike_rel(torch.log(ratio[h] / n), self.y[f"r{h}"].repeat(M))
        return loss.view(M, R).nanmean(dim=1).cpu().numpy()

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, bool))
        keys = [mask_key(m) for m in masks]
        todo = sorted({k: i for i, k in enumerate(keys) if k not in self.cache}.items())
        for s in range(0, len(todo), self.chunk):
            part = todo[s: s + self.chunk]
            vals = self._score(masks[[i for _, i in part]])
            for (k, _), v in zip(part, vals, strict=True):
                self.cache[k] = float(v)
            self.evaluations += len(part)
        return np.array([self.cache[k] for k in keys])


def fitness(loss: np.ndarray, base_loss: float, masks: np.ndarray, prev: np.ndarray | None,
            lam2: float, lam3: float) -> np.ndarray:
    """Eq. 7 transferred: loss change + λ2·(share of edges changed vs last winner) + λ3·density."""
    masks = np.atleast_2d(masks)
    E = masks.shape[1]
    change = np.zeros(len(masks)) if prev is None else (masks != np.asarray(prev, bool)).sum(1) / E
    return (np.asarray(loss) - base_loss) + lam2 * change + lam3 * masks.sum(1) / E


# ------------------------------------------------------------------ the market-time clock

@dataclass
class MarketClock:
    """Mutation rate on market time (Eq. 6): τ counts refits since the last reliability shift.

    The shift at a refit is the mean absolute change in the edges' reliability (Eq. 11). It is
    a **regime shift** when it exceeds ``k`` × the median of the last ``window`` shifts (and at
    least ``floor``); until ``min_history`` shifts exist, the fixed threshold ``eps`` applies.
    The first refit counts as a shift (nothing is known yet), so the search starts exploring.

    Relative, not fixed (D-37): the planted-break test (step 4.7) showed shifts of ~0.004 in a
    stable regime jumping to 0.057 at a break, which a fixed ε_c = 0.10 never registered.
    Reliability changes have no natural scale, so the threshold is set against their own history.
    """

    T: int = 6
    eps: float = 0.10
    k: float = 3.0
    floor: float = 0.01
    window: int = 12
    min_history: int = 3
    r_min: float = RATE_MIN
    r_max: float = RATE_MAX
    tau: int = 0
    prev_r: np.ndarray | None = None
    last_shift: float = float("nan")
    history: list[float] = field(default_factory=list)

    def threshold(self) -> float:
        if len(self.history) < self.min_history:
            return self.eps
        return max(self.k * float(np.median(self.history[-self.window:])), self.floor)

    def tick(self, r: np.ndarray) -> float:
        r = np.asarray(r, float)
        if self.prev_r is None:
            self.last_shift, fired = float("inf"), True
        else:
            self.last_shift = float(np.mean(np.abs(r - self.prev_r)))
            fired = self.last_shift > self.threshold()
            self.history.append(self.last_shift)
        self.tau = 0 if fired else self.tau + 1
        self.prev_r = r
        return self.rate()

    def rate(self) -> float:
        return self.r_min + (self.r_max - self.r_min) * (1 - min(self.tau, self.T) / self.T)


# ------------------------------------------------------------------ the search

class _Fitness(base.Fitness):
    weights = (-1.0,)  # minimise


class _Mask(list):
    def __init__(self, bits):
        super().__init__(bits)
        self.fitness = _Fitness()


@dataclass
class Selection:
    mask: np.ndarray                 # [E] bool, the winner
    fitness: float
    loss: float                      # tail loss of the winner
    delta_loss: float                # vs all edges on
    finalists: list[np.ndarray]      # the top distinct masks by fitness (winner first)
    evaluations: int                 # distinct masks scored
    rate: float                      # mutation rate used
    history: list[float] = field(default_factory=list)  # best fitness per generation
    seconds: float = 0.0


def evolve(loss_fn: LossFn, r: np.ndarray, prev: np.ndarray | None = None, rate: float = 0.2,
           lam2: float = 0.0, lam3: float = 0.0, pop_size: int = POP_SIZE, generations: int = GENERATIONS,
           cx_prob: float = CX_PROB, n_warm: int = N_WARM, top_k: int = 3, seed: int = 0) -> Selection:
    """Search masks with DEAP operators; ``loss_fn`` scores a batch of masks on held-out rows."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    random.seed(seed)                                   # DEAP's operators use the global RNG
    E = len(r)
    base_loss = float(loss_fn(np.ones((1, E), bool))[0])
    seen: dict[tuple, float] = {}

    def evaluate(inds: list[_Mask]) -> None:
        todo = [ind for ind in inds if not ind.fitness.valid]
        if not todo:
            return
        masks = np.array(todo, bool)
        f = fitness(loss_fn(masks), base_loss, masks, prev, lam2, lam3)
        for ind, v in zip(todo, f, strict=True):
            ind.fitness.values = (float(v),)
            seen[tuple(ind)] = float(v)

    pop: list[_Mask] = []
    if prev is not None:                                # warm start: the winner and one-bit neighbours
        prev = np.asarray(prev, bool)
        pop.append(_Mask([int(b) for b in prev]))
        for e in rng.sample(range(E), min(n_warm - 1, E)):
            bits = [int(b) for b in prev]
            bits[e] = 1 - bits[e]
            pop.append(_Mask(bits))
    while len(pop) < pop_size:                          # Eq. 5: off with probability 1 − r_e
        pop.append(_Mask([int(rng.random() < r[e]) for e in range(E)]))
    evaluate(pop)
    best = tools.selBest(pop, 1)[0]
    history = [best.fitness.values[0]]
    for _ in range(generations):
        offspring = [_clone(ind) for ind in tools.selTournament(pop, len(pop), tournsize=TOURNAMENT)]
        for a, b in zip(offspring[::2], offspring[1::2], strict=False):
            if rng.random() < cx_prob:
                tools.cxTwoPoint(a, b)
                del a.fitness.values, b.fitness.values
        for ind in offspring:
            before = list(ind)
            tools.mutFlipBit(ind, indpb=rate)
            if list(ind) != before and ind.fitness.valid:
                del ind.fitness.values
        evaluate(offspring)
        worst = max(range(len(offspring)), key=lambda i: offspring[i].fitness.values[0])
        offspring[worst] = _clone(best)                 # elitism: the best mask always survives
        pop = offspring
        best = min(pop + [best], key=lambda ind: ind.fitness.values[0])
        history.append(best.fitness.values[0])

    ranked = sorted(seen.items(), key=lambda kv: kv[1])
    finalists = [np.array(k, bool) for k, _ in ranked[:top_k]]
    win = finalists[0]
    win_loss = float(loss_fn(win[None])[0])
    return Selection(win, ranked[0][1], win_loss, win_loss - base_loss, finalists, len(seen), rate, history,
                     time.perf_counter() - t0)


def exhaustive(loss_fn: LossFn, n_edges: int, prev: np.ndarray | None = None, lam2: float = 0.0,
               lam3: float = 0.0) -> tuple[np.ndarray, float]:
    """The fitness optimum over all 2^E masks: the check on the evolutionary search."""
    masks = all_masks(n_edges)
    base_loss = float(loss_fn(np.ones((1, n_edges), bool))[0])
    f = fitness(loss_fn(masks), base_loss, masks, prev, lam2, lam3)
    i = int(np.argmin(f))
    return masks[i], float(f[i])


def _clone(ind: _Mask) -> _Mask:
    c = _Mask(list(ind))
    if ind.fitness.valid:
        c.fitness.values = ind.fitness.values
    return c
