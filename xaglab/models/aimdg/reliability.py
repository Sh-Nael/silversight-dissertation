"""Edge reliability: PG-GFE's confidence score (Eqs. 2–3), re-aimed at edges (step 4.2, D-34).

PG-GFE scores a feature by how consistently it relates to the labels, penalised by how much
that relation varies across labels:

    c_i = σ( α · mean_j cos(h_i, h_j)  −  β · Var_l φ(h_i, h_l) )

AIM-DG scores an **edge** e (a driver at a lag) the same way, with time in place of labels.
φ_e(s) is the edge's *predictive contribution* in sub-window s: how much the joint loss on
held-out rows rises when e alone is removed from the graph (positive = the edge helps).
Sub-windows are consecutive blocks of 21 trading days (one market month). Then

    r_e = σ( α · mean_s φ̃_e(s)  −  β · Var_s φ̃_e(s) ),   φ̃ = φ / sd(φ over all e, s)

so a strong but erratic edge scores below a moderate but steady one. φ is standardised
because raw loss differences are tiny (~0.001 nats) and would leave every r_e ≈ 0.5.

Why a contribution rather than PG-GFE's literal cosine: our driver and silver states come
from *different* encoders (one per node type), so a cosine between them has no meaning.
The contribution is exactly the "degree of feature–label connection" φ stands for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from xaglab.models.neural.data import Prepared
from xaglab.models.neural.losses import joint_nll_rows

SUBWINDOW = 21
ALPHA = 0.5
BETA = 0.5


@dataclass(frozen=True)
class Reliability:
    r: np.ndarray            # [E] reliability in (0, 1)
    consistency: np.ndarray  # [E] mean standardised contribution
    uncertainty: np.ndarray  # [E] variance of the standardised contribution across sub-windows
    phi: np.ndarray          # [E, S] raw contribution per edge and sub-window (nats per row)


def subwindows(n_rows: int, size: int = SUBWINDOW) -> list[slice]:
    """Consecutive blocks of ``size`` rows ending at the last row; an incomplete oldest block
    is dropped, so every block is one full market month and the newest data always counts."""
    k = n_rows // size
    if k < 2:
        raise ValueError(f"need at least 2 sub-windows of {size} rows, got {n_rows} rows")
    start = n_rows - k * size
    return [slice(start + i * size, start + (i + 1) * size) for i in range(k)]


def reliability(contrib: np.ndarray, size: int = SUBWINDOW, alpha: float = ALPHA,
                beta: float = BETA) -> Reliability:
    """Reliability scores from per-row contributions ``contrib`` [rows, E] (rows in date order)."""
    contrib = np.asarray(contrib, float)
    phi = np.stack([np.nanmean(contrib[s], axis=0) for s in subwindows(len(contrib), size)], axis=1)
    scale = np.std(phi)
    z = phi / scale if scale > 0 else np.zeros_like(phi)
    consistency = z.mean(axis=1)
    uncertainty = z.var(axis=1)
    r = 1.0 / (1.0 + np.exp(-(alpha * consistency - beta * uncertainty)))
    return Reliability(r, consistency, uncertainty, phi)


@torch.no_grad()
def contributions(nets: list[nn.Module], prep: Prepared, y: dict[str, torch.Tensor], rows: np.ndarray,
                  batch: int = 1024) -> np.ndarray:
    """Per-row contribution of every edge, averaged over an ensemble of graph networks.

    contribution[row, e] = joint NLL with edge e removed − joint NLL with all edges on.
    The encoders run once per batch; only the cheap graph step is repeated per mask.
    """
    n_edges = len(nets[0].edges)
    dev = prep.X.device
    full = torch.ones(n_edges, dtype=torch.bool, device=dev)
    drop_one = [full.clone().index_fill_(0, torch.tensor([e], device=dev), False) for e in range(n_edges)]
    out = np.zeros((len(rows), n_edges))
    for net in nets:
        net.eval()
        for i in range(0, len(rows), batch):
            b = torch.as_tensor(rows[i: i + batch], device=dev)
            yb = {k: v[b] for k, v in y.items()}
            silver, sources = net.encode(prep.windows(b))
            base = joint_nll_rows(net.propagate(silver, sources, full), yb)
            for e, m in enumerate(drop_one):
                out[i: i + len(b), e] += (joint_nll_rows(net.propagate(silver, sources, m), yb) - base).cpu().numpy()
    return out / len(nets)
