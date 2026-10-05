"""Finalists: D-14 stage 2 (step 4.5). The screening stage scores masks with the edge-dropout
supernet; here the top masks are checked by **fine-tuning** each supernet with the mask fixed
and judging the resulting ensemble on the held-out tail.

Fine-tuning starts from the supernet's weights (not from scratch), stops early on the tail,
and never keeps a network that got worse on the tail than it started. So on the tail, a
finalist can only match or improve on its screened score.

Whether this stage earns its cost is an empirical question (step 4.3 found that on real data
retraining noise is as large as the differences between masks); the rule and the evidence
are in scripts/finalists_check.py (step 4.5).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from xaglab.models.neural import trainer
from xaglab.models.neural.data import Prepared

FINE_TUNE_EPOCHS = 20
FINE_TUNE_PATIENCE = 5


def clone(net: nn.Module) -> nn.Module:
    """A deep copy whose recurrent weights are compacted again: deepcopy leaves cuDNN LSTM weights
    in separate chunks, which cuDNN would otherwise re-pack (and re-allocate) on every call."""
    c = copy.deepcopy(net)
    for m in c.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()
    return c


def masked_copy(net: nn.Module, mask: np.ndarray) -> nn.Module:
    """A copy of ``net`` with ``mask`` fixed (edge dropout off)."""
    c = clone(net)
    c.set_mask(torch.as_tensor(np.asarray(mask, bool)))
    return c


def fine_tune(nets: list[nn.Module], mask: np.ndarray, prep: Prepared, y: dict[str, torch.Tensor],
              fit_rows: np.ndarray, tail: np.ndarray, cfg: trainer.TrainConfig, seed: int = 0
              ) -> list[nn.Module]:
    """Each supernet fine-tuned with ``mask`` fixed, early-stopped on ``tail``; a network whose
    tail loss got worse keeps its pre-fine-tuning weights."""
    ft_cfg = replace(cfg, max_epochs=FINE_TUNE_EPOCHS, patience=FINE_TUNE_PATIENCE)
    tail_idx = torch.as_tensor(tail, device=prep.X.device)
    out = []
    for k, net in enumerate(nets):
        start = masked_copy(net, mask)
        before = trainer.eval_loss(start, prep, y, tail_idx, cfg.lam)
        tuned = trainer.train(lambda s=start: clone(s), prep, y, fit_rows, tail, ft_cfg,
                              seed=seed * 101 + k).net
        after = trainer.eval_loss(tuned, prep, y, tail_idx, cfg.lam)
        out.append(tuned if after <= before else start.eval())
    return out


@dataclass(frozen=True)
class FinalistResult:
    mask: np.ndarray              # the winning mask
    nets: list[nn.Module]         # its fine-tuned ensemble
    tail_losses: list[float]      # per finalist, ensemble joint loss on the tail after fine-tuning
    winner: int                   # index into the finalists


def choose(finalists: list[np.ndarray], nets: list[nn.Module], prep: Prepared, y: dict[str, torch.Tensor],
           fit_rows: np.ndarray, tail: np.ndarray, cfg: trainer.TrainConfig, seed: int = 0) -> FinalistResult:
    """Fine-tune every finalist's ensemble; the lowest ensemble tail loss wins."""
    from xaglab.models.aimdg.evolution import MaskScorer

    tuned, losses = [], []
    for m in finalists:
        ens = fine_tune(nets, m, prep, y, fit_rows, tail, cfg, seed)
        tuned.append(ens)
        losses.append(float(MaskScorer(ens, prep, y, tail)(np.asarray(m, bool)[None])[0]))
    w = int(np.argmin(losses))
    return FinalistResult(np.asarray(finalists[w], bool), tuned[w], losses, w)
