"""The multi-task loss (design note §2).

  L = BCE(up1) + BCE(up5) + λ · [QLIKE_rel(vol1) + QLIKE_rel(vol5)]

Direction heads output logits; binary cross-entropy is the log-loss they are scored on.

Volatility heads output z, the log of forecast variance relative to the causal EWMA
baseline. With f = baseline·e^z and r = rv / baseline, QLIKE = ln f + rv/f becomes
    ln(baseline) + z + r·e^{-z},
and ln(baseline) does not depend on the network, so the loss is QLIKE_rel = z + r·e^{-z}.
It is minimised at z = ln r: the network is trained on exactly the evaluation metric.

Rows with an unknown label (NaN) are left out of that term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def qlike_rel(z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return z + r * torch.exp(-z)


def _wmean(v: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    return v.mean() if w is None else (v * w).sum() / w.sum().clamp(min=1e-12)


def multitask_loss(out: dict[str, torch.Tensor], y: dict[str, torch.Tensor], lam: float,
                   w: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """``w``: optional per-row weights (recency weighting, D-37 / Phase 5 A4); each term becomes
    a weighted mean over the rows whose label is known."""
    parts: dict[str, torch.Tensor] = {}
    for h in (1, 5):
        t = y[f"up{h}"]
        m = ~torch.isnan(t)
        bce = F.binary_cross_entropy_with_logits(out[f"logit{h}"][m], t[m], reduction="none")
        parts[f"bce{h}"] = _wmean(bce, None if w is None else w[m])
        r = y[f"r{h}"]
        m = ~torch.isnan(r)
        parts[f"qlike{h}"] = _wmean(qlike_rel(out[f"z{h}"][m], r[m]), None if w is None else w[m])
    total = parts["bce1"] + parts["bce5"] + lam * (parts["qlike1"] + parts["qlike5"])
    return total, {k: float(v.detach()) for k, v in parts.items()}


def joint_nll_rows(out: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-row joint negative log-likelihood, the evaluation score of D-28 without averaging:
    BCE(up1) + BCE(up5) + ½·[QLIKE_rel(vol1) + QLIKE_rel(vol5)]. QLIKE_rel differs from QLIKE by
    ln(baseline), a per-row constant, so differences between two masks equal QLIKE differences.
    Rows with an unknown label get NaN."""
    total = 0.0
    for h in (1, 5):
        total = total + F.binary_cross_entropy_with_logits(out[f"logit{h}"], y[f"up{h}"], reduction="none")
        total = total + 0.5 * qlike_rel(out[f"z{h}"], y[f"r{h}"])
    return total
