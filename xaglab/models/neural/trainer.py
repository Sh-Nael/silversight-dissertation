"""Training loop, early stopping, prediction and seed ensembles.

* Device: the GPU if available; set XAGLAB_DEVICE=cpu to force the CPU (tests do).
* Early stopping: the last rows of the given training rows (15%, at least 126) are held
  out, separated by a 5-row purge (the longest label horizon), and training stops once
  their loss hasn't improved for `patience` epochs. The best epoch's weights are kept.
  With no early-stopping rows (`val_rows` empty), `train` runs exactly `max_epochs`
  epochs and keeps the final weights: the second stage of "refit on all rows" (D-30).
* Reproducibility: every fit seeds Python, NumPy and PyTorch and asks cuDNN for
  deterministic kernels. On the CPU a fixed seed reproduces results exactly; on the GPU
  tiny differences can remain, which the 5-seed ensembles average out.
* Ensembles: probabilities are averaged; variances are averaged on the variance scale
  (z = log mean exp z), since the mean of forecasts is what gets scored.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from xaglab.models.neural.data import Prepared
from xaglab.models.neural.losses import multitask_loss

PURGE = 5
#: Rows per forward pass when scoring or predicting (no gradients). Each window is computed
#: independently, so this sets peak GPU memory only, never the result. 1,024 keeps 6 parallel
#: static_gnn workers inside 12 GB (2,048/4,096 triggered allocator retries on 27 Sep).
EVAL_BATCH = 1024


def device() -> torch.device:
    want = os.environ.get("XAGLAB_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def early_stop_split(rows: np.ndarray, frac: float = 0.15, min_val: int = 126, purge: int = PURGE
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Split sorted training rows into (fit rows, early-stopping rows), purged between."""
    n_val = max(min_val, int(len(rows) * frac))
    if n_val + purge + 50 > len(rows):
        raise ValueError("too few training rows for an early-stopping split")
    return rows[: len(rows) - n_val - purge], rows[len(rows) - n_val:]


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lam: float = 0.5
    batch_size: int = 256
    max_epochs: int = 150
    patience: int = 12
    clip: float = 1.0


@dataclass
class TrainResult:
    net: nn.Module
    best_epoch: int
    epochs: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)  # per epoch: train, val


def _batches(idx: torch.Tensor, size: int, shuffle: bool, gen: torch.Generator | None):
    order = idx[torch.randperm(len(idx), generator=gen, device="cpu").to(idx.device)] if shuffle else idx
    for i in range(0, len(order), size):
        yield order[i: i + size]


def eval_loss(net: nn.Module, prep: Prepared, y: dict[str, torch.Tensor], idx: torch.Tensor, lam: float) -> float:
    """Mean training loss (λ-weighted) on rows ``idx``, no gradients: early stopping's yardstick."""
    net.eval()
    with torch.no_grad():
        total, n = 0.0, 0
        for b in _batches(idx, EVAL_BATCH, False, None):
            loss, _ = multitask_loss(net(prep.windows(b)), {k: v[b] for k, v in y.items()}, lam)
            total += float(loss) * len(b)
            n += len(b)
    return total / max(n, 1)


def recency_weights(rows: np.ndarray, half_life: float, n: int, device: torch.device) -> torch.Tensor:
    """Per-row weights 0.5^(age / half_life), age in rows before the last training row, scaled
    to mean 1 over ``rows``; full length ``n`` (other rows get 0). Recency-weighted training
    (Phase 5 A4, D-37): recent regimes count more, so a new relationship is learned sooner."""
    w = np.zeros(n)
    age = rows[-1] - rows
    w[rows] = 0.5 ** (age / half_life)
    w[rows] /= w[rows].mean()
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def train(build_net: Callable[[], nn.Module], prep: Prepared, y: dict[str, torch.Tensor],
          fit_rows: np.ndarray, val_rows: np.ndarray | None, cfg: TrainConfig, seed: int,
          weights: torch.Tensor | None = None) -> TrainResult:
    """``weights``: optional full-length per-row training weights (early stopping stays unweighted)."""
    seed_all(seed)
    dev = prep.X.device
    net = build_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fit_idx = torch.as_tensor(fit_rows, device=dev)
    stop_early = val_rows is not None and len(val_rows) > 0
    val_idx = torch.as_tensor(val_rows if stop_early else [], device=dev, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, history, wait = float("inf"), 0, None, [], 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        run, n = 0.0, 0
        for b in _batches(fit_idx, cfg.batch_size, True, gen):
            loss, _ = multitask_loss(net(prep.windows(b)), {k: v[b] for k, v in y.items()}, cfg.lam,
                                     None if weights is None else weights[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
            opt.step()
            run += float(loss.detach()) * len(b)
            n += len(b)
        if not stop_early:
            history.append({"epoch": epoch, "train": run / n, "val": float("nan")})
            best_epoch = epoch
            continue
        val = eval_loss(net, prep, y, val_idx, cfg.lam)
        history.append({"epoch": epoch, "train": run / n, "val": val})
        if val < best - 1e-6:
            best, best_epoch, wait = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if stop_early:
        net.load_state_dict(best_state)
    net.eval()
    return TrainResult(net, best_epoch, len(history), time.perf_counter() - t0, history)


def predict(nets: list[nn.Module], prep: Prepared, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Ensemble forecast for `rows`: up1/up5 probabilities, vol1/vol5 relative log-variances."""
    idx = torch.as_tensor(rows, device=prep.X.device)
    probs = {1: [], 5: []}
    ratios = {1: [], 5: []}
    with torch.no_grad():
        for net in nets:
            net.eval()
            outs = [net(prep.windows(b)) for b in _batches(idx, EVAL_BATCH, False, None)]
            for h in (1, 5):
                probs[h].append(torch.cat([torch.sigmoid(o[f"logit{h}"]) for o in outs]))
                ratios[h].append(torch.cat([torch.exp(o[f"z{h}"]) for o in outs]))
    out = {}
    for h in (1, 5):
        out[f"up{h}"] = torch.stack(probs[h]).mean(0).cpu().numpy().astype(float)
        out[f"vol{h}"] = torch.log(torch.stack(ratios[h]).mean(0)).cpu().numpy().astype(float)
    return out
