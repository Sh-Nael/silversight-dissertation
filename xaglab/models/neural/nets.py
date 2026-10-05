"""Networks.

SequenceNet: the `lstm` / `gru` baselines. All features of all nodes, stacked per day,
go through a 1–2-layer recurrent network over the W-day window; the last hidden state
feeds four small heads (logit1, logit5, z1, z5). Flat in structure, but with memory.

StaticGraphNet: the `static_gnn` baseline and the network AIM-DG will drive (D-33). Time, then graph:
  1. encode: each node's own columns → a node-specific input layer → the LSTM of its
     node *type* (weights shared within the type) over W + 20 days;
  2. graph: silver (the target) plus 15 source nodes = 5 drivers × lags 1, 5, 21, where
     lag ℓ is the driver's encoded state ℓ − 1 days before the origin;
  3. message passing: per edge type a GraphConv (lin_rel(source) + lin_root(silver)),
     averaged over the *active* edges (exactly PyG HeteroConv with aggr="mean"; a unit
     test checks this), with a residual and LayerNorm; with 2 layers, reverse edges first
     let each source see silver;
  4. the same four heads as SequenceNet, read from silver.
The edge mask (default: all on) is what AIM-DG sets. Two AIM-DG options (Phase 4):
  * ``edge_dropout=(lo, hi)``: in training mode each sample draws a keep-rate q ~ U(lo, hi)
    and keeps each edge with probability q, so one network (the "supernet") learns to work
    under any mask and a mask can be scored by a forward pass (step 4.3, D-14 stage 1);
  * ``set_mask(m)``: a fixed mask used whenever no mask is passed, in training and in
    prediction, and it switches edge dropout off (fine-tuning finalists, step 4.5).
"""

from __future__ import annotations

import math

import torch
from torch import nn

OUTPUTS = ("logit1", "logit5", "z1", "z5")


class Heads(nn.Module):
    """Four small heads on a shared representation."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.out = nn.ModuleDict({k: nn.Linear(dim, 1) for k in OUTPUTS})

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.drop(h)
        return {k: layer(h).squeeze(-1) for k, layer in self.out.items()}


class SequenceNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1, dropout: float = 0.1,
                 cell: str = "lstm"):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.heads = Heads(hidden, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq, _ = self.rnn(x)          # [B, W, hidden]
        return self.heads(seq[:, -1])  # last time step = the forecast origin


#: Node types for the type-shared encoders (Ch3 §3.4): the target, traded metals, macro-
#: financial series, and the risk (mood) gauge.
NODE_TYPES = {"silver": "target", "gold": "metal", "copper": "metal", "dxy": "macro",
              "real_yield_10y": "macro", "vix": "risk"}
#: Edge lags in trading days: lag ℓ reads the driver's state ℓ − 1 days before the origin.
LAGS = (1, 5, 21)
TARGET = "silver"


def column_groups(columns: list[str]) -> dict[str, list[int]]:
    """Column positions per node from pooled names ``<node>__<feature>[__missing]``;
    event columns (``event__…``) belong to silver's input (D-33 decision 6)."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(columns):
        node = c.split("__", 1)[0]
        groups.setdefault(TARGET if node == "event" else node, []).append(i)
    unknown = set(groups) - set(NODE_TYPES)
    if unknown:
        raise ValueError(f"no node type for {sorted(unknown)}")
    return groups


class EdgeTypeLinear(nn.Module):
    """One linear map per edge type, stored as a stack so all edge types run in one op.
    Initialised like nn.Linear."""

    def __init__(self, n_types: int, dim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_types, dim, dim))
        self.bias = nn.Parameter(torch.empty(n_types, dim)) if bias else None
        for k in range(n_types):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / math.sqrt(dim), 1 / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, E, dim] (a different input per edge type) or [B, dim] (the same input)."""
        eq = "beh,eoh->beo" if x.dim() == 3 else "bh,eoh->beo"
        out = torch.einsum(eq, x, self.weight)
        return out if self.bias is None else out + self.bias


class StaticGraphNet(nn.Module):
    def __init__(self, groups: dict[str, list[int]], hidden: int = 32, layers: int = 1,
                 gnn_layers: int = 1, dropout: float = 0.1, cell: str = "lstm", window: int = 60,
                 edge_dropout: tuple[float, float] | None = None):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.window, self.gnn_layers, self.edge_dropout = window, gnn_layers, edge_dropout
        self.nodes = list(groups)
        for n, idx in groups.items():  # column positions travel with the module (device moves)
            self.register_buffer(f"cols_{n}", torch.as_tensor(idx), persistent=False)
        self.drivers = [n for n in NODE_TYPES if n != TARGET and n in groups]
        self.edges = [(d, lag) for d in self.drivers for lag in LAGS]
        self.proj = nn.ModuleDict({n: nn.Linear(len(idx), hidden) for n, idx in groups.items()})
        types = sorted({NODE_TYPES[n] for n in groups})
        self.enc = nn.ModuleDict({t: rnn(hidden, hidden, num_layers=layers, batch_first=True,
                                         dropout=dropout if layers > 1 else 0.0) for t in types})
        E = len(self.edges)
        # layer k: sources -> silver (per edge type), then, if another layer follows, silver -> sources
        self.rel = nn.ModuleList([EdgeTypeLinear(E, hidden, bias=True) for _ in range(gnn_layers)])
        self.root = nn.ModuleList([EdgeTypeLinear(E, hidden, bias=False) for _ in range(gnn_layers)])
        self.rev_rel = nn.ModuleList([EdgeTypeLinear(E, hidden, bias=True) for _ in range(gnn_layers - 1)])
        self.rev_root = nn.ModuleList([EdgeTypeLinear(E, hidden, bias=False) for _ in range(gnn_layers - 1)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(gnn_layers)])
        self.src_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(gnn_layers - 1)])
        self.drop = nn.Dropout(dropout)
        self.heads = Heads(hidden, dropout)
        self.register_buffer("fixed_mask", None)

    def set_mask(self, mask: torch.Tensor | None) -> None:
        """Fix the active edges (bool [E]) for training and prediction; None = back to default."""
        self.fixed_mask = None if mask is None else torch.as_tensor(mask, dtype=torch.bool,
                                                                    device=self.rel[0].weight.device)

    def sample_masks(self, n: int, device: torch.device) -> torch.Tensor:
        """Edge dropout: per sample a keep-rate q ~ U(lo, hi), each edge kept with probability q."""
        lo, hi = self.edge_dropout
        q = torch.rand(n, 1, device=device) * (hi - lo) + lo
        return torch.rand(n, len(self.edges), device=device) < q

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, T, F] (T = W + 20) → silver's state [B, H] and source states [B, E, H]."""
        B = x.shape[0]
        seqs: dict[str, torch.Tensor] = {}
        for t, enc in self.enc.items():
            nodes = [n for n in self.nodes if NODE_TYPES[n] == t]
            z = torch.cat([self.proj[n](x[:, :, getattr(self, f"cols_{n}")]) for n in nodes])  # [B·k, T, H]
            out, _ = enc(z)
            for i, n in enumerate(nodes):
                seqs[n] = out[i * B: (i + 1) * B]
        silver = seqs[TARGET][:, -1]
        sources = torch.stack([seqs[d][:, -lag] for d, lag in self.edges], dim=1)
        return silver, sources

    def conv(self, k: int, silver: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Layer k's message into silver: mean over active edge types e of
        lin_rel_e(source_e) + lin_root_e(silver), i.e. HeteroConv(GraphConv, aggr="mean")."""
        msg = self.rel[k](sources) + self.root[k](silver)          # [B, E, H]
        m = mask.to(msg.dtype).expand(msg.shape[0], -1).unsqueeze(-1)
        return (msg * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        silver, sources = self.encode(x)
        return self.propagate(silver, sources, mask)

    def propagate(self, silver: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor | None = None
                  ) -> dict[str, torch.Tensor]:
        """Graph layers + heads from encoded states. Split from ``encode`` because the encoders
        don't depend on the mask: AIM-DG encodes once and tries many masks (``mask`` [E] for all
        samples, or [B, E] per sample)."""
        if mask is None:
            if self.fixed_mask is not None:
                mask = self.fixed_mask
            elif self.training and self.edge_dropout is not None:
                mask = self.sample_masks(silver.shape[0], silver.device)
            else:
                mask = torch.ones(len(self.edges), dtype=torch.bool, device=silver.device)
        for k in range(self.gnn_layers):
            new_silver = self.norm[k](silver + self.drop(torch.relu(self.conv(k, silver, sources, mask))))
            if k < self.gnn_layers - 1:  # reverse edges: each source hears silver (same, synchronous step)
                upd = self.rev_rel[k](silver) + self.rev_root[k](sources)
                sources = self.src_norm[k](sources + self.drop(torch.relu(upd)))
            silver = new_silver
        return self.heads(silver)
