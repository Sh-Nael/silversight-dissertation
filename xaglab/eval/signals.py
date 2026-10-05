"""The signal and risk layer (fixed 28 Sep 2026): from a model's forecasts to
BUY / SELL / NO TRADE decisions with a stop-loss, a take-profit and a position size, and a
backtest of those decisions on daily bars. It is what the app will show for "today", and
what tells us whether a 53–58% signal is usable: profit comes from reward-to-risk and
selectivity, not from the hit rate alone.

Rules (fixed before any result was computed; none of them is tuned):
  signal        confident days only: the causal gate of D-31 (a forecast is confident when its
                |p − 0.5| is above the (1 − coverage) quantile of the model's own previous 252
                forecasts); BUY if p ≥ 0.5, SELL otherwise
  expected move σ = sqrt(var_h), the model's forecast standard deviation of the h-day log return
  entry         the close of the forecast day t (the forecast is for (t, t + h], as the labels)
  stop / target 1.0 σ / 1.5 σ from the entry, in log-price terms
  holding       days t + 1 … t + h; if neither level is hit, exit at the close of t + h
  both hit on the same day  counted as the STOP (a loss): daily bars don't say which came first
  costs         a round-trip cost in basis points (default 10) taken from every trade
  sizing        risk 1% of capital per trade: size = 1% / stop distance, capped at 1× capital
  account       one position at a time (a new signal is skipped while a trade is open)

Per-trade statistics treat every signal as an independent trade (the signal's quality);
the equity curve runs the account rules (what a single account would have experienced).
Buy and sell trades are always reported separately (D-32). The per-signal frame also carries
the causal win rate: the win rate of that model's earlier signals on the same side, known by
the time the signal is issued, which is the "past success rate" the app displays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xaglab.eval.report import COVERAGES, causal_selection
from xaglab.eval.stats import permutation_test

STOP_MULT = 1.0
TARGET_MULT = 1.5
RISK_PER_TRADE = 0.01
MAX_SIZE = 1.0
COST_BPS = 10.0


@dataclass(frozen=True)
class Rules:
    stop: float = STOP_MULT
    target: float = TARGET_MULT
    risk: float = RISK_PER_TRADE
    max_size: float = MAX_SIZE
    cost_bps: float = COST_BPS


DEFAULT_RULES = Rules()


def trade_path(entry: float, side: int, sigma: float, highs: np.ndarray, lows: np.ndarray,
               closes: np.ndarray, rules: Rules = DEFAULT_RULES) -> tuple[float, str, int]:
    """Outcome of one trade: (log return before costs, exit reason, days held).

    ``side`` is +1 (buy) or −1 (sell); ``highs/lows/closes`` are the bars of days t+1 … t+h.
    Exit reasons: "target", "stop", "timeout". A day that touches both levels is a stop.
    """
    stop_lvl = entry * np.exp(-side * rules.stop * sigma)
    target_lvl = entry * np.exp(side * rules.target * sigma)
    for k in range(len(closes)):
        hit_stop = lows[k] <= stop_lvl if side > 0 else highs[k] >= stop_lvl
        hit_target = highs[k] >= target_lvl if side > 0 else lows[k] <= target_lvl
        if hit_stop:                       # both on one day counts as the stop
            return -rules.stop * sigma, "stop", k + 1
        if hit_target:
            return rules.target * sigma, "target", k + 1
    return side * float(np.log(closes[-1] / entry)), "timeout", len(closes)


def signals(pred: pd.DataFrame, bars: pd.DataFrame, h: int, coverage: float, rules: Rules = DEFAULT_RULES,
            window: int = 252, min_history: int = 63) -> pd.DataFrame:
    """Every signal of one run at horizon h as an independent trade.

    ``pred`` is a run's prediction frame (``p_up{h}``, ``var{h}`` by date); ``bars`` the target's
    daily open/high/low/close over the whole panel (dates ⊇ pred's, and h days beyond).
    Columns: date, side (+1/−1), p, sigma, entry, stop, target, ret (log, before costs),
    net (after costs), exit, days, size, pnl (fraction of capital), causal_win_rate.
    """
    d = pred[[f"p_up{h}", f"var{h}"]].dropna()
    p = d[f"p_up{h}"].to_numpy(float)
    sigma = np.sqrt(d[f"var{h}"].to_numpy(float))
    sel = causal_selection(np.abs(p - 0.5), coverage, window, min_history)
    pos = bars.index.get_indexer(d.index)
    if (pos < 0).any():
        raise ValueError("prediction dates missing from the bars")
    hi, lo, cl = (bars[c].to_numpy(float) for c in ("high", "low", "close"))
    rows = []
    for i in np.flatnonzero(sel):
        t = pos[i]
        if t + h >= len(bars):
            continue                        # the holding period runs past the data
        side = 1 if p[i] >= 0.5 else -1
        entry = cl[t]
        ret, why, days = trade_path(entry, side, sigma[i], hi[t + 1: t + h + 1], lo[t + 1: t + h + 1],
                                    cl[t + 1: t + h + 1], rules)
        net = ret - rules.cost_bps / 1e4
        size = min(rules.risk / (1 - np.exp(-rules.stop * sigma[i])), rules.max_size)
        rows.append({"date": d.index[i], "side": side, "p": p[i], "sigma": sigma[i], "entry": entry,
                     "stop": entry * np.exp(-side * rules.stop * sigma[i]),
                     "target": entry * np.exp(side * rules.target * sigma[i]),
                     "ret": ret, "net": net, "exit": why, "days": days, "size": size, "pnl": size * net})
    out = pd.DataFrame(rows, columns=["date", "side", "p", "sigma", "entry", "stop", "target", "ret", "net",
                                      "exit", "days", "size", "pnl"])
    out["causal_win_rate"] = _causal_win_rate(out, h)
    return out


def _causal_win_rate(trades: pd.DataFrame, h: int) -> np.ndarray:
    """For each trade, the share of the same side's earlier trades that won, counting only
    trades whose outcome was known (closed) before this trade's entry."""
    out = np.full(len(trades), np.nan)
    if len(trades) == 0:
        return out
    dates = trades["date"].to_numpy()
    for side in (1, -1):
        idx = np.flatnonzero(trades["side"].to_numpy() == side)
        if len(idx) == 0:
            continue
        won = (trades["net"].to_numpy()[idx] > 0).astype(float)
        closed = dates[idx] + np.timedelta64(int(np.ceil(h * 7 / 5)) + 2, "D")  # a safe calendar bound
        for j, i in enumerate(idx):
            known = closed[:j] < dates[i]
            if known.sum() >= 10:
                out[i] = won[:j][known].mean()
    return out


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    """Per-trade statistics of a set of independent trades (after costs)."""
    if len(trades) == 0:
        return {"n": 0}
    net = trades["net"].to_numpy(float)
    wins, losses = net[net > 0], net[net <= 0]
    perm = permutation_test(-net, np.zeros(len(net)))          # mean(net) > 0 ?
    p_pos = perm.p_value / 2 if perm.stat < 0 else 1 - perm.p_value / 2
    return {
        "n": len(net),
        "win_rate": float((net > 0).mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "mean_net_bps": float(net.mean() * 1e4),
        "mean_pnl_bps": float(trades["pnl"].mean() * 1e4),      # per trade, as a fraction of capital
        "p_positive": float(p_pos),
        "target_rate": float((trades["exit"] == "target").mean()),
        "stop_rate": float((trades["exit"] == "stop").mean()),
        "avg_win_bps": float(wins.mean() * 1e4) if len(wins) else float("nan"),
        "avg_loss_bps": float(losses.mean() * 1e4) if len(losses) else float("nan"),
    }


def equity_curve(trades: pd.DataFrame, index: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """One position at a time: a signal is skipped while a trade is open. Equity in units of
    starting capital, marked at trade exits (flat between), on the given date index."""
    eq = pd.Series(1.0, index=index)
    pos = index.get_indexer(trades["date"]) if len(trades) else np.array([], int)
    busy_until, taken = -1, []
    for k, t in enumerate(pos):
        if t <= busy_until:
            continue
        busy_until = t + int(trades["days"].iloc[k])
        taken.append(k)
        eq.iloc[busy_until:] *= 1 + trades["pnl"].iloc[k]
    running_max = eq.cummax()
    dd = eq / running_max - 1
    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    return pd.DataFrame({"equity": eq, "drawdown": dd}).assign(
        trades_taken=len(taken), total_return=eq.iloc[-1] - 1, annual_return=eq.iloc[-1] ** (1 / years) - 1,
        max_drawdown=float(dd.min()))


def evaluate(pred: pd.DataFrame, bars: pd.DataFrame, h: int, coverages: tuple[float, ...] = COVERAGES,
             rules: Rules = DEFAULT_RULES) -> pd.DataFrame:
    """Per coverage level and side (buy, sell, both): trade statistics and the account's
    total return and maximum drawdown."""
    rows = []
    for c in coverages:
        tr = signals(pred, bars, h, c, rules)
        for side, name in ((None, "both"), (1, "buy"), (-1, "sell")):
            sub = tr if side is None else tr[tr["side"] == side]
            stats = trade_stats(sub)
            acct = equity_curve(sub, bars.index[bars.index >= pred.index[0]], h) if len(sub) else None
            rows.append({"coverage_target": c, "side": name, **stats,
                         "total_return": None if acct is None else float(acct["total_return"].iloc[0]),
                         "annual_return": None if acct is None else float(acct["annual_return"].iloc[0]),
                         "max_drawdown": None if acct is None else float(acct["max_drawdown"].iloc[0])})
    return pd.DataFrame(rows)
