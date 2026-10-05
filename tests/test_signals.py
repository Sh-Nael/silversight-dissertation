"""Tests for the signal and risk layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaglab.eval.signals import Rules, equity_curve, evaluate, signals, trade_path, trade_stats

R = Rules(stop=1.0, target=1.5, risk=0.01, max_size=1.0, cost_bps=10.0)


def test_trade_path_by_hand():
    s = 0.02
    e = 100.0
    # buy: target 1.5σ = 103.05 hit on day 2 (high 104), stop 98.02 never touched
    ret, why, days = trade_path(e, 1, s, highs=np.array([101, 104, 105.]), lows=np.array([99.5, 99, 99.]),
                                closes=np.array([100.5, 103, 104.]), rules=R)
    assert why == "target" and days == 2 and np.isclose(ret, 1.5 * s)
    # buy: stop hit on day 1 (low 97), even though the target is also touched that day -> stop
    ret, why, days = trade_path(e, 1, s, np.array([104.0]), np.array([97.0]), np.array([100.0]), R)
    assert why == "stop" and days == 1 and np.isclose(ret, -s)
    # sell: price falls to the target (100 * exp(-0.03) = 97.04) on day 3
    ret, why, days = trade_path(e, -1, s, np.array([101, 100, 99.]), np.array([99, 98, 96.]), np.array([100, 99, 97.]), R)
    assert why == "target" and days == 3 and np.isclose(ret, 1.5 * s)
    # sell: nothing hit -> exit at the last close, return = -log(close / entry)
    ret, why, days = trade_path(e, -1, s, np.array([101, 101.]), np.array([99, 99.]), np.array([100, 99.]), R)
    assert why == "timeout" and days == 2 and np.isclose(ret, -np.log(0.99))


def _market(n=1200, seed=0, skill=0.0):
    """Daily bars and a run's forecasts; `skill` tilts p_up toward the realised next-day move."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    r = rng.normal(0, 0.015, n)
    close = 30 * np.exp(np.cumsum(r))
    high = close * np.exp(np.abs(rng.normal(0, 0.008, n)))
    low = close * np.exp(-np.abs(rng.normal(0, 0.008, n)))
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    nxt = np.r_[r[1:], 0.0]
    p = np.clip(0.5 + skill * np.sign(nxt) * rng.uniform(0, 0.2, n) + rng.normal(0, 0.05, n), 0.02, 0.98)
    pred = pd.DataFrame({"p_up1": p, "var1": np.full(n, 0.015 ** 2), "p_up5": p, "var5": np.full(n, 5 * 0.015 ** 2)},
                        index=idx)
    return bars, pred


def test_signals_frame_costs_sizing_and_causal_win_rate():
    bars, pred = _market()
    tr = signals(pred, bars, h=1, coverage=0.3, rules=R)
    assert 0 < len(tr) < len(pred)
    assert set(tr["side"]) <= {1, -1} and set(tr["exit"]) <= {"target", "stop", "timeout"}
    assert np.allclose(tr["net"], tr["ret"] - 0.001)                     # 10 bps round trip
    want = np.minimum(0.01 / (1 - np.exp(-0.015)), 1.0)
    assert np.allclose(tr["size"], want) and np.allclose(tr["pnl"], tr["size"] * tr["net"])
    # buy stops sit below the entry, sell stops above
    buys, sells = tr[tr.side == 1], tr[tr.side == -1]
    assert (buys.stop < buys.entry).all() and (sells.stop > sells.entry).all()
    # the causal win rate of trade i uses only earlier, closed trades of the same side
    k = buys.index[-1]
    earlier = buys[buys.date < tr.loc[k, "date"] - pd.Timedelta(days=4)]  # closed = entry + ceil(h·7/5) + 2 days
    assert np.isclose(tr.loc[k, "causal_win_rate"], (earlier.net > 0).mean())
    assert tr["causal_win_rate"].iloc[:10].isna().all()                # nothing known at first


def test_skill_shows_up_as_profit_and_noise_does_not():
    bars, good = _market(seed=1, skill=1.0)
    st = trade_stats(signals(good, bars, 1, 0.3, R))
    assert st["win_rate"] > 0.55 and st["profit_factor"] > 1.3 and st["p_positive"] < 0.01
    bars, noise = _market(seed=2, skill=0.0)
    sn = trade_stats(signals(noise, bars, 1, 0.3, R))
    assert sn["profit_factor"] < 1.1 and sn["p_positive"] > 0.05        # costs and the stop-first rule bite


def test_equity_curve_one_position_at_a_time_and_evaluate_table():
    bars, pred = _market(seed=3, skill=1.0)
    tr = signals(pred, bars, 5, 0.5, R)
    acct = equity_curve(tr, bars.index, 5)
    assert acct["trades_taken"].iloc[0] < len(tr)                       # overlapping signals skipped
    assert acct["equity"].iloc[0] == 1.0 and acct["max_drawdown"].iloc[0] <= 0
    assert np.isclose(acct["equity"].iloc[-1] - 1, acct["total_return"].iloc[0])
    table = evaluate(pred, bars, 1, coverages=(0.3, 1.0), rules=R)
    assert list(table["side"]) == ["both", "buy", "sell"] * 2
    both = table[(table.side == "both") & (table.coverage_target == 1.0)].iloc[0]
    assert both["n"] == table[(table.coverage_target == 1.0) & (table.side != "both")]["n"].sum()
