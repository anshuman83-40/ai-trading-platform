"""Tests for the trading costs and the backtest engine (step 5). Hand-made prices, no internet.
Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
import backtest as bt                    # noqa: E402
import costs as cs                       # noqa: E402

FREE = dict(plan=cs.NO_COSTS, slippage=0.0, stop_loss=None, max_drawdown_halt=None, max_per_industry=None)


# ───────────── costs ─────────────
def test_groww_buy_cost_worked_by_hand():
    c = cs.cost_breakdown("buy", 10_000.0)
    assert c["brokerage"] == pytest.approx(10.0)                     # 0.1% of 10,000 (between the Rs 5 minimum and Rs 20 cap)
    assert c["stt"] == pytest.approx(10.0)                           # 0.1%
    assert c["stamp"] == pytest.approx(1.5)                          # 0.015%
    assert c["exchange"] == pytest.approx(0.297) and c["sebi"] == pytest.approx(0.01) and c["ipft"] == pytest.approx(0.01)
    assert c["dp"] == 0.0
    assert c["gst"] == pytest.approx(0.18 * (10.0 + 0.297 + 0.01 + 0.01))
    assert c["total"] == pytest.approx(10 + 10 + 1.5 + 0.297 + 0.01 + 0.01 + 1.85706)


def test_groww_sell_cost_has_dp_charge_and_no_stamp_duty():
    c = cs.cost_breakdown("sell", 10_000.0)
    assert c["stamp"] == 0.0 and c["dp"] == 20.0 and c["stt"] == pytest.approx(10.0)
    assert c["gst"] == pytest.approx(0.18 * (10.0 + 20.0 + 0.297 + 0.01 + 0.01))
    assert c["total"] == pytest.approx(10 + 10 + 0.297 + 0.01 + 0.01 + 20 + 5.45706)


def test_brokerage_minimum_and_cap_and_small_sell_without_dp():
    assert cs.cost_breakdown("buy", 1_000.0)["brokerage"] == 5.0     # 0.1% would be Rs 1 -> minimum Rs 5
    assert cs.cost_breakdown("buy", 500_000.0)["brokerage"] == 20.0  # 0.1% would be Rs 500 -> cap Rs 20
    assert cs.cost_breakdown("sell", 99.0)["dp"] == 0.0
    assert cs.trade_cost("buy", 0) == 0.0
    with pytest.raises(ValueError):
        cs.trade_cost("hold", 100)


def test_small_trades_cost_a_much_bigger_share():
    assert cs.round_trip_pct(2_500) > cs.round_trip_pct(10_000) > cs.round_trip_pct(50_000) > 0.2


# ───────────── engine helpers ─────────────
def make_prices(table, start="2024-01-01", volume=1e6, dividends=None):
    """table: {symbol: [open/close price per day]} (open = close = that price unless a tuple (open, close))."""
    n = len(next(iter(table.values())))
    days = pd.bdate_range(start, periods=n)
    rows = []
    for sym, vals in table.items():
        for d, v in zip(days, vals):
            if v is None:
                continue
            o, c = v if isinstance(v, tuple) else (v, v)
            rows.append({"date": d, "symbol": sym, "open": float(o), "high": float(max(o, c)), "low": float(min(o, c)),
                         "close": float(c), "volume": float(volume),
                         "dividends": float((dividends or {}).get((sym, d), 0.0))})
    return pd.DataFrame(rows), days


def const_scores(days, scores):
    return pd.DataFrame([{"date": d, "symbol": s, "score": v} for d in days for s, v in scores.items()])


# ───────────── engine ─────────────
def test_orders_are_filled_at_the_NEXT_days_open_not_on_the_decision_day():
    prices, days = make_prices({"A": [(100, 100), (200, 200), (200, 200), (200, 200)]})
    res = bt.run(const_scores(days, {"A": 1.0}), prices, bt.Settings(capital=1000, top_k=1, **FREE))
    t = res["trades"][0]
    assert t["buy_date"] == str(days[1].date()) and t["buy_price"] == 200.0     # not day 0 at 100
    assert t["shares"] == 5 and res["equity"]["equity"].iloc[-1] == pytest.approx(1000.0)


def test_flat_prices_and_no_costs_keep_the_money_unchanged():
    prices, days = make_prices({"A": [50] * 12, "B": [20] * 12, "C": [10] * 12})
    res = bt.run(const_scores(days, {"A": 3, "B": 2, "C": 1}), prices, bt.Settings(capital=10_000, top_k=2, **FREE))
    assert (res["equity"]["equity"].round(6) == 10_000).all()
    assert bt.stats(res)["total_return"] == 0.0 and res["costs"] == 0.0


def test_costs_reduce_the_money_by_exactly_the_charges():
    prices, days = make_prices({"A": [100] * 8})
    s = bt.Settings(capital=10_000, top_k=1, slippage=0.0, stop_loss=None, max_drawdown_halt=None)
    res = bt.run(const_scores(days, {"A": 1.0}), prices, s)
    shares = res["trades"][0]["shares"]
    expected = cs.trade_cost("buy", shares * 100) + cs.trade_cost("sell", shares * 100)
    assert res["costs"] == pytest.approx(expected)
    assert res["equity"]["equity"].iloc[-1] == pytest.approx(10_000 - expected)
    assert shares * 100 + cs.trade_cost("buy", shares * 100) <= 10_000          # never spends more than it has


def test_slippage_makes_buys_dearer_and_sells_cheaper():
    prices, days = make_prices({"A": [100] * 6})
    s = bt.Settings(capital=10_000, top_k=1, plan=cs.NO_COSTS, slippage=0.01, stop_loss=None, max_drawdown_halt=None)
    t = bt.run(const_scores(days, {"A": 1.0}), prices, s)["trades"][0]
    assert t["buy_price"] == pytest.approx(101.0) and t["sell_price"] == pytest.approx(99.0) and t["pnl"] < 0


def test_profit_is_counted_correctly():
    prices, days = make_prices({"A": [100, 100, 110, 121, 121, 121]})
    res = bt.run(const_scores(days, {"A": 1.0}), prices, bt.Settings(capital=1000, top_k=1, **FREE))
    st = bt.stats(res)
    assert res["trades"][0]["shares"] == 10 and st["end_capital"] == pytest.approx(1210.0)
    assert st["total_return"] == pytest.approx(0.21) and st["win_rate"] == 1.0 and st["max_drawdown"] == 0.0


def test_rebalance_sells_the_stock_that_dropped_out_and_buys_the_new_one():
    prices, days = make_prices({"A": [10] * 13, "B": [10] * 13})
    sc = pd.DataFrame([{"date": d, "symbol": s, "score": (1.0 if (s == "A") == (i < 5) else 0.0)}
                       for i, d in enumerate(days) for s in ("A", "B")])     # A is best for 5 days, then B
    res = bt.run(sc, prices, bt.Settings(capital=1000, top_k=1, rebalance_every=5, **FREE))
    assert [(t["symbol"], t["reason"]) for t in res["trades"]] == [("A", "rebalance"), ("B", "end_of_test")]
    assert res["trades"][0]["sell_date"] == str(days[6].date()) == res["trades"][1]["buy_date"]


def test_stop_loss_sells_at_the_next_open_after_a_bad_close():
    prices, days = make_prices({"A": [100, 100, 100, (100, 85), (80, 80), 80, 80]})
    s = bt.Settings(capital=1000, top_k=1, plan=cs.NO_COSTS, slippage=0.0, stop_loss=0.10, max_drawdown_halt=None)
    t = bt.run(const_scores(days, {"A": 1.0}), prices, s)["trades"][0]
    assert t["reason"] == "stop_loss" and t["sell_date"] == str(days[4].date()) and t["sell_price"] == 80.0


def test_maximum_loss_rule_sells_everything_and_pauses():
    table = {"A": [100, 100] + [70] * 30, "B": [100, 100] + [70] * 30}
    prices, days = make_prices(table)
    s = bt.Settings(capital=10_000, top_k=2, plan=cs.NO_COSTS, slippage=0.0, stop_loss=None,
                    max_drawdown_halt=0.20, halt_days=10, max_per_industry=None)
    res = bt.run(const_scores(days, {"A": 2.0, "B": 1.0}), prices, s)
    first = [t for t in res["trades"] if t["reason"] == "max_loss_rule"]
    assert len(first) == 2 and all(t["sell_date"] == str(days[3].date()) for t in first)
    held = res["equity"]["holdings"]
    assert (held.iloc[3:13] == 0).all() and held.iloc[-2] > 0        # out of the market during the pause, back later


def test_whole_shares_only_and_too_expensive_stocks_are_skipped():
    prices, days = make_prices({"A": [600] * 8, "B": [90] * 8})
    res = bt.run(const_scores(days, {"A": 2.0, "B": 1.0}), prices, bt.Settings(capital=1000, top_k=2, **FREE))
    assert res["skipped_buys"] >= 1                                  # position size is Rs 500, one A share costs 600
    assert {t["symbol"] for t in res["trades"]} == {"B"} and res["trades"][0]["shares"] == 5


def test_industry_limit_is_respected():
    prices, days = make_prices({s: [10] * 8 for s in "ABCD"})
    ind = {"A": "Bank", "B": "Bank", "C": "Bank", "D": "IT"}
    s = bt.Settings(capital=1000, top_k=3, plan=cs.NO_COSTS, slippage=0.0, stop_loss=None, max_drawdown_halt=None, max_per_industry=2)
    res = bt.run(const_scores(days, {"A": 4, "B": 3, "C": 2, "D": 1}), prices, s, industry=ind)
    assert {t["symbol"] for t in res["trades"]} == {"A", "B", "D"}


def test_dividends_are_received_as_cash():
    prices, days = make_prices({"A": [100] * 6})
    prices.loc[(prices["date"] == days[3]), "dividends"] = 2.0
    res = bt.run(const_scores(days, {"A": 1.0}), prices, bt.Settings(capital=1000, top_k=1, **FREE))
    assert res["equity"]["equity"].iloc[-1] == pytest.approx(1000 + 10 * 2.0)


def test_stock_that_does_not_trade_is_not_bought_and_a_sell_waits():
    prices, days = make_prices({"A": [10, None, 10, 10, 10, 10, 10, None, 10, 10, 10, 10], "B": [10] * 12})
    sc = pd.DataFrame([{"date": d, "symbol": s, "score": (1.0 if (s == "A") == (i < 2 or i >= 5) else 0.0)}
                       for i, d in enumerate(days) for s in ("A", "B") if not (s == "A" and i in (1, 7))])
    res = bt.run(sc, prices, bt.Settings(capital=1000, top_k=1, rebalance_every=5, **FREE))
    assert res["skipped_buys"] == 1                                  # A had no price on day 1: not bought
    assert res["trades"][0]["symbol"] == "A" and res["trades"][0]["buy_date"] == str(days[6].date())


def test_unadjusted_corporate_action_is_not_counted_as_a_loss():
    prices, days = make_prices({"A": [100, 100, 100, 40, 40, 40]})
    res = bt.run(const_scores(days, {"A": 1.0}), prices, bt.Settings(capital=1000, top_k=1, **FREE),
                 breaks={"A": [str(days[3].date())]})
    assert res["trades"][0]["reason"] == "corporate_action" and res["trades"][0]["pnl"] == 0.0
    assert res["equity"]["equity"].iloc[3] == pytest.approx(1000.0)


def test_future_scores_and_prices_do_not_change_earlier_results():
    rng = np.random.default_rng(5)
    table = {s: list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))) for s in "ABCDEFGH"}
    prices, days = make_prices(table)
    sc = pd.DataFrame([{"date": d, "symbol": s, "score": rng.random()} for d in days for s in table])
    s = bt.Settings(capital=50_000, top_k=3, max_per_industry=None)
    full = bt.run(sc, prices, s)["equity"]["equity"]
    cut = days[40]
    p2, s2 = prices.copy(), sc.copy()
    p2.loc[p2["date"] > cut, ["open", "high", "low", "close"]] *= 5
    s2.loc[s2["date"] > cut, "score"] = 0.5
    changed = bt.run(s2, p2, s)["equity"]["equity"]
    pd.testing.assert_series_equal(full[full.index <= cut], changed[changed.index <= cut])


def test_max_drawdown_and_sharpe_inputs():
    assert bt.max_drawdown([100, 120, 90, 130]) == pytest.approx(90 / 120 - 1)
    assert bt.max_drawdown([100, 101, 102]) == 0.0


def test_buy_and_hold_benchmark():
    s = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2024-01-01", periods=4))
    st = bt.buy_and_hold(s, 1000.0, "2024-01-01", "2024-12-31")
    assert st["end_capital"] == pytest.approx(1200.0) and st["max_drawdown"] == pytest.approx(99 / 110 - 1, abs=1e-4)


def test_average_stock_benchmark_uses_yesterdays_eligibility():
    prices, days = make_prices({"A": [100, 110, 121, 121], "B": [100, 100, 100, 50]})
    prices["adj_close"] = prices["close"]
    elig = pd.DataFrame([{"date": d, "symbol": s, "eligible": not (s == "B" and i >= 2)}
                         for i, d in enumerate(days) for s in ("A", "B")])     # B stops being eligible on day 2
    st = bt.average_stock(prices, elig, 1000.0, str(days[0].date()), str(days[-1].date()))
    # day 1: both (A +10%, B 0%) -> +5%; day 2: both were eligible on day 1 -> (+10%, 0%) -> +5%; day 3: only A -> 0%
    assert st["end_capital"] == pytest.approx(1000 * 1.05 * 1.05 * 1.0)


def test_stats_report_how_concentrated_the_profit_is():
    prices, days = make_prices({"A": [100, 100, 150, 150, 150, 150]})
    res = bt.run(const_scores(days, {"A": 1.0}), prices, bt.Settings(capital=1000, top_k=1, **FREE))
    st = bt.stats(res)
    assert st["best_stock"] == {"symbol": "A", "profit": 500.0} and st["profit_without_10_best_trades"] == 0.0
