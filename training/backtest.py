"""
TradePulse ML — step 5: backtest. Follow the model's picks with pretend money and subtract every real cost.

The strategy that is tested (decided before looking at any result):
  - Every 5 trading days, after the close, rank the eligible stocks by score and choose the best 10
    (at most 3 from one industry).
  - Next morning at the OPEN: sell holdings that are no longer chosen, buy the new ones.
    Each new position gets 1/10 of the portfolio value. Whole shares only. No borrowing.
  - Stop-loss: a position that closes 10% or more below its buy price is sold at the next open.
  - Maximum-loss rule: if the portfolio falls 20% from its highest value, everything is sold at the next
    open and no new positions are opened for 20 trading days.
  - Prices: buys pay 0.05% more than the open and sells get 0.05% less (slippage).
  - Costs: Groww's delivery charges and all taxes (training/costs.py).
  - Dividends are received as cash. At the end everything is sold, with costs.

What daily data cannot model (so reality would be somewhat worse): the exact price of a market order at the
open, orders that are too large for the stock, upper/lower circuit days when a stock cannot be traded, and
stop-losses hit inside the day. Survivorship bias also remains (see training/README.md).

Run from the project folder, after train_model.py:
    python training/backtest.py

Output: models/xgb_v1/backtest.json
"""
import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import config as cfg                                   # noqa: E402
import costs as cs                                     # noqa: E402

RISK_FREE = 0.065        # yearly return with no risk (about a 1-year FD / T-bill in India), for the Sharpe ratio


@dataclass
class Settings:
    capital: float = 100_000.0
    top_k: int = 10                    # number of stocks held
    rebalance_every: int = cfg.HORIZON
    slippage: float = 0.0005           # 0.05% worse than the open on every buy and sell
    stop_loss: float = 0.10            # None = no stop-loss
    max_drawdown_halt: float = 0.20    # None = no maximum-loss rule
    halt_days: int = 20
    max_daily_loss: float = None       # e.g. 0.06: if the portfolio loses 6% in ONE day, sell everything next open
    daily_loss_pause: int = 5          # … and open nothing new for this many trading days
    max_per_industry: int = 3          # None = no industry limit
    plan: dict = field(default_factory=lambda: cs.GROWW_DELIVERY)


def _pivot(prices, col, days):
    return prices.pivot(index="date", columns="symbol", values=col).reindex(days)


def run(scores, prices, s=None, industry=None, breaks=None, start=None, end=None, liquidate_at_end=True):
    """Simulate the strategy. scores: table [date, symbol, score] with ELIGIBLE rows only (optional column
    'tie' breaks equal scores, higher first). prices: the raw daily price table. Returns a dict with the daily
    portfolio value, every completed trade and the settings.
    liquidate_at_end=False is for paper trading: positions stay open on the last day, are valued at the close,
    and the orders decided on the last day are returned as 'pending' (they would be placed the next morning)."""
    s = s or Settings()
    industry = industry or {}
    days = pd.DatetimeIndex(sorted(prices["date"].unique()))
    days = days[(days >= pd.Timestamp(start or days[0])) & (days <= pd.Timestamp(end or days[-1]))]
    if len(days) < (2 if liquidate_at_end else 1):
        raise ValueError("not enough trading days")
    px = prices[prices["date"].isin(days)]
    O, C, V, D = (_pivot(px, c, days) for c in ("open", "close", "volume", "dividends"))
    Cff = C.ffill()                                         # last known close, for valuing a stock on a day it did not trade
    break_set = {(sym, pd.Timestamp(d)) for sym, ds in (breaks or {}).items() for d in ds}

    sc = scores[scores["date"].isin(days)].copy()
    if "tie" not in sc:
        sc["tie"] = 0.0
    sc = sc.sort_values(["date", "score", "tie", "symbol"], ascending=[True, False, False, True])
    ranked = {d: g["symbol"].tolist() for d, g in sc.groupby("date")}

    cash, pos = float(s.capital), {}
    pending_sell, pending_buy, target_size = {}, [], 0.0
    equity_curve, trades, skipped = [], [], 0
    peak, halted_until, cost_total, bought_value = float(s.capital), -1, 0.0, 0.0
    prev_equity, events = float(s.capital), []

    def price_ok(sym, d):
        o = O.at[d, sym] if sym in O.columns else np.nan
        v = V.at[d, sym] if sym in V.columns else np.nan
        return o == o and o > 0 and v == v and v > 0

    def close_position(sym, d, price, reason, with_costs=True):
        nonlocal cash, cost_total
        p = pos.pop(sym)
        value = p["shares"] * price
        cost = cs.trade_cost("sell", value, s.plan) if with_costs else 0.0
        cash += value - cost
        cost_total += cost
        pnl = value - cost - p["paid"] + p["dividends"]
        trades.append({"symbol": sym, "buy_date": str(p["date"].date()), "sell_date": str(d.date()),
                       "shares": p["shares"], "buy_price": round(p["price"], 4), "sell_price": round(price, 4),
                       "days_held": int(days.get_loc(d) - days.get_loc(p["date"])),
                       "pnl": round(pnl, 2), "return_pct": round(100 * pnl / p["paid"], 3), "reason": reason})

    for i, d in enumerate(days):
        # 0) unadjusted corporate action on a held stock: the price change is not a real loss, so the position
        #    is taken out at yesterday's close with no gain, no loss and no cost (see config.PRICE_BREAKS)
        for sym in [x for x in pos if (x, d) in break_set]:
            close_position(sym, d, float(Cff.at[days[i - 1], sym]), "corporate_action", with_costs=False)
            pending_sell.pop(sym, None)

        # 1) sells decided earlier — at today's open (stay pending if the stock does not trade today)
        for sym in sorted(pending_sell):
            if sym not in pos:
                pending_sell.pop(sym)
            elif price_ok(sym, d):
                close_position(sym, d, float(O.at[d, sym]) * (1 - s.slippage), pending_sell.pop(sym))

        # 2) buys decided yesterday — at today's open, whole shares, never more cash than we have
        for sym in pending_buy:
            if len(pos) >= s.top_k or sym in pos:
                continue
            if not price_ok(sym, d):
                skipped += 1
                continue
            price = float(O.at[d, sym]) * (1 + s.slippage)
            shares = int(min(target_size, cash) // price)
            while shares > 0 and shares * price + cs.trade_cost("buy", shares * price, s.plan) > min(target_size, cash):
                shares -= 1
            if shares <= 0:
                skipped += 1                                    # one share costs more than the position size
                continue
            value = shares * price
            cost = cs.trade_cost("buy", value, s.plan)
            cash -= value + cost
            cost_total += cost
            bought_value += value
            pos[sym] = {"shares": shares, "price": price, "date": d, "paid": value + cost, "dividends": 0.0}
        pending_buy = []

        # 3) dividends on shares held today
        for sym, p in pos.items():
            div = D.at[d, sym]
            if div == div and div > 0:
                cash += p["shares"] * div
                p["dividends"] += p["shares"] * div

        # 4) value of everything at today's close
        holdings = sum(p["shares"] * float(Cff.at[d, sym]) for sym, p in pos.items())
        equity = cash + holdings
        equity_curve.append((d, equity, holdings))

        # 5) stop-loss, checked on the close, sold tomorrow
        if s.stop_loss is not None:
            for sym, p in pos.items():
                if float(Cff.at[d, sym]) <= p["price"] * (1 - s.stop_loss) and sym not in pending_sell:
                    pending_sell[sym] = "stop_loss"

        # 6) maximum-loss rule
        if i == halted_until:
            peak = equity                                       # start counting again after the pause
        peak = max(peak, equity)
        if s.max_drawdown_halt is not None and i > halted_until and equity <= peak * (1 - s.max_drawdown_halt):
            for sym in pos:
                pending_sell[sym] = "max_loss_rule"
            halted_until = i + s.halt_days
            events.append({"date": str(d.date()), "event": "max_loss_rule", "pause_days": s.halt_days})
        # 6b) one-day loss rule
        if s.max_daily_loss is not None and i > halted_until and pos and equity <= prev_equity * (1 - s.max_daily_loss):
            for sym in pos:
                pending_sell[sym] = "daily_loss_rule"
            halted_until = i + s.daily_loss_pause
            events.append({"date": str(d.date()), "event": "daily_loss_rule", "pause_days": s.daily_loss_pause})
        prev_equity = equity

        # 7) rebalance day: choose tomorrow's portfolio from TODAY's scores
        if i % s.rebalance_every == 0 and i > halted_until and (i < len(days) - 1 or not liquidate_at_end):
            chosen, per_ind = [], {}
            for sym in ranked.get(d, []):
                ind = industry.get(sym)
                if s.max_per_industry is not None and ind and per_ind.get(ind, 0) >= s.max_per_industry:
                    continue
                chosen.append(sym)
                per_ind[ind] = per_ind.get(ind, 0) + 1
                if len(chosen) == s.top_k:
                    break
            if chosen:
                for sym in pos:
                    if sym not in chosen and sym not in pending_sell:
                        pending_sell[sym] = "rebalance"
                pending_buy = [x for x in chosen if x not in pos or x in pending_sell]
                target_size = equity / s.top_k

    last = days[-1]
    eq = pd.DataFrame(equity_curve, columns=["date", "equity", "holdings"]).set_index("date")
    open_positions = []
    if liquidate_at_end:
        # the end: sell everything at the last close, with costs, so the final number is real money
        for sym in list(pos):
            close_position(sym, last, float(Cff.at[last, sym]) * (1 - s.slippage), "end_of_test")
        eq.iloc[-1, eq.columns.get_loc("equity")] = cash
        eq.iloc[-1, eq.columns.get_loc("holdings")] = 0.0
    else:
        for sym, p in sorted(pos.items()):
            px = float(Cff.at[last, sym])
            open_positions.append({"symbol": sym, "shares": p["shares"], "buy_date": str(p["date"].date()),
                                   "buy_price": round(p["price"], 2), "last_price": round(px, 2),
                                   "value": round(p["shares"] * px, 2),
                                   "pnl": round(p["shares"] * px - p["paid"] + p["dividends"], 2),
                                   "return_pct": round(100 * (p["shares"] * px - p["paid"] + p["dividends"]) / p["paid"], 2)})
    return {"equity": eq, "trades": trades, "costs": cost_total, "bought_value": bought_value,
            "skipped_buys": skipped, "settings": s, "cash": cash, "open_positions": open_positions,
            "pending": {"sell": dict(pending_sell), "buy": list(pending_buy)}, "events": events,
            "paused_until": str(days[halted_until].date()) if 0 <= halted_until < len(days) and halted_until >= len(days) - 1 else
                            ("after %d more trading days" % (halted_until - len(days) + 1) if halted_until >= len(days) else None)}


# ───────────────────────── statistics ─────────────────────────
def max_drawdown(equity):
    """Biggest fall from an earlier high, as a negative fraction (-0.25 = fell 25%)."""
    e = np.asarray(equity, dtype="float64")
    return float((e / np.maximum.accumulate(e) - 1).min())


def stats(result, risk_free=RISK_FREE):
    eq = result["equity"]["equity"]
    capital = result["settings"].capital
    years = len(eq) / 252
    r = eq.pct_change().dropna()
    r = pd.concat([pd.Series([eq.iloc[0] / capital - 1]), r])
    total = float(eq.iloc[-1] / capital - 1)
    vol = float(r.std(ddof=1) * math.sqrt(252)) if len(r) > 2 else float("nan")
    sharpe = float((r.mean() - risk_free / 252) / r.std(ddof=1) * math.sqrt(252)) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan")
    t = pd.DataFrame([x for x in result["trades"] if x["reason"] != "corporate_action"])
    wins = t[t["pnl"] > 0]["pnl"].sum() if len(t) else 0.0
    losses = -t[t["pnl"] < 0]["pnl"].sum() if len(t) else 0.0
    exposure = float((result["equity"]["holdings"] / result["equity"]["equity"]).mean())
    by_year = {str(y): round(float(g.iloc[-1] / (eq[eq.index < g.index[0]].iloc[-1] if (eq.index < g.index[0]).any() else capital) - 1), 4)
               for y, g in eq.groupby(eq.index.year)}
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "years": round(years, 2),
        "start_capital": round(capital, 2), "end_capital": round(float(eq.iloc[-1]), 2),
        "profit": round(float(eq.iloc[-1] - capital), 2),
        "total_return": round(total, 4),
        "yearly_return": round(float((1 + total) ** (1 / years) - 1), 4) if total > -1 else -1.0,
        "yearly_volatility": round(vol, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown(np.r_[capital, eq.to_numpy()]), 4),
        "trades": int(len(t)),
        "win_rate": round(float((t["pnl"] > 0).mean()), 4) if len(t) else None,
        "profit_factor": round(float(wins / losses), 2) if losses > 0 else None,
        "avg_days_held": round(float(t["days_held"].mean()), 1) if len(t) else None,
        "avg_trade_return_pct": round(float(t["return_pct"].mean()), 3) if len(t) else None,
        "stop_loss_exits": int((t["reason"] == "stop_loss").sum()) if len(t) else 0,
        "max_loss_rule_exits": int((t["reason"] == "max_loss_rule").sum()) if len(t) else 0,
        "costs_paid": round(result["costs"], 2),
        "costs_pct_of_start_capital": round(100 * result["costs"] / capital, 2),
        "costs_pct_per_year": round(100 * result["costs"] / capital / years, 2),
        "turnover_per_year": round(result["bought_value"] / float(eq.mean()) / years, 1),
        "avg_invested": round(exposure, 3),
        "skipped_buys": int(result["skipped_buys"]),
        "stocks_traded": int(t["symbol"].nunique()) if len(t) else 0,
        "profit_without_10_best_trades": round(float(t["pnl"].sum() - t["pnl"].nlargest(10).sum()), 2) if len(t) else 0.0,
        "best_stock": (lambda g: {"symbol": str(g.idxmax()), "profit": round(float(g.max()), 2)})(t.groupby("symbol")["pnl"].sum()) if len(t) else None,
        "return_by_year": by_year,
    }


def buy_and_hold(series, capital, start, end):
    """An index bought on the first day and never touched (no costs). Returns stats in the same shape."""
    s = series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))].dropna()
    eq = capital * s / s.iloc[0]
    res = {"equity": pd.DataFrame({"equity": eq, "holdings": eq}), "trades": [], "costs": 0.0,
           "bought_value": capital, "skipped_buys": 0, "settings": Settings(capital=capital)}
    out = stats(res)
    out["trades"], out["turnover_per_year"] = 0, 0.0
    return out


def average_stock(prices, eligible, capital, start, end):
    """Every eligible stock held in equal amounts, rebalanced daily, NO costs. Not a strategy anyone can run —
    it shows how the average stock IN OUR DATA did, which is what survivorship bias inflates."""
    p = prices.sort_values(["symbol", "date"]).copy()
    p["r"] = p.groupby("symbol")["adj_close"].pct_change()
    e = eligible.sort_values(["symbol", "date"]).copy()
    e["was_eligible"] = e.groupby("symbol")["eligible"].shift(1)              # decided the day before: no look-ahead
    m = p.merge(e[["date", "symbol", "was_eligible"]], on=["date", "symbol"])
    m = m[m["was_eligible"].eq(True) & (m["date"] >= pd.Timestamp(start)) &
          (m["date"] <= pd.Timestamp(end)) & (m["r"].abs() < 0.4)]
    eq = capital * (1 + m.groupby("date")["r"].mean()).cumprod()
    res = {"equity": pd.DataFrame({"equity": eq, "holdings": eq}), "trades": [], "costs": 0.0,
           "bought_value": capital, "skipped_buys": 0, "settings": Settings(capital=capital)}
    out = stats(res)
    out["trades"], out["turnover_per_year"] = 0, 0.0
    return out


# ───────────────────────── main ─────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest the saved model with real trading costs.")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "xgb_v1"))
    ap.add_argument("--random-seeds", type=int, default=20)
    args = ap.parse_args(argv)
    from ml import features as ft
    import evaluate_model as ev
    import train_model as tm

    data = pd.read_csv(os.path.join(args.data, "dataset_v1.csv.gz"), parse_dates=["date"])
    prices = pd.read_csv(os.path.join(args.data, "prices_daily.csv.gz"), parse_dates=["date"])
    market = pd.read_csv(os.path.join(args.data, "market_daily.csv.gz"), parse_dates=["date"])
    uni = pd.read_csv(os.path.join(args.data, "universe_nifty200.csv"))
    industry = dict(zip(uni["symbol"], uni["industry"]))
    booster, meta, schema = ev.load_model(args.model)

    elig = data[data["eligible"]].copy()
    elig["tie"] = elig["turnover_20_log"]                       # equal scores: the more heavily traded stock first
    elig["model"] = ev.model_scores(booster, meta, schema, elig)
    elig["volatility"], elig["momentum"], elig["rule"] = elig["atr_14"], elig["ret_126"], ft.rule_score(elig)

    test_start = str((pd.Timestamp(cfg.VALID_END) + pd.Timedelta(days=1)).date())
    end = str(prices["date"].max().date())
    val_start = str((pd.Timestamp(cfg.TRAIN_END) + pd.Timedelta(days=1)).date())

    # walk-forward scores: for each test year the model is retrained on the past only
    lab = tm.load_labelled(os.path.join(args.data, "dataset_v1.csv.gz"))
    elig["wf"] = np.nan
    cal = pd.DatetimeIndex(sorted(lab["date"].unique()))
    for y in sorted(elig.loc[elig["date"] >= test_start, "date"].dt.year.unique()):
        tr_end, va_end = pd.Timestamp(f"{y - 2}-12-31"), pd.Timestamp(f"{y - 1}-12-31")
        d = lab[lab["date"] <= va_end].copy()
        d["split"] = np.where(d["date"] <= tr_end, "train", "valid")
        for e in (tr_end, va_end):
            d.loc[d["date"].isin(cal[cal <= e][-cfg.EMBARGO_DAYS:]), "split"] = "gap"
        m, _ = tm.train(d)
        rows = elig["date"].dt.year == y
        elig.loc[rows, "wf"] = m.predict_proba(elig.loc[rows, ft.FEATURES])[:, 1]

    def bt(col, start=test_start, min_turnover=None, **kw):
        sc = elig[["date", "symbol", "tie"]].assign(score=elig[col]).dropna(subset=["score"])
        if min_turnover:                                    # only stocks that were ALREADY heavily traded on that day
            sc = sc[elig.loc[sc.index, "turnover_20_log"] >= math.log10(min_turnover)]
        return stats(run(sc, prices, Settings(**kw), industry=industry, breaks=cfg.PRICE_BREAKS, start=start, end=end))

    out = {"model_version": meta["model_version"], "run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "period": {"start": test_start, "end": end},
           "strategy": "every 5 trading days hold the 10 best-scored eligible NIFTY 200 stocks (max 3 per industry), "
                       "trade at the next open, equal position size, whole shares, 10% stop-loss, "
                       "20% maximum-loss rule with a 20-day pause",
           "assumptions": {"slippage_each_side": Settings().slippage, "cost_plan": cs.GROWW_DELIVERY,
                           "risk_free_for_sharpe": RISK_FREE,
                           "sharpe": "daily portfolio returns minus risk-free/252, mean/std x sqrt(252)",
                           "round_trip_cost_pct": {str(v): round(cs.round_trip_pct(v), 3) for v in (2500, 10000, 50000)}},
           "not_modelled": ["exact fill price of a market order at the open", "orders too large for the stock",
                            "circuit-limit days when a stock cannot be traded", "stop-losses hit inside the day",
                            "survivorship bias: only today's NIFTY 200 members are in the data",
                            "today's charges are applied to all past years", "income tax on short-term gains (20%)"]}

    print("Running backtests…")
    res = {"AI model (xgb_v1)": bt("model"),
           "AI model, retrained each year": bt("wf"),
           "Most volatile stocks": bt("volatility"),
           "Best 6-month return": bt("momentum"),
           "Website rule score": bt("rule")}
    keys = ["total_return", "yearly_return", "yearly_volatility", "sharpe", "max_drawdown", "win_rate",
            "profit_factor", "costs_pct_per_year", "trades", "end_capital", "profit", "avg_invested"]
    rnd_name = "Random stocks (average of %d runs)" % args.random_seeds

    def random_picks(model_result, **kw):
        runs = []
        for seed in range(args.random_seeds):
            elig["rand"] = np.random.default_rng(seed).random(len(elig))
            runs.append(bt("rand", **kw))
        avg = {k: round(float(np.mean([r[k] for r in runs])), 4) for k in keys}
        avg["yearly_return_range"] = [round(min(r["yearly_return"] for r in runs), 4), round(max(r["yearly_return"] for r in runs), 4)]
        avg["runs_beaten_by_model"] = int(sum(model_result["yearly_return"] > r["yearly_return"] for r in runs))
        avg["runs"] = len(runs)
        return avg

    res[rnd_name] = random_picks(res["AI model (xgb_v1)"])
    res["Average eligible stock (no costs)"] = average_stock(prices, data[["date", "symbol", "eligible"]], Settings().capital, test_start, end)
    nifty = market[market["symbol"] == "NIFTY50"].set_index("date")["close"]
    res["NIFTY 50, buy and hold (no costs)"] = buy_and_hold(nifty, Settings().capital, test_start, end)
    out["results"] = res

    # Survivorship check. Our data has only TODAY's NIFTY 200 members. Some of them were small, thinly traded
    # companies in 2023 that are in the index now BECAUSE their price multiplied. Nobody using the NIFTY 200
    # list in 2023 could have picked them. Requiring heavy trading ON THE DAY removes most of those cases.
    print("Running the large-stocks-only check…")
    big = 5e8                                                   # Rs 50 crore traded per day (20-day average)
    large = {"min_turnover_rupees_per_day": big,
             "why": "only stocks that were already heavily traded on the decision day; reduces (does not remove) survivorship bias",
             "AI model (xgb_v1)": bt("model", min_turnover=big),
             "AI model, retrained each year": bt("wf", min_turnover=big),
             "Most volatile stocks": bt("volatility", min_turnover=big)}
    large[rnd_name] = random_picks(large["AI model (xgb_v1)"], min_turnover=big)
    large["stocks_per_day_median"] = int(elig[(elig["date"] >= test_start) & (elig["turnover_20_log"] >= math.log10(big))].groupby("date").size().median())
    out["large_stocks_only"] = large

    print("Running sensitivity checks…")
    out["sensitivity_model"] = {
        "no stop-loss, no maximum-loss rule": bt("model", stop_loss=None, max_drawdown_halt=None),
        "slippage 0.10% each side": bt("model", slippage=0.001),
        "no trading costs at all (not achievable)": bt("model", plan=cs.NO_COSTS, slippage=0.0),
        "start capital Rs 25,000": bt("model", capital=25_000.0),
        "start capital Rs 5,00,000": bt("model", capital=500_000.0),
        "hold 20 stocks instead of 10": bt("model", top_k=20),
    }
    out["validation_period_model"] = bt("model", start=val_start) | {"note": "2021 onwards; includes the validation years used for early stopping"}

    with open(os.path.join(args.model, "backtest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    def line(name, r):
        pf = "%.2f" % r["profit_factor"] if r.get("profit_factor") else "  - "
        wr = "%.0f%%" % (100 * r["win_rate"]) if r.get("win_rate") else " - "
        print(f"  {name:<42}{r['yearly_return'] * 100:>+8.1f}%{r['max_drawdown'] * 100:>9.1f}%{r['sharpe']:>8.2f}"
              f"{wr:>7}{pf:>7}{r.get('costs_pct_per_year', 0):>8.1f}%{int(r.get('trades', 0)):>7}  Rs {r['end_capital']:>11,.0f}")

    print(f"\nTEST {test_start} to {end}, start capital Rs {Settings().capital:,.0f}, AFTER costs")
    print(f"  {'strategy':<42}{'per year':>9}{'worst fall':>10}{'Sharpe':>8}{'wins':>7}{'PF':>7}{'costs/yr':>9}{'trades':>7}  end value")
    for k, v in res.items():
        line(k, v)
    print(f"  Large stocks only (at least Rs 50 crore traded per day; about {large['stocks_per_day_median']} stocks a day):")
    for k, v in large.items():
        if isinstance(v, dict):
            line("  " + k, v)
    print("  Sensitivity of the AI model:")
    for k, v in out["sensitivity_model"].items():
        line("  " + k, v)
    m = res["AI model (xgb_v1)"]
    print(f"  AI model: profit Rs {m['profit']:,.0f}; without its 10 best trades Rs {m['profit_without_10_best_trades']:,.0f}; "
          f"best stock {m['best_stock']['symbol']} Rs {m['best_stock']['profit']:,.0f}; {m['stocks_traded']} stocks traded")
    r0 = res[rnd_name]
    print(f"  Random picks: from {r0['yearly_return_range'][0] * 100:+.1f}% to {r0['yearly_return_range'][1] * 100:+.1f}% a year; "
          f"the model beat {r0['runs_beaten_by_model']} of {r0['runs']} runs")
    print("  AI model by year: " + "  ".join(f"{y}: {v * 100:+.1f}%" for y, v in m["return_by_year"].items()))
    print("  NIFTY 50 by year: " + "  ".join(f"{y}: {v * 100:+.1f}%" for y, v in res["NIFTY 50, buy and hold (no costs)"]["return_by_year"].items()))
    print(f"  Round-trip cost: Rs 2,500 -> {cs.round_trip_pct(2500):.2f}%   Rs 10,000 -> {cs.round_trip_pct(10000):.2f}%   Rs 50,000 -> {cs.round_trip_pct(50000):.2f}%")
    print(f"\nSaved {os.path.join(args.model, 'backtest.json')}")
    return out


if __name__ == "__main__":
    main()
