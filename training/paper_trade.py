"""
TradePulse ML — step 7: paper trading. Virtual money, real prices, on days that had not happened when the
model was built. No real order is ever placed; nothing here can reach a broker.

How it stays honest
  - The only scores used are the ones PUBLISHED each evening in models/xgb_v1/history/<date>.json
    (the git history shows when each file was made). Nothing is re-scored afterwards.
  - Every evening the whole paper portfolio is re-calculated from the first day with the same engine as the
    backtest (training/backtest.py). There is no hidden running state that could drift or be edited;
    a test checks that adding a new day never changes earlier days.
  - A decision made after the close of day D is traded at the OPEN of day D+1, with slippage and all
    Groww charges and taxes (training/costs.py).

The players (rules fixed on 12 Oct 2026, before any paper day existed)
  ai_weekly    the backtested strategy: best 10 by model score, changed every 5 trading days, Rs 1,00,000
  ai_monthly   the same, but changed every 20 trading days — far fewer trades and charges
  ai_small     ai_weekly with Rs 25,000 — shows what small capital does
  rule         best 10 by the website's rule score, every 5 trading days
  random       10 random eligible stocks, every 5 trading days (what luck alone gives, with the same charges)
Risk rules for every player: equal position size (1/10 of the portfolio), whole shares, no borrowing, at most
3 stocks per industry, 10% stop-loss per position, sell everything and pause 20 days after a 20% fall from
the high, sell everything and pause 5 days after losing 6% in one day.

Run from the project folder, after predict.py:
    python training/paper_trade.py

Output: models/xgb_v1/paper/paper.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import backtest as bt                                  # noqa: E402
import config as cfg                                   # noqa: E402
import data_collection as dc                           # noqa: E402
import predict as pr                                   # noqa: E402

COMMON = dict(top_k=10, slippage=0.0005, stop_loss=0.10, max_drawdown_halt=0.20, halt_days=20,
              max_per_industry=3, max_daily_loss=0.06, daily_loss_pause=5)
PLAYERS = [
    {"id": "ai_weekly", "name": "AI model, weekly", "score": "score", "capital": 100_000.0, "every": 5,
     "about": "Best 10 stocks by AI score, changed every 5 trading days. This is the strategy that was backtested."},
    {"id": "ai_monthly", "name": "AI model, monthly", "score": "score", "capital": 100_000.0, "every": 20,
     "about": "Best 10 stocks by AI score, changed every 20 trading days: far fewer trades and charges."},
    {"id": "ai_small", "name": "AI model, weekly, small capital", "score": "score", "capital": 25_000.0, "every": 5,
     "about": "The weekly AI strategy with Rs 25,000, to show what fixed charges do to small amounts."},
    {"id": "rule", "name": "Website rule signal", "score": "rule_score", "capital": 100_000.0, "every": 5,
     "about": "Best 10 stocks by the website's RSI / MACD / EMA rule score, changed every 5 trading days."},
    {"id": "random", "name": "Random stocks", "score": "random", "capital": 100_000.0, "every": 5,
     "about": "10 random eligible stocks, changed every 5 trading days: what luck gives with the same charges."},
]
RULES = [
    "Virtual money only. No real order is placed and nothing here can reach a broker.",
    "Scores are the ones published each evening; a decision after the close of day D is traded at the open of day D+1.",
    "Buys cost 0.05% more than the open and sells get 0.05% less; all Groww delivery charges and taxes are subtracted.",
    "10 stocks, equal size, whole shares, no borrowing, at most 3 per industry.",
    "Stop-loss: a position closing 10% below its buy price is sold at the next open.",
    "After a 20% fall from the portfolio's high: sell everything, pause 20 trading days.",
    "After losing 6% in one day: sell everything, pause 5 trading days.",
    "If a day's AI ranking is missing or a stock has no price, nothing is guessed: that order is simply not made.",
]


def random_score(day, symbol):
    """A fixed 'random' number for a stock on a day (the same every time it is re-calculated)."""
    h = hashlib.sha256(f"tradepulse-paper|{day}|{symbol}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def load_history(folder):
    """Published prediction files -> one table [date, symbol, score, rule_score, tie, random]."""
    rows, files = [], sorted(glob.glob(os.path.join(folder, "*.json")))
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            day = doc["as_of"]
            for s in doc["stocks"]:
                rows.append({"date": pd.Timestamp(day), "symbol": s["symbol"], "score": float(s["score"]),
                             "rule_score": float(s.get("rule_score", 0.0)), "tie": float(s.get("turnover_cr", 0.0)),
                             "random": random_score(day, s["symbol"])})
        except (OSError, ValueError, KeyError, TypeError):
            continue                                            # an unreadable file counts as a missing day
    cols = ["date", "symbol", "score", "rule_score", "tie", "random"]
    return pd.DataFrame(rows, columns=cols)


def drawdown_pct(values, start):
    return round(100 * bt.max_drawdown(np.r_[start, np.asarray(values, dtype="float64")]), 2)


def run_player(p, hist, prices, industry, start, end):
    sc = hist[["date", "symbol", "tie"]].assign(score=hist[p["score"]])
    s = bt.Settings(capital=p["capital"], rebalance_every=p["every"], **COMMON)
    res = bt.run(sc, prices, s, industry=industry, breaks=cfg.PRICE_BREAKS, start=start, end=end, liquidate_at_end=False)
    eq = res["equity"]["equity"]
    done = [t for t in res["trades"] if t["reason"] != "corporate_action"]
    wins = sum(1 for t in done if t["pnl"] > 0)
    return {
        "id": p["id"], "name": p["name"], "about": p["about"], "changes_every_days": p["every"],
        "start_capital": p["capital"], "value": round(float(eq.iloc[-1]), 2),
        "return_pct": round(100 * (float(eq.iloc[-1]) / p["capital"] - 1), 2),
        "cash": round(float(res["cash"]), 2), "invested_pct": round(100 * (1 - res["cash"] / float(eq.iloc[-1])), 1),
        "max_drawdown_pct": drawdown_pct(eq.to_numpy(), p["capital"]),
        "closed_trades": len(done), "winning_trades": wins,
        "realised_profit": round(sum(t["pnl"] for t in done), 2),
        "costs_paid": round(res["costs"], 2), "orders_not_filled": int(res["skipped_buys"]),
        "open_positions": res["open_positions"],
        "orders_for_next_open": {"buy": res["pending"]["buy"], "sell": res["pending"]["sell"]},
        "risk_events": res["events"], "paused": res["paused_until"],
        "trades": res["trades"],
        "equity": [[str(d.date()), round(float(v), 2)] for d, v in eq.items()],
    }


def benchmarks(prices, market, start, end, capital=100_000.0):
    out = []
    n = market[(market["symbol"] == "NIFTY50") & (market["date"] >= start) & (market["date"] <= end)].sort_values("date")
    if len(n):
        eq = capital * n["close"] / n["close"].iloc[0]
        out.append({"id": "nifty", "name": "NIFTY 50 (no trading, no costs)", "return_pct": round(float(100 * (eq.iloc[-1] / capital - 1)), 2),
                    "max_drawdown_pct": drawdown_pct(eq.to_numpy(), capital),
                    "equity": [[str(d.date()), round(float(v), 2)] for d, v in zip(n["date"], eq)]})
    p = prices.sort_values(["symbol", "date"]).copy()
    p["r"] = p.groupby("symbol")["adj_close"].pct_change()
    p = p[(p["date"] > start) & (p["date"] <= end) & (p["r"].abs() < 0.4)]
    days = pd.DatetimeIndex(sorted(prices.loc[(prices["date"] >= start) & (prices["date"] <= end), "date"].unique()))
    r = p.groupby("date")["r"].mean().reindex(days).fillna(0.0)
    eq = capital * (1 + r).cumprod()
    if len(eq):
        out.append({"id": "average", "name": "Average of all covered stocks (no trading, no costs)",
                    "return_pct": round(float(100 * (eq.iloc[-1] / capital - 1)), 2),
                    "max_drawdown_pct": drawdown_pct(eq.to_numpy(), capital),
                    "equity": [[str(d.date()), round(float(v), 2)] for d, v in eq.items()]})
    return out


def monitor(hist, prices, calendar, horizon=cfg.HORIZON, top_fraction=cfg.TOP_FRACTION):
    """Is the model still doing what the tests said? For every published day whose 5-day result is now known:
    winner rate of its top 10% picks, rank correlation, and the picks' return versus all scored stocks."""
    cal = pd.DatetimeIndex(sorted(pd.unique(calendar)))
    pos = {d: i for i, d in enumerate(cal)}
    px = prices.copy()
    px["adj_open"] = px["open"] * px["adj_close"] / px["close"]
    opens = px.pivot(index="date", columns="symbol", values="adj_open")
    days = []
    for d, g in hist.groupby("date"):
        i = pos.get(d)
        if i is None or i + 1 + horizon >= len(cal):
            continue                                            # the 5 days are not over yet
        e, x = cal[i + 1], cal[i + 1 + horizon]
        if e not in opens.index or x not in opens.index:
            continue
        fwd = (opens.loc[x] / opens.loc[e] - 1).reindex(g["symbol"]).to_numpy()
        f = g.assign(fwd=fwd).dropna(subset=["fwd"])
        if len(f) < 30:
            continue
        f = f.assign(label=(f["fwd"].rank(pct=True, method="first") > 1 - top_fraction).astype(float))
        top = f.nlargest(max(1, int(round(0.10 * len(f)))), "score")
        r = f[["score", "fwd"]].rank()
        days.append({"date": str(d.date()), "stocks": int(len(f)),
                     "winner_rate_top10pct": round(float(top["label"].mean()), 4),
                     "excess_5d": round(float(top["fwd"].mean() - f["fwd"].mean()), 5),
                     "rank_ic": round(float(np.corrcoef(r["score"], r["fwd"])[0, 1]), 4)})
    avg = None
    if days:
        avg = {k: round(float(np.mean([x[k] for x in days])), 4) for k in ("winner_rate_top10pct", "excess_5d", "rank_ic")}
    return {"days_judged": len(days), "average": avg, "days": days[-60:]}


def build(hist, prices, market, industry, as_of, expected=None, generated=None):
    """Pure function (no internet): published scores + prices -> the whole paper-trading document."""
    as_of = pd.Timestamp(as_of)
    start = hist["date"].min()
    prices = prices[prices["date"] <= as_of]
    market = market[market["date"] <= as_of]
    hist = hist[hist["date"] <= as_of]
    cal = pd.DatetimeIndex(sorted(market.loc[market["symbol"] == "NIFTY50", "date"].unique()))
    market_days = cal[(cal >= start) & (cal <= as_of)]
    have = set(hist["date"].unique())
    missing = [str(d.date()) for d in market_days if d not in have]
    players = [run_player(p, hist, prices, industry, start, as_of) for p in PLAYERS]
    mon = monitor(hist, prices, cal)
    mon["expected_from_tests"] = expected
    mon["note"] = ("Each published day is judged once its 5 trading days are over. A few days prove nothing: "
                   "weeks of results are needed before the averages mean anything.")
    return {
        "started": str(start.date()), "as_of": str(as_of.date()), "generated": generated,
        "market_days": int(len(market_days)), "rules": RULES,
        "players": players, "benchmarks": benchmarks(prices, market, start, as_of),
        "monitoring": mon,
        "health": {"prediction_days": int(len(have)), "missing_prediction_days": missing,
                   "stocks_with_prices": int(prices.loc[prices["date"] == as_of, "symbol"].nunique()),
                   "note": "On a day without a published ranking no new orders are made; open positions and stop-losses still apply."},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Update the paper-trading results.")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "xgb_v1"))
    args = ap.parse_args(argv)
    hist = load_history(os.path.join(args.model, "history"))
    if hist.empty:
        sys.exit("No published predictions yet — run training/predict.py first.")
    with open(os.path.join(args.model, "universe.json"), encoding="utf-8") as f:
        uni = json.load(f)
    expected = None
    ev_path = os.path.join(args.model, "evaluation.json")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            m = (((json.load(f).get("periods") or {}).get("test") or {}).get("rankings") or {}).get("model") or {}
        expected = {"winner_rate_top10pct": m.get("hit_rate_top10pct"), "excess_5d": m.get("excess_5d"), "rank_ic": m.get("rank_ic")}

    now = datetime.now(pr.IST)
    start = str((hist["date"].min() - timedelta(days=20)).date())      # a little earlier, for the first day's returns
    end = str(now.date())
    print(f"Downloading prices {start} to {end}…")
    market_t = dc.download([(n, t) for n, t in dc.MARKET_SERIES.items()], start, end)
    if not len(market_t.get("NIFTY50", [])):
        sys.exit("Could not download NIFTY 50 — paper trading not updated.")
    tables = dc.download_complete([(s, dc.yahoo_ticker(s)) for s in uni["symbols"]], start, end)
    prices = pd.concat([t for t in tables.values() if len(t)], ignore_index=True)
    market = pd.concat(market_t.values(), ignore_index=True)
    as_of = pr.last_completed_day(market.loc[market["symbol"] == "NIFTY50", "date"], now)
    got = prices.loc[prices["date"] == as_of, "symbol"].nunique()
    if got < 0.95 * len(uni["symbols"]):
        sys.exit(f"Only {got} of {len(uni['symbols'])} stocks have a price for {as_of.date()} — paper trading not updated.")

    doc = build(hist, prices, market, uni.get("industry", {}), as_of, expected, now.strftime("%Y-%m-%d %H:%M IST"))
    out = os.path.join(args.model, "paper")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "paper.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"Paper trading from {doc['started']} to {doc['as_of']} ({doc['market_days']} market days)")
    for p in doc["players"]:
        print(f"  {p['name']:<34} Rs {p['value']:>11,.2f}  {p['return_pct']:>+6.2f}%  open {len(p['open_positions']):>2}  "
              f"closed trades {p['closed_trades']:>3}  next open: buy {len(p['orders_for_next_open']['buy'])}, sell {len(p['orders_for_next_open']['sell'])}")
    for b in doc["benchmarks"]:
        print(f"  {b['name']:<54} {b['return_pct']:>+6.2f}%")
    print(f"  model check: {doc['monitoring']['days_judged']} day(s) judged; missing ranking days: {doc['health']['missing_prediction_days'] or 'none'}")
    print(f"Saved {os.path.join(out, 'paper.json')}")
    return doc


if __name__ == "__main__":
    main()
