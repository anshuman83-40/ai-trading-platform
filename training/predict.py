"""
TradePulse ML — daily predictions from the SAVED model (no training).

  1. Downloads recent daily prices for the companies the model was trained on.
  2. Builds the features with the same code as training (ml/features.py via build_dataset.build).
  3. Loads models/xgb_v1 and scores every eligible stock for the latest COMPLETED trading day.
  4. Writes models/xgb_v1/predictions.json (what the website shows) and a dated copy in history/.

The website only reads the JSON file. It never downloads training data, builds features or loads the model,
so the small free server stays light.

Refuses to write predictions when the data is not good enough: prices too old, too many stocks without a
fresh price, or too few eligible stocks. An old file then stays in place and the website reports it as stale.

Run from the project folder, after the market has closed (3:30 pm IST):
    python training/predict.py
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import config as cfg                                   # noqa: E402
import build_dataset as bd                             # noqa: E402
import data_collection as dc                           # noqa: E402
from ml import features as ft                          # noqa: E402
from ml.inference import load_model, model_scores      # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
HISTORY_DAYS = 1100          # calendar days downloaded: about 750 trading days, enough for every feature
MAX_DATA_AGE_DAYS = 4        # latest completed trading day must be this recent (covers a weekend + a holiday)
MAX_MISSING_SHARE = 0.05     # stop if more than 5% of the companies have no price for that day
MIN_ELIGIBLE = 100           # stop if fewer stocks than this can be scored
MARKET_CLOSE = (15, 45)      # IST; before this, today's candle is still changing and must not be used


def last_completed_day(calendar, now_ist):
    """The latest trading day whose prices are final. Today's row is dropped while the market is still open."""
    days = pd.DatetimeIndex(sorted(pd.unique(calendar)))
    today = pd.Timestamp(now_ist.date())
    if len(days) and days[-1] == today and (now_ist.hour, now_ist.minute) < MARKET_CLOSE:
        days = days[:-1]
    return days[-1] if len(days) else None


def make_predictions(prices, market, booster, meta, schema, as_of, groups=None, universe=None):
    """Pure function (no internet): price tables -> the predictions document for the day `as_of`.
    Only rows up to `as_of` are used, exactly as on that evening."""
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of]
    market = market[market["date"] <= as_of]
    panel = bd.build(prices, market)
    day = panel[panel["date"] == as_of].copy()
    universe = sorted(universe) if universe is not None else sorted(prices["symbol"].unique())
    fresh = set(day["symbol"])
    day = day[day["eligible"]].copy()
    if len(day):
        day["score"] = model_scores(booster, meta, schema, day)
        day["rule_score"] = ft.rule_score(day)
        day["rule_signal"] = ft.rule_signal(day["rule_score"])
        day = day.sort_values(["score", "turnover_20_log", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        day["rank"] = np.arange(1, len(day) + 1)
        day["group"] = np.ceil(day["score"].rank(pct=True, method="first") * 10).clip(1, 10).astype(int)
    rates = {g["group"]: g for g in (groups or [])}
    stocks = [{"symbol": r.symbol, "rank": int(r.rank), "of": int(len(day)), "group": int(r.group),
               "score": round(float(r.score), 4), "close": round(float(r.close), 2),
               "rule_signal": r.rule_signal, "rule_score": float(r.rule_score),
               "turnover_cr": round(float(10 ** r.turnover_20_log / 1e7), 1),      # Rs crore traded per day, 20-day average
               "historical_winner_rate": rates.get(int(r.group), {}).get("winner_rate")}
              for r in day.itertuples()]
    return {
        "model_version": meta["model_version"], "feature_version": schema["feature_version"],
        "as_of": str(as_of.date()),
        "horizon_trading_days": meta["horizon_trading_days"],
        "meaning": "rank 1 = highest score. The score estimates how likely the stock is to be among the top 30% of "
                   "these stocks over the next 5 trading days. It is a weak ranking from past prices — not advice, "
                   "not a calibrated probability and not a promise.",
        "base_rate": meta["top_fraction"],
        "universe_size": len(universe), "with_fresh_price": len(fresh), "eligible": len(stocks),
        "not_scored": sorted(set(universe) - {s["symbol"] for s in stocks}),
        "stocks": stocks,
    }


def check(pred, as_of, today):
    """Reasons NOT to publish these predictions (empty list = fine)."""
    problems = []
    age = (pd.Timestamp(today) - pd.Timestamp(as_of)).days
    if age > MAX_DATA_AGE_DAYS:
        problems.append(f"latest prices are {age} days old ({pd.Timestamp(as_of).date()})")
    missing = 1 - pred["with_fresh_price"] / max(1, pred["universe_size"])
    if missing > MAX_MISSING_SHARE:
        problems.append(f"{missing * 100:.0f}% of the companies have no price for {pred['as_of']}")
    if pred["eligible"] < MIN_ELIGIBLE:
        problems.append(f"only {pred['eligible']} stocks could be scored (need {MIN_ELIGIBLE})")
    scores = [s["score"] for s in pred["stocks"]]
    if scores and not all(0 <= x <= 1 for x in scores):
        problems.append("a score is outside 0..1")
    return problems


def already_published(pred, path):
    """A ranking that has been published is never changed or replaced by an older day. Late re-runs (GitHub can
    start the evening job hours late) saw slightly revised prices and quietly changed earlier rankings, and once
    published 9 Oct after 12 Oct. Returns a reason to stop, or None."""
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return None
    if old.get("as_of", "") > pred["as_of"]:
        return f"a newer ranking ({old['as_of']}) is already published; not going back to {pred['as_of']}"
    if old.get("as_of") == pred["as_of"]:
        return f"the ranking for {pred['as_of']} is already published and is kept unchanged"
    return None


def worse_than_published(pred, path):
    """A ranking already published for the same day must not be replaced by one made from fewer stocks."""
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return []
    if old.get("as_of") == pred["as_of"] and old.get("eligible", 0) > pred["eligible"]:
        return [f"the ranking already published for {pred['as_of']} has {old['eligible']} stocks, this one only {pred['eligible']}"]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write today's predictions from the saved model.")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "xgb_v1"))
    ap.add_argument("--allow-stale", action="store_true", help="write the file even if a check fails (for testing only)")
    args = ap.parse_args(argv)

    booster, meta, schema = load_model(args.model)
    if schema["features"] != ft.FEATURES:
        sys.exit("The feature code has changed since this model was trained — retrain before predicting.")
    with open(os.path.join(args.model, "universe.json"), encoding="utf-8") as f:
        universe = json.load(f)["symbols"]
    groups = []
    ev_path = os.path.join(args.model, "evaluation.json")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            groups = json.load(f).get("rank_groups_test", [])

    now = datetime.now(IST)
    start = str((now - timedelta(days=HISTORY_DAYS)).date())
    end = str(now.date())
    print(f"Downloading {len(universe)} companies, {start} to {end}…")
    market_t = dc.download_complete([(n, t) for n, t in dc.MARKET_SERIES.items()], start, end)
    if len(market_t.get("NIFTY50", [])) < 300:
        sys.exit("Could not download NIFTY 50 — no predictions written.")
    tables = dc.download_complete([(s, dc.yahoo_ticker(s)) for s in universe], start, end)
    prices = pd.concat([t for t in tables.values() if len(t)], ignore_index=True)
    market = pd.concat(market_t.values(), ignore_index=True)

    as_of = last_completed_day(market.loc[market["symbol"] == "NIFTY50", "date"], now)
    pred = make_predictions(prices, market, booster, meta, schema, as_of, groups, universe)
    pred["generated"] = now.strftime("%Y-%m-%d %H:%M IST")
    pred["data_source"] = "Yahoo Finance daily prices (free, not exchange-licensed)"
    done = already_published(pred, os.path.join(args.model, "predictions.json"))
    if done and not args.allow_stale:
        print("Nothing to do: " + done)
        return None
    problems = check(pred, as_of, now.date())
    problems += worse_than_published(pred, os.path.join(args.model, "predictions.json"))
    if problems and not args.allow_stale:
        sys.exit("No predictions written: " + "; ".join(problems))
    pred["warnings"] = problems

    hist = os.path.join(args.model, "history")
    os.makedirs(hist, exist_ok=True)
    day_file = os.path.join(hist, pred["as_of"] + ".json")
    if os.path.exists(day_file) and not args.allow_stale:
        sys.exit(f"{day_file} already exists — a published day is never rewritten.")
    for path in (os.path.join(args.model, "predictions.json"), day_file):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pred, f, indent=1)
    print(f"Predictions for {pred['as_of']}: {pred['eligible']} stocks scored, "
          f"{len(pred['not_scored'])} not scored ({', '.join(pred['not_scored'][:8])}{'…' if len(pred['not_scored']) > 8 else ''})")
    print("Top 10: " + ", ".join(f"{s['symbol']} {s['score']:.3f}" for s in pred["stocks"][:10]))
    print(f"Saved {os.path.join(args.model, 'predictions.json')}")
    return pred


if __name__ == "__main__":
    main()
