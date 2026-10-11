"""
TradePulse ML — step 2: turn the price file into a training table (features + label).

For every company and every day it writes:
  - the features (ml/features.py) — calculated from that day and earlier only;
  - fwd_ret — the return of the trade being judged (buy next open, sell HORIZON trading days later at the open);
  - label   — 1 if fwd_ret is in the top 30% of eligible stocks that day, else 0;
  - eligible — whether the row may be used at all (enough history, enough trading, decided from past data).

Run from the project folder, after data_collection.py:
    python training/build_dataset.py

Output: training/data/dataset_v1.csv.gz and dataset_manifest.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config as cfg                                   # noqa: E402
from ml import features as ft                          # noqa: E402
from data_collection import sha256                     # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
ID_COLS = ["date", "symbol", "series"]
OUT_COLS = ID_COLS + ft.FEATURES + ["eligible", "fwd_ret", "label"]


def add_series(prices, breaks):
    """Cut a company's history into separate series at each unadjusted corporate action (config.PRICE_BREAKS).
    RELIANCE -> series 'RELIANCE'; TMPV -> 'TMPV' before the break and 'TMPV#1' from the break date on."""
    prices = prices.copy()
    prices["series"] = prices["symbol"]
    for sym, dates in breaks.items():
        m = prices["symbol"] == sym
        n = sum((prices.loc[m, "date"] >= pd.Timestamp(d)).astype(int) for d in dates)
        prices.loc[m, "series"] = np.where(n > 0, sym + "#" + n.astype(str), sym)
    return prices


def add_forward_return(panel, calendar, horizon):
    """fwd_ret for a row on day t = adjusted open on market day t+1+horizon / adjusted open on market day t+1 − 1.
    Market days come from the NIFTY 50 calendar, so every stock is judged over the SAME dates.
    If the stock has no price on either date, or the dates fall in another series (price break), fwd_ret is empty.
    This is the ONLY place where future prices are used — it is the answer, never an input."""
    cal = pd.DatetimeIndex(sorted(pd.unique(calendar)))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    panel = panel.copy()
    panel["_i"] = panel["date"].map(pos)
    opens = panel[["series", "_i"]].copy()
    opens["_adj_open"] = panel["open"] * panel["adj_close"] / panel["close"]
    opens = opens.dropna(subset=["_i"])

    def open_at(offset):
        look = opens.rename(columns={"_adj_open": "_o"}).copy()
        look["_i"] = look["_i"] - offset                      # the row that sits `offset` market days earlier
        return panel[["series", "_i"]].merge(look, on=["series", "_i"], how="left")["_o"].to_numpy()

    entry, exit_ = open_at(1), open_at(1 + horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        panel["fwd_ret"] = np.where((entry > 0) & (exit_ > 0), exit_ / entry - 1, np.nan)
    return panel.drop(columns="_i")


def add_eligibility(panel):
    """Rows the model may use — decided only from information available on that day."""
    panel = panel.copy()
    panel["_n"] = panel.groupby("series").cumcount() + 1                        # days of history so far
    panel["eligible"] = ((panel["_n"] >= cfg.MIN_HISTORY) &
                         (panel["turnover_20_log"] >= np.log10(cfg.MIN_TURNOVER)) &
                         (panel["volume"] > 0) &
                         panel[ft.STOCK_FEATURES].notna().all(axis=1))
    return panel.drop(columns="_n")


def add_label(panel, top_fraction, min_stocks):
    """label = 1 if the stock's fwd_ret is in the top `top_fraction` of eligible stocks on that day."""
    panel = panel.copy()
    ok = panel["eligible"] & panel["fwd_ret"].notna()
    rank = panel.loc[ok].groupby("date")["fwd_ret"].rank(pct=True, method="first")
    count = panel.loc[ok].groupby("date")["fwd_ret"].transform("size")
    label = (rank > 1 - top_fraction).astype("float64").where(count >= min_stocks)
    panel["label"] = np.nan
    panel.loc[ok, "label"] = label
    return panel


def build(prices, market):
    prices = add_series(prices, cfg.PRICE_BREAKS)
    panel = ft.build_panel(prices, market, series_col="series")
    panel = add_eligibility(panel)
    calendar = market.loc[market["symbol"] == "NIFTY50", "date"]
    panel = add_forward_return(panel, calendar, cfg.HORIZON)
    panel = add_label(panel, cfg.TOP_FRACTION, cfg.MIN_STOCKS_PER_DAY)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def split_of(dates):
    """train / valid / test / gap for each date, using config's dates and embargo."""
    cal = pd.DatetimeIndex(sorted(pd.unique(dates)))
    out = pd.Series("test", index=cal)
    out[cal <= pd.Timestamp(cfg.VALID_END)] = "valid"
    out[cal <= pd.Timestamp(cfg.TRAIN_END)] = "train"
    for end in (cfg.TRAIN_END, cfg.VALID_END):                  # last days of a period: their labels reach into the next
        before = cal[cal <= pd.Timestamp(end)]
        out[before[-cfg.EMBARGO_DAYS:]] = "gap"
    return pd.Series(dates).map(out).to_numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the training table (features + label).")
    ap.add_argument("--data", default=DATA_DIR)
    args = ap.parse_args(argv)
    p_path, m_path = os.path.join(args.data, "prices_daily.csv.gz"), os.path.join(args.data, "market_daily.csv.gz")
    if not os.path.exists(p_path):
        sys.exit("No price file yet — run:  python training/data_collection.py --end YYYY-MM-DD")
    prices = pd.read_csv(p_path, parse_dates=["date"])
    market = pd.read_csv(m_path, parse_dates=["date"])

    panel = build(prices, market)
    data = panel[OUT_COLS]
    out = os.path.join(args.data, "dataset_v1.csv.gz")
    data.to_csv(out, index=False, date_format="%Y-%m-%d", float_format="%.8g",
                compression={"method": "gzip", "mtime": 0})

    lab = data[data["label"].notna()].copy()
    lab["split"] = split_of(lab["date"])
    per_split = {s: {"rows": int(len(g)), "days": int(g["date"].nunique()),
                     "first": str(g["date"].min().date()), "last": str(g["date"].max().date()),
                     "positive_rate": round(float(g["label"].mean()), 4)}
                 for s, g in lab.groupby("split")}
    manifest = {
        "feature_version": ft.FEATURE_VERSION, "features": ft.FEATURES,
        "target": cfg.TARGET, "horizon_days": cfg.HORIZON, "top_fraction": cfg.TOP_FRACTION,
        "eligibility": {"MIN_HISTORY": cfg.MIN_HISTORY, "MIN_TURNOVER": cfg.MIN_TURNOVER,
                        "MIN_STOCKS_PER_DAY": cfg.MIN_STOCKS_PER_DAY},
        "price_breaks": cfg.PRICE_BREAKS,
        "split": {"TRAIN_END": cfg.TRAIN_END, "VALID_END": cfg.VALID_END, "EMBARGO_DAYS": cfg.EMBARGO_DAYS},
        "rows_total": int(len(data)), "rows_eligible": int(data["eligible"].sum()), "rows_labelled": int(len(lab)),
        "symbols_labelled": int(lab["symbol"].nunique()),
        "stocks_per_day": {"min": int(lab.groupby("date").size().min()), "median": int(lab.groupby("date").size().median()),
                           "max": int(lab.groupby("date").size().max())},
        "per_split": per_split,
        "rows_per_year": {str(y): int(n) for y, n in lab.groupby(lab["date"].dt.year).size().items()},
        "inputs": {"prices_daily.csv.gz": sha256(p_path), "market_daily.csv.gz": sha256(m_path)},
        "output": {os.path.basename(out): sha256(out)},
    }
    with open(os.path.join(args.data, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"Rows: {len(data):,} total | {manifest['rows_eligible']:,} eligible | {len(lab):,} with a label "
          f"({lab['symbol'].nunique()} companies)")
    print(f"Stocks per day: min {manifest['stocks_per_day']['min']}, median {manifest['stocks_per_day']['median']}, "
          f"max {manifest['stocks_per_day']['max']}")
    for s in ("train", "gap", "valid", "test"):
        if s in per_split:
            x = per_split[s]
            print(f"  {s:<5} {x['first']} to {x['last']}  {x['rows']:>8,} rows  {x['days']:>5} days  "
                  f"label=1 on {x['positive_rate'] * 100:.1f}%")
    print(f"Saved {out}")
    return manifest


if __name__ == "__main__":
    main()
