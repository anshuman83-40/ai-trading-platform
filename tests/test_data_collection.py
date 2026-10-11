"""Tests for the data checks in training/data_collection.py — hand-made data, no internet needed.
Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import data_collection as dc


def make(n=600, start="2020-01-01", price=100.0, volume=1000):
    """A clean, boring stock: n business days, price drifts gently."""
    dates = pd.bdate_range(start, periods=n)
    close = price * (1 + 0.001 * np.arange(n))
    return pd.DataFrame({"date": dates, "symbol": "TEST", "open": close * 0.999, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "adj_close": close,
                         "volume": float(volume), "dividends": 0.0, "splits": 0.0})


def test_clean_stock_is_used():
    df = make()
    rep = dc.validate_symbol(df, df["date"])
    assert rep["problems"] == [] and rep["warnings"] == []
    assert rep["rows"] == 600 and rep["missing_days"] == 0


def test_empty_table_is_rejected():
    rep = dc.validate_symbol(pd.DataFrame(columns=dc.COLUMNS), pd.bdate_range("2020-01-01", periods=10))
    assert rep["problems"] == ["no data"]


def test_short_history_is_rejected():
    df = make(n=200)
    rep = dc.validate_symbol(df, df["date"])
    assert any("history" in p for p in rep["problems"])


def test_many_missing_days_are_rejected_but_a_few_are_fine():
    full = make(n=1000)
    few = full.drop(full.index[100:110])                    # 1% missing
    assert dc.validate_symbol(few, full["date"])["problems"] == []
    many = full.drop(full.index[100:300])                   # 20% missing
    rep = dc.validate_symbol(many, full["date"])
    assert rep["missing_days"] == 200 and any("missing" in p for p in rep["problems"])


def test_days_before_listing_do_not_count_as_missing():
    cal = pd.bdate_range("2015-01-01", periods=2000)
    df = make(n=600, start=str(cal[1400].date()))           # listed late
    assert dc.validate_symbol(df, cal)["missing_days"] == 0


def test_non_positive_price_is_rejected():
    df = make()
    df.loc[50, "low"] = 0.0
    assert any("non-positive" in p for p in dc.validate_symbol(df, df["date"])["problems"])


def test_untraded_stock_is_rejected():
    df = make()
    df.loc[:99, "volume"] = 0                               # 100 of 600 days = 16.7%
    assert any("volume" in p for p in dc.validate_symbol(df, df["date"])["problems"])


def test_unadjusted_split_is_flagged_as_warning():
    df = make()
    df.loc[300:, ["open", "high", "low", "close", "adj_close"]] /= 2     # looks like a 50% crash
    rep = dc.validate_symbol(df, df["date"])
    assert rep["problems"] == [] and len(rep["jumps"]) == 1
    assert rep["jumps"][0] == str(df["date"].iloc[300].date())


def test_validate_all_keeps_only_good_symbols():
    good, bad = make(), make(n=100)
    report, keep = dc.validate_all({"GOOD": good, "BAD": bad}, good["date"])
    assert keep == ["GOOD"] and report["BAD"]["used"] is False


def test_tidy_drops_empty_rows_and_duplicates_and_sorts():
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04"]).tz_localize("Asia/Kolkata")
    raw = pd.DataFrame({"Open": [3, 2, 2.5, np.nan], "High": [3, 2, 2.5, np.nan], "Low": [3, 2, 2.5, np.nan],
                        "Close": [3, 2, 2.5, np.nan], "Adj Close": [3, 2, 2.5, np.nan],   # Yahoo's column is ignored
                        "Volume": [10, 10, 10, 0]}, index=idx)
    out = dc.tidy(raw, "X")
    assert list(out.columns) == dc.COLUMNS
    assert [str(d.date()) for d in out["date"]] == ["2024-01-02", "2024-01-03"]
    assert out["close"].tolist() == [2.5, 3.0]              # the later duplicate wins
    assert (out["dividends"] == 0).all() and (out["symbol"] == "X").all()


def test_yahoo_ticker_mapping():
    assert dc.yahoo_ticker("RELIANCE") == "RELIANCE.NS"
    assert dc.yahoo_ticker("TATAMOTORS") == "TMPV.NS"


def test_dividend_adjustment_gives_total_return():
    # Price goes 100 -> 100 -> 90 (ex-dividend day, dividend 10) -> 99.
    # The raw close shows a fake 10% loss on the ex-dividend day; the holder lost nothing (90 + 10 = 100).
    close, div = [100.0, 100.0, 90.0, 99.0], [0.0, 0.0, 10.0, 0.0]
    adj = dc.dividend_adjusted_close(close, div)
    assert adj[-1] == 99.0 and adj[2] == 90.0                 # days on/after the ex-date are unchanged
    assert abs(adj[1] - 90.0) < 1e-9 and abs(adj[0] - 90.0) < 1e-9   # earlier days scaled by 1 - 10/100
    assert abs(adj[2] / adj[1] - 1) < 1e-9                    # no fake -10% drop on the ex-dividend day


def test_dividend_adjustment_without_dividends_changes_nothing():
    close = [10.0, 11.0, 12.5]
    assert list(dc.dividend_adjusted_close(close, [0, 0, 0])) == close


def test_dividend_adjustment_is_repeatable():
    df = make()
    df.loc[[100, 300], "dividends"] = 2.0
    a = dc.dividend_adjusted_close(df["close"], df["dividends"])
    b = dc.dividend_adjusted_close(df["close"], df["dividends"])
    assert (a == b).all() and a[0] < df["close"].iloc[0]
