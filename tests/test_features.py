"""Tests for the features and the label — above all: NO information from the future in the features.
Hand-made data, no internet.  Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from ml import features as ft            # noqa: E402
import build_dataset as bd               # noqa: E402
import config as cfg                     # noqa: E402
from data_collection import dividend_adjusted_close   # noqa: E402


def fake_stock(symbol="AAA", n=420, seed=0, start="2019-01-01", dividends=False):
    """A random-walk stock with realistic open/high/low/close/volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    div = np.zeros(n)
    if dividends:
        div[[150, 330]] = 1.5
    df = pd.DataFrame({"date": dates, "symbol": symbol, "open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.integers(200_000, 900_000, n).astype(float), "dividends": div, "splits": 0.0})
    df["adj_close"] = dividend_adjusted_close(df["close"], df["dividends"])
    return df


def fake_market(n=420, seed=99, start="2019-01-01"):
    nifty = fake_stock("NIFTY50", n, seed, start)
    vix = fake_stock("INDIAVIX", n, seed + 1, start)
    return pd.concat([nifty, vix], ignore_index=True)


# ───────────── no look-ahead ─────────────
@pytest.mark.parametrize("cut", [260, 300, 355, 419])
def test_stock_features_do_not_use_the_future(cut):
    """Features for day `cut` must be identical whether or not later days exist in the data."""
    df = fake_stock(dividends=True)
    full = ft.stock_features(df).iloc[cut]
    part = df.iloc[:cut + 1].copy()
    part["adj_close"] = dividend_adjusted_close(part["close"], part["dividends"])   # as it would be downloaded that day
    alone = ft.stock_features(part).iloc[-1]
    assert full.notna().all(), "expected every feature to be available after a year of history"
    np.testing.assert_allclose(full.to_numpy(), alone.to_numpy(), rtol=1e-9, atol=1e-12)


def test_changing_future_prices_does_not_change_past_features():
    df = fake_stock()
    before = ft.stock_features(df)
    changed = df.copy()
    changed.loc[300:, ["open", "high", "low", "close", "adj_close"]] *= 3.0     # wild future
    changed.loc[300:, "volume"] *= 10
    after = ft.stock_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:300], after.iloc[:300])


def test_market_features_do_not_use_the_future():
    m = fake_market()
    nifty, vix = m[m.symbol == "NIFTY50"], m[m.symbol == "INDIAVIX"]
    full = ft.market_features(nifty, vix)
    part = ft.market_features(nifty.iloc[:350], vix.iloc[:350])
    pd.testing.assert_frame_equal(full.iloc[:350], part)


def test_missing_vix_day_is_filled_from_the_past_not_the_future():
    m = fake_market()
    nifty, vix = m[m.symbol == "NIFTY50"], m[m.symbol == "INDIAVIX"].reset_index(drop=True)
    gap_day = vix["date"].iloc[300]
    out = ft.market_features(nifty, vix.drop(index=300))
    assert out.loc[gap_day, "vix"] == vix["close"].iloc[299]          # yesterday's value, never tomorrow's


def test_whole_panel_does_not_use_the_future():
    """Same check for the complete feature table (stock + market + relative + ranks) with several companies."""
    prices = pd.concat([fake_stock(s, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])], ignore_index=True)
    market = fake_market()
    day = prices["date"].sort_values().unique()[380]
    full = ft.build_panel(prices, market)
    part = ft.build_panel(prices[prices["date"] <= day], market[market["date"] <= day])
    a = full[full["date"] == day].sort_values("symbol")[ft.FEATURES].to_numpy()
    b = part[part["date"] == day].sort_values("symbol")[ft.FEATURES].to_numpy()
    assert not np.isnan(a).any()
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


# ───────────── feature sanity ─────────────
def test_feature_list_matches_what_is_produced():
    prices = pd.concat([fake_stock(s, seed=i) for i, s in enumerate(["AAA", "BBB"])], ignore_index=True)
    panel = ft.build_panel(prices, fake_market())
    assert all(c in panel.columns for c in ft.FEATURES)
    assert len(ft.FEATURES) == len(set(ft.FEATURES))


def test_features_do_not_depend_on_the_price_level():
    """A share at Rs 100 and the same share at Rs 1,000 must give the same features (only turnover differs)."""
    df = fake_stock()
    big = df.copy()
    big[["open", "high", "low", "close", "adj_close"]] *= 10
    a, b = ft.stock_features(df), ft.stock_features(big)
    cols = [c for c in ft.STOCK_FEATURES if c != "turnover_20_log"]
    np.testing.assert_allclose(a[cols].iloc[260:].to_numpy(), b[cols].iloc[260:].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose((b["turnover_20_log"] - a["turnover_20_log"]).iloc[260:].to_numpy(), 1.0, rtol=1e-9)


def test_known_values():
    df = fake_stock()
    f = ft.stock_features(df)
    c = df["adj_close"]
    assert f["ret_5"].iloc[300] == pytest.approx(c.iloc[300] / c.iloc[295] - 1)
    assert f["dist_sma200"].iloc[300] == pytest.approx(c.iloc[300] / c.iloc[101:301].mean() - 1)
    assert 0 <= f["rsi_14"].iloc[300] <= 100 and 0 <= f["close_pos"].iloc[300] <= 1
    assert f["ret_252"].iloc[:252].isna().all() and f["ret_252"].iloc[252:].notna().all()


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 60.0))
    assert ft._rsi(up).iloc[-1] == 100.0
    flat = pd.Series(np.full(60, 5.0))
    assert ft._rsi(flat).iloc[-1] == 50.0


def test_ranks_are_within_one_day_only():
    prices = pd.concat([fake_stock(s, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])], ignore_index=True)
    panel = ft.build_panel(prices, fake_market())
    day = panel[panel["date"] == panel["date"].max()]
    assert sorted(day["xs_ret_5"].round(6)) == [0.25, 0.5, 0.75, 1.0]
    assert day.sort_values("ret_5")["symbol"].tolist() == day.sort_values("xs_ret_5")["symbol"].tolist()


# ───────────── label ─────────────
def simple_panel(opens, series="AAA"):
    n = len(opens)
    return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n), "symbol": series.split("#")[0],
                         "series": series, "open": np.array(opens, float), "close": 1.0, "adj_close": 1.0})


def test_forward_return_is_next_open_to_open_after_horizon():
    opens = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    p = simple_panel(opens)
    out = bd.add_forward_return(p, p["date"], horizon=5)
    assert out["fwd_ret"].iloc[0] == pytest.approx(16 / 11 - 1)      # buy day 1 open (11), sell day 6 open (16)
    assert out["fwd_ret"].iloc[3] == pytest.approx(19 / 14 - 1)
    assert out["fwd_ret"].iloc[4:].isna().all()                      # the future is not known yet: no answer


def test_forward_return_needs_the_exact_market_days():
    """If the stock did not trade on the entry or exit day, there is no label (no silent use of another day)."""
    p = simple_panel([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    cal = p["date"]
    gone = p.drop(index=6).reset_index(drop=True)                    # no price on market day 6
    out = bd.add_forward_return(gone, cal, horizon=5).set_index("date")["fwd_ret"]
    assert np.isnan(out.loc[cal.iloc[0]])                            # exit day missing
    assert np.isnan(out.loc[cal.iloc[5]])                            # entry day missing
    assert out.loc[cal.iloc[1]] == pytest.approx(17 / 12 - 1)        # unaffected row


def test_forward_return_does_not_cross_a_price_break():
    a = simple_panel([10, 11, 12, 13, 14, 15, 16, 17], "TMPV")
    b = simple_panel([5, 5, 5, 5, 5, 5, 5, 5], "TMPV#1")
    b["date"] = pd.bdate_range(a["date"].iloc[-1] + pd.Timedelta(days=1), periods=8)
    both = pd.concat([a, b], ignore_index=True)
    out = bd.add_forward_return(both, both["date"], horizon=5)
    first = out[out["series"] == "TMPV"]["fwd_ret"]
    assert first.iloc[0] == pytest.approx(16 / 11 - 1) and first.iloc[2:].isna().all()


def test_add_series_cuts_at_the_break_date():
    p = pd.DataFrame({"date": pd.to_datetime(["2025-10-10", "2025-10-13", "2025-10-14", "2025-10-15"] * 2),
                      "symbol": ["TMPV"] * 4 + ["TCS"] * 4})
    out = bd.add_series(p, {"TMPV": ["2025-10-14"]})
    assert out["series"].tolist() == ["TMPV", "TMPV", "TMPV#1", "TMPV#1"] + ["TCS"] * 4


def test_label_marks_the_top_30_percent_each_day():
    rows = []
    for d, shift in (("2024-01-01", 0.0), ("2024-01-02", 5.0)):      # day 2: every return is higher — ranking must not care
        for i in range(10):
            rows.append({"date": pd.Timestamp(d), "symbol": f"S{i}", "eligible": True, "fwd_ret": i / 100 + shift})
    out = bd.add_label(pd.DataFrame(rows), top_fraction=0.30, min_stocks=5)
    for _, day in out.groupby("date"):
        assert day["label"].sum() == 3
        assert set(day.loc[day["label"] == 1, "symbol"]) == {"S7", "S8", "S9"}


def test_label_ignores_ineligible_rows_and_thin_days():
    rows = [{"date": pd.Timestamp("2024-01-01"), "symbol": f"S{i}", "eligible": i < 6, "fwd_ret": i / 100} for i in range(10)]
    out = bd.add_label(pd.DataFrame(rows), top_fraction=0.30, min_stocks=5)
    assert out.loc[~out["eligible"], "label"].isna().all()           # best performers S6-S9 are not eligible: no label
    assert out["label"].sum() == 2                                   # top 30% of the 6 eligible = 2 stocks (rank > 0.7)
    thin = bd.add_label(pd.DataFrame(rows), top_fraction=0.30, min_stocks=8)
    assert thin["label"].isna().all()                                # too few eligible stocks that day


def test_split_is_chronological_with_a_gap():
    dates = pd.bdate_range("2020-11-01", "2023-03-31")
    s = pd.Series(bd.split_of(dates), index=dates)
    assert s[s == "train"].index.max() < s[s == "valid"].index.min() < s[s == "test"].index.min()
    assert (s[dates <= cfg.TRAIN_END].isin(["train", "gap"])).all()
    assert (s == "gap").sum() == 2 * cfg.EMBARGO_DAYS
    last_train = s[s == "train"].index.max()
    first_valid = s[s == "valid"].index.min()
    assert len(dates[(dates > last_train) & (dates < first_valid)]) == cfg.EMBARGO_DAYS


def test_full_build_on_fake_data_has_no_label_where_the_future_is_unknown():
    prices = pd.concat([fake_stock(f"S{i:02d}", n=420, seed=i) for i in range(40)], ignore_index=True)
    prices["volume"] *= 100                                          # liquid enough to be eligible
    panel = bd.build(prices, fake_market())
    last_days = sorted(panel["date"].unique())[-(cfg.HORIZON + 1):]
    assert panel[panel["date"].isin(last_days)]["label"].isna().all()
    lab = panel[panel["label"].notna()]
    assert len(lab) > 0 and lab["eligible"].all()
    assert lab.groupby("date")["label"].mean().between(0.28, 0.32).all()
