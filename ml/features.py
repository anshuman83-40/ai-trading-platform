"""
TradePulse ML — feature definitions (the numbers the model studies).

ONE place for feature code, used by BOTH the training pipeline and (later) the live server, so the model
always sees numbers calculated the same way.

The rule that matters most: a feature for day t may use prices up to and including day t — never later.
Everything here only looks backwards (rolling windows, moving averages, shift by a positive number).
tests/test_features.py proves it by cutting the data at day t and checking the numbers do not change.

Input tables use the columns written by training/data_collection.py:
    date, open, high, low, close, adj_close, volume      (one company, sorted by date)
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = "f1"

STOCK_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "ret_252",     # past returns
    "rsi_14", "macd_line", "macd_hist",                                        # momentum
    "dist_ema20", "dist_ema50", "dist_sma200",                                 # trend: price vs its averages
    "atr_14", "vol_21", "vol_63", "vol_ratio",                                 # how much the price swings
    "volume_ratio_20", "turnover_20_log",                                      # trading activity
    "dist_high_252", "dist_low_252", "close_pos", "gap_1", "up_days_10",       # position and recent behaviour
]
MARKET_FEATURES = ["mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_dist_sma200", "mkt_vol_21", "vix", "vix_chg_5"]
RELATIVE_FEATURES = ["rel_ret_5", "rel_ret_21", "rel_ret_63"]                  # stock minus market
RANK_FEATURES = ["xs_ret_5", "xs_ret_21", "xs_ret_126", "xs_vol_21", "xs_volume_ratio_20"]   # rank among all stocks that day
FEATURES = STOCK_FEATURES + MARKET_FEATURES + RELATIVE_FEATURES + RANK_FEATURES


def _rsi(close, n=14):
    """Wilder's RSI — same formula as server.py, for the whole series."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)
    return rsi


def stock_features(df):
    """Features for ONE company. Returns a table with the same row order as `df` and the STOCK_FEATURES columns.
    Prices are scaled by adj_close/close first, so dividends do not look like price falls. Every feature is a
    ratio, so the scaling itself cannot leak information."""
    f = pd.DataFrame(index=df.index)
    k = (df["adj_close"] / df["close"]).astype("float64")
    c = df["adj_close"].astype("float64")
    o, h, l = df["open"] * k, df["high"] * k, df["low"] * k
    r1 = c.pct_change()

    for n in (1, 5, 10, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1

    f["rsi_14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False, min_periods=26).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd_line"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c

    f["dist_ema20"] = c / c.ewm(span=20, adjust=False, min_periods=20).mean() - 1
    f["dist_ema50"] = c / c.ewm(span=50, adjust=False, min_periods=50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    f["vol_21"] = r1.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = r1.rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / f["vol_63"].replace(0, np.nan)

    vol = df["volume"].astype("float64")
    f["volume_ratio_20"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    turnover = (df["close"].astype("float64") * vol).rolling(20).mean()          # rupees traded per day (raw prices)
    f["turnover_20_log"] = np.log10(turnover.where(turnover > 0))

    f["dist_high_252"] = c / h.rolling(252).max() - 1
    f["dist_low_252"] = c / l.rolling(252).min() - 1
    rng = (h - l)
    f["close_pos"] = ((c - l) / rng.where(rng > 0)).fillna(0.5)                  # 1 = closed at the day's high
    f["gap_1"] = o / prev - 1
    f["up_days_10"] = (r1 > 0).astype("float64").rolling(10).mean().where(r1.notna().rolling(10).sum() == 10)
    return f[STOCK_FEATURES]


def market_features(nifty, vix):
    """Market-wide features, one row per date (index = date). `nifty` and `vix`: tables with date and close."""
    n = nifty.sort_values("date").set_index("date")["close"].astype("float64")
    m = pd.DataFrame(index=n.index)
    r1 = n.pct_change()
    m["mkt_ret_1"], m["mkt_ret_5"], m["mkt_ret_21"] = r1, n / n.shift(5) - 1, n / n.shift(21) - 1
    m["mkt_ret_63"] = n / n.shift(63) - 1                                        # helper for rel_ret_63
    m["mkt_dist_sma200"] = n / n.rolling(200).mean() - 1
    m["mkt_vol_21"] = r1.rolling(21).std() * np.sqrt(252)
    v = vix.sort_values("date").set_index("date")["close"].astype("float64").reindex(n.index).ffill()   # past values only
    m["vix"] = v
    m["vix_chg_5"] = v / v.shift(5) - 1
    return m


def add_relative(panel):
    """Stock return minus market return. `panel` already has the stock and market columns."""
    for n in (5, 21, 63):
        panel[f"rel_ret_{n}"] = panel[f"ret_{n}"] - panel[f"mkt_ret_{n}"]
    return panel


def add_ranks(panel):
    """Where each stock stands among all stocks ON THE SAME DAY (0 = lowest, 1 = highest)."""
    by_day = panel.groupby("date")
    for col in ("ret_5", "ret_21", "ret_126", "vol_21", "volume_ratio_20"):
        panel["xs_" + col] = by_day[col].rank(pct=True)
    return panel


def build_panel(prices, market, series_col="symbol"):
    """All features for all companies. `prices`: many companies stacked; `market`: NIFTY50 and INDIAVIX rows.
    `series_col` names the column that separates one price series from another."""
    prices = prices.sort_values([series_col, "date"]).reset_index(drop=True)
    parts = [stock_features(g) for _, g in prices.groupby(series_col, sort=False)]
    feats = pd.concat(parts).loc[prices.index]
    panel = pd.concat([prices, feats], axis=1)
    mkt = market_features(market[market["symbol"] == "NIFTY50"], market[market["symbol"] == "INDIAVIX"])
    panel = panel.merge(mkt, left_on="date", right_index=True, how="left")
    panel = add_ranks(add_relative(panel))
    return panel
