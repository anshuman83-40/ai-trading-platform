"""
TradePulse ML — step 1: collect and check historical DAILY prices.

What it does
  1. Gets the list of NIFTY 200 companies (from NSE; a saved copy is reused if NSE does not answer).
  2. Downloads daily prices for each company, plus NIFTY 50 and India VIX, from Yahoo Finance.
  3. Checks every company's data for problems (too little history, missing days, impossible prices,
     suspicious jumps) and leaves out the ones that are not usable.
  4. Saves one data file, a validation report and a manifest (what was downloaded, when, with which settings).

It never touches the website or server.py. Output goes to training/data/ (not committed to GitHub).

Run from the project folder:
    python training/data_collection.py --end 2026-10-09
    python training/data_collection.py --symbols RELIANCE,TCS,INFY --end 2026-10-09      (quick try)

Prices: Yahoo's Open/High/Low/Close are already adjusted for share splits and bonuses, so a 2-for-1 split
does not look like a 50% crash. "adj_close" is additionally adjusted for dividends (use it for returns);
we calculate it ourselves from close and dividends so that the file is identical on every run.

Known limits (also written into the manifest):
  - Survivorship bias: only TODAY's NIFTY 200 members are downloaded. Companies that dropped out of the
    index (often after falling a lot) are missing, which makes any backtest look better than reality.
  - Yahoo can revise old adjusted prices when a new dividend or split happens, so two downloads on
    different dates may differ slightly. The manifest stores a checksum of the file so this is visible.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
# name used in our files -> Yahoo ticker
MARKET_SERIES = {"NIFTY50": "^NSEI", "INDIAVIX": "^INDIAVIX"}
# NSE symbol -> Yahoo ticker, where it is not simply SYMBOL.NS (same special case as server.py)
YAHOO_OVERRIDES = {"TATAMOTORS": "TMPV.NS"}

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]

# ── validation settings ──
MIN_ROWS = 500              # about 2 years of trading days; less than this is not enough to learn from
MAX_MISSING_PCT = 5.0       # % of market days (after the stock's first day) without a price
MAX_ZERO_VOLUME_PCT = 5.0   # % of days with no trading
JUMP_PCT = 40.0             # a 1-day move bigger than this is reported as suspicious (not removed)


# ───────────────────────── universe ─────────────────────────
def yahoo_ticker(symbol):
    return YAHOO_OVERRIDES.get(symbol, symbol + ".NS")


def load_universe(path):
    """NIFTY 200 members as a table [symbol, company, industry]. Saved to `path` so later runs are identical."""
    if os.path.exists(path):
        return pd.read_csv(path), "saved copy " + os.path.basename(path)
    import requests
    r = requests.get(NIFTY200_URL, timeout=(10, 30), headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/csv, */*"})
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    raw.columns = [c.strip() for c in raw.columns]
    uni = pd.DataFrame({
        "symbol": raw["Symbol"].astype(str).str.strip(),
        "company": raw.get("Company Name", raw["Symbol"]).astype(str).str.strip(),
        "industry": raw.get("Industry", pd.Series([""] * len(raw))).astype(str).str.strip(),
    })
    uni.to_csv(path, index=False)
    return uni, "NSE " + NIFTY200_URL


# ───────────────────────── download ─────────────────────────
def dividend_adjusted_close(close, dividends):
    """Closing prices adjusted backwards for dividends (the standard method: on each ex-dividend day,
    every EARLIER price is multiplied by 1 − dividend / previous close), so returns include dividends.
    We calculate this ourselves instead of using Yahoo's "Adj Close": Yahoo's values change in the 6th-7th
    digit from one download to the next, which made the saved file different on every run."""
    close = pd.Series(close, dtype="float64").reset_index(drop=True)
    div = pd.Series(dividends, dtype="float64").reset_index(drop=True).fillna(0.0)
    prev = close.shift(1)
    factor = pd.Series(1.0, index=close.index)
    ok = (div > 0) & (prev > 0) & (div < prev)
    factor[ok] = 1.0 - div[ok] / prev[ok]
    later = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)     # product of the factors of all LATER days
    return (close * later).to_numpy()


def tidy(df, symbol):
    """One Yahoo table (index = dates) -> our long format. Rows without a closing price are dropped."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Adj Close": "adj_close", "Volume": "volume",
                            "Dividends": "dividends", "Stock Splits": "splits"})
    for c in ("dividends", "splits"):
        if c not in df:
            df[c] = 0.0
    df = df.dropna(subset=["close"]).copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["adj_close"] = dividend_adjusted_close(df["close"].to_numpy(), df["dividends"].to_numpy())
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df["date"] = idx.normalize()
    df["symbol"] = symbol
    df = df[COLUMNS].drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df.reset_index(drop=True)


def download(pairs, start, end, chunk=25, pause=1.0):
    """pairs: [(symbol, yahoo ticker)] -> {symbol: tidy table}. Yahoo's `end` is exclusive, so +1 day."""
    import yfinance as yf
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out = {}
    for i in range(0, len(pairs), chunk):
        part = pairs[i:i + chunk]
        tickers = [t for _, t in part]
        for attempt in range(3):
            try:
                data = yf.download(tickers, start=start, end=end_excl, interval="1d", group_by="ticker",
                                   auto_adjust=False, actions=True, progress=False, threads=4)
                break
            except Exception as e:                       # network trouble: wait and try again
                if attempt == 2:
                    print(f"  [!] download failed for {tickers[:3]}…: {e}")
                    data = None
                time.sleep(5 * (attempt + 1))
        for sym, t in part:
            try:
                one = data[t] if data is not None and isinstance(data.columns, pd.MultiIndex) else data
                out[sym] = tidy(one, sym)
            except Exception:
                out[sym] = pd.DataFrame(columns=COLUMNS)
        print(f"  downloaded {min(i + chunk, len(pairs))}/{len(pairs)}")
        time.sleep(pause)
    return out


# ───────────────────────── validation (pure functions — tested in tests/) ─────────────────────────
def validate_symbol(df, calendar):
    """Check one company's tidy table against the market calendar (list of dates the market was open).
    Returns a dict of facts and problems. Does not change the data."""
    rep = {"rows": int(len(df)), "problems": [], "warnings": []}
    if len(df) == 0:
        rep["problems"].append("no data")
        return rep
    d = df.sort_values("date")
    rep["first"], rep["last"] = str(d["date"].iloc[0].date()), str(d["date"].iloc[-1].date())

    if d["date"].duplicated().any():
        rep["problems"].append("duplicate dates")
    if len(d) < MIN_ROWS:
        rep["problems"].append(f"only {len(d)} days of history (need {MIN_ROWS})")

    px = d[["open", "high", "low", "close"]]
    bad_px = int((px <= 0).any(axis=1).sum() + px.isna().any(axis=1).sum())
    rep["bad_price_rows"] = bad_px
    if bad_px:
        rep["problems"].append(f"{bad_px} rows with missing or non-positive prices")
    tol = 1e-6
    bad_range = int(((d["high"] + tol < d[["open", "close", "low"]].max(axis=1)) |
                     (d["low"] - tol > d[["open", "close", "high"]].min(axis=1))).sum())
    rep["inconsistent_rows"] = bad_range
    if bad_range > max(3, 0.005 * len(d)):
        rep["problems"].append(f"{bad_range} rows where high/low do not contain open/close")
    elif bad_range:
        rep["warnings"].append(f"{bad_range} rows where high/low do not contain open/close")

    cal = pd.DatetimeIndex(calendar)
    cal = cal[(cal >= d["date"].iloc[0]) & (cal <= d["date"].iloc[-1])]
    missing = cal.difference(pd.DatetimeIndex(d["date"]))
    rep["missing_days"] = int(len(missing))
    rep["missing_pct"] = round(100 * len(missing) / max(1, len(cal)), 2)
    if rep["missing_pct"] > MAX_MISSING_PCT:
        rep["problems"].append(f"{rep['missing_pct']}% of market days missing")

    zero = int((d["volume"].fillna(0) <= 0).sum())
    rep["zero_volume_days"] = zero
    rep["zero_volume_pct"] = round(100 * zero / len(d), 2)
    if rep["zero_volume_pct"] > MAX_ZERO_VOLUME_PCT:
        rep["problems"].append(f"{rep['zero_volume_pct']}% of days have no trading volume")

    ret = d["close"].pct_change().abs() * 100
    jumps = d.loc[ret > JUMP_PCT, "date"]
    rep["jumps"] = [str(x.date()) for x in jumps][:10]
    if len(jumps):
        rep["warnings"].append(f"{len(jumps)} one-day moves above {JUMP_PCT:.0f}% (possible unadjusted split or bad tick)")
    return rep


def validate_all(tables, calendar):
    """-> (report per symbol, list of usable symbols)"""
    report, keep = {}, []
    for sym, df in tables.items():
        rep = validate_symbol(df, calendar)
        rep["used"] = not rep["problems"]
        report[sym] = rep
        if rep["used"]:
            keep.append(sym)
    return report, keep


# ───────────────────────── main ─────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download and validate daily prices for the TradePulse ML dataset.")
    ap.add_argument("--start", default="2012-01-01", help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)),
                    help="last date, inclusive (YYYY-MM-DD). Give a fixed date to make the run repeatable.")
    ap.add_argument("--symbols", default="", help="comma-separated NSE symbols for a quick try (default: NIFTY 200)")
    ap.add_argument("--out", default=DATA_DIR, help="output folder")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # 1) universe
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        uni, uni_src = pd.DataFrame({"symbol": syms, "company": syms, "industry": ""}), "command line"
    else:
        uni, uni_src = load_universe(os.path.join(args.out, "universe_nifty200.csv"))
    print(f"Universe: {len(uni)} companies ({uni_src})")

    # 2) market series first — NIFTY 50's dates are the market calendar
    print("Downloading market series…")
    market = download([(name, t) for name, t in MARKET_SERIES.items()], args.start, args.end)
    if len(market.get("NIFTY50", [])) < 100:
        sys.exit("Could not download NIFTY 50 — no market calendar, stopping. Check the internet connection.")
    calendar = market["NIFTY50"]["date"]

    # 3) stocks
    print("Downloading stocks…")
    tables = download([(s, yahoo_ticker(s)) for s in uni["symbol"]], args.start, args.end)

    # 4) validate
    report, keep = validate_all(tables, calendar)
    dropped = {s: r["problems"] for s, r in report.items() if not r["used"]}

    # 5) save
    prices = pd.concat([tables[s] for s in keep], ignore_index=True) if keep else pd.DataFrame(columns=COLUMNS)
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    mkt = pd.concat(market.values(), ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    p_path, m_path = os.path.join(args.out, "prices_daily.csv.gz"), os.path.join(args.out, "market_daily.csv.gz")
    # fixed float format + no gzip timestamp, so the same data always gives the same file (same checksum)
    for frame, path in ((prices, p_path), (mkt, m_path)):
        frame.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.6f",
                     compression={"method": "gzip", "mtime": 0})

    import yfinance as yf
    manifest = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Yahoo Finance via yfinance " + yf.__version__,
        "interval": "1d", "start": args.start, "end": args.end,
        "price_adjustment": "open/high/low/close are split- and bonus-adjusted by Yahoo; adj_close is also dividend-adjusted (calculated here from close and dividends)",
        "universe": uni_src, "universe_size": int(len(uni)),
        "symbols_used": len(keep), "symbols_dropped": len(dropped),
        "rows": int(len(prices)),
        "date_range": [str(prices["date"].min().date()), str(prices["date"].max().date())] if len(prices) else None,
        "market_series": {k: int(len(v)) for k, v in market.items()},
        "validation_settings": {"MIN_ROWS": MIN_ROWS, "MAX_MISSING_PCT": MAX_MISSING_PCT,
                                "MAX_ZERO_VOLUME_PCT": MAX_ZERO_VOLUME_PCT, "JUMP_PCT": JUMP_PCT},
        "files": {os.path.basename(p_path): sha256(p_path), os.path.basename(m_path): sha256(m_path)},
        "known_limits": [
            "Survivorship bias: only today's NIFTY 200 members are included; past members that left the index are missing.",
            "Yahoo may revise adjusted history after new dividends/splits; compare checksums between runs.",
            "Free data: not exchange-licensed, may contain errors. For research and education only.",
        ],
    }
    with open(os.path.join(args.out, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    # 6) summary
    warn = sum(1 for r in report.values() if r["used"] and r["warnings"])
    print(f"\nUsed {len(keep)} of {len(uni)} companies  |  {len(prices):,} rows  |  "
          f"{manifest['date_range'][0] if len(prices) else '-'} to {manifest['date_range'][1] if len(prices) else '-'}")
    print(f"Left out {len(dropped)}:")
    for s, why in sorted(dropped.items()):
        print(f"  {s:<12} {'; '.join(why)}")
    print(f"Used but with warnings: {warn} (see validation_report.json)")
    print(f"Saved to {args.out}")
    return manifest


if __name__ == "__main__":
    main()
