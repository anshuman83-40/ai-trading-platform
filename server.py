# ============================================================
#  TradePulse Backend — server.py
#  REAL market data via Yahoo Finance (yfinance) — same source
#  as Market_sentement/sentiment_analyzer.py
#  Run: python server.py   →   http://localhost:5000
# ============================================================

from flask import Flask, jsonify, request, send_from_directory, abort, redirect
from flask_cors import CORS
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import threading
import io
import requests
import time
import os
import re
import json
import gc
import html as html_lib
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
except ImportError:
    VADER = None

app = Flask(__name__)
CORS(app)  # Allow your website to call this server

# ============================================================
#  GROWW API (only used for placing real orders)
# ============================================================
GROWW_ACCESS_TOKEN = "YOUR_GROWW_ACCESS_TOKEN_HERE"
GROWW_BASE_URL = "https://api.groww.in/v1"
HEADERS = {
    "Authorization": f"Bearer {GROWW_ACCESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def is_groww_configured():
    return GROWW_ACCESS_TOKEN != "YOUR_GROWW_ACCESS_TOKEN_HERE" and len(GROWW_ACCESS_TOKEN) > 10

# Optional: NewsAPI key (newsapi.org). If empty, Google News RSS is used (free, no key).
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# ============================================================
#  SYMBOLS
#  Website symbol → Yahoo Finance ticker
# ============================================================
STOCKS = {
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "WIPRO":      "WIPRO.NS",
    "TATAMOTORS": "TMPV.NS",      # Tata Motors demerged in 2025 — now TMPV on NSE
    "SBIN":       "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ITC":        "ITC.NS",
    "LT":         "LT.NS",
    "AXISBANK":   "AXISBANK.NS",
    "MARUTI":     "MARUTI.NS",
}
INDICES = {
    "sensex":    ("BSE SENSEX", "^BSESN"),
    "nifty":     ("NIFTY 50",   "^NSEI"),
    "banknifty": ("BANK NIFTY", "^NSEBANK"),
}
NEWS_NAMES = {
    "RELIANCE": "Reliance Industries", "TCS": "TCS", "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank", "WIPRO": "Wipro", "TATAMOTORS": "Tata Motors",
    "SBIN": "SBI", "BAJFINANCE": "Bajaj Finance", "ITC": "ITC",
    "LT": "Larsen & Toubro", "AXISBANK": "Axis Bank", "MARUTI": "Maruti Suzuki",
}

# ============================================================
#  SIMPLE TTL CACHE (so we don't hammer Yahoo every 2 seconds)
# ============================================================
_cache = {}
_lock = threading.Lock()

def cached(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = fn()
    if value is not None:
        with _lock:
            _cache[key] = (now, value)
    return value

def peek(key):
    """Return a cached value without loading it."""
    with _lock:
        hit = _cache.get(key)
    return hit[1] if hit else None

def store(key, value):
    with _lock:
        _cache[key] = (time.time(), value)

# ============================================================
#  DATA-SOURCE HEALTH — what worked / failed last time (see /api/health)
# ============================================================
IST = timezone(timedelta(hours=5, minutes=30))
_health = {}

def _mark(source, ok, info=""):
    _health[source] = {"ok": ok, "at": datetime.now(IST).strftime("%d %b %H:%M:%S IST"), "info": str(info)[:200]}

# NSE / BSE often silently ignore requests from cloud servers (like Render) instead of refusing them,
# so every outside website gets a short connect + read time limit and is only called from background threads
QUICK = (5, 15)

def _get(url, total=30, session=None, **kw):
    """GET with a hard overall deadline. requests' own timeout only limits each read, so a site that
    trickles data slowly could otherwise hold a thread forever."""
    deadline = time.time() + total
    kw.setdefault("timeout", QUICK)
    r = (session or requests).get(url, stream=True, **kw)
    try:
        chunks = []
        for chunk in r.iter_content(65536):
            chunks.append(chunk)
            if time.time() > deadline:
                raise TimeoutError(f"{url.split('/')[2]} took longer than {total}s")
        r._content, r._content_consumed = b"".join(chunks), True    # lets r.text / r.json() work
        return r
    finally:
        r.close()

# ============================================================
#  ALL INDIAN STOCKS — official NSE + BSE lists (~5,300 companies)
#  Merged by ISIN so a company listed on both appears once (NSE preferred)
#  Loaded by the background thread — user requests never wait for it
# ============================================================
NSE_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_LIST_URL = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                "?Group=&Scripcode=&industry=&segment=Equity&status=Active")
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com",
    "Accept": "application/json, text/csv, */*",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BSE_SNAPSHOT = os.path.join(DATA_DIR, "bse_scrips.json")   # saved copy for servers that BSE blocks (e.g. Render)
FALLBACK_UNIVERSE = {s: {"name": NEWS_NAMES.get(s, s), "exchange": "NSE", "yahoo": y, "bse_code": None}
                     for s, y in STOCKS.items()}

def _nse_csv(url):
    r = _get(url, headers=BROWSER_HEADERS)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    return df

def _bse_rows():
    """Active BSE equities as [symbol, name, isin, code]: live from BSE, else the saved copy."""
    err = "empty response"
    try:
        live = _get(BSE_LIST_URL, headers=BROWSER_HEADERS).json()
        rows = [[(r.get("scrip_id") or "").strip().upper(), (r.get("Scrip_Name") or "").strip(),
                 (r.get("ISIN_NUMBER") or "").strip(), r.get("SCRIP_CD")] for r in live if r.get("scrip_id")]
        if rows:
            _mark("bse_list", True, f"live from BSE: {len(rows)} scrips")
            try:   # refresh the saved copy whenever BSE answers (e.g. on your laptop) — push it to update Render
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(BSE_SNAPSHOT, "w", encoding="utf-8") as f:
                    json.dump({"saved": datetime.now(IST).strftime("%Y-%m-%d"), "rows": rows}, f, separators=(",", ":"))
            except OSError:
                pass
            return rows
    except Exception as e:
        err = e
    try:
        with open(BSE_SNAPSHOT, encoding="utf-8") as f:
            snap = json.load(f)
        _mark("bse_list", True, f"saved copy from {snap.get('saved')} ({len(snap['rows'])} scrips); live BSE failed: {err}")
        return snap["rows"]
    except Exception as e:
        _mark("bse_list", False, f"live BSE failed: {err} | no saved copy: {e}")
        return []

def _load_universe():
    out, nse_by_isin = {}, {}
    try:
        for _, r in _nse_csv(NSE_LIST_URL).iterrows():
            sym = str(r["SYMBOL"]).strip()
            out[sym] = {"name": str(r["NAME OF COMPANY"]).strip(), "exchange": "NSE",
                        "yahoo": STOCKS.get(sym, sym + ".NS"), "bse_code": None}
            nse_by_isin[str(r.get("ISIN NUMBER", "")).strip()] = sym
        _mark("nse_list", True, f"{len(out)} companies")
    except Exception as e:
        _mark("nse_list", False, e)
    for sym, name, isin, code in _bse_rows():
        if not sym:
            continue
        if isin in nse_by_isin:                 # listed on both → keep NSE entry, mark it
            v = out[nse_by_isin[isin]]
            v["exchange"], v["bse_code"] = "NSE+BSE", code
            continue
        if sym in out:                          # same symbol, different company → keep NSE one
            continue
        out[sym] = {"name": name or sym, "exchange": "BSE", "yahoo": sym + ".BO", "bse_code": code}
    return out or None

def refresh_universe():
    u = _load_universe()
    if u:
        store("all_stocks", u)

def all_stocks():
    """{SYMBOL: {name, exchange, yahoo, bse_code}} — every NSE + BSE stock once loaded (never waits on the network)."""
    return peek("all_stocks") or FALLBACK_UNIVERSE

def search_stocks(q, limit=10):
    """Match by symbol, company name or BSE scrip code (e.g. 'tata steel', 'TATASTEEL', '500470')."""
    q = q.upper().strip()
    if not q:
        return []
    qs = q.replace(" ", "")
    exact, prefix, name_hits = [], [], []
    for sym, info in all_stocks().items():
        row = {"symbol": sym, "name": info["name"], "exchange": info["exchange"]}
        if sym == qs or info["bse_code"] == q:
            exact.append(row)
        elif sym.startswith(qs):
            prefix.append(row)
        elif q in info["name"].upper():
            name_hits.append(row)
    prefix.sort(key=lambda r: len(r["symbol"]))
    return (exact + prefix + name_hits)[:limit]

def resolve_symbol(q):
    """Turn whatever the user typed into a known symbol."""
    q = q.upper().strip()
    if q in STOCKS or q in all_stocks() or q.startswith("^"):
        return q
    hits = search_stocks(q, 1)
    return hits[0]["symbol"] if hits else q.replace(" ", "")

def yahoo_candidates(symbol):
    """Yahoo tickers to try, in order: known listing first, then NSE, then BSE."""
    symbol = symbol.upper().strip()
    if symbol.startswith("^") or "." in symbol:
        return [symbol]
    info = all_stocks().get(symbol)
    first = info["yahoo"] if info else STOCKS.get(symbol, symbol + ".NS")
    return list(dict.fromkeys([first, symbol + ".NS", symbol + ".BO"]))

def to_yahoo(symbol):
    return yahoo_candidates(symbol)[0]

# ============================================================
#  INDICATORS — real RSI(14), MACD(12,26,9), ATR(14)
# ============================================================
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).fillna(50).iloc[-1])

def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return float(line.iloc[-1]), float(signal.iloc[-1])

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])

def build_quote(symbol, df):
    """Turn 1 year of daily OHLCV into the quote + signal the website uses."""
    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    price = float(last["Close"])
    change = (price - float(prev["Close"])) / float(prev["Close"]) * 100

    r = rsi(df["Close"])
    m_line, m_sig = macd(df["Close"])
    a = atr(df)
    ema20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])

    # Score: momentum (MACD), trend (EMA20), RSI zone
    score = 0
    score += 1 if m_line > m_sig else -1
    score += 1 if price > ema20 else -1
    if r < 30: score += 1          # oversold → bounce likely
    elif r > 70: score -= 1        # overbought → pullback likely
    elif r > 55: score += 0.5
    elif r < 45: score -= 0.5

    if score >= 1.5:
        signal = "BUY"
        entry, stop, target = price - 0.25 * a, price - 1.5 * a, price + 2.5 * a
    elif score <= -1.5:
        signal = "SELL"
        entry, stop, target = price + 0.25 * a, price + 1.5 * a, price - 2.5 * a
    else:
        signal = "HOLD"
        entry, stop, target = price - 0.5 * a, price - 1.5 * a, price + 1.5 * a

    macd_lbl = "BULLISH" if m_line > m_sig and m_line > 0 else \
               "BEARISH" if m_line < m_sig and m_line < 0 else "NEUTRAL"

    return {
        "symbol":     symbol,
        "price":      round(price, 2),
        "change":     round(change, 2),
        "change_pct": round(change, 2),
        "high":       round(float(last["High"]), 2),
        "low":        round(float(last["Low"]), 2),
        "open":       round(float(last["Open"]), 2),
        "prev_close": round(float(prev["Close"]), 2),
        "volume":     int(last["Volume"] or 0),
        "w52high":    round(float(df["High"].max()), 2),
        "w52_high":   round(float(df["High"].max()), 2),
        "w52_low":    round(float(df["Low"].min()), 2),
        "rsi":        round(r),
        "macd":       macd_lbl,
        "signal":     signal,
        "entry":      round(entry, 2),
        "stop":       round(stop, 2),
        "target":     round(target, 2),
        "as_of":      str(df.index[-1].date()),
        "source":     "yahoo_finance",
    }

# ============================================================
#  DATA FETCHERS
# ============================================================
_yf_lock = threading.Lock()  # yf.download uses global state — never run two at once
_last_rate_limit = 0.0       # when Yahoo last answered "Too Many Requests"
_waiting = [0]               # visitors' requests waiting for Yahoo — the background scan steps aside for them
_waiting_lock = threading.Lock()

def _download(tickers, threads=True, background=False, **kw):
    global _last_rate_limit
    if background:                       # let any waiting visitor go first (max ~15 s)
        for _ in range(75):
            if not _waiting[0]:
                break
            time.sleep(0.2)
    else:
        with _waiting_lock:
            _waiting[0] += 1
    try:
        with _yf_lock:
            data = yf.download(tickers, group_by="ticker", progress=False,
                               threads=threads, auto_adjust=False, **kw)
            try:
                errors = getattr(yf.shared, "_ERRORS", {}) or {}
                if any("rate limit" in str(e).lower() or "too many requests" in str(e).lower() for e in errors.values()):
                    _last_rate_limit = time.time()
            except Exception:
                pass
            return data
    finally:
        if not background:
            with _waiting_lock:
                _waiting[0] -= 1

def fetch_all_quotes():
    """One batched Yahoo call for all watchlist stocks (~2 seconds)."""
    def load():
        data = _download(list(STOCKS.values()), period="1y", interval="1d")
        out = {}
        for sym, yt in STOCKS.items():
            try:
                q = build_quote(sym, data[yt])
                if q: out[sym] = q
            except Exception as e:
                print(f"  [!] {sym}: {e}")
        return out or None
    return cached("all_quotes", 20, load) or {}

def _label(q, symbol, yahoo):
    info = all_stocks().get(symbol, {})
    q["name"] = info.get("name") or NEWS_NAMES.get(symbol, symbol)
    q["exchange"] = "BSE" if yahoo.endswith(".BO") else "NSE"
    q["listed_on"] = info.get("exchange", q["exchange"])
    return q

def fetch_quote(symbol):
    symbol = resolve_symbol(symbol)
    if symbol in STOCKS:
        q = fetch_all_quotes().get(symbol)
        if q: return _label(q, symbol, STOCKS[symbol])
    def load():
        for yt in yahoo_candidates(symbol):     # NSE first, then BSE
            df = yf.Ticker(yt).history(period="1y", interval="1d", auto_adjust=False)
            if not df.empty:
                q = build_quote(symbol, df)
                if q: return _label(q, symbol, yt)
        return None
    return cached(f"quote:{symbol}", 20, load)

def fetch_history(symbol, interval):
    symbol = resolve_symbol(symbol)
    period = {"1m": "1d", "5m": "1d", "15m": "5d", "1h": "1mo"}.get(interval, "1d")
    def load():
        df = pd.DataFrame()
        for yt in yahoo_candidates(symbol):
            df = yf.Ticker(yt).history(period=period, interval=interval).dropna(subset=["Close"])
            if not df.empty: break
        if df.empty: return None
        if interval == "15m": df = df.tail(60)
        if interval == "1h":  df = df.tail(70)
        fmt = "%H:%M" if interval in ("1m", "5m") else "%d %b %H:%M"
        return {
            "symbol": symbol.upper(), "interval": interval,
            "labels": [t.strftime(fmt) for t in df.index],
            "close":  [round(float(c), 2) for c in df["Close"]],
        }
    return cached(f"hist:{symbol}:{interval}", 30, load)

# ============================================================
#  ROUTES — prices
# ============================================================
@app.route("/api/live/all")
@app.route("/api/prices")
def live_all_prices():
    return jsonify(fetch_all_quotes())

@app.route("/api/live/<symbol>")
@app.route("/api/price/<symbol>")
def live_price(symbol):
    q = fetch_quote(symbol)
    if not q:
        return jsonify({"error": f"No data found for {symbol.upper()} on NSE or BSE"}), 404
    return jsonify(q)

@app.route("/api/history/<symbol>")
def history(symbol):
    interval = request.args.get("interval", "5m")
    if interval not in ("1m", "5m", "15m", "1h"):
        interval = "5m"
    h = fetch_history(symbol, interval)
    if not h:
        return jsonify({"error": "No intraday data"}), 404
    return jsonify(h)

@app.route("/api/indices")
def get_indices():
    def load():
        tickers = [t for _, t in INDICES.values()]
        daily = _download(tickers, period="5d", interval="1d")
        intra = _download(tickers, period="1d", interval="5m")
        out = {}
        for key, (name, t) in INDICES.items():
            try:
                c = daily[t]["Close"].dropna()
                val, prev = float(c.iloc[-1]), float(c.iloc[-2])
                spark = intra[t]["Close"].dropna()
                out[key] = {
                    "name":   name,
                    "value":  round(val, 2),
                    "change": round((val - prev) / prev * 100, 2),
                    "points": round(val - prev, 2),
                    "spark":  [round(float(x), 2) for x in spark.tolist()],
                }
            except Exception as e:
                print(f"  [!] index {key}: {e}")
        return out or None
    return jsonify(cached("indices", 20, load) or {})

@app.route("/api/search")
def search():
    return jsonify(search_stocks(request.args.get("q", ""), 12))

@app.route("/api/universe")
def universe():
    s = all_stocks()
    count = lambda e: sum(1 for v in s.values() if v["exchange"] == e)
    return jsonify({"total": len(s), "nse_only": count("NSE"), "bse_only": count("BSE"), "both": count("NSE+BSE"),
                    "loaded": peek("all_stocks") is not None})

# ============================================================
#  MARKET MOVERS — choose NIFTY 50 / NIFTY 200 / NIFTY 500 / ALL INDIA (NSE+BSE)
#  A background thread keeps today's % change for every stock in memory:
#    every 5 min the background thread fetches today's change for ALL ~5,300 NSE + BSE stocks
#    (NIFTY 500 first) from Yahoo's lightweight spark endpoint; every list is built from that
# ============================================================
INDEX_LISTS = {
    "NIFTY50":  "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY500": NIFTY500_URL,
}
UNIVERSE_NAMES = {"NIFTY50": "NIFTY 50", "NIFTY200": "NIFTY 200", "NIFTY500": "NIFTY 500", "ALL": "ALL INDIA (NSE+BSE)"}
_moves = {}          # symbol -> {"symbol", "price", "change_pct", "volume", "exchange"}
ON_RENDER = bool(os.environ.get("RENDER"))   # Render sets this; its free server is small, so start gently

def refresh_index_lists():
    for key, url in INDEX_LISTS.items():
        try:
            store(f"members:{key}", [str(s).strip() for s in _nse_csv(url)["Symbol"]])
            _mark(f"list_{key.lower()}", True, "ok")
        except Exception as e:
            _mark(f"list_{key.lower()}", False, e)

def index_members(key):
    """Index constituents once loaded by the background thread (never waits on the network)."""
    return peek(f"members:{key}") or []

# Yahoo's "spark" endpoint (the one behind its mini-charts) returns today's price and % change for up to
# 20 stocks per request as small JSON — no login cookie and no heavy data tables, so it also works on Render.
# All of India (~5,300 stocks) = ~270 small requests, a few seconds with 4 parallel connections.
SPARK_URL = "https://query2.finance.yahoo.com/v8/finance/spark"
_spark = requests.Session()
_spark.headers.update({"User-Agent": BROWSER_HEADERS["User-Agent"]})
_full_pass_done = False

def _spark_batch(pairs):
    """pairs: [(our symbol, yahoo ticker)] (max 20) → list of mover rows."""
    r = _get(SPARK_URL, total=20, session=_spark,
             params={"symbols": ",".join(t for _, t in pairs), "range": "1d", "interval": "1d"})
    if r.status_code == 429:
        raise RuntimeError("Yahoo rate limit (HTTP 429)")
    r.raise_for_status()
    data, stale, rows = r.json(), time.time() - 4 * 86400, []
    for sym, t in pairs:
        v = data.get(t) or {}
        price, prev = v.get("fulldayPrice"), v.get("chartPreviousClose")
        if not price or not prev or (v.get("timestamp") or [0])[-1] < stale:   # no price / not traded for days
            continue
        rows.append({"symbol": sym, "price": round(float(price), 2),
                     "change_pct": round(float(v.get("fulldayChangePercent") or (price - prev) / prev * 100), 2),
                     "volume": None, "exchange": "BSE" if t.endswith(".BO") else "NSE"})
    return rows

def refresh_moves(symbols):
    """Today's price + % change for these symbols, 20 per request, 4 requests at a time."""
    from concurrent.futures import ThreadPoolExecutor
    stocks = all_stocks()
    pairs = [(s, stocks.get(s, {}).get("yahoo") or STOCKS.get(s, s + ".NS")) for s in symbols
             if not re.search(r"-RE\d*$", s)]                  # skip rights entitlements (not normal shares)
    batches = [pairs[i:i + 20] for i in range(0, len(pairs), 20)]
    errors = []
    def work(b):
        try:
            return _spark_batch(b)
        except Exception as e:
            errors.append(str(e))
            return []
    with ThreadPoolExecutor(4) as ex:
        for rows in ex.map(work, batches):
            with _lock:
                for row in rows:
                    _moves[row["symbol"]] = row
    if errors:
        _mark("yahoo_movers", False, f"{len(errors)} of {len(batches)} requests failed: {errors[0]}")
    return len(batches) - len(errors)

def _movers_worker():
    """Background thread: stock lists once a day; live movers for ALL of India every 5 min."""
    global _full_pass_done
    loop = 0
    while True:
        try:
            if loop % 288 == 0:                                  # at start, then once a day
                refresh_universe()
                refresh_index_lists()
            n500 = index_members("NIFTY500") or list(STOCKS)
            n500_set = set(n500)
            refresh_moves(n500)                                  # NIFTY lists first (ready in seconds)…
            refresh_moves([s for s in all_stocks() if s not in n500_set])   # …then the rest of India
            _full_pass_done = True
            _mark("yahoo_movers", True, f"all India updated ({len(_moves)} stocks)")
        except Exception as e:
            _mark("yahoo_movers", False, e)
        loop += 1
        time.sleep(300)

def get_movers(universe="NIFTY500"):
    universe = universe.upper().replace(" ", "")
    if universe not in UNIVERSE_NAMES:
        universe = "NIFTY500"
    with _lock:
        snapshot = dict(_moves)
    if universe == "ALL":
        rows = list(snapshot.values())
    else:
        rows = [snapshot[s] for s in index_members(universe) if s in snapshot]
    if not rows:  # background thread hasn't finished its first pass yet
        rows = [{"symbol": x["symbol"], "price": x["price"], "change_pct": x["change"],
                 "volume": x["volume"], "exchange": "NSE"} for x in fetch_all_quotes().values()]
        return rows, "WATCHLIST", universe, True
    return rows, UNIVERSE_NAMES[universe], universe, universe == "ALL" and not _full_pass_done

@app.route("/api/movers")
def movers():
    n = min(int(request.args.get("n", 10)), 50)
    rows, name, key, loading = get_movers(request.args.get("universe", "NIFTY500"))
    rows = sorted(rows, key=lambda x: x["change_pct"], reverse=True)
    return jsonify({"universe": name, "key": key, "count": len(rows), "loading": loading,
                    "gainers": [r for r in rows if r["change_pct"] > 0][:n],
                    "losers": [r for r in reversed(rows) if r["change_pct"] < 0][:n]})

@app.route("/api/gainers")
def get_gainers():
    q = sorted(get_movers(request.args.get("universe", "NIFTY500"))[0], key=lambda x: x["change_pct"], reverse=True)
    return jsonify([x for x in q if x["change_pct"] > 0][:5])

@app.route("/api/losers")
def get_losers():
    q = sorted(get_movers(request.args.get("universe", "NIFTY500"))[0], key=lambda x: x["change_pct"])
    return jsonify([x for x in q if x["change_pct"] < 0][:5])

# ============================================================
#  ROUTE — real news + VADER sentiment (like sentiment_analyzer.py)
# ============================================================
def score_headline(text):
    s = VADER.polarity_scores(text)["compound"] if VADER else 0.0
    tag = "BULLISH" if s >= 0.05 else "BEARISH" if s <= -0.05 else "NEUTRAL"
    return round(s, 3), tag

def news_from_newsapi(query):
    r = requests.get("https://newsapi.org/v2/everything", timeout=8, params={
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": 12, "apiKey": NEWS_API_KEY})
    items = []
    for a in r.json().get("articles", []):
        title = a.get("title") or ""
        if not title or "[Removed]" in title: continue
        items.append({"title": title, "url": a.get("url", "#"),
                      "source": (a.get("source") or {}).get("name", ""),
                      "desc": (a.get("description") or "")[:160],
                      "published": a.get("publishedAt", "")})
    return items

def news_from_google(query):
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        title = it.findtext("title", "")
        source = it.findtext("source", "")
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        items.append({"title": title, "url": it.findtext("link", "#"),
                      "source": source, "desc": "",
                      "published": it.findtext("pubDate", "")})
        if len(items) >= 12: break
    return items

@app.route("/api/news")
def get_news():
    stock = request.args.get("stock", "").upper().strip()
    if stock:
        name = NEWS_NAMES.get(stock) or all_stocks().get(stock, {}).get("name", stock)
        for suffix in (" Limited", " Ltd.", " Ltd"):
            name = name.replace(suffix, "")
        query = f"{name} share price"
    else:
        query = "Sensex Nifty stock market"
    def load():
        try:
            items = news_from_newsapi(query) if NEWS_API_KEY else news_from_google(query)
        except Exception as e:
            print(f"  [!] news: {e}")
            return None
        for n in items:
            n["score"], n["tag"] = score_headline(n["title"])
            try:
                dt = parsedate_to_datetime(n["published"]) if "," in n["published"] \
                    else pd.Timestamp(n["published"]).to_pydatetime()
                n["time"], n["ts"] = dt.astimezone().strftime("%d %b %H:%M"), dt.timestamp()
            except Exception:
                n["time"], n["ts"] = "", 0
        return sorted(items, key=lambda n: n["ts"], reverse=True)
    return jsonify(cached(f"news:{stock}", 300, load) or [])

# ============================================================
#  IPO CENTER — ongoing / upcoming / closed IPOs with live GMP
#  Main source : investorgain.com live GMP report (GMP + full timetable)
#  Fallback    : NSE official IPO API (dates, price, subscription — no GMP)
#  Refund date is not published in either source, so it is calculated with
#  SEBI's T+3 rule: T = close, T+1 allotment, T+2 refunds, T+3 listing
# ============================================================
IPO_GMP_URL = "https://www.investorgain.com/report/live-ipo-gmp/331/"
IPO_SITE = "https://www.investorgain.com"

def ist_today():
    return datetime.now(IST).date()

def _strip(fragment):
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment or ""))).strip()

def _cf_decode(m):
    """Undo Cloudflare 'email protection', which hides text like 'L@450' on the page."""
    h = m.group(2)
    k = int(h[:2], 16)
    return "".join(chr(int(h[i:i + 2], 16) ^ k) for i in range(2, len(h), 2))

def _num(s):
    m = re.search(r"-?\d[\d,]*\.?\d*", s or "")
    return float(m.group(0).replace(",", "")) if m else None

def _ipo_day(s, today):
    """'5-Oct' → date. The page has no year, so pick the year closest to today."""
    m = re.match(r"\s*(\d{1,2})-([A-Za-z]{3})", s or "")
    if not m:
        return None
    best = None
    for y in (today.year - 1, today.year, today.year + 1):
        try:
            d = datetime.strptime(f"{m.group(1)}-{m.group(2)}-{y}", "%d-%b-%Y").date()
        except ValueError:
            continue
        if best is None or abs((d - today).days) < abs((best - today).days):
            best = d
    return best

# Fixed-date NSE/BSE trading holidays (festival holidays like Diwali/Holi change every year)
FIXED_HOLIDAYS = {(1, 26), (5, 1), (8, 15), (10, 2), (12, 25)}

def _next_workday(d):
    d += timedelta(days=1)
    while d.weekday() >= 5 or (d.month, d.day) in FIXED_HOLIDAYS:   # skip weekends + holidays
        d += timedelta(days=1)
    return d

def _refund_day(allotment, listing):
    """T+2: first working day after allotment (must be before listing)."""
    if not allotment:
        return None
    r = _next_workday(allotment)
    return r if (listing is None or r < listing) else None

def _status_by_dates(today, open_d, close_d, listing):
    bidding_over = datetime.now(IST).hour >= 17          # IPO bidding ends ~5 PM on the last day
    if open_d and today < open_d:
        return "upcoming"
    if close_d and (today < close_d or (today == close_d and not bidding_over)):
        return "open"
    if listing and today >= listing:
        return "listed"
    return "closed" if close_d else "upcoming"

def _ipo_symbol(name):
    """Find the stock symbol of a listed IPO so the user can analyse it."""
    key = re.sub(r"[^A-Z0-9 ]", "", name.upper()).strip()
    if len(key) < 3:
        return None
    for sym, info in all_stocks().items():
        if re.sub(r"[^A-Z0-9 ]", "", info["name"].upper()).startswith(key):
            return sym
    return None

def _ipo_row(name, kind, status, price, lot, size_cr, sub, gmp, gmp_pct, gmp_low, gmp_high,
             open_d, close_d, allot_d, listing_d, listed_price=None, listing_gain=None,
             allotted=False, updated="", url="", symbol=None):
    iso = lambda d: d.isoformat() if d else None
    return {
        "name": name, "kind": kind, "type": "Mainboard" if kind == "IPO" else "SME",
        "status": status, "allotted": allotted,
        "price": price, "lot": lot, "min_invest": round(price * lot) if price and lot else None,
        "size_cr": size_cr, "sub": sub,
        "gmp": gmp, "gmp_pct": gmp_pct, "gmp_low": gmp_low, "gmp_high": gmp_high,
        "est_listing": round(price + gmp, 2) if price and gmp is not None else None,
        "open": iso(open_d), "close": iso(close_d), "allotment": iso(allot_d),
        "refund": iso(_refund_day(allot_d, listing_d)), "listing": iso(listing_d),
        "listed_price": listed_price, "listing_gain": listing_gain,
        "updated": updated, "url": url, "symbol": symbol,
    }

def _ipos_investorgain():
    r = _get(IPO_GMP_URL, headers={
        "User-Agent": BROWSER_HEADERS["User-Agent"], "Accept": "text/html", "Accept-Language": "en-US,en;q=0.9"})
    r.raise_for_status()
    page = re.sub(r'<(a|span)[^>]*data-cfemail="([0-9a-f]+)"[^>]*>.*?</\1>', _cf_decode, r.text, flags=re.S)
    i = page.find("<table")
    if i < 0:
        return None
    table = page[i:page.find("</table>", i)]
    today = ist_today()
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        cells = dict(re.findall(r'<td[^>]*data-label="([^"]+)"[^>]*>(.*?)</td>', tr, re.S))
        if "Name" not in cells:
            continue
        name_html = cells["Name"]
        link = re.search(r'<a href="(/gmp/[^"]+)"[^>]*>(.*?)</a>', name_html, re.S)
        name = _strip(link.group(2)) if link else _strip(name_html)
        badges = [b.strip() for b in re.findall(r'class="badge[^"]*">([^<]+)</span>', name_html)]
        kind = badges[0] if badges else "IPO"            # IPO (mainboard) / NSE SME / BSE SME
        flags = badges[1:]                               # U / O / C / Allotted
        listed = re.search(r"L@\s*([\d.,]+)\s*\(\s*(-?[\d.]+)%", _strip(name_html))

        gmp_txt = _strip(cells.get("GMP"))
        g = re.search(r"₹\s*(--|-?[\d.]+)", gmp_txt)
        pct = re.search(r"\((-?[\d.]+)%\)", gmp_txt)
        rng = re.search(r"(-?[\d.]+)\s*↓\s*/\s*(-?[\d.]+)\s*↑", gmp_txt)
        gmp = float(g.group(1)) if g and g.group(1) != "--" else None

        day = lambda label: _ipo_day(_strip((cells.get(label) or "").split("<br")[0]), today)
        open_d, close_d, allot_d, listing_d = day("Open"), day("Close"), day("BoA Dt"), day("Listing")

        if listed:
            status = "listed"
        elif "O" in flags:
            status = "open"
        elif "U" in flags:
            status = "upcoming"
        elif "C" in flags:
            status = "closed"
        else:
            status = _status_by_dates(today, open_d, close_d, listing_d)

        price = _num(_strip(cells.get("Price (₹)"))) or None
        lot = _num(_strip(cells.get("Lot")))
        out.append(_ipo_row(
            name=name, kind=kind, status=status,
            price=price, lot=int(lot) if lot else None,
            size_cr=_num(_strip(cells.get("IPO Size"))), sub=_num(_strip(cells.get("Sub"))),
            gmp=gmp, gmp_pct=float(pct.group(1)) if pct and gmp is not None else None,
            gmp_low=float(rng.group(1)) if rng else None, gmp_high=float(rng.group(2)) if rng else None,
            open_d=open_d, close_d=close_d, allot_d=allot_d, listing_d=listing_d,
            listed_price=float(listed.group(1).replace(",", "")) if listed else None,
            listing_gain=float(listed.group(2)) if listed else None,
            allotted="Allotted" in flags, updated=_strip(cells.get("Updated-On")),
            url=IPO_SITE + link.group(1) if link else IPO_SITE,
            symbol=_ipo_symbol(name) if listed else None))
    return out or None

def _ipos_nse():
    """Fallback: official NSE data. No GMP; allotment/refund/listing estimated with T+3."""
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_HEADERS["User-Agent"], "Accept": "application/json, text/plain, */*",
                      "Referer": "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"})
    _get("https://www.nseindia.com", total=20, session=s)   # sets the cookies NSE requires
    base = "https://www.nseindia.com/api/"
    today = ist_today()
    nse_day = lambda t: datetime.strptime(t.strip(), "%d-%b-%Y").date() if t and t.strip() not in ("-", "") else None
    out, seen = [], set()

    def add(r, start, end, price_txt, sub=None, listing_txt=None):
        sym = (r.get("symbol") or "").strip()
        if not sym or sym in seen:
            return
        seen.add(sym)
        open_d, close_d = nse_day(start), nse_day(end)
        allot_d = _next_workday(close_d) if close_d else None
        listing_d = nse_day(listing_txt) or (_next_workday(_next_workday(allot_d)) if allot_d else None)
        prices = re.findall(r"\d+(?:\.\d+)?", price_txt or "")
        out.append(_ipo_row(
            name=(r.get("companyName") or r.get("company") or sym).replace(" Limited", ""),
            kind="IPO" if (r.get("series") or r.get("securityType")) == "EQ" else "NSE SME",
            status=_status_by_dates(today, open_d, close_d, listing_d),
            price=float(prices[-1]) if prices else None, lot=None, size_cr=None,
            sub=round(float(sub), 2) if sub not in (None, "") else None,
            gmp=None, gmp_pct=None, gmp_low=None, gmp_high=None,
            open_d=open_d, close_d=close_d, allot_d=allot_d, listing_d=listing_d,
            url=f"https://www.nseindia.com/market-data/all-upcoming-issues-ipo", symbol=sym))

    for r in _get(base + "ipo-current-issue", total=20, session=s).json() + \
             _get(base + "all-upcoming-issues?category=ipo", total=20, session=s).json():
        if r.get("series") in ("EQ", "SME"):
            add(r, r.get("issueStartDate"), r.get("issueEndDate"), r.get("issuePrice"), r.get("noOfTime"))
    past = [r for r in _get(base + "public-past-issues", total=20, session=s).json() if r.get("securityType") in ("EQ", "SME")]
    for r in past[:40]:
        add(r, r.get("ipoStartDate"), r.get("ipoEndDate"), r.get("issuePrice") if (r.get("issuePrice") or "-").strip() != "-"
            else r.get("priceRange"), listing_txt=r.get("listingDate"))
    return out or None

# ── Allotment status: every IPO has a registrar whose website shows allotment results ──
# Official allotment-status pages of the common registrars (used when the IPO page has no direct link)
REGISTRAR_LINKS = [
    ("kfin",         "https://ipostatus.kfintech.com/"),
    ("intime",       "https://in.mpms.mufg.com/Initial_Offer/public-issues.html"),  # MUFG Intime (formerly Link Intime)
    ("cameo",        "https://ipostatus1.cameoindia.com/"),
    ("bigshare",     "https://ipo.bigshareonline.com/IPO_Status.html"),
    ("skyline",      "https://www.skylinerta.com/ipo.php"),
    ("mas services", "https://www.masserv.com/opt.asp"),
    ("purva",        "https://www.purvashare.com/investor-service/ipo-query"),
    ("maashitla",    "https://maashitla.com/allotment-status/public-issues"),
]
BSE_ALLOTMENT = "https://www.bseindia.com/investors/appli_check.aspx"   # official BSE allotment check
IPO_DETAIL_RE = re.compile(r"^https://www\.investorgain\.com/gmp/[a-z0-9-]+/\d+/$")
_registrars = {}          # IPO detail page -> {"registrar": name, "url": allotment page}
_prefetching = threading.Lock()

def ipo_registrar(detail_url):
    """Registrar name + allotment-status page for one IPO (cached — an IPO's registrar never changes)."""
    if not IPO_DETAIL_RE.match(detail_url or ""):
        return None
    if detail_url in _registrars:
        return _registrars[detail_url]
    page = _get(detail_url, total=20, headers={
        "User-Agent": BROWSER_HEADERS["User-Agent"], "Accept": "text/html"}).text
    names = re.findall(r'ipo-registrar-review/\d+/\d+/\\?"\s+title=\\?"([^"\\]+?) Review', page)
    direct = [u for u in re.findall(r'\\?"ipo_allotment_url\\?"\s*:\s*\\?"([^"\\]*)', page) if re.match(r"^https?://", u)]
    name = names[0] if names else ""
    url = direct[0] if direct else next((u for key, u in REGISTRAR_LINKS if key in name.lower()), "")
    info = {"registrar": name, "url": url}
    with _lock:
        _registrars[detail_url] = info
    return info

def _prefetch_registrars(rows):
    """Look up registrars for open / closed IPOs in the background, 1 page per second."""
    if not _prefetching.acquire(blocking=False):
        return
    try:
        for r in rows:
            if r["status"] in ("open", "closed") and r["url"] not in _registrars and IPO_DETAIL_RE.match(r["url"] or ""):
                try:
                    ipo_registrar(r["url"])
                except Exception as e:
                    print(f"  [!] registrar for {r['name']}: {e}")
                time.sleep(1)
    finally:
        _prefetching.release()

def _short_registrar(name):
    if not name:
        return None
    words = name.replace(".", " ").split()
    return " ".join(words[:2]) if words[0].upper() == "MUFG" else words[0]

_ipo_tried = False   # has the background thread finished at least one attempt?

def refresh_ipos():
    """Fetch IPO data (investorgain → NSE fallback). Runs in the background thread, never on a user request."""
    global _ipo_tried
    for source, fn in (("investorgain.com", _ipos_investorgain), ("nseindia.com", _ipos_nse)):
        tag = "ipo_" + source.split(".")[0]
        try:
            rows = fn()
            if rows:
                store("ipos", {"source": source, "rows": rows,
                               "fetched": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")})
                _mark(tag, True, f"{len(rows)} IPOs")
                _ipo_tried = True
                if source == "investorgain.com":
                    _prefetch_registrars(rows)        # registrar links for open / closed IPOs
                return
            _mark(tag, False, "no IPO rows found")
        except Exception as e:
            _mark(tag, False, e)
    _ipo_tried = True

def _ipo_worker():
    while True:
        refresh_ipos()
        time.sleep(600)                    # every 10 minutes

def fetch_ipos():
    return peek("ipos")

def _with_registrar(r):
    info = _registrars.get(r["url"]) or {}
    allot = info.get("url") or None
    if not allot and not IPO_DETAIL_RE.match(r["url"] or "") and r["kind"] == "IPO":
        allot = BSE_ALLOTMENT                      # NSE-fallback rows: BSE's official allotment check
    return dict(r, registrar=info.get("registrar") or None,
                registrar_short=_short_registrar(info.get("registrar")), allotment_url=allot)

@app.route("/api/ipo")
def ipo():
    d = fetch_ipos()
    if not d:
        return jsonify({"error": "unavailable" if _ipo_tried else "loading"}), 503
    rows = [_with_registrar(r) for r in d["rows"]]
    far = "9999-12-31"
    pick = lambda st: [r for r in rows if r["status"] == st]
    return jsonify({
        "source": d["source"], "fetched": d["fetched"], "has_gmp": d["source"] == "investorgain.com",
        "source_url": IPO_GMP_URL if d["source"] == "investorgain.com" else "https://www.nseindia.com/market-data/all-upcoming-issues-ipo",
        "open":     sorted(pick("open"), key=lambda r: r["close"] or far),                     # closing soonest first
        "upcoming": sorted(pick("upcoming"), key=lambda r: r["open"] or far),                  # opening soonest first
        "closed":   sorted(pick("closed"), key=lambda r: r["listing"] or r["close"] or far),   # listing soonest first
        "listed":   sorted(pick("listed"), key=lambda r: r["listing"] or "", reverse=True),    # newest listing first
    })

@app.route("/api/ipo/allotment")
def ipo_allotment():
    """Send the user to the right registrar's allotment-status page for this IPO."""
    detail = request.args.get("u", "")
    try:
        info = ipo_registrar(detail)
    except Exception as e:
        print(f"  [!] allotment lookup: {e}")
        info = None
    target = (info or {}).get("url") or (detail if IPO_DETAIL_RE.match(detail) else BSE_ALLOTMENT)
    if not re.match(r"^https?://", target):
        target = BSE_ALLOTMENT
    return redirect(target, code=302)

# ============================================================
#  ROUTE — POST /api/order  (simulated unless Groww token set)
# ============================================================
@app.route("/api/order", methods=["POST"])
def place_order():
    body = request.get_json()
    symbol     = body.get("symbol", "").upper()
    order_type = body.get("type", "BUY")
    quantity   = body.get("quantity", 1)
    price      = body.get("price", 0)

    if not symbol or not quantity:
        return jsonify({"success": False, "message": "Symbol and quantity required"}), 400

    if not is_groww_configured():
        return jsonify({
            "success":  True,
            "order_id": f"SIM{int(time.time())}",
            "symbol":   symbol, "type": order_type,
            "quantity": quantity, "price": price,
            "status":   "SIMULATED",
            "message":  f"Simulated {order_type} order for {quantity} shares of {symbol} at ₹{price}. Add Groww API key for real orders."
        })

    try:
        order_payload = {
            "trading_symbol":   symbol,
            "exchange":         "NSE",
            "segment":          "CASH",
            "transaction_type": order_type,
            "order_type":       "MARKET",
            "quantity":         quantity,
            "product":          "INTRADAY",
        }
        res = requests.post(f"{GROWW_BASE_URL}/orders/place", headers=HEADERS, json=order_payload, timeout=10)
        data = res.json()
        if res.status_code == 200 and data.get("status") == "SUCCESS":
            return jsonify({
                "success":  True,
                "order_id": data.get("payload", {}).get("order_id", ""),
                "symbol":   symbol, "type": order_type, "quantity": quantity,
                "status":   "PLACED",
                "message":  f"✅ {order_type} order placed for {quantity} shares of {symbol}"
            })
        return jsonify({"success": False, "message": data.get("message", "Order failed")}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/portfolio/groww")
def get_groww_portfolio():
    if not is_groww_configured():
        return jsonify({"message": "Add Groww API key to see real portfolio", "holdings": []})
    try:
        res = requests.get(f"{GROWW_BASE_URL}/portfolio/holdings", headers=HEADERS, timeout=10)
        holdings = res.json().get("payload", {}).get("holdings", [])
        return jsonify({"holdings": holdings, "source": "groww"})
    except Exception as e:
        return jsonify({"error": str(e), "holdings": []}), 500

# ============================================================
#  WEBSITE — open http://<this-pc-ip>:5000 on any device on your Wi-Fi
#  (only the website files are served — never server.py itself)
# ============================================================
SITE_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_FILES = {"index.html", "login.html", "style.css", "script.js", "login.js", "login.css",
              # phone-app (PWA) files
              "manifest.json", "sw.js", "icon-192.png", "icon-512.png", "apple-touch-icon.png"}
MIME = {".js": "text/javascript", ".json": "application/manifest+json", ".png": "image/png"}

@app.route("/")
def site_home():
    return send_from_directory(SITE_DIR, "login.html")

@app.route("/<name>")
def site_file(name):
    if name not in SITE_FILES:
        abort(404)
    resp = send_from_directory(SITE_DIR, name, mimetype=MIME.get(os.path.splitext(name)[1]))
    if name in ("sw.js", "index.html", "login.html", "manifest.json"):
        resp.headers["Cache-Control"] = "no-cache"      # phones always pick up new versions
    return resp

@app.route("/api/status")
def get_status():
    return jsonify({
        "backend":    "running",
        "data":       "yahoo_finance (real NSE prices, ~15 min delay possible)",
        "groww_api":  "connected" if is_groww_configured() else "not configured — orders simulated",
        "version":    "TradePulse v3.2"
    })

@app.route("/api/health")
def health():
    """Which outside data sources work right now — useful when the site runs on Render."""
    try:
        import resource                                   # Linux only (Render)
        peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        peak_mb = None
    with _lock:
        movers_known = len(_moves)
    d = peek("ipos")
    out = {
        "sources": _health,
        "stocks_loaded": len(all_stocks()), "movers_known": movers_known,
        "ipo_source": d["source"] if d else None, "registrars_known": len(_registrars),
        "peak_memory_mb": peak_mb, "on_render": ON_RENDER, "pid": os.getpid(),
    }
    if request.args.get("debug"):    # where is each background thread right now?
        import sys, traceback
        frames = sys._current_frames()
        out["threads"] = {
            t.name: ([f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
                      for f in traceback.extract_stack(frames[t.ident])][-8:] if t.ident in frames else "not running")
            for t in threading.enumerate() if t.name.endswith("worker")}
        out["workers_started"] = _workers_started
    return jsonify(out)

# Start the background threads inside the process that answers visitors, on its first request.
# (Starting them while the module loads breaks under gunicorn --preload, which Render uses: the threads
#  would live in gunicorn's master process, and the worker that serves visitors would never see their data.)
_workers_started = []
_workers_pid = None
_workers_lock = threading.Lock()

@app.before_request
def _start_workers_once():
    global _workers_pid
    if _workers_pid == os.getpid():
        return
    with _workers_lock:
        if _workers_pid == os.getpid():
            return
        _workers_pid = os.getpid()
        for fn, name in ((_movers_worker, "movers-worker"), (_ipo_worker, "ipo-worker")):
            threading.Thread(target=fn, name=name, daemon=True).start()
            _workers_started.append(f"{name} (pid {os.getpid()})")


if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "YOUR-PC-IP"
    print("\n  ✅ TradePulse backend — REAL prices from Yahoo Finance")
    print("  💻 On this PC : http://localhost:5000")
    print(f"  📱 On phone   : http://{lan_ip}:5000   (same Wi-Fi)")
    print("  🌐 API status : http://localhost:5000/api/status")
    print("  📊 http://localhost:5000/api/live/all\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
