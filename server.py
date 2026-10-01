# ============================================================
#  TradePulse Backend — server.py
#  REAL market data via Yahoo Finance (yfinance) — same source
#  as Market_sentement/sentiment_analyzer.py
#  Run: python server.py   →   http://localhost:5000
# ============================================================

from flask import Flask, jsonify, request
from flask_cors import CORS
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import threading
import requests
import time
import os

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

def to_yahoo(symbol):
    symbol = symbol.upper().strip()
    if symbol in STOCKS:
        return STOCKS[symbol]
    if symbol.startswith("^") or "." in symbol:
        return symbol
    return symbol + ".NS"

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
def _download(tickers, **kw):
    return yf.download(tickers, group_by="ticker", progress=False,
                       threads=True, auto_adjust=False, **kw)

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

def fetch_quote(symbol):
    symbol = symbol.upper()
    if symbol in STOCKS:
        q = fetch_all_quotes().get(symbol)
        if q: return q
    def load():
        df = yf.Ticker(to_yahoo(symbol)).history(period="1y", interval="1d", auto_adjust=False)
        return build_quote(symbol, df) if not df.empty else None
    return cached(f"quote:{symbol}", 20, load)

def fetch_history(symbol, interval):
    period = {"1m": "1d", "5m": "1d", "15m": "5d", "1h": "1mo"}.get(interval, "1d")
    def load():
        df = yf.Ticker(to_yahoo(symbol)).history(period=period, interval=interval)
        df = df.dropna(subset=["Close"])
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
        return jsonify({"error": f"No data found for {symbol.upper()} on NSE"}), 404
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

@app.route("/api/gainers")
def get_gainers():
    q = sorted(fetch_all_quotes().values(), key=lambda x: x["change"], reverse=True)
    return jsonify([{"symbol": x["symbol"], "price": x["price"], "change_pct": x["change"],
                     "volume": x["volume"]} for x in q if x["change"] > 0][:5])

@app.route("/api/losers")
def get_losers():
    q = sorted(fetch_all_quotes().values(), key=lambda x: x["change"])
    return jsonify([{"symbol": x["symbol"], "price": x["price"], "change_pct": x["change"],
                     "volume": x["volume"]} for x in q if x["change"] < 0][:5])

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
    query = f"{NEWS_NAMES.get(stock, stock)} share price" if stock else "Sensex Nifty stock market"
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

@app.route("/api/status")
def get_status():
    return jsonify({
        "backend":    "running",
        "data":       "yahoo_finance (real NSE prices, ~15 min delay possible)",
        "groww_api":  "connected" if is_groww_configured() else "not configured — orders simulated",
        "version":    "TradePulse v3.1"
    })


if __name__ == "__main__":
    print("\n  ✅ TradePulse backend — REAL prices from Yahoo Finance")
    print("  🌐 http://localhost:5000/api/status")
    print("  📊 http://localhost:5000/api/live/all\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
