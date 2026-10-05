# TradePulse — AI Trading Platform

An intraday trading dashboard for Indian stocks (NSE/BSE) with real market prices, technical-indicator signals, news sentiment and an IPO center with live GMP.

**Live site:** https://tradepulse-yx71.onrender.com

## Features
- **Real prices for ~5,300 Indian companies**: every active NSE and BSE stock via Yahoo Finance (`yfinance`)
- **Search by symbol, company name or BSE code**, with autocomplete (e.g. `tata steel`, `TATASTEEL`, `500325`)
- **Live indices**: SENSEX, NIFTY 50 and BANK NIFTY with intraday sparklines
- **Signal analyser**: BUY / SELL / HOLD from RSI(14), MACD(12,26,9) and EMA20, with ATR-based entry, stop loss and target
- **Intraday chart** (1m / 5m / 15m / 1h)
- **Market movers**: top 25 gainers and losers for NIFTY 50, NIFTY 200, NIFTY 500 or all of India (NSE + BSE)
- **IPO center**: ongoing, upcoming, closed and listed IPOs (mainboard and SME) with live GMP, subscription, price, lot size, open / close / allotment / refund / listing dates, and a one-click link to each IPO's registrar to check allotment status
- **News sentiment**: latest headlines scored with VADER (bullish / bearish / neutral)
- Watchlist, portfolio P&L tracker and price alerts

## Tech stack
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Backend:** Python, Flask, yfinance, pandas, VADER
- **Hosting:** Render (gunicorn)

## How to run
1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the backend and keep this terminal open:
   ```
   python server.py
   ```
3. Open http://localhost:5000 and click **DEMO ACCESS**.

The footer shows **REAL NSE · YAHOO FINANCE** when the website is connected to the backend.

## API endpoints
| Endpoint | Description |
|---|---|
| `/api/live/all` | Quotes + signals for the watchlist |
| `/api/live/<SYMBOL>` | Quote + signal for any NSE or BSE stock (symbol, name or BSE code) |
| `/api/search?q=<text>` | Search all NSE + BSE companies |
| `/api/history/<SYMBOL>?interval=5m` | Intraday prices for the chart |
| `/api/indices` | SENSEX, NIFTY 50, BANK NIFTY |
| `/api/movers?universe=NIFTY50\|NIFTY200\|NIFTY500\|ALL&n=25` | Top gainers and losers |
| `/api/ipo` | Ongoing, upcoming, closed and listed IPOs with GMP, timetable and registrar |
| `/api/ipo/allotment?u=<IPO page>` | Redirects to that IPO registrar's allotment-status page |
| `/api/news?stock=<SYMBOL>` | Headlines with sentiment scores |

## Data sources and notes
- Prices: Yahoo Finance; data can be delayed by up to ~15 minutes.
- Stock lists: official NSE equity list and BSE active scrips list, merged by ISIN.
- IPO data and GMP: [investorgain.com](https://www.investorgain.com/report/live-ipo-gmp/331/); falls back to NSE's official IPO data (no GMP) if unavailable.
- **GMP (grey market premium) is unofficial and unregulated** — an indicator, not a guarantee.
- Refund date is calculated as T+2 under SEBI's T+3 listing timeline (one working day after allotment).
- Orders are **simulated** unless a Groww API token is set in `server.py`. The order book (market depth) is simulated.

> ⚠️ For educational purposes only. Not financial advice.
