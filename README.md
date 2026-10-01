# TradePulse — AI Trading Platform

An intraday trading dashboard for Indian stocks (NSE/BSE) with real market prices, technical-indicator signals and news sentiment.

## Features
- **Real prices** for any NSE stock via Yahoo Finance (`yfinance`)
- **Live indices**: SENSEX, NIFTY 50 and BANK NIFTY with intraday sparklines
- **Signal analyser**: BUY / SELL / HOLD from RSI(14), MACD(12,26,9) and EMA20, with ATR-based entry, stop loss and target
- **Intraday chart** (1m / 5m / 15m / 1h)
- **Market movers**, watchlist, portfolio P&L tracker and price alerts
- **News sentiment**: latest headlines scored with VADER (bullish / bearish / neutral)

## Tech stack
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Backend:** Python, Flask, yfinance, pandas, VADER

## How to run
1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the backend and keep this terminal open:
   ```
   python server.py
   ```
3. Open `login.html` (e.g. VS Code → right-click → **Open with Live Server**) and click **DEMO ACCESS**.

The footer shows **REAL NSE · YAHOO FINANCE** when the website is connected to the backend.

## API endpoints
| Endpoint | Description |
|---|---|
| `/api/live/all` | Quotes + signals for the watchlist |
| `/api/live/<SYMBOL>` | Quote + signal for any NSE stock |
| `/api/history/<SYMBOL>?interval=5m` | Intraday prices for the chart |
| `/api/indices` | SENSEX, NIFTY 50, BANK NIFTY |
| `/api/gainers`, `/api/losers` | Top movers |
| `/api/news?stock=<SYMBOL>` | Headlines with sentiment scores |

## Notes
- Yahoo Finance data can be delayed by up to ~15 minutes.
- Orders are **simulated** unless a Groww API token is set in `server.py`.
- The order book (market depth) is simulated.

> ⚠️ For educational purposes only. Not financial advice.
