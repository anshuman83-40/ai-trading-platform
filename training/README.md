# TradePulse ML — training pipeline

Goal: a reproducible swing-trading model (hold about 5 trading days) for NIFTY 200 stocks, tested honestly
on past data with all trading costs subtracted. This folder is separate from the website: nothing here runs
on the live server, and the server never retrains a model.

**Status: step 1 of the pipeline (data collection) is done. No model exists yet.**

## Setup (once)

```bash
pip install -r requirements-train.txt
```

## Step 1 — collect and check daily prices

```bash
python training/data_collection.py --end 2026-10-09
```

- `--end` fixes the last date so the run can be repeated. `--start` defaults to 2012-01-01.
- Quick try with a few stocks: add `--symbols RELIANCE,TCS,INFY`.

Output in `training/data/` (not committed to GitHub):

| File | What it is |
|---|---|
| `universe_nifty200.csv` | The company list used. Reused on later runs so the universe does not silently change. Delete it to fetch a fresh list. |
| `prices_daily.csv.gz` | One row per company per day: open, high, low, close, adj_close, volume, dividends, splits |
| `market_daily.csv.gz` | NIFTY 50 and India VIX, same format |
| `validation_report.json` | Every company's checks: history length, missing days, zero-volume days, suspicious jumps, used or left out |
| `manifest.json` | Settings, date range, row counts, file checksums, known limits |

### What the checks do

A company is **left out** if it has under 500 trading days, more than 5% of market days missing, more than 5%
of days with no volume, or missing / non-positive prices. One-day moves above 40% are **reported** (possible
unadjusted split or bad data) but not removed.

### Known limits

- **Survivorship bias.** Only today's NIFTY 200 members are downloaded. Companies that left the index are
  missing, so a backtest on this data looks better than reality.
- **Free data.** Yahoo Finance data is not exchange-licensed and can contain errors. Yahoo can also revise old
  adjusted prices, so compare the checksums in `manifest.json` between runs.
- **Daily only.** Free intraday history is too short to train on (5-minute bars: 60 days).

## Tests

```bash
python -m pytest tests -q
```

## Next steps (not built yet)

2. Features without future information + the label (5-day return, top 30% of the universe)
3. XGBoost v1, saved to `models/` with its metadata
4. Evaluation against the existing rule signal and simple baselines
5. Backtest with brokerage, taxes and slippage
