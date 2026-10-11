# TradePulse ML — training pipeline

Goal: a reproducible swing-trading model (hold about 5 trading days) for NIFTY 200 stocks, tested honestly
on past data with all trading costs subtracted. This folder is separate from the website: nothing here runs
on the live server, and the server never retrains a model.

**Status: steps 1–3 are done. Model `xgb_v1` is trained and saved; it has NOT been evaluated on the test period or backtested yet.**

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

## Step 2 — features and label

```bash
python training/build_dataset.py
```

Writes `training/data/dataset_v1.csv.gz` (one row per company per day) and `dataset_manifest.json`.

**What the model will be asked** (all settings are in `training/config.py`):

- Decision after the close of day *t*, using data up to day *t* only.
- Trade being judged: buy at the open of day *t+1*, sell at the open 5 trading days later (dividends included).
- `label = 1` if that return is in the top 30% of eligible stocks that day, else 0.
- Eligible = at least 252 days of history, at least Rs 1 crore traded per day (20-day average), traded that day.

**Features** (`ml/features.py`, 39 numbers): past returns over 1 to 252 days, RSI, MACD, distance from
EMA 20 / EMA 50 / SMA 200, ATR, volatility, volume and turnover, distance from the 52-week high and low,
NIFTY 50 returns and trend, India VIX, stock-minus-market returns, and the stock's rank among all stocks that day.
The same file will be used by the live server later, so training and live numbers are calculated identically.

**No future information in the features** — this is tested, not assumed (`tests/test_features.py`):
features for a day are identical when all later days are deleted from the data, and do not change when
future prices are changed. The tests were also checked the other way round: four deliberate look-ahead
mistakes were put into the code and the tests failed each time.

**Unadjusted corporate actions.** Four price series are cut in two at a date listed in `config.PRICE_BREAKS`
(Tata Motors and Vedanta demergers, a Motilal Oswal bonus issue, Patanjali's relisting). No feature or label
crosses the cut. Large moves that were real (for example Yes Bank in March 2020) are kept.

**Split by time, never shuffled:** train to 2020, validation 2021–2022, test 2023 onwards, with a 6-day gap
between periods because a label looks 6 days ahead.

## Step 3 — train and save the model

```bash
python training/train_model.py
```

Writes `models/xgb_v1/`: `model.json` (the model), `feature_schema.json` (the 39 features, in order) and
`metadata.json` (target, periods, settings, library versions, input checksums, validation metrics).
Training takes a few seconds on a CPU; `--device cuda` works but is not needed at this data size.

- Learns from the train period only. The validation period only decides when to stop adding trees.
- The test period is not read by this script.
- Settings are fixed defaults, not the winner of a large search (which would overfit the validation data).
- Same data + same code = the same `model.json` (checked by training twice and comparing checksums).

**Validation result of xgb_v1 (2021–2022), before any costs:**

| Ranking by | Winners among its top 10% picks | Rank correlation with the 5-day return |
|---|---|---|
| Nothing (base rate) | 30.3% | 0 |
| xgb_v1 score | 37.5% | 0.028 |
| Volatility alone (`atr_14`) | 36.0% | 0.003 |
| 5-day reversal alone (`-ret_5`) | 32.5% | 0.031 |

The model is better than chance, but only slightly better than two one-line rules. Much of its hit rate comes
from choosing volatile stocks, which are more often in the top 30% and also more often in the bottom.
Whether anything is left after trading costs is the question for steps 4 and 5.

## Tests

```bash
python -m pytest tests -q
```

## Next steps (not built yet)

4. Evaluation on the untouched test period, against the existing rule signal and simple baselines
5. Backtest with brokerage, taxes and slippage
