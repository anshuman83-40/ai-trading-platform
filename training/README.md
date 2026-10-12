# TradePulse ML — training pipeline

Goal: a reproducible swing-trading model (hold about 5 trading days) for NIFTY 200 stocks, tested honestly
on past data with all trading costs subtracted. This folder is separate from the website: nothing here runs
on the live server, and the server never retrains a model.

**Status: steps 1–6 are done. The model's ranking is shown in TradePulse as a research experiment. The backtest does NOT show a strategy that can be relied on to make money (see step 5).**

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

## Step 4 — evaluation on unseen data

```bash
python training/evaluate_model.py
```

Writes `models/xgb_v1/evaluation.json`. The test period (Jan 2023 – Sep 2026) was opened for the first time
on 12 Oct 2026, after the model and its settings were fixed. Nothing was tuned afterwards.

**Test period, picking the 10% highest-scored stocks each day, BEFORE trading costs:**

| Picking by | Winners among picks | 5-day return | Versus all stocks | t-stat | Rank IC |
|---|---|---|---|---|---|
| AI model (xgb_v1) | 37.6% | 0.95% | +0.48% | 3.4 | +0.033 |
| Most volatile (`atr_14`) | 37.3% | 0.91% | +0.44% | 2.6 | +0.016 |
| Best 6-month return | 35.7% | 0.69% | +0.22% | 1.5 | +0.009 |
| Fell most in 5 days | 31.6% | 0.56% | +0.09% | 0.6 | +0.021 |
| Website rule score | 31.2% | 0.44% | −0.03% | 0.5 | +0.001 |
| All eligible stocks | 30.2% | 0.47% | — | — | — |
| NIFTY 50 | — | 0.13% | — | — | — |

- **Website BUY / SELL rule:** BUY stocks did −0.05% versus all stocks (no skill); SELL stocks did −0.11%.
- **Year by year (model versus all stocks):** 2023 +0.55%, 2024 +0.58%, 2025 +0.31%, 2026 +0.49%.
- **Walk-forward** (model retrained each year on the past only): +0.56% versus all stocks, rank IC +0.033.
- **Calibration:** scores run only from about 0.27 to 0.35, and the real winner rate in those groups runs from
  0.25 to 0.37. The score is a weak ranking, not a confident probability.
- t-statistics use every 5th day only, because consecutive 5-day trades overlap.

**How to read this honestly**

- The model's picks beat the average stock by about half a percent per 5-day trade, in every test year.
- Simply buying the most volatile stocks did almost as well. So most of the model's result is "it prefers
  risky stocks during a rising market", which is extra risk, not skill. Its ranking (IC) is better than
  volatility alone, which is the part that may be real skill.
- A round trip in delivery costs roughly 0.25–0.4% in taxes, charges and slippage before brokerage. That
  would take most of a +0.48% edge.
- "All eligible stocks" earned 0.47% per 5 days, which is unusually high: these were strong years, and the
  data only contains companies that are in the NIFTY 200 today (survivorship bias).

## Step 5 — backtest with real costs

```bash
python training/backtest.py
```

Writes `models/xgb_v1/backtest.json`. Costs are in `training/costs.py` (Groww delivery charges and all taxes,
copied from https://groww.in/pricing on 12 Oct 2026).

**The strategy tested** (fixed before looking at results): every 5 trading days hold the 10 best-scored
eligible stocks (at most 3 per industry); trade at the next day's open with 0.05% slippage each side; equal
position sizes, whole shares, no borrowing; 10% stop-loss per position; if the portfolio falls 20% from its
high, sell everything and pause 20 trading days; dividends received; everything sold at the end.

**Test period 2 Jan 2023 – 9 Oct 2026, starting with Rs 1,00,000, AFTER costs:**

| Strategy | Per year | Worst fall | Sharpe | Winning trades | Profit factor | Costs per year | End value |
|---|---|---|---|---|---|---|---|
| AI model (xgb_v1) | +20.1% | −25.6% | 0.63 | 53% | 1.24 | 23.0% | Rs 1,96,680 |
| AI model, retrained each year | +38.7% | −25.8% | 1.20 | 51% | 1.45 | 25.9% | Rs 3,34,606 |
| Most volatile stocks | +15.5% | −33.0% | 0.44 | 46% | 1.22 | 14.1% | Rs 1,70,321 |
| Best 6-month return | +6.2% | −43.6% | 0.10 | 37% | 1.11 | 8.9% | Rs 1,24,703 |
| Website rule score | −12.2% | −45.6% | −1.14 | 38% | 0.82 | 18.6% | Rs 61,858 |
| Random stocks (average of 20 runs) | −15.9% | −54.1% | −1.44 | 43% | 0.80 | 25.7% | Rs 53,458 |
| Average eligible stock, **no costs** | +22.8% | −22.3% | 0.97 | — | — | 0 | Rs 2,13,675 |
| NIFTY 50 buy and hold, **no costs** | +5.9% | −15.8% | 0.01 | — | — | 0 | Rs 1,23,756 |

Sharpe = daily portfolio returns minus 6.5%/252, mean ÷ standard deviation × √252. "Costs per year" is a
percentage of the starting capital. Model by year: 2023 +32.9%, 2024 +31.2%, 2025 +14.3%, 2026 −1.3%.

**The same model with one thing changed:**

| Change | Per year | Worst fall |
|---|---|---|
| Start with Rs 25,000 | −15.4% | −56.4% |
| Start with Rs 5,00,000 | +29.9% | −27.4% |
| No stop-loss and no maximum-loss rule | +31.9% | −29.8% |
| Slippage 0.10% each side | +18.3% | −26.3% |
| Hold 20 stocks instead of 10 | +11.2% | −25.0% |
| No costs at all (not achievable) | +48.2% | −23.3% |
| Only stocks already trading Rs 50 crore a day | +14.9% | −27.3% |

**Why this is NOT proof that the model makes money**

1. **It did not beat doing nothing clever.** The average stock in the data rose 22.8% a year with no trading.
   The model made 20.1% after costs. Its picks were better before costs (+48%), but trading every 5 days cost
   about 23% of the capital per year and used the advantage up.
2. **The data is biased upwards (survivorship).** It contains only today's NIFTY 200 members. Several were
   small companies in 2023 and are in the index now *because* their price multiplied. One of them (GVT&D)
   produced Rs 40,312 of the Rs 96,680 profit; the 10 best trades produced Rs 77,046. Nobody using the NIFTY 200
   list in 2023 could have chosen those stocks. The Rs 50-crore check removes only part of this bias.
3. **The result swings widely with details that should not matter**: from −15% to +39% a year across the rows
   above. A real edge would be steadier.
4. **Small capital loses.** A Rs 2,500 position costs 1.64% per round trip (Rs 10,000: 0.69%; Rs 50,000: 0.36%).
   With Rs 25,000 the same strategy lost 15% a year.
5. **One period, mostly a rising market**, and looked at only once. 2026 was slightly negative.
6. **Not included:** income tax on short-term gains (20%), circuit-limit days, the real price of market orders
   at the open, stop-losses hit inside the day. Today's charges are applied to all past years.

What does hold up: the model's picks beat random picks with the same costs in all 20 random runs, and the
website's existing rule signal lost money. So the ranking has some information; it is not yet a strategy.

## Step 6 — predictions in TradePulse

```bash
python training/predict.py
```

Run it after the market closes (3:30 pm IST). It downloads recent prices for the 187 companies in
`models/xgb_v1/universe.json`, builds the features with the same code as training, scores them with the saved
model and writes `models/xgb_v1/predictions.json` plus a dated copy in `models/xgb_v1/history/`.
Then publish the new file:

```bash
git add models/xgb_v1/predictions.json models/xgb_v1/history
```

```bash
git commit -m "AI predictions update"
```

```bash
git push
```

**How the website uses it**

- The server never loads the model, builds features or trains. `ml/api.py` only reads the small JSON files,
  so the free server stays light (no xgboost is installed there).
- `GET /api/ml/predict/<symbol>` — rank, group (1–10), score, the website's rule signal for the same day,
  model version, horizon and the historical winner rate of that group.
- `GET /api/ml/top?n=10` — the highest-ranked stocks. `GET /api/ml/info` — model details and test results.
- If the file is more than 5 days old, missing or unreadable, the answer is `stale` / `missing` with **no score**.
- In the app: Analyse tab → "AI research score" card; Markets tab → "AI Research".

**Checks before a file is written:** today's unfinished candle is never used; prices must be at most 4 days
old; at least 80% of the companies need a price for the day; at least 100 stocks must be scorable.

**Same numbers live and in training.** For 9 Oct 2026 the live-style run (1,100 days of history) and the
training table (history from 2012) gave the same 185 stocks, the same top 10 and scores within 0.00005.

**What a stock's group means** (test period, winner = top 30% over the next 5 trading days):

| Group (10 = highest scores of the day) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Were winners | 24% | 27% | 27% | 28% | 29% | 30% | 33% | 33% | 35% | 38% |

## Tests

```bash
python -m pytest tests -q
```

## Next steps (not built yet)

7. Paper trading with virtual money — the only fair test left, because it uses data that does not exist yet
