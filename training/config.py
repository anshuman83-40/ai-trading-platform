"""TradePulse ML — every decision about the model's task in one place (so it is written down, not hidden in code)."""

# ── What the model predicts ──
# Decision is made after the close of day t, using data up to day t only.
# Trade that is being judged: buy at the OPEN of day t+1, sell at the OPEN of day t+1+HORIZON.
HORIZON = 5                 # trading days the stock is held
# Label = 1 if the stock's return over that trade is in the top TOP_FRACTION of all eligible stocks that day.
TOP_FRACTION = 0.30
TARGET = (f"1 if the {HORIZON}-trading-day return (next day's open to the open {HORIZON} trading days later, "
          f"dividends included) is in the top {int(TOP_FRACTION * 100)}% of eligible NIFTY 200 stocks that day, else 0")

# ── Which rows are eligible (decided with past data only) ──
MIN_HISTORY = 252           # trading days of history before a stock's rows are used (long features need a year)
MIN_TURNOVER = 1e7          # average rupees traded per day over the last 20 days (Rs 1 crore): must be easy to buy/sell
MIN_STOCKS_PER_DAY = 30     # days with fewer eligible stocks get no labels (ranking would be meaningless)

# ── Price breaks: corporate actions Yahoo did NOT adjust for ──
# Found by the >40% one-day-move check in data_collection.py and reviewed by hand on 11 Oct 2026.
# The price series is cut into two separate series at each date; no feature or label crosses the cut.
# (Large moves that were REAL — PNB Oct 2017, INDUSINDBK and YESBANK Mar 2020 — are deliberately not listed.)
PRICE_BREAKS = {
    "TMPV":       ["2025-10-14"],   # Tata Motors demerger: 660.75 -> 395.45 overnight
    "VEDL":       ["2026-04-30"],   # Vedanta demerger: 773.60 -> 271.55 overnight
    "MOTILALOFS": ["2024-01-01"],   # unadjusted bonus issue: 1240.85 -> 315.01 overnight
    "PATANJALI":  ["2020-01-27"],   # relisting after insolvency: frozen at 111.67 with zero volume, then 5.67
}

# ── Chronological split (never shuffled). A gap of EMBARGO_DAYS is left between the periods because a label
#    made on the last training day looks HORIZON+1 days into the future. ──
TRAIN_END = "2020-12-31"    # train:  first usable date .. TRAIN_END
VALID_END = "2022-12-31"    # valid:  after TRAIN_END .. VALID_END   (for choosing settings)
                            # test:   after VALID_END .. last date   (looked at once, at the end)
EMBARGO_DAYS = HORIZON + 1
