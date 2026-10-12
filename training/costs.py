"""
TradePulse ML — what one buy or one sell really costs (equity DELIVERY, NSE).

Rates copied from Groww's public pricing page https://groww.in/pricing on 12 Oct 2026.
They are applied to every past year in the backtest (a simplification: some rates were different earlier).
Statutory rounding rules (for example STT rounded to the nearest rupee) are ignored — the effect is paise.

To test another broker, copy GROWW_DELIVERY, change the numbers and pass it as `plan`.
"""

GROWW_DELIVERY = {
    "name": "Groww equity delivery (NSE)",
    "as_of": "2026-10-12",
    "source": "https://groww.in/pricing",
    "brokerage_pct": 0.001,        # 0.1% of the order value …
    "brokerage_cap": 20.0,         # … but never more than Rs 20 …
    "brokerage_min": 5.0,          # … and never less than Rs 5, per executed order
    "stt_pct": 0.001,              # Securities Transaction Tax, 0.1% on BUY and on SELL
    "stamp_buy_pct": 0.00015,      # stamp duty, 0.015% on BUY only
    "exchange_pct": 0.0000297,     # NSE transaction charge, 0.00297% on both sides
    "sebi_pct": 0.000001,          # SEBI turnover charge, 0.0001% on both sides
    "ipft_pct": 0.000001,          # Investor Protection Fund Trust charge, 0.0001% on both sides
    "dp_sell": 20.0,               # per company per sell day: depository Rs 3.5 + Groww Rs 16.5
    "dp_min_value": 100.0,         # no DP charge when the shares sold are worth less than Rs 100
    "gst": 0.18,                   # on brokerage, DP, exchange, SEBI and IPFT charges (not on STT or stamp duty)
}

NO_COSTS = {"name": "no costs (for tests and comparison only)", "brokerage_pct": 0.0, "brokerage_cap": 0.0,
            "brokerage_min": 0.0, "stt_pct": 0.0, "stamp_buy_pct": 0.0, "exchange_pct": 0.0, "sebi_pct": 0.0,
            "ipft_pct": 0.0, "dp_sell": 0.0, "dp_min_value": 0.0, "gst": 0.0}


def cost_breakdown(side, value, plan=GROWW_DELIVERY):
    """Every charge for ONE executed order. side: 'buy' or 'sell'. value: shares x price in rupees."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if value <= 0:
        return {"brokerage": 0.0, "stt": 0.0, "stamp": 0.0, "exchange": 0.0, "sebi": 0.0, "ipft": 0.0,
                "dp": 0.0, "gst": 0.0, "total": 0.0}
    brokerage = max(plan["brokerage_min"], min(plan["brokerage_cap"], plan["brokerage_pct"] * value))
    stt = plan["stt_pct"] * value
    stamp = plan["stamp_buy_pct"] * value if side == "buy" else 0.0
    exchange, sebi, ipft = plan["exchange_pct"] * value, plan["sebi_pct"] * value, plan["ipft_pct"] * value
    dp = plan["dp_sell"] if side == "sell" and value >= plan["dp_min_value"] else 0.0
    gst = plan["gst"] * (brokerage + dp + exchange + sebi + ipft)
    out = {"brokerage": brokerage, "stt": stt, "stamp": stamp, "exchange": exchange, "sebi": sebi, "ipft": ipft,
           "dp": dp, "gst": gst}
    out["total"] = sum(out.values())
    return out


def trade_cost(side, value, plan=GROWW_DELIVERY):
    return cost_breakdown(side, value, plan)["total"]


def round_trip_pct(value, plan=GROWW_DELIVERY):
    """Cost of buying and later selling the same amount, as a percentage of that amount (no price change)."""
    return 100 * (trade_cost("buy", value, plan) + trade_cost("sell", value, plan)) / value
