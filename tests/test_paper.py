"""Tests for paper trading (step 7). Made-up prices and rankings, no internet.
Run from the project folder:  python -m pytest tests -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
import paper_trade as pt                 # noqa: E402
from test_features import fake_stock, fake_market   # noqa: E402

N_STOCKS, N_DAYS = 40, 60


@pytest.fixture(scope="module")
def world():
    prices = pd.concat([fake_stock(f"S{i:02d}", n=N_DAYS + 20, seed=i, start="2026-01-01") for i in range(N_STOCKS)], ignore_index=True)
    prices["volume"] *= 100
    market = fake_market(n=N_DAYS + 20, start="2026-01-01")
    days = sorted(prices["date"].unique())[20:]                       # paper trading starts on day 20
    rng = np.random.default_rng(7)
    rows = []
    for d in days:
        for s in sorted(prices["symbol"].unique()):
            rows.append({"date": d, "symbol": s, "score": float(rng.random()), "rule_score": float(rng.integers(-3, 4)),
                         "tie": float(rng.random() * 100), "random": pt.random_score(str(pd.Timestamp(d).date()), s)})
    hist = pd.DataFrame(rows)
    industry = {f"S{i:02d}": f"IND{i % 12}" for i in range(N_STOCKS)}
    return prices, market, hist, industry, [pd.Timestamp(d) for d in days]


def test_document_has_every_player_and_money_adds_up(world):
    prices, market, hist, industry, days = world
    doc = pt.build(hist, prices, market, industry, days[-1])
    assert [p["id"] for p in doc["players"]] == ["ai_weekly", "ai_monthly", "ai_small", "rule", "random"]
    assert doc["started"] == str(days[0].date()) and doc["as_of"] == str(days[-1].date()) and doc["market_days"] == len(days)
    for p in doc["players"]:
        held = sum(x["value"] for x in p["open_positions"])
        assert p["cash"] >= -0.01                                     # never borrows
        assert p["value"] == pytest.approx(p["cash"] + held, abs=0.5)
        assert len(p["open_positions"]) <= 10 and len(p["equity"]) == len(days)
        assert p["equity"][0][1] == p["start_capital"]                # day 1: only a plan, no trade yet
        per_industry = pd.Series([industry[x["symbol"]] for x in p["open_positions"]]).value_counts()
        assert per_industry.empty or per_industry.max() <= 3
    json.dumps(doc)                                                   # the whole document can be saved as JSON


def test_first_day_only_plans_orders_for_the_next_open(world):
    prices, market, hist, industry, days = world
    doc = pt.build(hist, prices, market, industry, days[0])
    for p in doc["players"]:
        assert p["value"] == p["start_capital"] and p["open_positions"] == [] and p["closed_trades"] == 0
        assert len(p["orders_for_next_open"]["buy"]) == 10


def test_adding_a_day_never_changes_earlier_days(world):
    """The key honesty property: results are re-calculated every evening and the past must stay fixed."""
    prices, market, hist, industry, days = world
    full = pt.build(hist, prices, market, industry, days[-1])
    for n in (5, 21, 40):
        part = pt.build(hist, prices, market, industry, days[n - 1])
        for a, b in zip(part["players"], full["players"]):
            assert a["equity"] == b["equity"][:n]
            assert a["trades"] == [t for t in b["trades"] if t["sell_date"] <= str(days[n - 1].date())]
        assert part["benchmarks"][0]["equity"] == full["benchmarks"][0]["equity"][:n]


def test_monthly_player_trades_less_than_the_weekly_one(world):
    prices, market, hist, industry, days = world
    doc = pt.build(hist, prices, market, industry, days[-1])
    p = {x["id"]: x for x in doc["players"]}
    assert p["ai_monthly"]["closed_trades"] < p["ai_weekly"]["closed_trades"]
    assert p["ai_monthly"]["costs_paid"] < p["ai_weekly"]["costs_paid"]


def test_a_day_without_a_published_ranking_is_reported_and_makes_no_orders(world):
    prices, market, hist, industry, days = world
    gap = days[5]                                                     # a weekly decision day (0, 5, 10, …)
    doc = pt.build(hist[hist["date"] != gap], prices, market, industry, days[7])
    assert doc["health"]["missing_prediction_days"] == [str(gap.date())]
    with_gap = {p["id"]: p for p in doc["players"]}["ai_weekly"]
    normal = {p["id"]: p for p in pt.build(hist, prices, market, industry, days[7])["players"]}["ai_weekly"]
    assert all(t["reason"] != "rebalance" for t in with_gap["trades"])            # no new ranking: nothing was swapped
    assert any(t["reason"] == "rebalance" for t in normal["trades"])              # with the ranking, stocks were swapped
    first_plan = set(pt.build(hist, prices, market, industry, days[0])["players"][0]["orders_for_next_open"]["buy"])
    assert {x["symbol"] for x in with_gap["open_positions"]} <= first_plan        # still holding only the first picks


def test_monitoring_only_judges_days_whose_five_days_are_over(world):
    prices, market, hist, industry, days = world
    doc = pt.build(hist, prices, market, industry, days[-1], expected={"winner_rate_top10pct": 0.376})
    mon = doc["monitoring"]
    assert mon["days_judged"] == len(days) - 6                        # the last 6 published days cannot be judged yet
    assert mon["days"][-1]["date"] == str(days[-7].date())
    assert 0 <= mon["average"]["winner_rate_top10pct"] <= 1 and abs(mon["average"]["rank_ic"]) < 0.2   # random scores
    assert mon["expected_from_tests"] == {"winner_rate_top10pct": 0.376}
    early = pt.build(hist, prices, market, industry, days[3])["monitoring"]
    assert early["days_judged"] == 0 and early["average"] is None


def test_monitoring_recognises_a_perfect_ranking(world):
    prices, market, hist, industry, days = world
    cal = pd.DatetimeIndex(sorted(market.loc[market["symbol"] == "NIFTY50", "date"].unique()))
    px = prices.assign(adj_open=prices["open"] * prices["adj_close"] / prices["close"]).pivot(index="date", columns="symbol", values="adj_open")
    perfect = hist.copy()
    pos = {d: i for i, d in enumerate(cal)}
    def future(row):
        i = pos[row["date"]]
        return px.at[cal[i + 6], row["symbol"]] / px.at[cal[i + 1], row["symbol"]] - 1 if i + 6 < len(cal) else 0.0
    perfect["score"] = perfect.apply(future, axis=1)
    mon = pt.monitor(perfect, prices, cal)
    assert mon["average"]["winner_rate_top10pct"] == 1.0 and mon["average"]["rank_ic"] == pytest.approx(1.0, abs=1e-6)
    assert mon["average"]["excess_5d"] > 0


def test_random_scores_are_fixed_and_spread_out():
    a = pt.random_score("2026-10-12", "RELIANCE")
    assert a == pt.random_score("2026-10-12", "RELIANCE") and 0 <= a < 1
    assert a != pt.random_score("2026-10-13", "RELIANCE") != pt.random_score("2026-10-13", "TCS")
    xs = [pt.random_score("2026-10-12", f"S{i}") for i in range(2000)]
    assert 0.45 < np.mean(xs) < 0.55


def test_load_history_reads_published_files_and_skips_broken_ones(tmp_path):
    good = {"as_of": "2026-10-12", "stocks": [{"symbol": "AAA", "score": 0.36, "rule_score": 1.5, "turnover_cr": 120.5},
                                              {"symbol": "BBB", "score": 0.30}]}
    (tmp_path / "2026-10-12.json").write_text(json.dumps(good))
    (tmp_path / "2026-10-13.json").write_text("{ broken")
    h = pt.load_history(str(tmp_path))
    assert h["date"].nunique() == 1 and h["symbol"].tolist() == ["AAA", "BBB"]
    assert h["tie"].tolist() == [120.5, 0.0] and h["rule_score"].tolist() == [1.5, 0.0]
    assert pt.load_history(str(tmp_path / "nothing")).empty


def test_ranking_made_after_the_next_market_open_is_not_used():
    cal = pd.to_datetime(["2026-10-12", "2026-10-13", "2026-10-14"])
    d = pd.Timestamp("2026-10-12")
    assert pt.on_time(pt.made_at("2026-10-12 17:05 IST"), d, cal)
    assert pt.on_time(pt.made_at("2026-10-13 09:10 IST"), d, cal)          # just before the 9:15 open
    assert not pt.on_time(pt.made_at("2026-10-13 09:20 IST"), d, cal)
    assert not pt.on_time(pt.made_at("2026-10-14 01:49 IST"), d, cal)      # the late re-run seen on 14 Oct
    assert pt.on_time(None, d, cal) and pt.on_time(pt.made_at("2026-10-15 10:00 IST"), pd.Timestamp("2026-10-14"), cal)
    assert pt.made_at("nonsense") is None


def test_late_rankings_are_left_out_and_reported(world):
    prices, market, hist, industry, days = world
    h = hist.copy()
    h["made"] = h["date"] + pd.Timedelta(hours=17)                         # normally made at 5 pm on the day
    late_day = days[3]
    h.loc[h["date"] == late_day, "made"] = late_day + pd.Timedelta(days=2)  # this one only two days later
    doc = pt.build(h, prices, market, industry, days[10])
    assert doc["health"]["late_rankings_not_used"] == [str(late_day.date())]
    assert str(late_day.date()) in doc["health"]["missing_prediction_days"]
    ok = pt.build(h[h["date"] != late_day], prices, market, industry, days[10])
    assert [p["equity"] for p in doc["players"]] == [p["equity"] for p in ok["players"]]   # exactly as if it never existed
