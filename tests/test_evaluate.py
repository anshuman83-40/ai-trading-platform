"""Tests for the evaluation code (step 4). Made-up data, no internet.
Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from ml import features as ft            # noqa: E402
import evaluate_model as ev              # noqa: E402
import train_model as tm                 # noqa: E402
import config as cfg                     # noqa: E402
from test_train import fake_dataset, SMALL   # noqa: E402


def server_rule(macd_above_signal, price_above_ema20, r):
    """The scoring written out exactly as in server.py build_quote()."""
    score = 0
    score += 1 if macd_above_signal else -1
    score += 1 if price_above_ema20 else -1
    if r < 30: score += 1
    elif r > 70: score -= 1
    elif r > 55: score += 0.5
    elif r < 45: score -= 0.5
    return score, ("BUY" if score >= 1.5 else "SELL" if score <= -1.5 else "HOLD")


def test_rule_score_matches_the_website_logic_for_every_case():
    rows = [(m, e, r) for m in (0.01, -0.01) for e in (0.02, -0.02) for r in (10, 29.9, 30, 40, 45, 50, 55, 56, 70, 71, 95)]
    panel = pd.DataFrame(rows, columns=["macd_hist", "dist_ema20", "rsi_14"])
    score = ft.rule_score(panel)
    signal = ft.rule_signal(score)
    for i, (m, e, r) in enumerate(rows):
        want_score, want_signal = server_rule(m > 0, e > 0, r)
        assert score.iloc[i] == want_score and signal.iloc[i] == want_signal, (m, e, r)


def day_frame(days=40, stocks=50, seed=3):
    rng = np.random.default_rng(seed)
    parts = []
    for d in pd.bdate_range("2024-01-01", periods=days):
        r = rng.normal(0, 0.03, stocks)
        parts.append(pd.DataFrame({"date": d, "symbol": [f"S{i:02d}" for i in range(stocks)], "fwd_ret": r,
                                   "label": (pd.Series(r).rank(pct=True) > 0.7).astype(float),
                                   "perfect": r, "noise": rng.normal(size=stocks), "flat": 1.0}))
    return pd.concat(parts, ignore_index=True)


def test_per_day_and_summarise_on_perfect_noise_and_constant_scores():
    f = day_frame()
    good = ev.summarise(ev.per_day(f, "perfect"))
    assert good["hit_rate_top10pct"] == 1.0 and good["rank_ic"] == pytest.approx(1.0)
    assert good["excess_5d"] > 0.03 and good["excess_positive_days"] == 1.0 and good["excess_t_stat"] > 5
    noise = ev.summarise(ev.per_day(f, "noise"))
    assert abs(noise["rank_ic"]) < 0.08 and abs(noise["excess_t_stat"]) < 3
    flat = ev.per_day(f, "flat")                                     # every stock has the same score
    assert flat["ic"].isna().all()                                   # no ranking -> no correlation, not a crash


def test_picks_are_exactly_the_top_10_percent_and_ties_are_broken_by_name():
    f = day_frame(days=1, stocks=50)
    f["tie"] = 0.0
    d = ev.per_day(f, "tie")
    first5 = f.sort_values("symbol").head(5)                         # 10% of 50 = 5 stocks, alphabetical on a full tie
    assert d["ret_top"].iloc[0] == pytest.approx(first5["fwd_ret"].mean())


def test_nifty_forward_uses_next_open_to_open_after_horizon():
    m = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10), "symbol": "NIFTY50",
                      "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]})
    out = ev.nifty_forward(m, 5)
    assert out.iloc[0] == pytest.approx(106 / 101 - 1) and out.iloc[4:].isna().all()


def test_calibration_bins_report_score_and_actual_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.6, 20000)
    f = pd.DataFrame({"model": p, "label": (rng.uniform(size=p.size) < p).astype(float)})
    cal = ev.calibration(f)
    assert len(cal) == 10 and sum(c["rows"] for c in cal) == 20000
    assert all(abs(c["mean_score"] - c["actual_rate"]) < 0.04 for c in cal)      # honest scores look calibrated
    f["model"] = p * 0.5                                                          # scores that are too low
    assert all(c["actual_rate"] > c["mean_score"] for c in ev.calibration(f))


def test_walk_forward_only_learns_from_the_past(monkeypatch):
    data = fake_dataset()
    seen = []
    real_train = tm.train

    def spy(d, params=None, device="cpu", n_jobs=8):
        used = d[d["split"].isin(["train", "valid"])]
        test = d[d["split"] == "test"]
        seen.append((used["date"].max(), test["date"].min(), test["date"].dt.year.unique().tolist()))
        return real_train(d, params=SMALL, n_jobs=1)

    monkeypatch.setattr(ev.tm, "train", spy)
    wf, info = ev.walk_forward(data, [2022, 2023])
    assert [i["test_year"] for i in info] == [2022, 2023]
    for last_used, first_test, test_years in seen:
        assert last_used < first_test                                # nothing from the test year is learned from
        assert len(test_years) == 1
        gap = pd.bdate_range(last_used, first_test)
        assert len(gap) - 2 >= cfg.EMBARGO_DAYS - 1                  # the gap between learning and testing is kept
    assert set(wf["date"].dt.year) == {2022, 2023} and wf["wf"].between(0, 1).all()
    assert not wf.duplicated(["date", "symbol"]).any()
