"""Tests for the daily prediction step: same numbers as training, no future data, refuses bad data.
Made-up prices, no internet.  Run from the project folder:  python -m pytest tests -q
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from ml import features as ft            # noqa: E402
from ml import inference as inf          # noqa: E402
import build_dataset as bd               # noqa: E402
import predict as pr                     # noqa: E402
import train_model as tm                 # noqa: E402
from test_features import fake_stock, fake_market   # noqa: E402


@pytest.fixture(scope="module")
def world():
    """40 liquid fake stocks, a market, and a small model trained on them."""
    prices = pd.concat([fake_stock(f"S{i:02d}", n=520, seed=i, start="2021-01-01") for i in range(40)], ignore_index=True)
    prices["volume"] *= 100
    market = fake_market(n=520, start="2021-01-01")
    panel = bd.build(prices, market)
    lab = panel[panel["label"].notna()].copy()
    cut = sorted(lab["date"].unique())[int(lab["date"].nunique() * 0.7)]
    lab["split"] = np.where(lab["date"] <= cut, "train", "valid")
    model, _ = tm.train(lab, params=dict(tm.PARAMS, n_estimators=30, early_stopping_rounds=10, min_child_weight=5), n_jobs=1)
    meta = {"model_version": "test", "trees_used": int(model.best_iteration + 1), "horizon_trading_days": 5, "top_fraction": 0.3}
    schema = {"feature_version": ft.FEATURE_VERSION, "features": ft.FEATURES}
    return prices, market, panel, model.get_booster(), meta, schema


def test_prediction_scores_equal_the_scores_from_the_training_table(world):
    """No difference between 'training numbers' and 'live numbers' for the same day."""
    prices, market, panel, booster, meta, schema = world
    day = sorted(panel["date"].unique())[-30]
    pred = pr.make_predictions(prices, market, booster, meta, schema, day)
    rows = panel[(panel["date"] == day) & panel["eligible"]].copy()
    rows["score"] = inf.model_scores(booster, meta, schema, rows)
    want = rows.set_index("symbol")["score"].astype(float).to_dict()
    got = {s["symbol"]: s["score"] for s in pred["stocks"]}
    assert set(got) == set(want) and len(got) > 30
    assert all(abs(got[k] - want[k]) < 6e-5 for k in got)            # the file stores 4 decimals


def test_predictions_for_a_day_ignore_everything_after_it(world):
    prices, market, _, booster, meta, schema = world
    day = sorted(prices["date"].unique())[-30]
    a = pr.make_predictions(prices, market, booster, meta, schema, day)
    wild = prices.copy()
    wild.loc[wild["date"] > day, ["open", "high", "low", "close", "adj_close"]] *= 7
    b = pr.make_predictions(wild, market, booster, meta, schema, day)
    assert a == b


def test_ranks_groups_and_fields(world):
    prices, market, _, booster, meta, schema = world
    day = prices["date"].max()
    groups = [{"group": g, "winner_rate": 0.2 + g / 100} for g in range(1, 11)]
    pred = pr.make_predictions(prices, market, booster, meta, schema, day, groups)
    st = pred["stocks"]
    assert [s["rank"] for s in st] == list(range(1, len(st) + 1))
    assert all(st[i]["score"] >= st[i + 1]["score"] for i in range(len(st) - 1))
    assert st[0]["group"] == 10 and st[-1]["group"] == 1 and st[0]["historical_winner_rate"] == pytest.approx(0.30)
    assert all(s["rule_signal"] in ("BUY", "HOLD", "SELL") and 0 <= s["score"] <= 1 for s in st)
    assert pred["as_of"] == str(day.date()) and pred["eligible"] == len(st) and pred["model_version"] == "test"


def test_stock_without_a_price_that_day_is_not_scored(world):
    prices, market, _, booster, meta, schema = world
    day = prices["date"].max()
    gone = prices[~((prices["symbol"] == "S05") & (prices["date"] == day))]
    pred = pr.make_predictions(gone, market, booster, meta, schema, day)
    assert "S05" in pred["not_scored"] and "S05" not in {s["symbol"] for s in pred["stocks"]}
    assert pred["with_fresh_price"] == 39


def test_todays_unfinished_candle_is_not_used():
    cal = pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09"])
    during = datetime(2026, 10, 9, 11, 0, tzinfo=pr.IST)             # market still open on the 9th
    after = datetime(2026, 10, 9, 16, 0, tzinfo=pr.IST)
    weekend = datetime(2026, 10, 10, 10, 0, tzinfo=pr.IST)
    assert pr.last_completed_day(cal, during) == pd.Timestamp("2026-10-08")
    assert pr.last_completed_day(cal, after) == pd.Timestamp("2026-10-09")
    assert pr.last_completed_day(cal, weekend) == pd.Timestamp("2026-10-09")


def test_bad_data_is_refused():
    ok = {"as_of": "2026-10-09", "universe_size": 187, "with_fresh_price": 185, "eligible": 180,
          "stocks": [{"score": 0.3}, {"score": 0.31}]}
    assert pr.check(ok, "2026-10-09", "2026-10-11") == []
    assert any("days old" in p for p in pr.check(ok, "2026-10-09", "2026-10-20"))
    assert any("no price" in p for p in pr.check(dict(ok, with_fresh_price=100), "2026-10-09", "2026-10-10"))
    assert any("no price" in p for p in pr.check(dict(ok, with_fresh_price=162), "2026-10-09", "2026-10-10"))   # 13% missing
    assert any("could be scored" in p for p in pr.check(dict(ok, eligible=40), "2026-10-09", "2026-10-10"))
    assert any("outside" in p for p in pr.check(dict(ok, stocks=[{"score": 1.4}]), "2026-10-09", "2026-10-10"))


def test_scoring_refuses_missing_or_infinite_features(world):
    _, _, panel, booster, meta, schema = world
    rows = panel[panel["eligible"]].tail(5).copy()
    with pytest.raises(ValueError, match="missing"):
        inf.model_scores(booster, meta, schema, rows.drop(columns=["rsi_14"]))
    rows.loc[rows.index[0], "ret_5"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        inf.model_scores(booster, meta, schema, rows)


def test_companies_that_failed_to_download_are_counted_as_missing(world):
    """25 of 187 companies missing from a download once changed the ranking silently — it must be refused."""
    prices, market, _, booster, meta, schema = world
    day = prices["date"].max()
    everyone = sorted(prices["symbol"].unique())
    partial = prices[~prices["symbol"].isin(everyone[:6])]            # 6 of 40 never arrived
    pred = pr.make_predictions(partial, market, booster, meta, schema, day, universe=everyone)
    assert pred["universe_size"] == 40 and pred["with_fresh_price"] == 34
    assert set(everyone[:6]) <= set(pred["not_scored"])
    assert any("no price" in p for p in pr.check(pred, day, day))


def test_a_published_ranking_is_not_replaced_by_one_with_fewer_stocks(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text('{"as_of": "2026-10-12", "eligible": 185}')
    assert pr.worse_than_published({"as_of": "2026-10-12", "eligible": 160}, str(path))
    assert pr.worse_than_published({"as_of": "2026-10-12", "eligible": 185}, str(path)) == []
    assert pr.worse_than_published({"as_of": "2026-10-13", "eligible": 150}, str(path)) == []     # a new day is fine
    assert pr.worse_than_published({"as_of": "2026-10-12", "eligible": 1}, str(tmp_path / "none.json")) == []


def test_published_rankings_are_never_changed_or_replaced_by_an_older_day(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text('{"as_of": "2026-10-12", "eligible": 185}')
    assert "already published" in pr.already_published({"as_of": "2026-10-12"}, str(path))
    assert "newer ranking" in pr.already_published({"as_of": "2026-10-09"}, str(path))
    assert pr.already_published({"as_of": "2026-10-13"}, str(path)) is None
    assert pr.already_published({"as_of": "2026-10-13"}, str(tmp_path / "none.json")) is None
