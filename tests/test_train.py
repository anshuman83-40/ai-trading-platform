"""Tests for training: the test period is never used, results repeat, and the saved model gives the same scores.
Small made-up data, no internet.  Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from ml import features as ft            # noqa: E402
import train_model as tm                 # noqa: E402
import config as cfg                     # noqa: E402

SMALL = dict(tm.PARAMS, n_estimators=40, early_stopping_rounds=10, min_child_weight=5)


def fake_dataset(seed=0, stocks=40):
    """Random features; the label depends weakly on one of them, so there is something to learn."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-06-01", "2023-06-30")
    n = len(dates) * stocks
    data = pd.DataFrame(rng.normal(size=(n, len(ft.FEATURES))), columns=ft.FEATURES)
    data["date"] = np.repeat(dates, stocks)
    data["symbol"] = np.tile([f"S{i:02d}" for i in range(stocks)], len(dates))
    data["fwd_ret"] = 0.01 * data["ret_5"] + rng.normal(0, 0.03, n)
    data["label"] = (data.groupby("date")["fwd_ret"].rank(pct=True) > 1 - cfg.TOP_FRACTION).astype(float)
    data["split"] = tm.split_of(data["date"])
    return data


def test_training_never_scores_or_uses_test_rows():
    data = fake_dataset()
    assert (data["split"] == "test").any()
    _, scored = tm.train(data, params=SMALL, n_jobs=1)
    assert set(scored["split"]) == {"train", "valid"}
    assert scored["date"].max() <= pd.Timestamp(cfg.VALID_END)


def test_changing_test_rows_does_not_change_the_model():
    data = fake_dataset()
    a, _ = tm.train(data, params=SMALL, n_jobs=1)
    wild = data.copy()
    is_test = wild["split"] == "test"
    wild.loc[is_test, ft.FEATURES] = 999.0
    wild.loc[is_test, "label"] = 1.0
    b, _ = tm.train(wild, params=SMALL, n_jobs=1)
    x = data[ft.FEATURES].head(500)
    np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))


def test_training_is_repeatable():
    data = fake_dataset()
    a, _ = tm.train(data, params=SMALL, n_jobs=1)
    b, _ = tm.train(data, params=SMALL, n_jobs=1)
    x = data[ft.FEATURES].head(500)
    np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))


def test_saved_model_gives_the_same_scores(tmp_path):
    data = fake_dataset()
    model, _ = tm.train(data, params=SMALL, n_jobs=1)
    path = str(tmp_path / "model.json")
    model.get_booster().save_model(path)
    loaded = xgb.Booster()
    loaded.load_model(path)
    x = data[ft.FEATURES].head(500)
    again = loaded.predict(xgb.DMatrix(x, feature_names=ft.FEATURES), iteration_range=(0, model.best_iteration + 1))
    np.testing.assert_allclose(model.predict_proba(x)[:, 1], again, rtol=1e-6)


def test_model_learns_the_planted_signal_and_metrics_are_sane():
    data = fake_dataset()
    _, scored = tm.train(data, params=SMALL, n_jobs=1)
    m = tm.metrics(scored[scored["split"] == "valid"])
    assert m["auc"] > 0.52 and m["rank_ic"] > 0                      # ret_5 really drives the label here
    assert 0.25 < m["base_rate"] < 0.35 and 0 <= m["hit_rate_top10pct"] <= 1


def test_daily_metrics_on_a_perfect_and_a_useless_score():
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=30):
        r = rng.normal(0, 0.03, 50)
        rows.append(pd.DataFrame({"date": d, "fwd_ret": r, "label": (pd.Series(r).rank(pct=True) > 0.7).astype(float),
                                  "perfect": r, "useless": rng.normal(size=50)}))
    frame = pd.concat(rows, ignore_index=True)
    good = tm.daily_metrics(frame, "perfect")
    assert good["hit_rate_top10pct"] == 1.0 and good["rank_ic"] == pytest.approx(1.0)
    bad = tm.daily_metrics(frame, "useless")
    assert abs(bad["rank_ic"]) < 0.1
