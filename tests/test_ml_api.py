"""Tests for the website's prediction endpoints (ml/api.py). Uses a temporary model folder — no model, no internet.
Run from the project folder:  python -m pytest tests -q
"""
import json
import os
import sys
from datetime import date

import pytest
from flask import Flask

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ml import api                        # noqa: E402

TODAY = date(2026, 10, 13)


def predictions(as_of="2026-10-12"):
    stocks = [{"symbol": s, "rank": i + 1, "of": 3, "group": g, "score": sc, "close": 100.0 + i,
               "rule_signal": sig, "rule_score": 0.0, "historical_winner_rate": wr}
              for i, (s, g, sc, sig, wr) in enumerate([("AAA", 10, 0.36, "SELL", 0.38), ("TMPV", 5, 0.30, "HOLD", 0.29),
                                                       ("CCC", 1, 0.27, "BUY", 0.24)])]
    return {"model_version": "xgb_v1", "as_of": as_of, "horizon_trading_days": 5, "base_rate": 0.3,
            "eligible": 3, "generated": "2026-10-12 16:13 IST", "not_scored": ["VEDL"], "stocks": stocks}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(api, "today_ist", lambda: TODAY)
    api._cache.clear()
    (tmp_path / "universe.json").write_text(json.dumps({"symbols": ["AAA", "TMPV", "CCC", "VEDL"]}))
    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "xgb_v1", "algorithm": "XGBoost", "n_features": 39}))

    def write(doc):
        (tmp_path / "predictions.json").write_text(json.dumps(doc))
        api._cache.clear()
    app = Flask(__name__)
    app.register_blueprint(api.bp)
    c = app.test_client()
    c.write = write
    return c


def test_fresh_prediction_for_a_covered_stock(client):
    client.write(predictions())
    r = client.get("/api/ml/predict/aaa").get_json()
    assert r["status"] == "fresh" and r["age_days"] == 1 and r["model_version"] == "xgb_v1"
    assert r["horizon_trading_days"] == 5 and r["base_rate"] == 0.3 and "not investment advice" in r["disclaimer"]
    p = r["prediction"]
    assert p == {"symbol": "AAA", "rank": 1, "of": 3, "group": 10, "score": 0.36, "close": 100.0,
                 "rule_signal": "SELL", "historical_winner_rate": 0.38}


def test_website_symbol_alias(client):
    client.write(predictions())
    r = client.get("/api/ml/predict/TATAMOTORS").get_json()
    assert r["symbol"] == "TMPV" and r["prediction"]["rank"] == 2


def test_stale_file_gives_no_score(client):
    client.write(predictions(as_of="2026-10-01"))                    # 12 days old
    r = client.get("/api/ml/predict/AAA").get_json()
    assert r["status"] == "stale" and r["prediction"] is None and "too old" in r["reason"]
    assert client.get("/api/ml/top").get_json()["stocks"] == []
    assert client.get("/api/ml/info").get_json()["status"] == "stale"


def test_future_dated_file_is_not_trusted(client):
    client.write(predictions(as_of="2026-10-20"))
    assert client.get("/api/ml/predict/AAA").get_json()["status"] == "stale"


def test_missing_or_broken_file(client, tmp_path):
    r = client.get("/api/ml/predict/AAA").get_json()
    assert r["status"] == "missing" and r["prediction"] is None
    (tmp_path / "predictions.json").write_text("{ not json")
    api._cache.clear()
    assert client.get("/api/ml/top").get_json()["status"] == "missing"
    assert client.get("/api/ml/info").status_code == 200


def test_stock_outside_the_model_and_stock_without_fresh_data(client):
    client.write(predictions())
    r = client.get("/api/ml/predict/ZZZNOTREAL").get_json()
    assert r["status"] == "not_covered" and r["prediction"] is None
    r = client.get("/api/ml/predict/VEDL").get_json()
    assert r["status"] == "not_scored" and r["prediction"] is None


def test_top_list_is_limited_and_ordered(client):
    client.write(predictions())
    r = client.get("/api/ml/top?n=2").get_json()
    assert [s["symbol"] for s in r["stocks"]] == ["AAA", "TMPV"] and r["status"] == "fresh"
    assert len(client.get("/api/ml/top?n=9999").get_json()["stocks"]) == 3
    assert len(client.get("/api/ml/top?n=abc").get_json()["stocks"]) == 3
    assert "rule_score" not in r["stocks"][0]


def test_info_reports_tests_and_backtest_when_present(client, tmp_path):
    client.write(predictions())
    (tmp_path / "evaluation.json").write_text(json.dumps({
        "note": "before costs", "rank_groups_test": [{"group": 10, "winner_rate": 0.38}],
        "periods": {"test": {"first": "2023-01-02", "last": "2026-09-30", "rankings": {"model": {"excess_5d": 0.0048}},
                             "rule_signal": {"BUY": {"excess_5d": -0.0005}}}}}))
    (tmp_path / "backtest.json").write_text(json.dumps({
        "period": {"start": "2023-01-01", "end": "2026-10-09"}, "strategy": "top 10",
        "results": {"AI model (xgb_v1)": {"yearly_return": 0.201, "max_drawdown": -0.256, "equity_curve": [1, 2, 3]}},
        "sensitivity_model": {"start capital Rs 25,000": {"yearly_return": -0.154, "end_capital": 13476, "start_capital": 25000},
                              "hold 20 stocks instead of 10": {"yearly_return": 0.112}}}))
    api._cache.clear()
    r = client.get("/api/ml/info").get_json()
    assert r["tests"]["rank_groups"][0]["winner_rate"] == 0.38 and r["tests"]["period"] == ["2023-01-02", "2026-09-30"]
    assert r["backtest"]["results"]["AI model (xgb_v1)"] == {"yearly_return": 0.201, "max_drawdown": -0.256}
    assert list(r["backtest"]["small_capital"]) == ["start capital Rs 25,000"]
    assert r["model"]["features"] == 39 and r["eligible"] == 3


def test_file_change_is_picked_up_without_restart(client, tmp_path):
    client.write(predictions())
    assert client.get("/api/ml/predict/AAA").get_json()["prediction"]["score"] == 0.36
    doc = predictions()
    doc["stocks"][0]["score"] = 0.41
    (tmp_path / "predictions.json").write_text(json.dumps(doc))
    os.utime(tmp_path / "predictions.json", (1_900_000_000, 1_900_000_000))     # a clearly different modified time
    assert client.get("/api/ml/predict/AAA").get_json()["prediction"]["score"] == 0.41


def paper_doc(as_of="2026-10-12"):
    trades = [{"symbol": f"S{i}", "sell_date": f"2026-10-{i + 1:02d}", "pnl": i} for i in range(25)]
    return {"started": "2026-09-01", "as_of": as_of, "generated": "x", "market_days": 30, "rules": ["Virtual money only."],
            "players": [{"id": "ai_weekly", "name": "AI model, weekly", "value": 101000.0, "return_pct": 1.0,
                         "open_positions": [{"symbol": "AAA"}], "trades": trades, "equity": [["2026-09-01", 100000.0]]}],
            "benchmarks": [{"id": "nifty", "return_pct": 0.5}], "monitoring": {"days_judged": 3}, "health": {"missing_prediction_days": []}}


def test_paper_endpoint_returns_players_and_only_recent_trades(client, tmp_path):
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "paper.json").write_text(json.dumps(paper_doc()))
    api._cache.clear()
    r = client.get("/api/ml/paper").get_json()
    assert r["status"] == "fresh" and r["age_days"] == 1 and r["started"] == "2026-09-01" and r["market_days"] == 30
    p = r["players"][0]
    assert p["value"] == 101000.0 and "trades" not in p and len(p["recent_trades"]) == 20
    assert p["recent_trades"][0]["symbol"] == "S24"                   # newest first
    assert r["benchmarks"][0]["id"] == "nifty" and r["rules"] == ["Virtual money only."] and "not investment advice" in r["disclaimer"]


def test_paper_endpoint_when_missing_or_old(client, tmp_path):
    assert client.get("/api/ml/paper").get_json() == {"status": "missing", "disclaimer": api.DISCLAIMER, "players": [], "benchmarks": []}
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "paper.json").write_text(json.dumps(paper_doc(as_of="2026-09-20")))
    api._cache.clear()
    r = client.get("/api/ml/paper").get_json()
    assert r["status"] == "stale" and len(r["players"]) == 1          # old results are still shown, marked as out of date
