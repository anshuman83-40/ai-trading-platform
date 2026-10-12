"""
TradePulse ML — the website's read-only prediction endpoints.

The server does NOT load the model, build features or train. It only reads small JSON files written by
training/predict.py, training/evaluate_model.py and training/backtest.py, so it stays light on the free plan.

  GET /api/ml/info               what the model is, how it did in tests, and whether today's file is fresh
  GET /api/ml/top?n=10           the highest-scored stocks of the latest prediction day
  GET /api/ml/predict/<symbol>   one stock: rank, group, score, the website's rule signal for comparison
  GET /api/ml/paper              paper trading (virtual money): every player's value, positions and trades

A prediction is only returned when the file is recent enough. Otherwise the answer says "stale" and carries
no score — an old ranking must never be shown as if it were today's.
"""
import json
import os
from datetime import date, datetime, timedelta, timezone

from flask import Blueprint, jsonify, request

bp = Blueprint("ml", __name__)

MODEL_DIR = os.environ.get("TP_MODEL_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgb_v1")
MAX_AGE_DAYS = 5          # calendar days after the prediction day (a prediction is for the next 5 trading days)
ALIASES = {"TATAMOTORS": "TMPV"}     # website symbol -> the symbol in the model's company list
IST = timezone(timedelta(hours=5, minutes=30))
DISCLAIMER = ("Research and education only — not investment advice. The score is a weak ranking learned from past "
              "prices. In tests it did not beat trading costs with small capital.")

_cache = {}               # path -> (modified time, parsed JSON)


def _read(name):
    """A JSON file from the model folder, re-read only when the file changes. None if missing or unreadable."""
    path = os.path.join(MODEL_DIR, name)
    try:
        mtime = os.path.getmtime(path)
        hit = _cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _cache[path] = (mtime, data)
        return data
    except (OSError, ValueError):
        return None


def today_ist():
    return datetime.now(IST).date()


def freshness(pred, today=None):
    """-> (status, age in days). status: 'fresh', 'stale' (too old) or 'missing' (no usable file)."""
    if not pred or not pred.get("stocks") or not pred.get("as_of"):
        return "missing", None
    try:
        age = ((today or today_ist()) - date.fromisoformat(pred["as_of"])).days
    except ValueError:
        return "missing", None
    return ("fresh" if 0 <= age <= MAX_AGE_DAYS else "stale"), age


def _header(pred, status, age):
    return {"status": status, "model_version": (pred or {}).get("model_version"),
            "as_of": (pred or {}).get("as_of"), "age_days": age,
            "horizon_trading_days": (pred or {}).get("horizon_trading_days"),
            "base_rate": (pred or {}).get("base_rate"), "disclaimer": DISCLAIMER}


def _public(stock):
    return {k: stock.get(k) for k in ("symbol", "rank", "of", "group", "score", "close", "rule_signal",
                                      "historical_winner_rate")}


@bp.route("/api/ml/info")
def ml_info():
    pred, meta = _read("predictions.json"), _read("metadata.json") or {}
    ev, back = _read("evaluation.json") or {}, _read("backtest.json") or {}
    status, age = freshness(pred)
    out = _header(pred, status, age)
    out["model_version"] = out["model_version"] or meta.get("model_version")
    out["model"] = {"algorithm": meta.get("algorithm"), "target": meta.get("target"), "universe": meta.get("universe"),
                    "trained_on": (meta.get("periods") or {}).get("train"), "features": meta.get("n_features"),
                    "data_source": meta.get("data_source"), "created": meta.get("created")}
    test = (ev.get("periods") or {}).get("test") or {}
    out["tests"] = {
        "period": [test.get("first"), test.get("last")],
        "note": ev.get("note"),
        "rank_groups": ev.get("rank_groups_test"),
        "picks_top10pct": (test.get("rankings") or {}).get("model"),
        "website_rule_signal": test.get("rule_signal"),
    }
    res = back.get("results") or {}
    keep = ("yearly_return", "max_drawdown", "sharpe", "end_capital", "start_capital", "costs_pct_per_year",
            "profit", "profit_without_10_best_trades", "trades", "win_rate", "profit_factor")
    out["backtest"] = {
        "period": back.get("period"), "strategy": back.get("strategy"),
        "results": {name: {k: r.get(k) for k in keep if k in r} for name, r in res.items()},
        "small_capital": {k: {x: v.get(x) for x in ("yearly_return", "max_drawdown", "end_capital", "start_capital")}
                          for k, v in (back.get("sensitivity_model") or {}).items() if k.startswith("start capital")},
        "not_modelled": back.get("not_modelled"),
    }
    if pred:
        out["eligible"], out["generated"] = pred.get("eligible"), pred.get("generated")
    return jsonify(out)


@bp.route("/api/ml/top")
def ml_top():
    pred = _read("predictions.json")
    status, age = freshness(pred)
    out = _header(pred, status, age)
    try:
        n = max(1, min(int(request.args.get("n", 10)), 50))
    except ValueError:
        n = 10
    out["stocks"] = [_public(s) for s in pred["stocks"][:n]] if status == "fresh" else []
    return jsonify(out)


@bp.route("/api/ml/paper")
def ml_paper():
    """Paper-trading results written by training/paper_trade.py. Virtual money only."""
    doc = _read(os.path.join("paper", "paper.json"))
    if not doc or not doc.get("players"):
        return jsonify({"status": "missing", "disclaimer": DISCLAIMER, "players": [], "benchmarks": []})
    try:
        age = (today_ist() - date.fromisoformat(doc["as_of"])).days
    except (KeyError, ValueError):
        age = None
    out = {k: doc.get(k) for k in ("started", "as_of", "generated", "market_days", "rules", "benchmarks", "monitoring", "health")}
    out["status"] = "fresh" if age is not None and 0 <= age <= MAX_AGE_DAYS else "stale"
    out["age_days"], out["disclaimer"] = age, DISCLAIMER
    out["players"] = [dict({k: v for k, v in p.items() if k != "trades"}, recent_trades=(p.get("trades") or [])[-20:][::-1])
                      for p in doc["players"]]
    return jsonify(out)


@bp.route("/api/ml/predict/<symbol>")
def ml_predict(symbol):
    pred = _read("predictions.json")
    status, age = freshness(pred)
    out = _header(pred, status, age)
    sym = str(symbol).upper().strip()
    sym = ALIASES.get(sym, sym)
    out["symbol"], out["prediction"] = sym, None
    if status != "fresh":
        out["reason"] = ("No prediction file yet." if status == "missing"
                         else f"The latest predictions are from {pred['as_of']} — too old to show.")
        return jsonify(out)
    row = next((s for s in pred["stocks"] if s["symbol"] == sym), None)
    if row is None:
        covered = _read("universe.json") or {}
        if sym in (pred.get("not_scored") or []) or sym in (covered.get("symbols") or []):
            out["status"], out["reason"] = "not_scored", "Not enough recent, reliable price data for this stock today."
        else:
            out["status"], out["reason"] = "not_covered", "The model only covers the NIFTY 200 companies it was trained on."
        return jsonify(out)
    out["prediction"] = _public(row)
    return jsonify(out)
