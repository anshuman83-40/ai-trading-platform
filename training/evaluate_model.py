"""
TradePulse ML — step 4: the exam. Compare the saved model with simple alternatives on data it never saw.

Compared, on the validation period and on the TEST period (2023 onwards, first opened here):
  - the saved model (models/xgb_v1)
  - the website's existing rule-based BUY / SELL signal
  - one-line rules: most volatile, biggest 5-day fall (reversal), best 6-month return (momentum)
  - all eligible stocks equally (what you get with no skill) and NIFTY 50 over the same days
Also: a walk-forward check (the model is retrained each year on the past only), results year by year,
and whether the model's scores can be read as probabilities (calibration).

All returns here are BEFORE trading costs. Costs, position sizes and drawdowns are step 5 (backtest.py).

Run from the project folder, after train_model.py:
    python training/evaluate_model.py

Output: models/xgb_v1/evaluation.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import config as cfg                                   # noqa: E402
from ml import features as ft                          # noqa: E402
import train_model as tm                               # noqa: E402

TOP = 0.10      # "the picks" = the 10% of stocks with the highest score each day


def load_model(model_dir):
    import xgboost as xgb
    with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, "feature_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.json"))
    return booster, meta, schema


def model_scores(booster, meta, schema, frame):
    import xgboost as xgb
    feats = schema["features"]
    dm = xgb.DMatrix(frame[feats], feature_names=feats)
    return booster.predict(dm, iteration_range=(0, meta["trees_used"]))


def per_day(frame, col):
    """One row per day for a ranking score: its top-10% picks versus all stocks, and the rank correlation."""
    def one(g):
        n_top = max(1, int(round(TOP * len(g))))
        order = g.sort_values([col, "symbol"], ascending=[False, True])        # ties broken by name, not by luck
        top = order.head(n_top)
        r = g[[col, "fwd_ret"]].rank()
        ic = np.corrcoef(r[col], r["fwd_ret"])[0, 1] if r[col].nunique() > 1 else np.nan
        return pd.Series({"hit": top["label"].mean(), "ret_top": top["fwd_ret"].mean(),
                          "ret_all": g["fwd_ret"].mean(), "ic": ic})
    return frame.groupby("date")[[col, "symbol", "label", "fwd_ret"]].apply(one)


def summarise(d):
    """Averages over days. Days overlap (each trade lasts 5 days), so for the t-statistic only every 5th day
    is used — otherwise the same price move would be counted five times and look more certain than it is."""
    excess = d["ret_top"] - d["ret_all"]
    ind = excess.iloc[::cfg.HORIZON]
    ic_ind = d["ic"].dropna().iloc[::cfg.HORIZON]
    t = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")
    return {"days": int(len(d)),
            "hit_rate_top10pct": round(float(d["hit"].mean()), 4),
            "ret_top10pct_5d": round(float(d["ret_top"].mean()), 5),
            "ret_all_5d": round(float(d["ret_all"].mean()), 5),
            "excess_5d": round(float(excess.mean()), 5),
            "excess_t_stat": round(t(ind), 2),
            "excess_positive_days": round(float((excess > 0).mean()), 4),
            "rank_ic": round(float(d["ic"].mean()), 4),
            "rank_ic_t_stat": round(t(ic_ind), 2)}


def rule_summary(frame):
    """The website's signal is BUY / HOLD / SELL, not a ranking, so judge each group by what it did next."""
    out = {}
    for sig, g in frame.groupby("rule_signal"):
        by_day = g.groupby("date")
        all_ret = frame.groupby("date")["fwd_ret"].mean()
        diff = (by_day["fwd_ret"].mean() - all_ret.reindex(by_day["fwd_ret"].mean().index))
        ind = diff.iloc[::cfg.HORIZON]
        out[sig] = {"share_of_rows": round(float(len(g) / len(frame)), 4),
                    "winner_rate": round(float(g["label"].mean()), 4),
                    "ret_5d": round(float(by_day["fwd_ret"].mean().mean()), 5),
                    "excess_5d": round(float(diff.mean()), 5),
                    "excess_t_stat": round(float(ind.mean() / ind.std(ddof=1) * np.sqrt(len(ind))), 2) if len(ind) > 2 else None}
    return out


def nifty_forward(market, horizon):
    """NIFTY 50 over the same trade: next day's open to the open `horizon` market days later."""
    n = market[market["symbol"] == "NIFTY50"].sort_values("date").set_index("date")["open"]
    return (n.shift(-(1 + horizon)) / n.shift(-1) - 1).rename("nifty_fwd")


def calibration(frame, bins=10):
    """Do the scores mean what they say? Group rows by score and compare the average score with how often
    the stock really was a winner. Close numbers = the score can be read as a probability."""
    q = pd.qcut(frame["model"], bins, labels=False, duplicates="drop")
    g = frame.groupby(q).agg(mean_score=("model", "mean"), actual_rate=("label", "mean"), rows=("label", "size"))
    return [{"bin": int(i) + 1, "mean_score": round(float(r.mean_score), 4), "actual_rate": round(float(r.actual_rate), 4),
             "rows": int(r.rows)} for i, r in g.iterrows()]


def walk_forward(data, years):
    """For each test year Y: learn from everything up to Y-2, use Y-1 to decide when to stop, score year Y.
    Nothing from year Y or later is seen. Returns the test rows with a 'wf' score column."""
    parts, info = [], []
    cal = pd.DatetimeIndex(sorted(data["date"].unique()))
    for y in years:
        tr_end, va_end = pd.Timestamp(f"{y - 2}-12-31"), pd.Timestamp(f"{y - 1}-12-31")
        d = data[data["date"] <= pd.Timestamp(f"{y}-12-31")].copy()
        d["split"] = np.where(d["date"] <= tr_end, "train", np.where(d["date"] <= va_end, "valid", "test"))
        for end in (tr_end, va_end):                                            # same gap rule as the main split
            d.loc[d["date"].isin(cal[cal <= end][-cfg.EMBARGO_DAYS:]), "split"] = "gap"
        model, _ = tm.train(d)
        te = d[d["split"] == "test"].copy()
        te["wf"] = model.predict_proba(te[ft.FEATURES])[:, 1]
        parts.append(te)
        info.append({"test_year": int(y), "train_to": str(tr_end.date()), "valid_year": int(y - 1),
                     "trees": int(model.best_iteration + 1), "test_rows": int(len(te))})
    return pd.concat(parts), info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the saved model against baselines on unseen data.")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", tm.MODEL_VERSION))
    args = ap.parse_args(argv)
    from sklearn.metrics import roc_auc_score

    data = tm.load_labelled(os.path.join(args.data, "dataset_v1.csv.gz"))
    market = pd.read_csv(os.path.join(args.data, "market_daily.csv.gz"), parse_dates=["date"])
    booster, meta, schema = load_model(args.model)
    assert schema["features"] == ft.FEATURES, "feature list changed since the model was trained"

    data["model"] = model_scores(booster, meta, schema, data)
    data["volatility"] = data["atr_14"]
    data["reversal"] = -data["ret_5"]
    data["momentum"] = data["ret_126"]
    data["rule"] = ft.rule_score(data)
    data["rule_signal"] = ft.rule_signal(data["rule"])
    nifty = nifty_forward(market, cfg.HORIZON)
    scores = ["model", "volatility", "reversal", "momentum", "rule"]
    names = {"model": "AI model (xgb_v1)", "volatility": "Most volatile", "reversal": "Fell most in 5 days",
             "momentum": "Best 6-month return", "rule": "Website rule score"}

    result = {"model_version": meta["model_version"], "evaluated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "note": "All returns are BEFORE trading costs. 'excess' = the picks' 5-day return minus the average of all eligible stocks.",
              "top_fraction_picked": TOP, "periods": {}}
    daily = {}
    for period in ("valid", "test"):
        f = data[data["split"] == period]
        res = {"first": str(f["date"].min().date()), "last": str(f["date"].max().date()), "rows": int(len(f)),
               "base_rate": round(float(f["label"].mean()), 4),
               "model_auc": round(float(roc_auc_score(f["label"], f["model"])), 4),
               "nifty50_5d": round(float(nifty.reindex(sorted(f["date"].unique())).mean()), 5),
               "rankings": {}, "rule_signal": rule_summary(f)}
        for s in scores:
            daily[(period, s)] = per_day(f, s)
            res["rankings"][s] = summarise(daily[(period, s)])
        result["periods"][period] = res

    test = data[data["split"] == "test"]
    result["test_by_year"] = {}
    for y, f in test.groupby(test["date"].dt.year):
        result["test_by_year"][str(y)] = {s: summarise(per_day(f, s)) for s in ("model", "volatility", "reversal")}
        result["test_by_year"][str(y)]["model_auc"] = round(float(roc_auc_score(f["label"], f["model"])), 4)
    result["calibration_test"] = calibration(test)

    years = sorted(test["date"].dt.year.unique())
    wf, wf_info = walk_forward(data, years)
    result["walk_forward"] = {"folds": wf_info, "all_years": summarise(per_day(wf, "wf")),
                              "auc": round(float(roc_auc_score(wf["label"], wf["wf"])), 4),
                              "fixed_model_same_rows": summarise(per_day(wf, "model"))}

    with open(os.path.join(args.model, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    # ── printed report ──
    for period in ("valid", "test"):
        r = result["periods"][period]
        print(f"\n{period.upper()}  {r['first']} to {r['last']}  ({r['rows']:,} rows)   model AUC {r['model_auc']}   "
              f"winners by definition {r['base_rate'] * 100:.1f}%")
        print(f"  {'picking the top 10% by':<24}{'winners':>9}{'5d return':>11}{'vs all':>9}{'t':>7}{'rank IC':>9}{'t':>7}")
        for s in scores:
            m = r["rankings"][s]
            print(f"  {names[s]:<24}{m['hit_rate_top10pct'] * 100:>8.1f}%{m['ret_top10pct_5d'] * 100:>10.2f}%"
                  f"{m['excess_5d'] * 100:>+8.2f}%{m['excess_t_stat']:>7.2f}{m['rank_ic']:>+9.4f}{m['rank_ic_t_stat']:>7.2f}")
        print(f"  {'All eligible stocks':<24}{r['base_rate'] * 100:>8.1f}%{r['rankings']['model']['ret_all_5d'] * 100:>10.2f}%")
        print(f"  {'NIFTY 50':<24}{'':>9}{r['nifty50_5d'] * 100:>10.2f}%")
        print("  Website signal:  " + "   ".join(
            f"{k}: {v['share_of_rows'] * 100:.0f}% of rows, winners {v['winner_rate'] * 100:.1f}%, vs all {v['excess_5d'] * 100:+.2f}% (t {v['excess_t_stat']})"
            for k, v in sorted(r["rule_signal"].items())))
    print("\nTEST by year — AI model:  " + "   ".join(
        f"{y}: vs all {v['model']['excess_5d'] * 100:+.2f}%, IC {v['model']['rank_ic']:+.3f}" for y, v in result["test_by_year"].items()))
    w = result["walk_forward"]
    print(f"WALK-FORWARD (retrained each year): vs all {w['all_years']['excess_5d'] * 100:+.2f}% (t {w['all_years']['excess_t_stat']}), "
          f"IC {w['all_years']['rank_ic']:+.4f} (t {w['all_years']['rank_ic_t_stat']}), AUC {w['auc']}   "
          f"| fixed model on the same rows: vs all {w['fixed_model_same_rows']['excess_5d'] * 100:+.2f}%, IC {w['fixed_model_same_rows']['rank_ic']:+.4f}")
    print("CALIBRATION on test (average score -> how often it really was a winner):")
    print("  " + "  ".join(f"{c['mean_score']:.2f}->{c['actual_rate']:.2f}" for c in result["calibration_test"]))
    print(f"\nSaved {os.path.join(args.model, 'evaluation.json')}")
    return result


if __name__ == "__main__":
    main()
