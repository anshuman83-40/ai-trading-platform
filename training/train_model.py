"""
TradePulse ML — step 3: train the first model (XGBoost v1) and save it.

  - Learns from the TRAIN period only.
  - The VALIDATION period decides when to stop adding trees (early stopping). It is not used for learning.
  - The TEST period is not opened here at all. It is looked at once, in evaluate_model.py.
  - Settings are fixed, sensible defaults — no search over hundreds of combinations, because picking the
    best of many tries on the same validation data would overfit to it.

Run from the project folder, after build_dataset.py:
    python training/train_model.py                       (CPU — takes about a minute)
    python training/train_model.py --device cuda         (GPU, if available; not needed at this data size)

Output: models/xgb_v1/  model.json, feature_schema.json, metadata.json
The website never trains: it will only load these saved files.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import config as cfg                                   # noqa: E402
from ml import features as ft                          # noqa: E402
from build_dataset import split_of                     # noqa: E402
from data_collection import sha256                     # noqa: E402

MODEL_VERSION = "xgb_v1"
SEED = 42
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 4,              # shallow trees: stock data is noisy, deep trees memorise noise
    "min_child_weight": 200,     # a rule must hold for many rows before it is trusted
    "subsample": 0.8,            # each tree sees 80% of the rows …
    "colsample_bytree": 0.8,     # … and 80% of the features
    "reg_lambda": 5.0,
    "n_estimators": 2000,        # upper limit; early stopping normally ends much sooner
    "early_stopping_rounds": 100,
    "random_state": SEED,
}


def load_labelled(path):
    """Rows that have an answer, with their period (train / valid / test / gap)."""
    data = pd.read_csv(path, parse_dates=["date"])
    data = data[data["label"].notna()].copy()
    data["split"] = split_of(data["date"])
    return data


def daily_metrics(frame, score_col="score"):
    """Numbers that matter for picking stocks — calculated per day, then averaged over days."""
    def one_day(g):
        n_top = max(1, int(round(0.10 * len(g))))
        top = g.nlargest(n_top, score_col)
        r = g[[score_col, "fwd_ret"]].rank()
        return pd.Series({"hit_top10": top["label"].mean(),                     # of the 10% best-scored stocks, how many were winners
                          "ret_top10": top["fwd_ret"].mean(),                   # their average 5-day return (before costs)
                          "ret_all": g["fwd_ret"].mean(),                       # average of every eligible stock that day
                          "rank_ic": np.corrcoef(r[score_col], r["fwd_ret"])[0, 1]})
    d = frame.groupby("date")[[score_col, "label", "fwd_ret"]].apply(one_day)
    return {"days": int(len(d)),
            "hit_rate_top10pct": round(float(d["hit_top10"].mean()), 4),
            "mean_5d_return_top10pct": round(float(d["ret_top10"].mean()), 5),
            "mean_5d_return_all_stocks": round(float(d["ret_all"].mean()), 5),
            "rank_ic": round(float(d["rank_ic"].mean()), 4),
            "rank_ic_positive_days": round(float((d["rank_ic"] > 0).mean()), 4)}


def metrics(frame):
    from sklearn.metrics import log_loss, roc_auc_score
    base = float(frame["label"].mean())
    out = {"rows": int(len(frame)), "base_rate": round(base, 4),
           "auc": round(float(roc_auc_score(frame["label"], frame["score"])), 4),
           "logloss": round(float(log_loss(frame["label"], frame["score"])), 5),
           "logloss_if_always_base_rate": round(float(log_loss(frame["label"], np.full(len(frame), base))), 5)}
    out.update(daily_metrics(frame))
    return out


def train(data, params=None, device="cpu", n_jobs=8):
    """-> (fitted model, table of train+valid rows with a 'score' column). Never touches test rows."""
    import xgboost as xgb
    params = dict(PARAMS if params is None else params)
    tr, va = data[data["split"] == "train"], data[data["split"] == "valid"]
    model = xgb.XGBClassifier(**params, device=device, n_jobs=n_jobs)
    model.fit(tr[ft.FEATURES], tr["label"], eval_set=[(va[ft.FEATURES], va["label"])], verbose=False)
    scored = pd.concat([tr, va]).copy()
    scored["score"] = model.predict_proba(scored[ft.FEATURES])[:, 1]
    return model, scored


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train and save XGBoost v1.")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", MODEL_VERSION))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args(argv)
    d_path = os.path.join(args.data, "dataset_v1.csv.gz")
    if not os.path.exists(d_path):
        sys.exit("No dataset yet — run:  python training/build_dataset.py")
    import sklearn
    import xgboost as xgb

    data = load_labelled(d_path)
    t0 = time.time()
    model, scored = train(data, device=args.device)
    seconds = round(time.time() - t0, 1)
    tr, va = scored[scored["split"] == "train"], scored[scored["split"] == "valid"]
    m_train, m_valid = metrics(tr), metrics(va)

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model.json")
    model.get_booster().save_model(model_path)
    with open(os.path.join(args.out, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_version": ft.FEATURE_VERSION, "features": ft.FEATURES,
                   "note": "Order matters. Calculated by ml/features.py from data up to the decision day only."}, f, indent=1)
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    with open(os.path.join(args.data, "dataset_manifest.json"), encoding="utf-8") as f:
        ds = json.load(f)
    meta = {
        "model_version": MODEL_VERSION, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "XGBoost gradient-boosted trees (binary classifier)",
        "target": cfg.TARGET, "horizon_trading_days": cfg.HORIZON, "top_fraction": cfg.TOP_FRACTION,
        "output_meaning": "score between 0 and 1: higher = more likely to be in the top 30% over the next 5 trading days. "
                          "NOT a calibrated probability and NOT a promise of profit.",
        "universe": "NIFTY 200 members as of the data download (survivorship bias: past members are missing)",
        "data_source": "Yahoo Finance daily prices via yfinance (free, not exchange-licensed)",
        "feature_version": ft.FEATURE_VERSION, "n_features": len(ft.FEATURES),
        "periods": {s: {"first": str(g["date"].min().date()), "last": str(g["date"].max().date()), "rows": int(len(g))}
                    for s, g in data.groupby("split")},
        "test_period_used_in_training": False,
        "params": PARAMS, "seed": SEED, "device": args.device,
        "trees_used": int(model.best_iteration + 1), "training_seconds": seconds,
        "metrics": {"train": m_train, "valid": m_valid},
        "top_features_by_gain": [{"feature": k, "share": round(v / total, 4)}
                                 for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:10]],
        "inputs": {"dataset_v1.csv.gz": sha256(d_path), **ds.get("inputs", {})},
        "files": {"model.json": sha256(model_path)},
        "libraries": {"xgboost": xgb.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__, "python": sys.version.split()[0]},
        "git_commit": git_commit(),
        "reproduce": ["python training/data_collection.py --end " + ds.get("per_split", {}).get("test", {}).get("last", "YYYY-MM-DD"),
                      "python training/build_dataset.py", "python training/train_model.py"],
    }
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"Trained {MODEL_VERSION} on {len(tr):,} rows in {seconds}s ({args.device}); "
          f"stopped at {meta['trees_used']} trees")
    print(f"{'':<10}{'AUC':>8}{'logloss':>10}{'(no-skill)':>12}{'hit top10%':>12}{'rank IC':>9}")
    for name, m in (("train", m_train), ("valid", m_valid)):
        print(f"{name:<10}{m['auc']:>8.4f}{m['logloss']:>10.5f}{m['logloss_if_always_base_rate']:>12.5f}"
              f"{m['hit_rate_top10pct']:>12.4f}{m['rank_ic']:>9.4f}")
    print(f"Base rate (share of winners by definition): {m_valid['base_rate']:.3f}.  AUC 0.5 = no skill.")
    print("Top features:", ", ".join(f"{x['feature']} {x['share'] * 100:.0f}%" for x in meta["top_features_by_gain"][:6]))
    print(f"Saved to {args.out}")
    return meta


if __name__ == "__main__":
    main()
