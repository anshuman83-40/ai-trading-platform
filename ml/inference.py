"""
TradePulse ML — loading the saved model and scoring rows. Nothing here trains anything.

Used by the evaluation, the backtest and the daily prediction script, so they all score in exactly one way.
"""
import json
import os

import numpy as np


def load_model(model_dir):
    """-> (booster, metadata, feature schema) from a models/<version>/ folder."""
    import xgboost as xgb
    with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, "feature_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.json"))
    return booster, meta, schema


def model_scores(booster, meta, schema, frame):
    """Score every row of `frame`. The feature columns must be exactly the ones the model was trained on
    (same names, same order — taken from the saved schema, not from the current code)."""
    import xgboost as xgb
    feats = schema["features"]
    missing = [c for c in feats if c not in frame.columns]
    if missing:
        raise ValueError(f"cannot score: {len(missing)} feature column(s) missing, e.g. {missing[:3]}")
    x = frame[feats].to_numpy(dtype="float64")
    if np.isinf(x).any():
        raise ValueError("cannot score: infinite values in the features")
    dm = xgb.DMatrix(x, feature_names=feats)
    return booster.predict(dm, iteration_range=(0, meta["trees_used"]))
