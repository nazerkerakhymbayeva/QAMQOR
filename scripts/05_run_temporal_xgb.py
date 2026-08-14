#!/usr/bin/env python3
"""Stage 5 -- temporal XGBoost baseline (balancing + resume + low peak memory).

Class balancing
---------------
With ``--balanced``, inverse-frequency per-sample weights are attached to the
training matrix (same mechanism as the other models), so cross-model comparison
is fair.


Examples
--------
    python scripts/05_run_temporal_xgb.py                # unbalanced reference
    python scripts/05_run_temporal_xgb.py --balanced     # balanced variant (resumable)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import load_split
from qamqor.features import temporal_features
from qamqor.metrics import evaluate

N_ROUNDS = 300
MAX_BIN = 256
BASE_PARAMS = {"max_depth": 8, "eta": 0.03, "tree_method": "hist", "max_bin": MAX_BIN}


def train_and_predict(X_train_t, y_train_t, X_test_raw, window, task, weight=None):
    """Train on the engineered features and predict, keeping peak memory low.

    ``X_train_t`` is the dense engineered training matrix; it is freed as soon as
    the ``QuantileDMatrix`` has been built, before boosting. ``X_test_raw`` is the
    *raw* test matrix -- the engineered test features are built only after
    training, so the two large dense matrices never coexist with the booster.
    """
    params = dict(BASE_PARAMS)
    if task == "multiclass":
        params["objective"] = "multi:softprob"
        params["num_class"] = int(len(np.unique(y_train_t)))
    else:
        params["objective"] = "binary:logistic"

    t0 = time.time()
    try:
        dtrain = xgb.QuantileDMatrix(X_train_t, label=y_train_t, weight=weight)
    except Exception:
        dtrain = xgb.DMatrix(X_train_t, label=y_train_t, weight=weight)

    # Free the dense training matrix BEFORE boosting: the quantized DMatrix holds
    # everything the booster needs, so this releases ~1.7 GB for Mediapipe.
    X_train_t = None
    gc.collect()

    booster = xgb.train(params, dtrain, num_boost_round=N_ROUNDS)
    del dtrain
    gc.collect()

    # Build the engineered TEST features only now that training is done.
    X_test_t = temporal_features(X_test_raw, window)
    raw = booster.predict(xgb.DMatrix(X_test_t))
    elapsed = time.time() - t0

    pred = raw.argmax(axis=1) if task == "multiclass" else (raw >= 0.5).astype(int)
    del booster, raw, X_test_t
    gc.collect()
    return pred, elapsed


def load_done(out):
    """Return (rows, done_keys) for resuming from an existing output CSV."""
    if not os.path.exists(out):
        return [], set()
    prev = pd.read_csv(out)
    rows = prev.to_dict("records")
    done = {(r["Split"], r["Modality"], r["tool"], r["class"], r["Model"])
            for r in rows}
    print(f"[resume] {len(done)} configurations already in {out}; skipping them.")
    return rows, done


def parse_args():
    p = argparse.ArgumentParser(description="Run the temporal XGBoost baseline.")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default=None)
    p.add_argument("--task", choices=config.TASKS, default=None)
    p.add_argument("--window", type=int, default=config.WINDOW)
    p.add_argument("--balanced", action="store_true",
                   help="Train with balanced (inverse-frequency) sample weights.")
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS
    os.makedirs(args.results_dir, exist_ok=True)
    suffix = "-bal" if args.balanced else ""
    model_name = "Temporal XGBoost" + suffix
    print(f"balanced={args.balanced}")

    out = os.path.join(
        args.results_dir,
        "QAMQOR_temporal_xgb_bal.csv" if args.balanced
        else "QAMQOR_temporal_xgb.csv")
    rows, done = load_done(out)

    for tool in tools:
        for task in tasks:
            for modality in config.MODALITIES[tool]:
                mcode = config.MODALITY_CODE[modality]
                for split in config.SPLITS:
                    key = (split, mcode, tool, task, model_name)
                    if key in done:
                        print(f"[skip-done] {tool} {task} {split} {mcode}")
                        continue
                    try:
                        X_train, X_test, y_train, y_test = load_split(
                            split, modality, task, args.splits_dir)
                    except FileNotFoundError:
                        print(f"[skip] {tool} {task} {split} {mcode}: split not found")
                        continue

                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)
                    y_test = le.transform(y_test)

                    X_train = np.asarray(X_train, dtype=np.float32)
                    X_test = np.asarray(X_test, dtype=np.float32)   # kept raw until after training
                    X_train_t = temporal_features(X_train, args.window)
                    del X_train
                    gc.collect()

                    y_train_t = y_train[args.window:]
                    y_test_t = y_test[args.window:]
                    sw = (compute_sample_weight("balanced", y_train_t)
                          if args.balanced else None)

                    pred, elapsed = train_and_predict(
                        X_train_t, y_train_t, X_test, args.window, task, weight=sw)
                    del X_train_t, X_test
                    gc.collect()

                    metrics = evaluate(y_test_t, pred)
                    rows.append({
                        "Split": split, "Modality": mcode, "Model": model_name,
                        "tool": tool, "class": task,
                        **{k: round(v, 4) for k, v in metrics.items()},
                        "Time (sec)": round(elapsed, 3),
                    })
                    print(f"[ok] {tool:9s} {task:10s} {split:8s} {mcode} "
                          f"{model_name} Acc={metrics['Accuracy']:.4f} "
                          f"F1={metrics['F1-score']:.4f} "
                          f"BalAcc={metrics['Balanced Accuracy']:.4f} ({elapsed:.1f}s)")
                    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
