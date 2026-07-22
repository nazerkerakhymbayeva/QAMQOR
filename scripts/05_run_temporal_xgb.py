#!/usr/bin/env python3
"""Stage 5 -- temporal XGBoost baseline.

Bridges the frame-level tree ensembles and the recurrent model: it keeps the
gradient-boosted-tree classifier but replaces the per-frame features with the
engineered temporal summary from :func:`qamqor.features.temporal_features`
(current frame + windowed mean/std + first-order delta). The label vector is
shifted by ``window`` to align with the reduced feature matrix.

The training uses XGBoost's ``hist`` tree method via a memory-safe
``QuantileDMatrix``, which keeps peak memory bounded for the wide Mediapipe
modalities without altering the resulting metrics (``QuantileDMatrix`` is the
representation ``XGBClassifier(tree_method="hist")`` builds internally).

Examples
--------
    python scripts/05_run_temporal_xgb.py
    python scripts/05_run_temporal_xgb.py --tool Mediapipe --task binary
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import load_split
from qamqor.features import temporal_features
from qamqor.metrics import evaluate

N_ROUNDS = 300
MAX_BIN = 256
BASE_PARAMS = {"max_depth": 8, "eta": 0.03, "tree_method": "hist", "max_bin": MAX_BIN}


def train_predict(X_train_t, y_train_t, X_test_t, task):
    params = dict(BASE_PARAMS)
    if task == "multiclass":
        params["objective"] = "multi:softprob"
        params["num_class"] = int(len(np.unique(y_train_t)))
    else:
        params["objective"] = "binary:logistic"

    start = time.time()
    try:
        dtrain = xgb.QuantileDMatrix(X_train_t, label=y_train_t)
    except Exception:
        dtrain = xgb.DMatrix(X_train_t, label=y_train_t)

    booster = xgb.train(params, dtrain, num_boost_round=N_ROUNDS)
    del dtrain
    gc.collect()

    raw = booster.predict(xgb.DMatrix(X_test_t))
    elapsed = time.time() - start
    pred = raw.argmax(axis=1) if task == "multiclass" else (raw >= 0.5).astype(int)

    del booster, raw
    gc.collect()
    return pred, elapsed


def parse_args():
    p = argparse.ArgumentParser(description="Run the temporal XGBoost baseline.")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default=None)
    p.add_argument("--task", choices=config.TASKS, default=None)
    p.add_argument("--window", type=int, default=config.WINDOW)
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS
    os.makedirs(args.results_dir, exist_ok=True)

    rows = []
    for tool in tools:
        for task in tasks:
            for modality in config.MODALITIES[tool]:
                mcode = config.MODALITY_CODE[modality]
                for split in config.SPLITS:
                    try:
                        X_train, X_test, y_train, y_test = load_split(
                            split, modality, task, args.splits_dir)
                    except FileNotFoundError:
                        print(f"[skip] {tool} {task} {split} {modality}: "
                              f"split not found")
                        continue

                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)
                    y_test = le.transform(y_test)

                    X_train_t = temporal_features(X_train, args.window)
                    X_test_t = temporal_features(X_test, args.window)
                    y_train_t = y_train[args.window:]
                    y_test_t = y_test[args.window:]

                    pred, elapsed = train_predict(
                        X_train_t, y_train_t, X_test_t, task)
                    del X_train_t, X_test_t
                    gc.collect()

                    metrics = evaluate(y_test_t, pred)
                    rows.append({
                        "Split": split, "Modality": mcode,
                        "Model": "Temporal XGBoost", "tool": tool, "class": task,
                        **{k: round(v, 4) for k, v in metrics.items()},
                        "Time (sec)": round(elapsed, 3),
                    })
                    print(f"[ok] {tool:9s} {task:10s} {split:8s} {mcode} "
                          f"TemporalXGB Acc={metrics['Accuracy']:.4f} "
                          f"F1={metrics['F1-score']:.4f} ({elapsed:.1f}s)")

    out = os.path.join(args.results_dir, "QAMQOR_temporal_xgb.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
