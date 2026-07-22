#!/usr/bin/env python3
"""Stage 2 -- conventional machine-learning baselines.

Trains the four reference classifiers -- LightGBM, CatBoost, XGBoost, and
Logistic Regression -- on every predefined split/modality/task. These models are
intentionally conventional: the contribution of QAMQOR is the dataset and the
evaluation protocol, and these baselines exist to provide reproducible reference
points against which future deep and sequence-learning methods can be compared
on identical partitions.

Logistic Regression is trained on standardized features (fit on train only, to
avoid leakage); the tree ensembles are scale-invariant and use the raw features.
All hyper-parameters are fixed and reported in the manuscript for reproducibility.

Examples
--------
    python scripts/02_run_baselines.py
    python scripts/02_run_baselines.py --tool Mediapipe --task binary
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import load_split
from qamqor.metrics import evaluate

# Optional heavy dependencies are imported lazily so the script degrades
# gracefully if one is unavailable in the environment.
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

from xgboost import XGBClassifier


# --------------------------------------------------------------------------- #
# Model factory (fixed hyper-parameters -- see manuscript, Table of settings)
# --------------------------------------------------------------------------- #
def build_models(task: str):
    """Return the {name: estimator} mapping of reference classifiers.

    ``eval_metric`` is task-dependent for the gradient-boosted trees; every other
    hyper-parameter is fixed across tasks, tools, modalities, and splits.
    """
    xgb_eval = "logloss" if task == "binary" else "mlogloss"
    models = {}

    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, random_state=config.SEED,
            verbose=-1,
        )
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, learning_rate=0.03, random_seed=config.SEED, verbose=0,
        )

    models["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, eval_metric=xgb_eval,
        random_state=config.SEED, n_jobs=-1,
    )
    models["Logistic Regression"] = LogisticRegression(max_iter=3000)
    return models


def parse_args():
    p = argparse.ArgumentParser(description="Run conventional ML baselines.")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default=None)
    p.add_argument("--task", choices=config.TASKS, default=None)
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS
    os.makedirs(args.results_dir, exist_ok=True)

    if not _HAS_LGBM:
        print("[warn] lightgbm unavailable -- LightGBM baseline skipped.")
    if not _HAS_CATBOOST:
        print("[warn] catboost unavailable -- CatBoost baseline skipped.")

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
                              f"split not found (run 01_make_splits.py first)")
                        continue

                    # Integer-encode labels for the multiclass task.
                    if task == "multiclass":
                        le = LabelEncoder()
                        y_train = le.fit_transform(y_train)
                        y_test = le.transform(y_test)

                    # Standardize for LR only (fit on train).
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    for name, model in build_models(task).items():
                        start = time.time()
                        if name == "Logistic Regression":
                            model.fit(X_train_s, y_train)
                            pred = model.predict(X_test_s)
                        else:
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)
                        elapsed = time.time() - start

                        metrics = evaluate(y_test, pred)
                        rows.append({
                            "Split": split, "Modality": mcode, "Model": name,
                            "tool": tool, "class": task,
                            **{k: round(v, 4) for k, v in metrics.items()},
                            "Time (sec)": round(elapsed, 3),
                        })
                        print(f"[ok] {tool:9s} {task:10s} {split:8s} {mcode} "
                              f"{name:20s} Acc={metrics['Accuracy']:.4f} "
                              f"F1={metrics['F1-score']:.4f} ({elapsed:.1f}s)")

    out = os.path.join(args.results_dir, "QAMQOR_baselines.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
