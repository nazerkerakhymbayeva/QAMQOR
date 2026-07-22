#!/usr/bin/env python3
"""Stage 3 -- leave-children-out 5-fold cross-validation (XGBoost).

Complements the single held-out CHILD split with a 5-fold, subject-independent
cross-validation. :class:`~sklearn.model_selection.GroupKFold` on ``childID``
guarantees that no child appears in both the training and test partition of any
fold, so the reported mean +/- SD across folds estimates how the model
generalizes to *unseen children* -- the condition that matters for real-world
deployment -- while also quantifying the variance of that estimate.

The single CHILD split answers "how well does it do on one held-out group of
children?"; this stage answers "how stable is that answer, and is it an
artefact of one lucky partition?" -- directly supporting the benchmark's central
claim that the evaluation protocol, not only the classifier, drives reported
performance.

Examples
--------
    python scripts/03_run_children_cv.py
    python scripts/03_run_children_cv.py --tool OpenPose --folds 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import build_modalities, load_raw
from qamqor.metrics import evaluate


def run_children_cv(df, modalities, target, task, tool, n_folds, group_col):
    """Return one mean +/- SD summary row per modality."""
    rows = []
    groups_all = df[group_col].values

    for modality, cols in modalities.items():
        if not cols:
            print(f"  [skip] {modality}: no columns")
            continue

        X = df[cols].fillna(0).values
        y = LabelEncoder().fit_transform(df[target].values)
        groups = groups_all

        gkf = GroupKFold(n_splits=n_folds)
        fold_metrics = {k: [] for k in config.METRIC_KEYS}
        fold_times = []

        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
            assert set(np.unique(groups[tr])).isdisjoint(np.unique(groups[te])), \
                "Child leaked across a fold."

            model = XGBClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=8,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss" if task == "binary" else "mlogloss",
                random_state=config.SEED, n_jobs=-1,
            )
            t0 = time.time()
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            fold_times.append(time.time() - t0)

            for k, v in evaluate(y[te], pred).items():
                fold_metrics[k].append(v)
            print(f"  [{config.MODALITY_CODE[modality]}] fold {fold}/{n_folds} "
                  f"acc={fold_metrics['Accuracy'][-1]:.4f}")

        row = {
            "Split": config.CHILD_CV_SPLIT,
            "Modality": config.MODALITY_CODE[modality],
            "Model": "XGBoost", "tool": tool, "class": task,
        }
        for k in config.METRIC_KEYS:
            a = np.array(fold_metrics[k])
            row[k] = f"{a.mean():.4f} \u00b1 {a.std(ddof=1):.4f}"
            row[f"{k}_mean"] = round(float(a.mean()), 4)
            row[f"{k}_sd"] = round(float(a.std(ddof=1)), 4)
        t = np.array(fold_times)
        row["Time (sec)"] = f"{t.mean():.3f} \u00b1 {t.std(ddof=1):.3f}"
        rows.append(row)
        print(f"[{tool}] {task} {row['Modality']} -> Acc {row['Accuracy']}\n")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Leave-children-out 5-fold CV.")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default=None)
    p.add_argument("--task", choices=config.TASKS, default=None)
    p.add_argument("--folds", type=int, default=config.N_FOLDS)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS
    os.makedirs(args.results_dir, exist_ok=True)

    all_rows = []
    for tool in tools:
        df = load_raw(tool, args.data_dir)
        modalities = build_modalities(df, tool)
        for task in tasks:
            print(f"\n=== {tool} | {task} | {args.folds}-fold leave-children-out ===")
            all_rows += run_children_cv(
                df, modalities, config.TARGET_COL[task], task, tool,
                args.folds, config.GROUP_COL["CHILD"])

    out = os.path.join(args.results_dir, "QAMQOR_children_cv.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
