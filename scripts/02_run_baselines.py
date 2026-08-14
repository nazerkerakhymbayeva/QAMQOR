#!/usr/bin/env python3
"""Stage 2 -- conventional machine-learning baselines (with optional balancing).

Trains the four reference classifiers -- LightGBM, CatBoost, XGBoost, and
Logistic Regression -- on every predefined split/modality/task. 

Class balancing
---------------
With ``--balanced``, every classifier is trained with inverse-frequency
per-sample weights (``sklearn.utils.class_weight.compute_sample_weight``), the
same mechanism applied to the deep temporal models, so that the imbalance
handling is identical across all models and cross-model comparison is fair. 

Examples
--------
    python scripts/02_run_baselines.py                        # unbalanced reference
    python scripts/02_run_baselines.py --balanced             # balanced variants
    python scripts/02_run_baselines.py --balanced --tool Mediapipe --task binary
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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
def build_models(task: str, catboost_threads: int = 4,
                 catboost_ram_limit: str | None = "4gb",
                 catboost_border: int | None = None):
    
    xgb_eval = "logloss" if task == "binary" else "mlogloss"
    models = {}

    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, random_state=config.SEED,
            verbose=-1, n_jobs=-1,
        )
    if _HAS_CATBOOST:
        cb_kwargs = dict(
            iterations=300, learning_rate=0.03, random_seed=config.SEED,
            verbose=0, thread_count=catboost_threads,
        )
        if catboost_ram_limit:
            cb_kwargs["used_ram_limit"] = catboost_ram_limit
        if catboost_border:
            cb_kwargs["border_count"] = catboost_border
        models["CatBoost"] = CatBoostClassifier(**cb_kwargs)

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
    p.add_argument("--balanced", action="store_true",
                   help="Train with balanced (inverse-frequency) sample weights.")
    p.add_argument("--catboost-threads", type=int, default=4,
                   help="CatBoost thread_count (lower = less memory). Default 4.")
    p.add_argument("--catboost-ram", default="4gb",
                   help="CatBoost used_ram_limit, e.g. '4gb' or '2gb'. "
                        "Empty string disables the cap.")
    p.add_argument("--catboost-border", type=int, default=0,
                   help="CatBoost border_count (e.g. 128 to halve quantization "
                        "memory). 0 keeps the CatBoost default. Changing this "
                        "alters the model, so re-run ALL CatBoost rows.")
    p.add_argument("--skip-models", default="",
                   help="Comma-separated model names to skip entirely, e.g. "
                        "'CatBoost' when it will not fit in memory. Already-saved "
                        "rows for that model are kept.")
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args()


def load_done(out):
    """Return (rows, done_keys) for resuming from an existing output CSV."""
    if not os.path.exists(out):
        return [], set()
    prev = pd.read_csv(out)
    rows = prev.to_dict("records")
    done = {(r["Split"], r["Modality"], r["tool"], r["class"], r["Model"])
            for r in rows}
    print(f"[resume] {len(done)} model-configurations already in {out}; "
          f"skipping them.")
    return rows, done


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS
    os.makedirs(args.results_dir, exist_ok=True)
    suffix = "-bal" if args.balanced else ""
    cb_opts = dict(catboost_threads=args.catboost_threads,
                   catboost_ram_limit=(args.catboost_ram or None),
                   catboost_border=(args.catboost_border or None))
    skip = {m.strip() for m in args.skip_models.split(",") if m.strip()}
    if skip:
        print(f"skipping models: {sorted(skip)}")

    if not _HAS_LGBM:
        print("[warn] lightgbm unavailable -- LightGBM baseline skipped.")
    if not _HAS_CATBOOST:
        print("[warn] catboost unavailable -- CatBoost baseline skipped.")
    print(f"balanced={args.balanced}")

    out = os.path.join(
        args.results_dir,
        "QAMQOR_baselines_bal.csv" if args.balanced else "QAMQOR_baselines.csv")
    rows, done = load_done(out)

    for tool in tools:
        for task in tasks:
            for modality in config.MODALITIES[tool]:
                mcode = config.MODALITY_CODE[modality]
                # Names of the models this run would produce for this config.
                planned = [name + suffix for name in build_models(task, **cb_opts)
                           if name not in skip]
                for split in config.SPLITS:
                    # Skip the whole configuration only if every planned model
                    # is already present (avoids reloading the split needlessly).
                    if all((split, mcode, tool, task, m) in done for m in planned):
                        print(f"[skip-done] {tool} {task} {split} {mcode} (all models)")
                        continue
                    try:
                        X_train, X_test, y_train, y_test = load_split(
                            split, modality, task, args.splits_dir)
                    except FileNotFoundError:
                        print(f"[skip] {tool} {task} {split} {modality}: "
                              f"split not found (run 01_make_splits.py first)")
                        continue

                    X_train = np.asarray(X_train, dtype=np.float32)
                    X_test = np.asarray(X_test, dtype=np.float32)

                    # Integer-encode labels for the multiclass task.
                    if task == "multiclass":
                        le = LabelEncoder()
                        y_train = le.fit_transform(y_train)
                        y_test = le.transform(y_test)

                    # Balanced per-sample weights (identical mechanism to the
                    # deep models); None reproduces the unweighted reference.
                    sw = (compute_sample_weight("balanced", y_train)
                          if args.balanced else None)

                    models = build_models(task, **cb_opts)

                    def record(mname, pred, elapsed):
                        metrics = evaluate(y_test, pred)
                        rows.append({
                            "Split": split, "Modality": mcode,
                            "Model": mname, "tool": tool, "class": task,
                            **{k: round(v, 4) for k, v in metrics.items()},
                            "Time (sec)": round(elapsed, 3),
                        })
                        print(f"[ok] {tool:9s} {task:10s} {split:8s} {mcode} "
                              f"{mname:24s} Acc={metrics['Accuracy']:.4f} "
                              f"F1={metrics['F1-score']:.4f} "
                              f"BalAcc={metrics['Balanced Accuracy']:.4f} "
                              f"({elapsed:.1f}s)")
                        pd.DataFrame(rows).to_csv(out, index=False)

                    # Tree ensembles first, on the RAW features. Each model's
                    # memory is released before the next one is trained.
                    for name, model in models.items():
                        if name == "Logistic Regression" or name in skip:
                            continue
                        mname = name + suffix
                        if (split, mcode, tool, task, mname) in done:
                            continue
                        start = time.time()
                        model.fit(X_train, y_train, sample_weight=sw)
                        pred = model.predict(X_test)
                        record(mname, pred, time.time() - start)
                        del model, pred
                        gc.collect()

                    # Logistic Regression LAST: build the standardized copies only
                    # now, so the ~0.9 GB scaled matrices never coexist with the
                    # memory-heavy tree fits, and free them immediately after.
                    lr_name = "Logistic Regression"
                    lr_key = (split, mcode, tool, task, lr_name + suffix)
                    if (lr_name in models and lr_name not in skip
                            and lr_key not in done):
                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
                        X_test_s = scaler.transform(X_test).astype(np.float32)
                        lr = models[lr_name]
                        start = time.time()
                        lr.fit(X_train_s, y_train, sample_weight=sw)
                        pred = lr.predict(X_test_s)
                        record(lr_name + suffix, pred, time.time() - start)
                        del scaler, X_train_s, X_test_s, lr, pred
                        gc.collect()

                    del X_train, X_test
                    gc.collect()

    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
