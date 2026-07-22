#!/usr/bin/env python3
"""Stage 4 -- LSTM sequence baseline (Mediapipe modalities).

Provides a recurrent reference point that consumes the raw temporal window rather
than engineered summary statistics. Each frame is classified from the ``window``
frames preceding it; the label vector is shifted by ``window`` to stay aligned
with the generated sequences.

The architecture and training schedule are fixed (LSTM(64) -> Dropout(0.3) ->
Dense(32, ReLU) -> Dropout(0.2) -> output; Adam; early stopping on validation
loss with patience 10; learning-rate reduction on plateau). The output head is
task-dependent, matching the paper: a single sigmoid unit with binary
cross-entropy for the binary task, and a softmax over the classes with sparse
categorical cross-entropy for the multiclass task. All random seeds are set, so
runs are reproducible up to the nondeterminism inherent in GPU kernels.

Examples
--------
    python scripts/04_run_lstm.py
    python scripts/04_run_lstm.py --task binary --epochs 100
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import load_split
from qamqor.features import create_sequences
from qamqor.metrics import evaluate

# TensorFlow is imported lazily (inside the functions that need it) so that the
# command-line interface -- including ``--help`` -- works even in environments
# where TensorFlow is not installed.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seed(seed=config.SEED):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm(input_shape, n_classes, task, lstm_units=64, dense_units=32,
               dropout1=0.3, dropout2=0.2):
    """Build the LSTM with a task-dependent output head.

    Following the paper: the binary task uses a single sigmoid unit with binary
    cross-entropy; the multiclass task uses a softmax over ``n_classes`` with
    sparse categorical cross-entropy.
    """
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    if task == "binary":
        out_units, out_act, loss = 1, "sigmoid", "binary_crossentropy"
    else:
        out_units, out_act, loss = n_classes, "softmax", "sparse_categorical_crossentropy"

    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape),
        Dropout(dropout1),
        Dense(dense_units, activation="relu"),
        Dropout(dropout2),
        Dense(out_units, activation=out_act),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def run_one(split, modality, task, window, epochs, batch_size, splits_dir, verbose):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    X_train, X_test, y_train, y_test = load_split(split, modality, task, splits_dir)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, window)

    set_seed()
    model = build_lstm(
        input_shape=(window, X_train.shape[1]),
        n_classes=len(le.classes_),
        task=task,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    t0 = time.time()
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=callbacks, verbose=verbose)
    prob = model.predict(X_test_seq, verbose=0)
    if task == "binary":
        pred = (prob.ravel() >= 0.5).astype(int)
    else:
        pred = np.argmax(prob, axis=1)
    elapsed = time.time() - t0

    metrics = evaluate(y_test_seq, pred)
    return metrics, elapsed


def parse_args():
    p = argparse.ArgumentParser(description="Run the LSTM sequence baseline.")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default="Mediapipe",
                   help="Tool whose modalities to use (default: Mediapipe).")
    p.add_argument("--task", choices=config.TASKS, default=None)
    p.add_argument("--window", type=int, default=config.WINDOW)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [args.task] if args.task else config.TASKS
    os.makedirs(args.results_dir, exist_ok=True)

    rows = []
    for task in tasks:
        for modality in config.MODALITIES[args.tool]:
            mcode = config.MODALITY_CODE[modality]
            for split in config.SPLITS:
                try:
                    metrics, elapsed = run_one(
                        split, modality, task, args.window, args.epochs,
                        args.batch_size, args.splits_dir, args.verbose)
                except FileNotFoundError:
                    print(f"[skip] {task} {split} {modality}: split not found")
                    continue
                rows.append({
                    "Split": split, "Modality": mcode, "Model": "LSTM",
                    "tool": args.tool, "class": task,
                    **{k: round(v, 4) for k, v in metrics.items()},
                    "Time (sec)": round(elapsed, 3),
                })
                print(f"[ok] {task:10s} {split:8s} {mcode} LSTM "
                      f"Acc={metrics['Accuracy']:.4f} F1={metrics['F1-score']:.4f} "
                      f"({elapsed:.1f}s)")

    out = os.path.join(args.results_dir, "QAMQOR_lstm.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
