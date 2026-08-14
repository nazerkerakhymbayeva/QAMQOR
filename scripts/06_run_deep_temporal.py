#!/usr/bin/env python3
"""Stage 9 -- modern temporal deep baselines (TCN and Transformer encoder).

Two temporal architectures evaluated under the identical protocol as every other
model in the benchmark, consuming the same serialized ``.npy`` splits:

* ``--arch tcn``          -- Temporal Convolutional Network (dilated causal
  residual 1-D convolutions; Bai et al., 2018).
* ``--arch transformer``  -- self-attention encoder (Vaswani et al., 2017).
* ``--arch lstm``         -- the published recurrent baseline (LSTM(64) ->
  Dropout -> Dense -> head), for an imbalance-aware variant under the same
  standardization / class-weighting pipeline as the other deep models.

Imbalance handling (disclosed, uniform, no per-split tuning)
-----------------------------------------------------------
QAMQOR is strongly imbalanced, and unregularized deep models collapse to
majority-class prediction (high accuracy, macro-F1 / balanced accuracy / kappa at
the trivial baseline). Two standard, disclosed steps counter this and target the
imbalance-aware metrics the paper reports -- NOT raw accuracy, which a majority
predictor already maximizes:

* ``--standardize`` (default on): z-score each feature using statistics fit on
  the TRAINING split only (no test leakage), which stabilizes optimization.
* ``--class-weight`` (default on): balanced class weights (inverse frequency)
  computed from the training labels, applied as per-sample weights to BOTH the
  training and validation streams, so that early stopping tracks a balanced
  objective rather than the collapsed majority solution. The test stream is left
  unweighted; all reported metrics are computed on the true test distribution.

These settings are fixed across all four protocols and all modalities. To keep
the benchmark fair, run the untuned reference (``--no-standardize
--no-class-weight``) and this imbalance-aware variant as separate, clearly
labelled models (see ``--model-name``); do not select settings on test scores.

Memory-safe windowing
---------------------
Windows are streamed lazily with ``tf.keras.utils.timeseries_dataset_from_array``
(the i-th sequence is ``X[i:i+window]`` with label ``y[i+window]``, matching
``qamqor.features.create_sequences``), so peak memory is ``O(batch*window*d)``
rather than ``O(N*window*d)``.

Only dependency is TensorFlow/Keras, which the LSTM baseline already requires.

Examples
--------
    # imbalance-aware variant (default), saved under a distinct model name
    python scripts/09_run_deep_temporal.py --arch tcn --model-name "TCN-bal"

    # untuned reference, identical to the fair baseline in the main table
    python scripts/09_run_deep_temporal.py --arch tcn --no-standardize --no-class-weight

After running::

    python scripts/06_combine_results.py
    python scripts/07_statistical_tests.py --metric F1-score --task binary
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import load_split
from qamqor.metrics import evaluate

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ARCH_NAME = {"tcn": "TCN", "transformer": "Transformer", "lstm": "LSTM"}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.SEED) -> None:
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# Memory-safe lazy windowing, with optional per-sample class weights
# --------------------------------------------------------------------------- #
def make_windowed_dataset(X, y, window, batch_size, shuffle,
                          seed=config.SEED, weight_vec=None):
    """Return a ``tf.data.Dataset`` of windowed sequences.

    If ``weight_vec`` (a length-``n_classes`` tensor) is given, each element is
    ``(sequence, label, sample_weight)`` with ``sample_weight = weight_vec[label]``;
    otherwise ``(sequence, label)``. Windows are generated lazily from the base
    array, so only one batch exists in memory at a time.
    """
    import tensorflow as tf

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n_seq = len(X) - window
    if n_seq <= 0:
        raise ValueError(f"segment too short for window={window}: len={len(X)}")

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X, targets=y[window:], sequence_length=window,
        batch_size=batch_size, shuffle=shuffle, seed=seed)
    if weight_vec is not None:
        w = tf.constant(weight_vec, dtype=tf.float32)
        ds = ds.map(lambda xb, yb: (xb, yb, tf.gather(w, tf.cast(yb, tf.int32))))
    return ds.prefetch(tf.data.AUTOTUNE), n_seq


# --------------------------------------------------------------------------- #
# Output head (identical to the LSTM head)
# --------------------------------------------------------------------------- #
def _head_config(task, n_classes):
    if task == "binary":
        return 1, "sigmoid", "binary_crossentropy"
    return n_classes, "softmax", "sparse_categorical_crossentropy"


# --------------------------------------------------------------------------- #
# Architecture: Temporal Convolutional Network (TCN)
# --------------------------------------------------------------------------- #
def build_tcn(input_shape, n_classes, task,
              filters=64, kernel_size=3, dilations=(1, 2, 4),
              dropout=0.2, dense_units=32):
    from tensorflow.keras.layers import (Activation, Add, Conv1D, Dense,
                                         Dropout, GlobalAveragePooling1D, Input,
                                         LayerNormalization)
    from tensorflow.keras.models import Model

    def residual_block(x, dilation):
        prev = x
        for _ in range(2):
            x = Conv1D(filters, kernel_size, padding="causal",
                       dilation_rate=dilation)(x)
            x = LayerNormalization()(x)
            x = Activation("relu")(x)
            x = Dropout(dropout)(x)
        if prev.shape[-1] != filters:
            prev = Conv1D(filters, 1, padding="same")(prev)
        return Add()([prev, x])

    inp = Input(shape=input_shape)
    x = inp
    for dilation in dilations:
        x = residual_block(x, dilation)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

    out_units, out_act, loss = _head_config(task, n_classes)
    out = Dense(out_units, activation=out_act)(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"],
                  weighted_metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# Architecture: Transformer encoder
# --------------------------------------------------------------------------- #
def _sinusoidal_positional_encoding_np(length, depth):
    half = depth // 2
    positions = np.arange(length)[:, None]
    dims = np.arange(max(half, 1))[None, :] / max(half, 1)
    angle_rads = positions * (1.0 / (10000.0 ** dims))
    pos = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    if pos.shape[-1] < depth:
        pos = np.pad(pos, ((0, 0), (0, depth - pos.shape[-1])))
    return pos.astype(np.float32)


def build_transformer(input_shape, n_classes, task,
                      d_model=64, num_heads=4, ff_dim=128, num_layers=2,
                      dropout=0.2, dense_units=32):
    import tensorflow as tf
    from tensorflow.keras.layers import (Add, Dense, Dropout,
                                         GlobalAveragePooling1D, Input, Layer,
                                         LayerNormalization, MultiHeadAttention)
    from tensorflow.keras.models import Model

    window = input_shape[0]

    class PositionalEncoding(Layer):
        def __init__(self, length, depth, **kw):
            super().__init__(**kw)
            self._length, self._depth = length, depth

        def build(self, input_shape):
            self.pos = tf.constant(
                _sinusoidal_positional_encoding_np(self._length, self._depth))
            super().build(input_shape)

        def call(self, x):
            return x + tf.cast(self.pos, x.dtype)

    def encoder_block(x):
        attn = MultiHeadAttention(num_heads=num_heads,
                                  key_dim=max(d_model // num_heads, 1),
                                  dropout=dropout)(x, x)
        x = LayerNormalization()(Add()([x, attn]))
        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        return LayerNormalization()(Add()([x, ff]))

    inp = Input(shape=input_shape)
    x = Dense(d_model)(inp)
    x = PositionalEncoding(window, d_model)(x)
    x = Dropout(dropout)(x)
    for _ in range(num_layers):
        x = encoder_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

    out_units, out_act, loss = _head_config(task, n_classes)
    out = Dense(out_units, activation=out_act)(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"],
                  weighted_metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# Architecture: LSTM (reproduces the published Stage-4 baseline exactly)
# --------------------------------------------------------------------------- #
def build_lstm(input_shape, n_classes, task,
               lstm_units=64, dense_units=32, dropout1=0.3, dropout2=0.2):
    """LSTM(64) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> head.

    Identical architecture and training schedule to ``04_run_lstm.py`` (the
    untuned reference); the only differences when the imbalance-aware flags are
    on are the shared input standardization and balanced class weighting applied
    in ``run_one``.
    """
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    out_units, out_act, loss = _head_config(task, n_classes)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units),
        Dropout(dropout1),
        Dense(dense_units, activation="relu"),
        Dropout(dropout2),
        Dense(out_units, activation=out_act),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"],
                  weighted_metrics=["accuracy"])
    return model


BUILDERS = {"tcn": build_tcn, "transformer": build_transformer,
            "lstm": build_lstm}


# --------------------------------------------------------------------------- #
# Balanced class weights (robust to classes absent from a split)
# --------------------------------------------------------------------------- #
def balanced_weight_vector(y, n_classes):
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    vec = np.ones(n_classes, dtype=np.float32)
    for c, wi in zip(present, w):
        vec[int(c)] = wi
    return vec


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #
def run_one(arch, split, modality, task, window, epochs, batch_size,
            val_split, splits_dir, verbose, standardize=True,
            use_class_weight=True):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    X_train, X_test, y_train, y_test = load_split(split, modality, task, splits_dir)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(le.classes_)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if standardize:                                    # fit on TRAIN only
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    cut = int(len(X_train) * (1.0 - val_split))
    cut = max(window + 1, min(cut, len(X_train) - (window + 1)))

    weight_vec = None
    if use_class_weight:
        weight_vec = balanced_weight_vector(y_train[:cut], n_classes)

    set_seed()
    train_ds, _ = make_windowed_dataset(
        X_train[:cut], y_train[:cut], window, batch_size,
        shuffle=True, weight_vec=weight_vec)
    val_ds, _ = make_windowed_dataset(
        X_train[cut:], y_train[cut:], window, batch_size,
        shuffle=False, weight_vec=weight_vec)          # weighted val -> balanced early stopping
    test_ds, _ = make_windowed_dataset(
        X_test, y_test, window, batch_size, shuffle=False, weight_vec=None)
    y_test_seq = y_test[window:]

    model = BUILDERS[arch](
        input_shape=(window, X_train.shape[1]), n_classes=n_classes, task=task)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=callbacks, verbose=verbose)
    prob = model.predict(test_ds, verbose=0)
    if task == "binary":
        pred = (prob.ravel() >= 0.5).astype(int)
    else:
        pred = np.argmax(prob, axis=1)
    elapsed = time.time() - t0

    metrics = evaluate(y_test_seq, pred)
    return metrics, elapsed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Modern temporal deep baselines (TCN / Transformer) with "
                    "optional, disclosed imbalance handling.")
    p.add_argument("--arch", choices=list(BUILDERS), default="tcn")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default=None)
    p.add_argument("--task", choices=config.TASKS, default=None)
    p.add_argument("--window", type=int, default=config.WINDOW)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Mini-batch size (256 default; 32 for LSTM parity).")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--standardize", action=argparse.BooleanOptionalAction,
                   default=True, help="Z-score features (fit on train only).")
    p.add_argument("--class-weight", dest="class_weight",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Balanced class weighting to counter majority collapse.")
    p.add_argument("--model-name", default=None,
                   help="Override the Model label / output filename, e.g. "
                        "'TCN-bal', to keep tuned and reference variants apart.")
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS
    model_name = args.model_name or ARCH_NAME[args.arch]
    slug = model_name.lower().replace(" ", "_")
    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"QAMQOR_{slug}.csv")

    print(f"model={model_name} | standardize={args.standardize} | "
          f"class_weight={args.class_weight} | batch={args.batch_size}")

    rows = []
    for tool in tools:
        for task in tasks:
            for modality in config.MODALITIES[tool]:
                mcode = config.MODALITY_CODE[modality]
                for split in config.SPLITS:
                    try:
                        metrics, elapsed = run_one(
                            args.arch, split, modality, task, args.window,
                            args.epochs, args.batch_size, args.val_split,
                            args.splits_dir, args.verbose,
                            standardize=args.standardize,
                            use_class_weight=args.class_weight)
                    except FileNotFoundError:
                        print(f"[skip] {tool} {task} {split} {modality}: "
                              f"split not found")
                        continue
                    rows.append({
                        "Split": split, "Modality": mcode, "Model": model_name,
                        "tool": tool, "class": task,
                        **{k: round(v, 4) for k, v in metrics.items()},
                        "Time (sec)": round(elapsed, 3),
                    })
                    print(f"[ok] {tool:9s} {task:10s} {split:8s} {mcode} "
                          f"{model_name} Acc={metrics['Accuracy']:.4f} "
                          f"F1={metrics['F1-score']:.4f} "
                          f"BalAcc={metrics['Balanced Accuracy']:.4f} "
                          f"kappa={metrics['Cohen Kappa']:.4f} ({elapsed:.1f}s)")
                    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
