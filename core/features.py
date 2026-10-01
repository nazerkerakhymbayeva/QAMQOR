"""Temporal feature construction for the sequence-aware baselines.

Two representations of the same sliding window are provided:

* :func:`temporal_features` -- a flat, engineered summary of each window
  (current frame, windowed mean, windowed standard deviation, and first-order
  frame delta) used by the temporal XGBoost baseline.
* :func:`create_sequences` -- the raw ``(window, n_features)`` tensor used by the
  LSTM baseline.

Both consume frame-ordered feature matrices and drop the first ``window`` frames,
so the label vector must be shifted by ``window`` to stay aligned (the scripts do
this explicitly). The engineered summary is memory-safe: it pre-allocates a
``float32`` output array rather than growing a Python list of ``float64`` rows,
which keeps peak memory bounded even for the wide Mediapipe modalities.
"""

from __future__ import annotations

import numpy as np

from .config import WINDOW


def temporal_features(X: np.ndarray, window: int = WINDOW,
                      dtype=np.float32) -> np.ndarray:
    """Engineer flat temporal features from a frame-ordered matrix.

    For each frame ``i >= window`` the feature vector concatenates four blocks of
    width ``d`` (the number of input features):

    #. the current frame ``X[i]``;
    #. the mean over the preceding ``window`` frames;
    #. the standard deviation over the preceding ``window`` frames;
    #. the first-order delta ``X[i] - X[i-1]``.

    Parameters
    ----------
    X : ndarray of shape (n_frames, d)
        Frame-ordered feature matrix.
    window : int
        Number of preceding frames summarized at each step.
    dtype : numpy dtype
        Output dtype. ``float32`` halves memory relative to ``float64`` and is
        lossless for downstream gradient-boosted trees, which cast to ``float32``
        internally.

    Returns
    -------
    ndarray of shape (n_frames - window, 4 * d)
    """
    X = np.asarray(X)
    n, d = X.shape
    out = np.empty((n - window, 4 * d), dtype=dtype)
    for idx in range(n - window):
        i = idx + window
        chunk = X[i - window:i]
        out[idx, 0 * d:1 * d] = X[i]
        out[idx, 1 * d:2 * d] = chunk.mean(axis=0)
        out[idx, 2 * d:3 * d] = chunk.std(axis=0)
        out[idx, 3 * d:4 * d] = X[i] - X[i - 1]
    return out


def create_sequences(X: np.ndarray, y: np.ndarray, window: int = WINDOW):
    """Build ``(window, n_features)`` sequences for a recurrent model.

    Frame ``i`` is represented by the ``window`` frames immediately preceding it,
    and its label is ``y[i]``. The first ``window`` frames therefore have no
    complete history and are dropped.

    Parameters
    ----------
    X : ndarray of shape (n_frames, d)
        Frame-ordered feature matrix.
    y : ndarray of shape (n_frames,)
        Frame-level integer labels.
    window : int
        Sequence length.

    Returns
    -------
    (ndarray of shape (n_frames - window, window, d), ndarray of shape (n_frames - window,))
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i - window:i])
        y_seq.append(y[i])
    return np.asarray(X_seq), np.asarray(y_seq)
