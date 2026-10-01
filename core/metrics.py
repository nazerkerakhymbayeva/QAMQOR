"""Evaluation metrics shared by every model in the benchmark.

A single ``evaluate`` function guarantees that all models -- conventional
baselines, LSTM, and temporal XGBoost -- are scored with identical metric
definitions and identical averaging. This is essential for fair comparison: a
difference in averaging convention alone can shift reported F1 by several points
and would otherwise confound the very comparison the benchmark is meant to
enable.

Averaging convention
--------------------
Precision, recall, and F1 are **macro-averaged for both tasks**. This is
deliberate and matches the QAMQOR paper: under the 70.3% binary majority class,
positive-class or frequency-weighted F1 is optimistically biased, so macro
averaging is used so that both the engaged and disengaged classes (and, in the
multiclass task, every engagement level) contribute equally. Consequently
``F1-score`` denotes macro-F1 throughout, ``Balanced Accuracy`` equals
macro-averaged recall, and ``Cohen's kappa`` provides chance-corrected agreement.
``Accuracy`` is retained for continuity but must be read against the
majority-class baseline, since a classifier that always predicts the dominant
class can score high accuracy with no discriminative ability.

Note: for the binary task, macro precision/recall/F1 differ from the
positive-class values. Using positive-class averaging instead would inflate the
reported binary F1 (e.g. from ~0.5 to ~0.85 on QAMQOR) and would not reproduce
the paper's Table 4 or its binary significance tests.
"""

from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, average: str = "macro") -> Dict[str, float]:
    """Compute the six benchmark metrics for a set of predictions.

    Precision, recall, and F1 are macro-averaged by default for both the binary
    and multiclass tasks, matching the paper's reporting convention.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Integer-encoded ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Integer-encoded predicted labels.
    average : str, default "macro"
        Averaging mode for precision/recall/F1. Left configurable for
        completeness, but the benchmark uses "macro"; changing it will not
        reproduce the published results.

    Returns
    -------
    dict
        Mapping from metric name to value, in the fixed order used in the paper
        tables.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }
