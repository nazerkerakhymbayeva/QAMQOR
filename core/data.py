"""Data loading, modality selection, and evaluation-split generation.

This module turns a raw anonymized keypoint table into the serialized train/test
arrays consumed by every model script. It implements the four evaluation
protocols that are the central methodological contribution of the benchmark:

* **RANDOM**   -- stratified frame-level split (no grouping). Optimistic, because
  frames from the same child/session/activity appear in both train and test.
* **CHILD**    -- subject-independent: no child appears in both partitions.
* **SESSION**  -- session-independent: no recording session is shared.
* **ACTIVITY** -- activity-independent: no robot activity is shared.

The grouped splits use :class:`~sklearn.model_selection.GroupShuffleSplit` with a
fixed seed so the partitions are identical on every machine.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config


# --------------------------------------------------------------------------- #
# Loading and modality construction
# --------------------------------------------------------------------------- #
def load_raw(tool: str, data_dir: str = None) -> pd.DataFrame:
    """Load the raw anonymized keypoint table for a given tool.

    Parameters
    ----------
    tool : {"OpenPose", "Mediapipe"}
    data_dir : str, optional
        Directory containing the raw CSV. Defaults to :data:`config.DATA_DIR`.
    """
    data_dir = data_dir or config.DATA_DIR
    path = os.path.join(data_dir, config.RAW_CSV[tool])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw keypoint table not found: {path}\n"
            f"Place the anonymized {tool} CSV in '{data_dir}/'."
        )
    return pd.read_csv(path)


def build_modalities(df: pd.DataFrame, tool: str) -> Dict[str, List[str]]:
    """Return the {modality_name: feature_columns} mapping for a tool.

    Columns are selected by substring on the feature names. The "body" modality
    is the upper-body pose (body + hand keypoints); "full" additionally includes
    the face keypoints.
    """
    suffix = tool.lower()
    face = [c for c in df.columns if "face" in c.lower()]
    body = [c for c in df.columns if ("pose" in c.lower() or "body" in c.lower())]
    hand = [c for c in df.columns if "hand" in c.lower()]

    full = face + body + hand
    body_full = body + hand  # upper-body pose

    return {
        f"face_{suffix}": face,
        f"body_{suffix}": body_full,
        f"full_{suffix}": full,
    }


# --------------------------------------------------------------------------- #
# Split generation
# --------------------------------------------------------------------------- #
def _grouped_split(X, y, groups, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                            random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx


def make_splits_for_modality(df, cols, target, seed=config.SEED):
    """Generate the four held-out partitions for one modality/target.

    Returns
    -------
    dict
        Mapping ``split_name -> (X_train, X_test, y_train, y_test)``.
    """
    X = df[cols].fillna(0).values
    y = df[target].values
    out = {}

    # RANDOM: stratified frame-level split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed
    )
    out["RANDOM"] = (X_tr, X_te, y_tr, y_te)

    # Grouped, held-out splits.
    for split_name, group_col in config.GROUP_COL.items():
        tr, te = _grouped_split(X, y, df[group_col].values, seed)
        # Safety: confirm the groups are disjoint across train/test.
        g = df[group_col].values
        assert set(np.unique(g[tr])).isdisjoint(np.unique(g[te])), (
            f"Group leakage detected in {split_name} split."
        )
        out[split_name] = (X[tr], X[te], y[tr], y[te])

    return out


def save_split(split, modality, task, arrays, splits_dir=config.SPLITS_DIR):
    """Persist one (X_train, X_test, y_train, y_test) tuple under the canonical
    naming scheme."""
    os.makedirs(splits_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = arrays
    np.save(config.split_path(split, modality, task, "Xtrain", splits_dir), X_train)
    np.save(config.split_path(split, modality, task, "Xtest", splits_dir), X_test)
    np.save(config.split_path(split, modality, task, "ytrain", splits_dir), y_train)
    np.save(config.split_path(split, modality, task, "ytest", splits_dir), y_test)


def load_split(split, modality, task, splits_dir=config.SPLITS_DIR):
    """Load a serialized (X_train, X_test, y_train, y_test) tuple."""
    X_train = np.load(config.split_path(split, modality, task, "Xtrain", splits_dir))
    X_test = np.load(config.split_path(split, modality, task, "Xtest", splits_dir))
    y_train = np.load(config.split_path(split, modality, task, "ytrain", splits_dir))
    y_test = np.load(config.split_path(split, modality, task, "ytest", splits_dir))
    return X_train, X_test, y_train, y_test
