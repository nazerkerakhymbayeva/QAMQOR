"""Central configuration for the QAMQOR engagement-recognition benchmark.

All constants that define the reproducible evaluation protocol live here so that
every script in ``scripts/`` shares one source of truth: the global random seed,
the four evaluation splits, the modality definitions, the fixed modality codes
used in the paper tables, and the canonical file-naming scheme for the
serialized train/test arrays.

Keeping these in one place is what makes the benchmark reproducible: changing a
split name, a seed, or a modality definition changes it for the entire pipeline
at once, and never silently in one script but not another.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root; override with env vars if needed)
# --------------------------------------------------------------------------- #
DATA_DIR: str = os.environ.get("QAMQOR_DATA_DIR", "data")
SPLITS_DIR: str = os.environ.get("QAMQOR_SPLITS_DIR", "splits")
RESULTS_DIR: str = os.environ.get("QAMQOR_RESULTS_DIR", "results")

# Raw anonymized keypoint tables (frame-level features + annotations + group IDs)
RAW_CSV = {
    "OpenPose": "appBased146Openpose.csv",
    "Mediapipe": "appBased136Mediapipe.csv",
}

# --------------------------------------------------------------------------- #
# Annotation columns
# --------------------------------------------------------------------------- #
TARGET_COL = {
    "multiclass": "engagement_x",
    "binary": "engagement_bin",
}

# Grouping columns that define the subject-, session-, and activity-independent
# evaluation protocols. RANDOM uses no grouping (frame-level stratified split).
GROUP_COL = {
    "CHILD": "childID",
    "SESSION": "sessionID",
    "ACTIVITY": "appID",
}

# Non-feature columns that must never be used as model inputs.
META_COLS = [
    "engagement_x",
    "engagement_bin",
    "childID",
    "sessionID",
    "frameID",
    "appID",
]

# --------------------------------------------------------------------------- #
# Evaluation protocol
# --------------------------------------------------------------------------- #
# Ordered from optimistic (RANDOM) to most demanding (ACTIVITY), matching the
# order used throughout the manuscript.
SPLITS = ["RANDOM", "CHILD", "SESSION", "ACTIVITY"]
CHILD_CV_SPLIT = "CHILD_5FOLD_CV"      # leave-children-out cross-validation
TASKS = ["multiclass", "binary"]
TEST_SIZE = 0.20
N_FOLDS = 5                            # folds for leave-children-out CV

# --------------------------------------------------------------------------- #
# Modality definitions and fixed codes (M1..M6) used in the paper tables
# --------------------------------------------------------------------------- #
# Column selection is by substring match on the feature names:
#   face  -> "face"
#   body  -> "pose" or "body"
#   hand  -> "hand"
# "full" = face + body + hand ; "body" modality = body + hand (upper-body pose).
MODALITY_CODE = {
    # OpenPose
    "face_openpose": "M1",
    "body_openpose": "M2",
    "full_openpose": "M3",
    # Mediapipe
    "face_mediapipe": "M4",
    "body_mediapipe": "M5",
    "full_mediapipe": "M6",
}

# Modality names to iterate over, per tool.
MODALITIES = {
    "OpenPose": ["face_openpose", "body_openpose", "full_openpose"],
    "Mediapipe": ["face_mediapipe", "body_mediapipe", "full_mediapipe"],
}

# Temporal-model window length (LSTM and temporal XGBoost).
WINDOW = 10

# The six metrics reported for every configuration, in table order.
METRIC_KEYS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Balanced Accuracy",
    "Cohen Kappa",
]


# --------------------------------------------------------------------------- #
# Canonical file-naming for serialized splits
# --------------------------------------------------------------------------- #
def split_path(split: str, modality: str, task: str, array: str,
               splits_dir: str = SPLITS_DIR) -> str:
    """Return the canonical path of a serialized split array.

    A single naming scheme is used for *both* tools and *both* tasks, replacing
    the several incompatible conventions in the original notebooks::

        splits/<SPLIT>_<modality>_<task>_<array>.npy

    where ``array`` is one of ``Xtrain, Xtest, ytrain, ytest``. The modality name
    already encodes the tool (e.g. ``face_openpose`` vs ``face_mediapipe``), so no
    separate tool field is needed.
    """
    fname = f"{split}_{modality}_{task}_{array}.npy"
    return os.path.join(splits_dir, fname)
