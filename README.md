# QAMQOR

**A reproducible benchmark for engagement recognition in Robot-Assisted Autism Therapy (RAAT).**

QAMQOR provides anonymized keypoint data, engagement annotations, predefined
evaluation splits, and a set of reference baselines for recognizing child
engagement from pose/face/hand keypoints during robot-assisted therapy sessions.

The contribution of this repository is a **dataset and an evaluation
methodology**, not a new classifier. The baselines exist to establish
reproducible reference results under a set of standardized evaluation protocols,
so that future methods can be compared fairly on identical partitions. A central
finding the benchmark is designed to expose is that **the evaluation protocol can
influence reported performance as much as, or more than, the choice of
classifier** — which is why subject-, session-, and activity-independent
evaluation is treated as a first-class concern here.

> The raw video recordings cannot be shared because participant consent did not
> permit future distribution of identifiable video data. To support
> reproducibility while protecting participant privacy, this repository releases
> the anonymized OpenPose and MediaPipe keypoints, the engagement annotations,
> and the predefined evaluation splits.

---

## Evaluation protocols

The core of the benchmark is four evaluation protocols of increasing difficulty,
plus a cross-validation variant:

| Split            | Grouping        | What it measures |
|------------------|-----------------|------------------|
| `RANDOM`         | none (stratified) | Optimistic: frames from the same child/session/activity may appear in both train and test. |
| `CHILD`          | `childID`       | Subject-independent: generalization to unseen children. |
| `SESSION`        | `sessionID`     | Session-independent: generalization to unseen recording sessions. |
| `ACTIVITY`       | `appID`         | Activity-independent: generalization to unseen robot activities. |
| `CHILD_5FOLD_CV` | `childID` (5-fold) | Subject-independent generalization with mean ± SD across folds. |

`RANDOM` typically yields the most optimistic estimates precisely because it
leaks child-, session-, and activity-specific information across the split; the
grouped protocols are the ones that reflect real-world deployment, where the
system must handle children and activities it has never seen.

## Modalities

| Code | Tool      | Modality | Keypoints |
|------|-----------|----------|-----------|
| M1   | OpenPose  | face     | face |
| M2   | OpenPose  | body     | body + hand (upper-body pose) |
| M3   | OpenPose  | full     | face + body + hand |
| M4   | MediaPipe | face     | face |
| M5   | MediaPipe | body     | body + hand (upper-body pose) |
| M6   | MediaPipe | full     | face + body + hand |

## Baselines

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | linear | trained on standardized features |
| LightGBM, CatBoost, XGBoost | gradient-boosted trees | frame-level features |
| Temporal XGBoost | GBT + temporal features | window = 10 (current frame + rolling mean/std + delta) |
| LSTM | recurrent | window = 10 raw-frame sequences |

All hyper-parameters are fixed in [`qamqor/config.py`](qamqor/config.py) and in
the per-script model factories, and all seeds are set, so results are
reproducible up to GPU-kernel nondeterminism for the LSTM.

---

## Installation

```bash
git clone https://github.com/nazerkerakhymbayeva/QAMQOR.git
cd QAMQOR
pip install -r requirements.txt
```

TensorFlow is only required for the LSTM baseline (`scripts/04_run_lstm.py`); the
other stages run without it. LightGBM and CatBoost degrade gracefully — if either
is unavailable, its baseline is skipped with a warning.

## Data

Place the released keypoint tables in `data/` (see [`data/README.md`](data/README.md)):

```
data/appBased146Openpose.csv
data/appBased136Mediapipe.csv
```

## Reproduce the benchmark

Run the whole pipeline:

```bash
bash run_all.sh
```

…or stage by stage:

```bash
python scripts/01_make_splits.py            # 1. generate predefined splits  -> splits/
python scripts/02_run_baselines.py          # 2. LightGBM / CatBoost / XGBoost / LR
python scripts/03_run_children_cv.py        # 3. leave-children-out 5-fold CV
python scripts/04_run_lstm.py               # 4. LSTM sequence baseline
python scripts/05_run_temporal_xgb.py       # 5. temporal XGBoost baseline
python scripts/06_combine_results.py        # 6. merge into results/QAMQOR_all_results.csv
python scripts/07_statistical_tests.py      # 7. Friedman + Wilcoxon post-hoc
python scripts/08_paper_tables.py           # 8. regenerate Table 4 and Table 5
```

Every script takes `--tool`, `--task`, and directory overrides; run any script
with `--help` for details. For example, to reproduce only the binary MediaPipe
baselines:

```bash
python scripts/02_run_baselines.py --tool Mediapipe --task binary
```

## Statistical analysis

`scripts/07_statistical_tests.py` computes the significance analysis directly
from the combined results table — nothing is hard-coded. For the chosen metric
and task it runs:

1. a **Friedman test across the four splits** (each model × modality × tool is a
   block) — this tests the benchmark's central claim that the protocol matters;
2. a **Friedman test across the models** (each split × modality × tool is a
   block);

each followed, when significant, by **Wilcoxon signed-rank post-hoc** comparisons
with **Holm–Bonferroni** correction. **Kendall's W** (omnibus) and the
**matched-pairs rank-biserial correlation** (pairwise) are reported so that
statistical significance is not conflated with practical significance.

```bash
python scripts/07_statistical_tests.py --metric F1-score --task multiclass
python scripts/07_statistical_tests.py --metric Accuracy  --task binary \
    --out results/posthoc_binary_accuracy.csv
```

## Repository layout

```
QAMQOR/
├── qamqor/                 # shared, importable core
│   ├── config.py           # seeds, splits, modalities, codes, file naming
│   ├── data.py             # loading, modality selection, split generation
│   ├── features.py         # temporal features + sequence windows
│   └── metrics.py          # the six benchmark metrics (shared by all models)
├── scripts/                # 01–07 pipeline stages (CLI, argparse)
├── data/                   # place the anonymized keypoint CSVs here
├── splits/                 # generated .npy partitions (git-ignored)
├── results/                # generated result tables
├── requirements.txt
└── run_all.sh
```

## Notes on reproducibility

* A single file-naming scheme (`qamqor/config.py::split_path`) is used for both
  tools and both tasks, so the generated partitions are unambiguous.
* All metrics use one shared `evaluate()` with **macro-averaged** precision,
  recall, and F1 for **both** the binary and multiclass tasks, matching the
  paper ("F1-score denotes macro-F1 throughout"). Under the 70.3% binary
  majority class, positive-class or frequency-weighted F1 is optimistically
  biased; macro averaging weights the engaged and disengaged classes equally.
  `Balanced Accuracy` equals macro-averaged recall and `Cohen's κ` gives
  chance-corrected agreement. Accuracy is read against the majority-class
  baseline (binary macro-F1 = 0.41, multiclass macro-F1 = 0.09).
* Grouped splits assert group-disjointness between train and test, so subject-,
  session-, and activity-independence is verified rather than assumed.

### Metrics convention (important for matching the paper)

The binary metrics are **macro-averaged**, not positive-class. Using
positive-class averaging inflates binary F1 from ≈0.5 to ≈0.85 on QAMQOR and
will not reproduce Table 4, Table 5, or the binary significance tests. If you
regenerate the combined results, confirm binary `Recall` equals binary
`Balanced Accuracy` (both are macro-recall) as a quick check that macro
averaging is in effect.

## Citation

If you use QAMQOR, please cite the accompanying paper (Frontiers in Robotics and
AI). A `CITATION.cff` / BibTeX entry will be added on publication.
