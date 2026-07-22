#!/usr/bin/env bash
# Reproduce the full QAMQOR benchmark end-to-end.
#
# Prerequisite: place the anonymized keypoint tables in data/
#   data/appBased146Openpose.csv
#   data/appBased136Mediapipe.csv
#
# Usage:  bash run_all.sh
set -euo pipefail

echo "==> Stage 1/6: generate evaluation splits"
python scripts/01_make_splits.py

echo "==> Stage 2/6: conventional ML baselines"
python scripts/02_run_baselines.py

echo "==> Stage 3/6: leave-children-out 5-fold cross-validation"
python scripts/03_run_children_cv.py

echo "==> Stage 4/6: LSTM sequence baseline (requires TensorFlow)"
python scripts/04_run_lstm.py

echo "==> Stage 5/6: temporal XGBoost baseline"
python scripts/05_run_temporal_xgb.py

echo "==> Stage 6/6: combine results"
python scripts/06_combine_results.py

echo "==> Statistical analysis (Friedman + Wilcoxon post-hoc)"
python scripts/07_statistical_tests.py --metric F1-score --task multiclass
python scripts/07_statistical_tests.py --metric F1-score --task binary
python scripts/07_statistical_tests.py --metric Accuracy  --task binary

echo "==> Regenerate paper Tables 4 and 5"
python scripts/08_paper_tables.py

echo "Done. Combined table: results/QAMQOR_all_results.csv"
