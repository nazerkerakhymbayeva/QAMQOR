#!/usr/bin/env python3
"""Stage 6 -- merge per-model result files into one benchmark table.

Concatenates the CSVs produced by stages 2-5 into a single tidy table with a
consistent schema, adding ``*_mean`` / ``*_sd`` columns (identical to the point
estimate for single-run rows, and the fold statistics for the cross-validation
rows) so the whole table can be consumed uniformly by the analysis script.

A ``Source`` column records which file each row came from, preserving provenance.

Examples
--------
    python scripts/06_combine_results.py
    python scripts/06_combine_results.py --out results/QAMQOR_all_results.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config

BASE_COLS = ["Split", "Modality", "Model", "tool", "class"]


def _ensure_mean_sd(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<metric>_mean`` / ``<metric>_sd`` columns if absent.

    For single-run rows the mean equals the point estimate and the SD is 0; for
    rows already carrying fold statistics (the CV output) the existing columns
    are preserved.
    """
    for k in config.METRIC_KEYS:
        if f"{k}_mean" not in df.columns:
            df[f"{k}_mean"] = pd.to_numeric(df[k], errors="coerce")
        if f"{k}_sd" not in df.columns:
            df[f"{k}_sd"] = 0.0
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Combine per-model result CSVs.")
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--pattern", default="QAMQOR_*.csv",
                   help="Glob of per-model files to merge.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or os.path.join(args.results_dir, "QAMQOR_all_results.csv")

    files = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(out)]
    if not files:
        print(f"[error] no result files matching '{args.pattern}' in "
              f"'{args.results_dir}/'. Run stages 2-5 first.")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["Source"] = os.path.basename(f)
        frames.append(_ensure_mean_sd(df))
        print(f"[read] {os.path.basename(f):40s} {len(df):4d} rows")

    combined = pd.concat(frames, ignore_index=True)

    # Stable ordering for readability.
    split_order = {s: i for i, s in enumerate(config.SPLITS + [config.CHILD_CV_SPLIT])}
    combined["_split_ord"] = combined["Split"].map(split_order).fillna(99)
    combined = (combined
                .sort_values(["class", "tool", "_split_ord", "Modality", "Model"])
                .drop(columns="_split_ord")
                .reset_index(drop=True))

    combined.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(combined)} rows, {combined['Model'].nunique()} models)")


if __name__ == "__main__":
    main()
