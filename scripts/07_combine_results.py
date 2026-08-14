#!/usr/bin/env python3
"""Stage 6 -- merge per-model result files, de-duplicate, and split into
balanced / imbalanced / combined tables.

Concatenates the CSVs produced by stages 2-5 into a tidy table with a consistent
schema (adding ``*_mean`` / ``*_sd`` columns and a ``Source`` column), removes
duplicate results, and writes three files so statistical tests can be run on each:

* ``QAMQOR_imbalanced.csv`` -- models trained on the natural class distribution.
* ``QAMQOR_balanced.csv``   -- imbalance-aware variants (model names ending "-bal").
* ``QAMQOR_combined.csv``   -- both, in one table.

A model is classified as *balanced* iff its ``Model`` name ends with ``-bal``.

Examples
--------
    python scripts/06_combine_results.py
    python scripts/06_combine_results.py --on-conflict mean
    python scripts/06_combine_results.py --on-conflict error
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config

KEY = ["Split", "Modality", "Model", "tool", "class"]
BAL_SUFFIX = "-bal"


def _ensure_mean_sd(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<metric>_mean`` / ``<metric>_sd`` columns if absent."""
    for k in config.METRIC_KEYS:
        if f"{k}_mean" not in df.columns:
            df[f"{k}_mean"] = pd.to_numeric(df.get(k), errors="coerce")
        if f"{k}_sd" not in df.columns:
            df[f"{k}_sd"] = 0.0
    return df


def _aggregate_conflicts(group: pd.DataFrame) -> pd.Series:
    """Collapse conflicting rows for one key into a single mean +/- sd row."""
    row = group.iloc[0].copy()
    for k in config.METRIC_KEYS:
        if k in group.columns:
            vals = pd.to_numeric(group[k], errors="coerce")
            row[k] = round(vals.mean(), 4)
            row[f"{k}_mean"] = round(vals.mean(), 4)
            row[f"{k}_sd"] = round(vals.std(ddof=1) if len(vals) > 1 else 0.0, 4)
    if "Time (sec)" in group.columns:
        row["Time (sec)"] = round(pd.to_numeric(
            group["Time (sec)"], errors="coerce").mean(), 3)
    row["Source"] = ";".join(sorted(str(s) for s in group["Source"].unique())) \
        + f" (mean of {len(group)})"
    return row


def dedup(combined: pd.DataFrame, on_conflict: str):
    """Return (deduped_df, n_exact, n_conflict_rows_removed)."""
    n_read = len(combined)

    # Stage 1: exact duplicates (same key AND same metric values).
    metric_cols = [c for c in config.METRIC_KEYS if c in combined.columns]
    exact_mask = combined.duplicated(subset=KEY + metric_cols, keep="first")
    n_exact = int(exact_mask.sum())
    combined = combined[~exact_mask].reset_index(drop=True)
    if n_exact:
        print(f"[dedup] removed {n_exact} exact-duplicate row(s).")

    # Stage 2: identity-key conflicts (same key, different values).
    conflict_mask = combined.duplicated(subset=KEY, keep=False)
    conflicts = combined[conflict_mask]
    if len(conflicts):
        show = "F1-score" if "F1-score" in conflicts.columns else metric_cols[0]
        n_keys = conflicts.groupby(KEY, sort=False).ngroups
        print(f"[conflict] {n_keys} identity key(s) differ across files:")
        for keyvals, g in conflicts.groupby(KEY, sort=False):
            tag = " | ".join(f"{c}={v}" for c, v in zip(KEY, keyvals))
            srcs = ", ".join(f"{r['Source']}:{r[show]}" for _, r in g.iterrows())
            print(f"  - {tag}  ->  {srcs}")
        if on_conflict == "error":
            print("\n[abort] --on-conflict=error: resolve the above "
                  "(rename a Model, delete a file, or choose --on-conflict).")
            sys.exit(1)

    before = len(combined)
    if len(conflicts) and on_conflict == "mean":
        keep = combined[~conflict_mask].copy()
        agg = pd.DataFrame([_aggregate_conflicts(g)
                            for _, g in conflicts.groupby(KEY, sort=False)])
        combined = pd.concat([keep, agg], ignore_index=True)
        print(f"[dedup] collapsed conflicting keys into mean +/- sd rows.")
    elif len(conflicts):  # first / last
        combined = combined.drop_duplicates(
            subset=KEY, keep=on_conflict).reset_index(drop=True)
        print(f"[dedup] kept '{on_conflict}' row per conflicting key.")

    n_conflict_removed = before - len(combined)
    return combined, n_exact, n_conflict_removed


def sort_table(df: pd.DataFrame) -> pd.DataFrame:
    order = {s: i for i, s in enumerate(config.SPLITS + [config.CHILD_CV_SPLIT])}
    df = df.copy()
    df["_o"] = df["Split"].map(order).fillna(99)
    return (df.sort_values(["class", "tool", "_o", "Modality", "Model"])
              .drop(columns="_o").reset_index(drop=True))


def parse_args():
    p = argparse.ArgumentParser(description="Combine, dedup, and split results.")
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--pattern", default="QAMQOR_*.csv")
    p.add_argument("--on-conflict", choices=["last", "first", "mean", "error"],
                   default="last")
    p.add_argument("--combined-out", default="QAMQOR_combined.csv")
    p.add_argument("--balanced-out", default="QAMQOR_balanced.csv")
    p.add_argument("--imbalanced-out", default="QAMQOR_imbalanced.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rd = args.results_dir
    outputs = {os.path.abspath(os.path.join(rd, n)) for n in
               (args.combined_out, args.balanced_out, args.imbalanced_out)}

    files = sorted(glob.glob(os.path.join(rd, args.pattern)))
    frames = []
    for f in files:
        if os.path.abspath(f) in outputs:
            continue                                   # never read our own output
        df = pd.read_csv(f)
        if "Source" in df.columns:                     # already-combined file
            print(f"[skip] {os.path.basename(f):32s} looks combined; ignoring.")
            continue
        df["Source"] = os.path.basename(f)
        frames.append(_ensure_mean_sd(df))
        print(f"[read] {os.path.basename(f):32s} {len(df):4d} rows")

    if not frames:
        print(f"[error] no per-model files matching '{args.pattern}' in '{rd}/'.")
        return

    combined = pd.concat(frames, ignore_index=True)
    n_read = len(combined)

    combined, n_exact, n_conflict = dedup(combined, args.on_conflict)
    combined = sort_table(combined)

    # Split balanced vs imbalanced by model-name suffix.
    is_bal = combined["Model"].astype(str).str.endswith(BAL_SUFFIX)
    balanced = combined[is_bal].reset_index(drop=True)
    imbalanced = combined[~is_bal].reset_index(drop=True)

    for name, table in [(args.combined_out, combined),
                        (args.balanced_out, balanced),
                        (args.imbalanced_out, imbalanced)]:
        path = os.path.join(rd, name)
        table.to_csv(path, index=False)

    print(f"\n{n_read} rows read -> {len(combined)} after dedup "
          f"({n_exact} exact + {n_conflict} conflict rows removed)")
    print(f"Saved:")
    print(f"  {args.combined_out:26s} {len(combined):4d} rows | "
          f"{combined['Model'].nunique()} models")
    print(f"  {args.imbalanced_out:26s} {len(imbalanced):4d} rows | "
          f"{sorted(imbalanced['Model'].unique())}")
    print(f"  {args.balanced_out:26s} {len(balanced):4d} rows | "
          f"{sorted(balanced['Model'].unique())}")


if __name__ == "__main__":
    main()
