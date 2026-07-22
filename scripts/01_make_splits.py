#!/usr/bin/env python3
"""Stage 1 -- generate the predefined evaluation splits.

Reads the raw anonymized keypoint table(s) and writes the serialized train/test
arrays for every combination of split x modality x task under ``splits/``, using
the canonical naming scheme in :func:`qamqor.config.split_path`.

Running this stage once fixes the exact partitions used by all downstream models,
which is what allows different classifiers -- and future methods contributed by
other groups -- to be compared on identical data.

Examples
--------
    python scripts/01_make_splits.py                     # both tools, both tasks
    python scripts/01_make_splits.py --tool OpenPose     # one tool only
    python scripts/01_make_splits.py --task binary       # one task only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.data import build_modalities, load_raw, make_splits_for_modality, save_split


def parse_args():
    p = argparse.ArgumentParser(description="Generate QAMQOR evaluation splits.")
    p.add_argument("--tool", choices=list(config.RAW_CSV), default=None,
                   help="Restrict to one tool (default: both).")
    p.add_argument("--task", choices=config.TASKS, default=None,
                   help="Restrict to one task (default: both).")
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--splits-dir", default=config.SPLITS_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    tools = [args.tool] if args.tool else list(config.RAW_CSV)
    tasks = [args.task] if args.task else config.TASKS

    for tool in tools:
        df = load_raw(tool, args.data_dir)
        modalities = build_modalities(df, tool)
        print(f"\n[{tool}] loaded {len(df):,} frames, "
              f"{df['childID'].nunique()} children")

        for task in tasks:
            target = config.TARGET_COL[task]
            for modality, cols in modalities.items():
                if not cols:
                    print(f"  [skip] {modality}: no matching columns")
                    continue
                splits = make_splits_for_modality(df, cols, target, seed=args.seed)
                for split_name, arrays in splits.items():
                    save_split(split_name, modality, task, arrays, args.splits_dir)
                print(f"  [ok] {task:10s} {modality:15s} "
                      f"({len(cols)} features) -> 4 splits written")

    print(f"\nAll splits written to '{args.splits_dir}/'.")


if __name__ == "__main__":
    main()
