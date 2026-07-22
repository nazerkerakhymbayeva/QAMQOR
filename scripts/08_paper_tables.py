#!/usr/bin/env python3
"""Stage 8 -- regenerate the paper's summary tables from the combined results.

Two aggregate tables in the manuscript are pure post-processing of the combined
per-configuration results, and are reproduced here so the reported summaries can
be traced directly to the raw numbers:

* **Table 4** -- overall benchmark performance: mean +/- SD of each metric per
  model, pooled across modalities and the four held-out splits, per task, with
  the majority-class baseline appended.
* **Table 5** -- static vs. temporal XGBoost: mean macro-F1 per split for the two
  models over the modalities common to both, with the per-split gain.

The majority-class baseline is computed from the label distribution when the raw
tables are available (``--data-dir``); otherwise the values reported in the paper
(binary macro-F1 = 0.41, multiclass macro-F1 = 0.09) are used as a fallback and
flagged as such.

Examples
--------
    python scripts/08_paper_tables.py
    python scripts/08_paper_tables.py --results results/QAMQOR_all_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config
from qamqor.metrics import evaluate

HELD_OUT = ["RANDOM", "CHILD", "SESSION", "ACTIVITY"]
# Paper-reported fallbacks if the raw label distribution is unavailable.
MAJORITY_FALLBACK = {
    "binary":     {"Accuracy": 70.3, "Macro-F1": 0.41, "Balanced Acc": 50.0, "Cohen's k": 0.00},
    "multiclass": {"Accuracy": 30.4, "Macro-F1": 0.09, "Balanced Acc": 20.0, "Cohen's k": 0.00},
}


def _num(df, col):
    if f"{col}_mean" in df.columns:
        v = pd.to_numeric(df[f"{col}_mean"], errors="coerce")
        if v.notna().any():
            return v
    return pd.to_numeric(df[col], errors="coerce")


def majority_baseline_from_data(task, data_dir):
    """Compute majority-class baseline metrics from the pooled label column."""
    target = config.TARGET_COL[task]
    ys = []
    for tool in config.RAW_CSV:
        path = os.path.join(data_dir, config.RAW_CSV[tool])
        if os.path.exists(path):
            ys.append(pd.read_csv(path, usecols=[target])[target].values)
    if not ys:
        return None
    y = np.concatenate(ys)
    maj = pd.Series(y).mode().iloc[0]
    y_pred = np.full_like(y, maj)
    m = evaluate(y, y_pred)
    return {"Accuracy": 100 * m["Accuracy"], "Macro-F1": m["F1-score"],
            "Balanced Acc": 100 * m["Balanced Accuracy"], "Cohen's k": m["Cohen Kappa"]}


def table4(df, task, data_dir):
    """Overall performance: mean +/- SD per model, pooled across held-out splits."""
    sub = df[(df["class"] == task) & (df["Split"].isin(HELD_OUT))].copy()
    rows = []
    for model, g in sub.groupby("Model"):
        acc, f1 = _num(g, "Accuracy"), _num(g, "F1-score")
        bacc, kap = _num(g, "Balanced Accuracy"), _num(g, "Cohen Kappa")
        rows.append({
            "Model": model,
            "Accuracy (%)": f"{100*acc.mean():.1f} \u00b1 {100*acc.std(ddof=1):.1f}",
            "Macro-F1": f"{f1.mean():.2f} \u00b1 {f1.std(ddof=1):.2f}",
            "Balanced Acc (%)": f"{100*bacc.mean():.1f} \u00b1 {100*bacc.std(ddof=1):.1f}",
            "Cohen's k": f"{kap.mean():.2f} \u00b1 {kap.std(ddof=1):.2f}",
        })

    # Child-Independent 5-fold CV row, if present.
    cv = df[(df["class"] == task) & (df["Split"] == config.CHILD_CV_SPLIT)]
    if len(cv):
        acc, f1 = _num(cv, "Accuracy"), _num(cv, "F1-score")
        bacc, kap = _num(cv, "Balanced Accuracy"), _num(cv, "Cohen Kappa")
        rows.append({
            "Model": "XGBoost (Child-Independent CV)",
            "Accuracy (%)": f"{100*acc.mean():.1f} \u00b1 {100*acc.std(ddof=1):.1f}",
            "Macro-F1": f"{f1.mean():.2f} \u00b1 {f1.std(ddof=1):.2f}",
            "Balanced Acc (%)": f"{100*bacc.mean():.1f} \u00b1 {100*bacc.std(ddof=1):.1f}",
            "Cohen's k": f"{kap.mean():.2f} \u00b1 {kap.std(ddof=1):.2f}",
        })

    base = majority_baseline_from_data(task, data_dir)
    flagged = base is None
    if flagged:
        base = MAJORITY_FALLBACK[task]
    kappa = base["Cohen's k"]
    rows.append({
        "Model": "Majority-class baseline" + (" (paper value)" if flagged else ""),
        "Accuracy (%)": f"{base['Accuracy']:.1f}",
        "Macro-F1": f"{base['Macro-F1']:.2f}",
        "Balanced Acc (%)": f"{base['Balanced Acc']:.1f}",
        "Cohen's k": f"{kappa:.2f}",
    })
    return pd.DataFrame(rows)


def table5(df):
    """Static vs Temporal XGBoost: mean macro-F1 per split over common modalities."""
    out = []
    for task in ["binary", "multiclass"]:
        sub = df[(df["class"] == task) &
                 (df["Model"].isin(["XGBoost", "Temporal XGBoost"])) &
                 (df["Split"].isin(HELD_OUT))].copy()
        sub["_f1"] = _num(sub, "F1-score")
        for split in HELD_OUT:
            s = sub[sub["Split"] == split]
            # restrict to modalities measured for BOTH models
            common = (s.groupby("Modality")["Model"].nunique() == 2)
            mods = common[common].index
            s = s[s["Modality"].isin(mods)]
            if s.empty:
                continue
            static = s[s["Model"] == "XGBoost"]["_f1"].mean()
            temporal = s[s["Model"] == "Temporal XGBoost"]["_f1"].mean()
            gain = temporal - static
            out.append({
                "Task": task, "Split": split,
                "Static macro-F1": round(static, 2),
                "Temporal macro-F1": round(temporal, 2),
                "Gain": f"{gain:+.2f}",
                "Better": "Temporal" if gain > 0 else "Static",
            })
    return pd.DataFrame(out)


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate paper Tables 4 and 5.")
    p.add_argument("--results", default=None)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    results = args.results or os.path.join(config.RESULTS_DIR,
                                           "QAMQOR_all_results.csv")
    if not os.path.exists(results):
        print(f"[error] combined results not found: {results}")
        return
    df = pd.read_csv(results)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("TABLE 4  |  Overall benchmark performance (mean \u00b1 SD)")
    print("=" * 70)
    for task in ["binary", "multiclass"]:
        print(f"\n--- {task} ---")
        t4 = table4(df, task, args.data_dir)
        print(t4.to_string(index=False))
        t4.to_csv(os.path.join(args.results_dir, f"table4_{task}.csv"), index=False)

    print("\n" + "=" * 70)
    print("TABLE 5  |  Static vs Temporal XGBoost (mean macro-F1)")
    print("=" * 70)
    t5 = table5(df)
    print(t5.to_string(index=False))
    t5.to_csv(os.path.join(args.results_dir, "table5_static_vs_temporal.csv"), index=False)

    print(f"\nSaved table CSVs to '{args.results_dir}/'.")


if __name__ == "__main__":
    main()
