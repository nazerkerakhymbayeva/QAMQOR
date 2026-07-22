#!/usr/bin/env python3
"""Stage 7 -- statistical significance analysis.

Runs the non-parametric significance tests reported in the manuscript directly on
the combined benchmark table, so every number in the statistical analysis is
reproducible from the released results. Two complementary questions are tested,
for a chosen metric and task:

1. **Does the evaluation protocol change performance?** (the benchmark's central
   claim) -- a Friedman test across the four splits (RANDOM, CHILD, SESSION,
   ACTIVITY), treating each (model x modality x tool) configuration as a block.

2. **Do the classifiers differ?** -- a Friedman test across models, treating each
   (split x modality x tool) configuration as a block.

Each Friedman test is followed, when significant, by Wilcoxon signed-rank
post-hoc comparisons with Holm-Bonferroni correction. Effect sizes are reported
throughout -- Kendall's W for the omnibus test and the matched-pairs rank-biserial
correlation for each pairwise comparison -- so that statistical significance is
never conflated with practical significance.

Only *complete* blocks (configurations measured for every treatment being
compared) are used, as the Friedman test requires. The script reports how many
blocks entered each test.

Examples
--------
    python scripts/07_statistical_tests.py --metric F1-score --task multiclass
    python scripts/07_statistical_tests.py --metric Accuracy --task binary \\
        --results results/QAMQOR_all_results.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qamqor import config

HELD_OUT_SPLITS = ["RANDOM", "CHILD", "SESSION", "ACTIVITY"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _numeric_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return the metric as a float Series, preferring a ``*_mean`` column."""
    if f"{metric}_mean" in df.columns:
        vals = pd.to_numeric(df[f"{metric}_mean"], errors="coerce")
        if vals.notna().any():
            return vals
    return pd.to_numeric(df[metric], errors="coerce")


def _pivot(df, index_cols, column, metric):
    """Wide table: one row per block (index_cols), one column per treatment."""
    tmp = df.copy()
    tmp["_val"] = _numeric_metric(tmp, metric)
    wide = tmp.pivot_table(index=index_cols, columns=column, values="_val")
    return wide.dropna(axis=0, how="any")  # complete blocks only


def kendalls_w(friedman_stat, n_blocks, k_treatments):
    """Kendall's W effect size for a Friedman test."""
    if n_blocks == 0 or k_treatments <= 1:
        return float("nan")
    return friedman_stat / (n_blocks * (k_treatments - 1))


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test.

    Computed from the signed ranks of the non-zero paired differences; positive
    values indicate ``x`` > ``y``. Robust and independent of scipy's internals.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = r_plus + r_minus
    return 0.0 if total == 0 else (r_plus - r_minus) / total


def paired_cohens_d(x, y):
    """Paired Cohen's d = mean(difference) / SD(difference).

    Reported for descriptive accuracy differences, as in the paper. Positive
    values indicate ``x`` > ``y``.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return 0.0 if sd == 0 else d.mean() / sd


def holm_correction(pairs):
    """Holm-Bonferroni step-down correction.

    Parameters
    ----------
    pairs : list of (label, p_value, effect_size)

    Returns
    -------
    list of (label, p_raw, p_holm, effect_size), ordered by raw p-value.
    """
    ordered = sorted(pairs, key=lambda t: t[1])
    m = len(ordered)
    out, running = [], 0.0
    for i, (label, p, es) in enumerate(ordered):
        p_holm = min(max((m - i) * p, running), 1.0)
        running = p_holm
        out.append((label, p, p_holm, es))
    return out


def interpret_w(w):
    if np.isnan(w):
        return "n/a"
    if w < 0.10:
        return "negligible"
    if w < 0.30:
        return "small"
    if w < 0.50:
        return "moderate"
    return "large"


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def friedman_over(wide, dimension_name):
    """Run a Friedman omnibus test over the columns of ``wide``."""
    treatments = list(wide.columns)
    if len(treatments) < 3 or len(wide) < 2:
        print(f"  [!] Not enough complete blocks/treatments for a Friedman test "
              f"over {dimension_name} "
              f"({len(wide)} blocks, {len(treatments)} treatments).")
        return None
    samples = [wide[t].values for t in treatments]
    stat, p = friedmanchisquare(*samples)
    w = kendalls_w(stat, len(wide), len(treatments))

    mean_ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    print(f"\n  Friedman test across {dimension_name} "
          f"({len(wide)} complete blocks, {len(treatments)} treatments)")
    print(f"    chi^2 = {stat:.3f},  p = {p:.3e},  "
          f"Kendall's W = {w:.3f} ({interpret_w(w)})")
    print(f"    mean rank (1 = best):")
    for name, r in mean_ranks.items():
        print(f"      {str(name):22s} {r:.2f}")
    return {"stat": stat, "p": p, "w": w, "wide": wide,
            "treatments": treatments, "mean_ranks": mean_ranks}


def wilcoxon_posthoc(wide, treatments):
    """Pairwise Wilcoxon signed-rank tests with Holm correction."""
    pairs, extra = [], {}
    for a, b in itertools.combinations(treatments, 2):
        x, y = wide[a].values, wide[b].values
        try:
            _, p = wilcoxon(x, y)
        except ValueError:
            p = 1.0  # all differences zero
        label = f"{a} vs {b}"
        pairs.append((label, p, rank_biserial(x, y)))
        extra[label] = paired_cohens_d(x, y)

    corrected = holm_correction(pairs)
    print("\n  Wilcoxon signed-rank post-hoc (Holm-corrected):")
    print(f"    {'comparison':32s} {'p_raw':>10s} {'p_holm':>10s} "
          f"{'rank-bis.':>10s} {'Cohen d':>9s}  sig")
    out = []
    for label, p_raw, p_holm, es in corrected:
        d = extra[label]
        sig = "*" if p_holm < 0.05 else ""
        print(f"    {label:32s} {p_raw:10.3e} {p_holm:10.3e} "
              f"{es:10.3f} {d:9.3f}  {sig}")
        out.append((label, p_raw, p_holm, es, d))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="QAMQOR statistical significance tests.")
    p.add_argument("--results", default=None,
                   help="Combined results CSV (default: results/QAMQOR_all_results.csv).")
    p.add_argument("--metric", default="F1-score", choices=config.METRIC_KEYS)
    p.add_argument("--task", default="multiclass", choices=config.TASKS)
    p.add_argument("--out", default=None,
                   help="Optional CSV to write the post-hoc tables to.")
    return p.parse_args()


def main():
    args = parse_args()
    results = args.results or os.path.join(config.RESULTS_DIR,
                                           "QAMQOR_all_results.csv")
    if not os.path.exists(results):
        print(f"[error] combined results not found: {results}\n"
              f"Run 06_combine_results.py first.")
        return

    df = pd.read_csv(results)
    df = df[df["class"] == args.task].copy()
    print("=" * 74)
    print(f"QAMQOR statistical analysis | metric = {args.metric} | task = {args.task}")
    print(f"source = {results}")
    print("=" * 74)

    export_rows = []

    # ---- (1) Effect of evaluation protocol (splits) ----------------------- #
    print("\n[1] Effect of the EVALUATION PROTOCOL (splits as treatments)")
    df_splits = df[df["Split"].isin(HELD_OUT_SPLITS)]
    wide_s = _pivot(df_splits, ["Model", "Modality", "tool"], "Split", args.metric)
    wide_s = wide_s[[s for s in HELD_OUT_SPLITS if s in wide_s.columns]]
    res_s = friedman_over(wide_s, "splits")
    if res_s and res_s["p"] < 0.05:
        ph = wilcoxon_posthoc(res_s["wide"], res_s["treatments"])
        for label, p_raw, p_holm, es, d in ph:
            export_rows.append({"analysis": "splits", "metric": args.metric,
                                "task": args.task, "comparison": label,
                                "p_raw": p_raw, "p_holm": p_holm,
                                "rank_biserial": es, "cohens_d": d})

    # ---- (2) Effect of classifier (models) -------------------------------- #
    print("\n[2] Effect of the CLASSIFIER (models as treatments)")
    df_models = df[df["Split"].isin(HELD_OUT_SPLITS)]
    wide_m = _pivot(df_models, ["Split", "Modality", "tool"], "Model", args.metric)
    res_m = friedman_over(wide_m, "models")
    if res_m and res_m["p"] < 0.05:
        ph = wilcoxon_posthoc(res_m["wide"], res_m["treatments"])
        for label, p_raw, p_holm, es, d in ph:
            export_rows.append({"analysis": "models", "metric": args.metric,
                                "task": args.task, "comparison": label,
                                "p_raw": p_raw, "p_holm": p_holm,
                                "rank_biserial": es, "cohens_d": d})

    if args.out and export_rows:
        pd.DataFrame(export_rows).to_csv(args.out, index=False)
        print(f"\nPost-hoc tables written to: {args.out}")

    print("\nNote: report Kendall's W / rank-biserial alongside p-values; a "
          "significant\np-value with a small effect size indicates statistical "
          "but not practical significance.")


if __name__ == "__main__":
    main()
