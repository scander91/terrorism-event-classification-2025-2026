#!/usr/bin/env python3
"""
MTL-CBERT Evaluation and Statistical Analysis
================================================
Aggregates results across seeds, computes mean ± std,
performs paired bootstrap significance tests, and generates
per-class performance analysis.

Usage:
    python3 evaluate.py
    python3 evaluate.py --results_dir outputs/
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from config import *


def load_results(results_dir):
    """Load all seed result files."""
    results = []
    for f in sorted(Path(results_dir).glob("results_seed*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def aggregate_metrics(results):
    """Compute mean ± std across seeds."""
    metrics_keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "mcc"]
    aggregated = {}

    for key in metrics_keys:
        values = [r["test_metrics"][key] for r in results]
        aggregated[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }

    return aggregated


def aggregate_per_class(results):
    """Compute per-class mean ± std across seeds."""
    classes = ATTACK_TYPES
    per_class = {}

    for cls in classes:
        f1_values = []
        prec_values = []
        rec_values = []

        for r in results:
            cls_data = r.get("per_class", {}).get(cls, {})
            if cls_data:
                f1_values.append(cls_data.get("f1-score", 0))
                prec_values.append(cls_data.get("precision", 0))
                rec_values.append(cls_data.get("recall", 0))

        if f1_values:
            per_class[cls] = {
                "f1": {"mean": np.mean(f1_values), "std": np.std(f1_values)},
                "precision": {"mean": np.mean(prec_values), "std": np.std(prec_values)},
                "recall": {"mean": np.mean(rec_values), "std": np.std(rec_values)},
            }

    return per_class


def paired_bootstrap_test(y_true, preds_a, preds_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap significance test.
    H0: model A and model B have equal macro-F1.
    Returns p-value.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    f1_a = f1_score(y_true, preds_a, average="macro")
    f1_b = f1_score(y_true, preds_b, average="macro")
    observed_diff = f1_a - f1_b

    count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        f1_a_boot = f1_score(y_true[idx], preds_a[idx], average="macro")
        f1_b_boot = f1_score(y_true[idx], preds_b[idx], average="macro")
        if (f1_b_boot - f1_a_boot) >= observed_diff:
            count += 1

    return count / n_bootstrap


def cohens_d(scores_a, scores_b):
    """Cohen's d effect size between two sets of F1 scores."""
    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
    diff = np.mean(scores_a) - np.mean(scores_b)
    pooled = np.sqrt((np.std(scores_a, ddof=1)**2 + np.std(scores_b, ddof=1)**2) / 2)
    return diff / pooled if pooled > 0 else float("inf")


def print_report(aggregated, per_class):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("MTL-CBERT Evaluation Report")
    print("=" * 60)

    print("\nOverall Metrics (Mean ± Std across 5 seeds):")
    for key, val in aggregated.items():
        print(f"  {key:>20s}: {val['mean']*100:.1f} ± {val['std']*100:.1f}")

    print("\nPer-Class F1 (Mean ± Std):")
    for cls, metrics in per_class.items():
        f1 = metrics["f1"]
        print(f"  {cls:>40s}: {f1['mean']*100:.1f} ± {f1['std']*100:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if not results:
        print("No results found. Run train.py first.")
        return

    print(f"Loaded {len(results)} seed results")

    aggregated = aggregate_metrics(results)
    per_class = aggregate_per_class(results)
    print_report(aggregated, per_class)

    summary = {
        "num_seeds": len(results),
        "overall": {k: {"mean": v["mean"], "std": v["std"]}
                    for k, v in aggregated.items()},
        "per_class": per_class,
    }

    output_path = Path(args.results_dir) / "evaluation_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved: {output_path}")


if __name__ == "__main__":
    main()
