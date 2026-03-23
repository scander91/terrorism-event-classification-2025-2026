#!/usr/bin/env python3
"""
Aggregate Cross-Domain Evaluation Results
==========================================
Reads individual result files and produces summary tables
showing Mean Accuracy (%) ± Standard Deviation for each
dataset, model, prompt, shot level, and selection method.

Usage:
    python3 03_aggregate_results.py

Output:
    results/cross_domain/summary_tables.txt
    results/cross_domain/summary_tables.json
"""

import os
import json
import numpy as np
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cross_domain")

DATASET_ORDER = [
    "nadi", "madar", "astd", "arsas", "asad", "labr",
    "semeval2016", "alomari", "arsentiment", "osact4", "adult_content"
]

DISPLAY_NAMES = {
    "nadi": "NADI", "madar": "MADAR", "astd": "ASTD",
    "arsas": "ArSAS", "asad": "ASAD", "labr": "LABR",
    "semeval2016": "SemEval", "alomari": "Alomari",
    "arsentiment": "ArSent", "osact4": "OSACT4",
    "adult_content": "Adult",
}


def load_all_results():
    """Load all result JSON files from the results directory."""
    results = []
    if not os.path.exists(RESULTS_DIR):
        return results
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_results.json"):
            with open(os.path.join(RESULTS_DIR, fname), "r") as f:
                results.extend(json.load(f))
    return results


def compute_statistics(results):
    """Group by configuration and compute mean ± std."""
    grouped = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["model"], r["prompt"],
               r["shots"], r["selection"])
        grouped[key].append(r["accuracy"])

    stats = {}
    for key, accs in grouped.items():
        stats[key] = {
            "mean": np.mean(accs) * 100,
            "std": np.std(accs) * 100,
            "n_runs": len(accs),
        }
    return stats


def format_table(stats, model, shots, selection):
    """Format one results table."""
    lines = []
    lines.append(f"\nModel: {model} | Shots: {shots} | Selection: {selection}")
    lines.append("-" * 70)

    header = f"{'Dataset':<12}"
    for p in ["P1", "P2", "P3"]:
        header += f"  {p:>18}"
    lines.append(header)
    lines.append("-" * 70)

    for ds in DATASET_ORDER:
        row = f"{DISPLAY_NAMES.get(ds, ds):<12}"
        for p in ["P1", "P2", "P3"]:
            key = (ds, model, p, shots, selection)
            if key in stats:
                v = stats[key]
                row += f"  {v['mean']:5.1f} +/- {v['std']:4.1f}    "
            else:
                row += f"  {'---':>18}"
        lines.append(row)

    return "\n".join(lines)


def main():
    results = load_all_results()
    if not results:
        print("No results found. Run 02_run_cross_domain.py first.")
        return

    print(f"Loaded {len(results)} result entries")
    stats = compute_statistics(results)

    report = []
    report.append("=" * 70)
    report.append("Cross-Domain Evaluation: Arabic NLP Benchmarks")
    report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
    report.append("=" * 70)

    for model in ["deepseek", "qwen3"]:
        for selection in ["random", "topk"]:
            for shots in [5, 8, 10]:
                report.append(format_table(stats, model, shots, selection))

    report_text = "\n".join(report)

    # Save text
    txt_path = os.path.join(RESULTS_DIR, "summary_tables.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"Saved: {txt_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "summary_tables.json")
    json_data = {str(k): v for k, v in stats.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")

    print(report_text)


if __name__ == "__main__":
    main()
