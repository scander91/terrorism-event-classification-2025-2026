#!/usr/bin/env python3
"""
Step 1: Raw Data Profiling
===========================
Loads the raw GTD dataset and computes all BEFORE-preprocessing metrics
that Chapter 2 claims. Compares claimed vs actual values.

Depends on: dataset_config.json from step0
Output: step1_raw_profile.json, step1_report.txt
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ── Load config ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/TerrorismNER_Project")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ch2_verification_results")
CONFIG_FILE = os.path.join(OUTPUT_DIR, "dataset_config.json")

# ── Chapter 2 CLAIMED values ──────────────────────────────────────────
CLAIMED = {
    "initial_samples": 208688,
    "initial_features": 135,
    "features_with_missing": 76,
    "feature_level_prevalence_pct": 56.3,  # 76/135
    "avg_missing_rate_per_feature_pct": 41.2,
    "overall_cell_missing_rate_pct": 38.7,  # Eq 2.4
    "table5_overall_missing_rate_pct": 56.2,  # Table 5 (conflated!)
    "final_samples": 180706,
    "final_features": 73,
    "final_numerical": 30,
    "final_categorical": 28,
    "final_text": 15,
    "data_retention_pct": 89.7,  # Table 5 & 8
    "post_missing_rate_pct": 4.8,  # Table 5 & 8
    "complete_cases_pct": 89.7,  # Table 5
    "features_over_90pct_missing": None,  # Not specified in chapter
    "skewness_threshold": 2.0,
    "correlation_threshold": 0.85,  # From algorithm
    "vif_threshold": 10.0,
    "features_removed_vif": 10,
}


def load_raw_data(config):
    """Load the raw GTD dataset."""
    raw_path = config.get("raw_gtd_path")
    if not raw_path or not os.path.exists(raw_path):
        print("ERROR: raw_gtd_path not set or file not found.")
        print(f"  Path: {raw_path}")
        print("  Please edit dataset_config.json and set 'raw_gtd_path'")
        sys.exit(1)

    print(f"Loading: {raw_path}")
    if raw_path.endswith(".csv"):
        df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    elif raw_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(raw_path)
    elif raw_path.endswith(".parquet"):
        df = pd.read_parquet(raw_path)
    elif raw_path.endswith((".pkl", ".pickle")):
        df = pd.read_pickle(raw_path)
    else:
        print(f"ERROR: Unsupported file format: {raw_path}")
        sys.exit(1)

    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def classify_features(df):
    """Classify features as numerical, categorical, or text."""
    numerical = []
    categorical = []
    text = []

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            numerical.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            # Heuristic: if average non-null string length > 50, treat as text
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                nunique = non_null.nunique()
                # Text if long strings or very high cardinality
                if avg_len > 50 or (nunique > 1000 and avg_len > 20):
                    text.append(col)
                else:
                    categorical.append(col)
            else:
                categorical.append(col)  # All null, can't determine
        else:
            categorical.append(col)

    return numerical, categorical, text


def compute_missing_metrics(df):
    """Compute all three missing rate metrics from Chapter 2 Equations 2.2-2.4."""
    n_samples, n_features = df.shape

    # Per-feature missing rates
    feature_missing_counts = df.isnull().sum()
    feature_missing_rates = feature_missing_counts / n_samples

    # Eq 2.2: Feature-level prevalence (proportion of features with ANY missing)
    features_with_any_missing = (feature_missing_counts > 0).sum()
    eta_features = features_with_any_missing / n_features

    # Eq 2.3: Average missing rate per feature
    eta_bar = feature_missing_rates.mean()

    # Eq 2.4: Overall dataset missing rate (cell-level)
    total_cells = n_samples * n_features
    total_missing = df.isnull().sum().sum()
    eta_overall = total_missing / total_cells

    # Additional: average missing rate across ONLY affected features
    affected_rates = feature_missing_rates[feature_missing_rates > 0]
    eta_bar_affected = affected_rates.mean() if len(affected_rates) > 0 else 0

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "features_with_any_missing": int(features_with_any_missing),
        "eta_features_pct": round(float(eta_features * 100), 2),
        "eta_bar_pct": round(float(eta_bar * 100), 2),
        "eta_bar_affected_only_pct": round(float(eta_bar_affected * 100), 2),
        "eta_overall_pct": round(float(eta_overall * 100), 2),
        "total_cells": int(total_cells),
        "total_missing_cells": int(total_missing),
        "feature_missing_rates": {col: round(float(r * 100), 2)
                                   for col, r in feature_missing_rates.items()
                                   if r > 0},
    }


def compute_features_over_threshold(df, threshold_pct=90):
    """Find features with missing rate above threshold."""
    n_samples = len(df)
    feature_missing_pcts = (df.isnull().sum() / n_samples * 100)
    over = feature_missing_pcts[feature_missing_pcts > threshold_pct]
    return over.sort_values(ascending=False).to_dict()


def compute_skewness(df, numerical_cols):
    """Compute skewness for numerical features (for imputation method selection)."""
    results = {}
    for col in numerical_cols:
        series = df[col].dropna()
        if len(series) > 2:
            skew = float(series.skew())
            method = "median" if abs(skew) > 2.0 else "mean"
            results[col] = {
                "skewness": round(skew, 4),
                "abs_skewness": round(abs(skew), 4),
                "imputation_method": method,
            }
    return results


def compute_correlation_matrix(df, numerical_cols):
    """Compute correlation matrix for numerical features."""
    # Only use columns with enough non-null data
    usable = [c for c in numerical_cols if df[c].dropna().shape[0] > 10]
    if not usable:
        return {}, []

    corr = df[usable].corr()

    # Find highly correlated pairs (|r| > threshold)
    threshold = CLAIMED["correlation_threshold"]
    high_corr_pairs = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            r = abs(corr.iloc[i, j])
            if r > threshold:
                high_corr_pairs.append({
                    "feature_1": usable[i],
                    "feature_2": usable[j],
                    "correlation": round(float(corr.iloc[i, j]), 4),
                    "abs_correlation": round(float(r), 4),
                })

    high_corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return {"threshold": threshold, "pairs": high_corr_pairs}, usable


def compute_complete_cases(df):
    """Compute complete cases (rows with NO missing values)."""
    complete = df.dropna().shape[0]
    total = df.shape[0]
    return {
        "complete_rows": int(complete),
        "total_rows": int(total),
        "complete_cases_pct": round(float(complete / total * 100), 2),
    }


def generate_report(results, report_path):
    """Generate a human-readable comparison report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  CHAPTER 2 VERIFICATION REPORT — STEP 1: RAW DATA PROFILE")
    lines.append(f"  Generated: {datetime.now().isoformat()}")
    lines.append("=" * 78)

    # ── Dataset dimensions ────────────────────────────────────────────
    lines.append("\n── 1. DATASET DIMENSIONS ─────────────────────────────────────────────")
    actual = results["missing_metrics"]
    lines.append(f"  {'Metric':<40} {'Chapter Claims':>15} {'Actual':>15} {'Match':>8}")
    lines.append(f"  {'-'*40} {'-'*15} {'-'*15} {'-'*8}")

    checks = [
        ("Samples (rows)", CLAIMED["initial_samples"], actual["n_samples"]),
        ("Features (columns)", CLAIMED["initial_features"], actual["n_features"]),
    ]
    for label, claimed, act in checks:
        match = "✅" if claimed == act else "❌ DIFFERS"
        lines.append(f"  {label:<40} {str(claimed):>15} {str(act):>15} {match:>8}")

    # ── Feature type classification ───────────────────────────────────
    lines.append("\n── 2. FEATURE CLASSIFICATION ─────────────────────────────────────────")
    fc = results["feature_classification"]
    lines.append(f"  Numerical features:  {fc['n_numerical']}")
    lines.append(f"  Categorical features: {fc['n_categorical']}")
    lines.append(f"  Text features:       {fc['n_text']}")
    lines.append(f"  Total:               {fc['n_numerical'] + fc['n_categorical'] + fc['n_text']}")

    # ── Missing rate metrics ──────────────────────────────────────────
    lines.append("\n── 3. MISSING RATE METRICS (THE CRITICAL ISSUE) ─────────────────────")
    lines.append(f"  {'Metric':<50} {'Claimed':>10} {'Actual':>10} {'Match':>8}")
    lines.append(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*8}")

    missing_checks = [
        ("Features with ANY missing (count)",
         CLAIMED["features_with_missing"], actual["features_with_any_missing"]),
        ("Feature-level prevalence η_features (%)",
         CLAIMED["feature_level_prevalence_pct"], actual["eta_features_pct"]),
        ("Avg missing rate per feature η̄ (%)",
         CLAIMED["avg_missing_rate_per_feature_pct"], actual["eta_bar_pct"]),
        ("Overall cell-level missing rate η_overall (%) [Eq 2.4]",
         CLAIMED["overall_cell_missing_rate_pct"], actual["eta_overall_pct"]),
        ("Table 5 'Overall Missing Rate' (%)",
         CLAIMED["table5_overall_missing_rate_pct"], "N/A — label error"),
    ]

    for label, claimed, act in missing_checks:
        if isinstance(act, str):
            match = "⚠️"
        elif isinstance(claimed, (int, float)) and isinstance(act, (int, float)):
            match = "✅" if abs(claimed - act) < 0.15 else "❌ DIFFERS"
        else:
            match = "?"
        lines.append(f"  {label:<50} {str(claimed):>10} {str(act):>10} {match:>8}")

    lines.append(f"\n  DIAGNOSIS:")
    lines.append(f"    76/135 = {76/135*100:.3f}% → rounds to 56.3% (NOT 56.2% as in Table 5)")
    lines.append(f"    Actual: {actual['features_with_any_missing']}/{actual['n_features']} = "
                 f"{actual['eta_features_pct']}%")
    lines.append(f"    Table 5 says 'Overall Missing Rate = 56.2%' — this is FEATURE-LEVEL")
    lines.append(f"    PREVALENCE, not the cell-level rate of {actual['eta_overall_pct']}%")

    # ── Additional: avg across affected features only ─────────────────
    lines.append(f"\n  NOTE: Chapter says '41.2% across affected features'")
    lines.append(f"    η̄ (all features):       {actual['eta_bar_pct']}%")
    lines.append(f"    η̄ (affected only):      {actual['eta_bar_affected_only_pct']}%")
    lines.append(f"    → Which one is 41.2%?")

    # ── Complete cases ────────────────────────────────────────────────
    lines.append("\n── 4. COMPLETE CASES & DATA RETENTION ───────────────────────────────")
    cc = results["complete_cases"]
    lines.append(f"  Complete rows (no missing): {cc['complete_rows']:,} / {cc['total_rows']:,}")
    lines.append(f"  Complete cases rate:        {cc['complete_cases_pct']}%")
    lines.append(f"  Chapter claims:             {CLAIMED['complete_cases_pct']}%")
    lines.append(f"  Data retention (180706/208688): {180706/208688*100:.1f}%")
    lines.append(f"  Chapter claims retention:   {CLAIMED['data_retention_pct']}%")

    retention_actual = 180706 / actual["n_samples"] * 100 if actual["n_samples"] > 0 else 0
    lines.append(f"  Actual if final=180706:     {retention_actual:.1f}%")

    # ── Features > 90% missing ────────────────────────────────────────
    lines.append("\n── 5. FEATURES WITH >90% MISSING (elimination candidates) ──────────")
    over90 = results["features_over_90pct"]
    if over90:
        lines.append(f"  {'Feature':<50} {'Missing %':>10}")
        lines.append(f"  {'-'*50} {'-'*10}")
        for feat, pct in over90.items():
            lines.append(f"  {feat:<50} {pct:>10.2f}%")
        lines.append(f"\n  Total features >90% missing: {len(over90)}")
    else:
        lines.append("  No features with >90% missing rate")

    # ── Skewness analysis ─────────────────────────────────────────────
    lines.append("\n── 6. SKEWNESS ANALYSIS (imputation method selection) ──────────────")
    skew = results["skewness"]
    median_count = sum(1 for v in skew.values() if v["imputation_method"] == "median")
    mean_count = sum(1 for v in skew.values() if v["imputation_method"] == "mean")
    lines.append(f"  Total numerical features analyzed: {len(skew)}")
    lines.append(f"  |skewness| > 2 → median imputation: {median_count} features")
    lines.append(f"  |skewness| ≤ 2 → mean imputation:   {mean_count} features")
    lines.append(f"\n  Top 10 most skewed features:")
    sorted_skew = sorted(skew.items(), key=lambda x: x[1]["abs_skewness"], reverse=True)
    lines.append(f"  {'Feature':<40} {'Skewness':>10} {'Method':>10}")
    lines.append(f"  {'-'*40} {'-'*10} {'-'*10}")
    for feat, info in sorted_skew[:10]:
        lines.append(f"  {feat:<40} {info['skewness']:>10.4f} {info['imputation_method']:>10}")

    # ── Correlation analysis ──────────────────────────────────────────
    lines.append("\n── 7. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.85) ────────────────")
    corr = results["correlation"]
    pairs = corr.get("pairs", [])
    if pairs:
        lines.append(f"  Total pairs with |r| > {corr['threshold']}: {len(pairs)}")
        lines.append(f"\n  {'Feature 1':<30} {'Feature 2':<30} {'Correlation':>12}")
        lines.append(f"  {'-'*30} {'-'*30} {'-'*12}")
        for p in pairs[:20]:
            lines.append(f"  {p['feature_1']:<30} {p['feature_2']:<30} {p['correlation']:>12.4f}")
        if len(pairs) > 20:
            lines.append(f"  ... and {len(pairs) - 20} more pairs")
    else:
        lines.append(f"  No pairs with |r| > {corr.get('threshold', 0.85)}")

    # ── Summary of issues found ───────────────────────────────────────
    lines.append(f"\n{'='*78}")
    lines.append("  ISSUES SUMMARY")
    lines.append(f"{'='*78}")

    issues = []
    if actual["n_samples"] != CLAIMED["initial_samples"]:
        issues.append(f"❌ Sample count: claimed {CLAIMED['initial_samples']:,}, "
                      f"actual {actual['n_samples']:,}")
    if actual["n_features"] != CLAIMED["initial_features"]:
        issues.append(f"❌ Feature count: claimed {CLAIMED['initial_features']}, "
                      f"actual {actual['n_features']}")
    if abs(actual["eta_features_pct"] - CLAIMED["feature_level_prevalence_pct"]) > 0.15:
        issues.append(f"❌ Feature prevalence: claimed {CLAIMED['feature_level_prevalence_pct']}%, "
                      f"actual {actual['eta_features_pct']}%")
    if abs(actual["eta_bar_pct"] - CLAIMED["avg_missing_rate_per_feature_pct"]) > 0.5:
        issues.append(f"❌ Avg missing rate: claimed {CLAIMED['avg_missing_rate_per_feature_pct']}%, "
                      f"actual {actual['eta_bar_pct']}%")
    if abs(actual["eta_overall_pct"] - CLAIMED["overall_cell_missing_rate_pct"]) > 0.5:
        issues.append(f"❌ Overall missing rate: claimed {CLAIMED['overall_cell_missing_rate_pct']}%, "
                      f"actual {actual['eta_overall_pct']}%")

    issues.append(f"⚠️  Table 5 'Overall Missing Rate = 56.2%' is feature-level prevalence, "
                  f"NOT cell-level rate ({actual['eta_overall_pct']}%)")

    retention = 180706 / actual["n_samples"] * 100 if actual["n_samples"] > 0 else 0
    if abs(retention - CLAIMED["data_retention_pct"]) > 0.5:
        issues.append(f"❌ Data retention: claimed {CLAIMED['data_retention_pct']}%, "
                      f"actual {retention:.1f}% (180706/{actual['n_samples']:,})")

    for i, issue in enumerate(issues, 1):
        lines.append(f"  {i}. {issue}")

    if not issues:
        lines.append("  ✅ All metrics match!")

    lines.append(f"\n{'='*78}")
    lines.append("  NEXT: Run step2_preprocessing_pipeline.py")
    lines.append(f"{'='*78}\n")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    return report


def main():
    print(f"{'='*70}")
    print(f"  STEP 1: RAW DATA PROFILING")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    # Load config
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Config not found. Run step0_discover_dataset.py first.")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Load data
    df = load_raw_data(config)

    # 1. Feature classification
    print("\n[1/6] Classifying features...")
    num_cols, cat_cols, text_cols = classify_features(df)
    feature_class = {
        "n_numerical": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_text": len(text_cols),
        "numerical": num_cols,
        "categorical": cat_cols,
        "text": text_cols,
    }
    print(f"  Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}, Text: {len(text_cols)}")

    # 2. Missing metrics
    print("\n[2/6] Computing missing rate metrics...")
    missing = compute_missing_metrics(df)

    # 3. Features over 90% missing
    print("\n[3/6] Finding features >90% missing...")
    over90 = compute_features_over_threshold(df, 90)
    print(f"  Found {len(over90)} features")

    # 4. Complete cases
    print("\n[4/6] Computing complete cases...")
    cc = compute_complete_cases(df)
    print(f"  Complete rows: {cc['complete_rows']:,} / {cc['total_rows']:,} ({cc['complete_cases_pct']}%)")

    # 5. Skewness
    print("\n[5/6] Computing skewness for numerical features...")
    skewness = compute_skewness(df, num_cols)
    print(f"  Analyzed {len(skewness)} numerical features")

    # 6. Correlation
    print("\n[6/6] Computing correlation matrix...")
    corr, usable_num = compute_correlation_matrix(df, num_cols)
    print(f"  Found {len(corr.get('pairs', []))} highly correlated pairs")

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "raw_data_path": config.get("raw_gtd_path"),
        "feature_classification": feature_class,
        "missing_metrics": missing,
        "features_over_90pct": over90,
        "complete_cases": cc,
        "skewness": skewness,
        "correlation": corr,
        "claimed_values": CLAIMED,
    }

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "step1_raw_profile.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # Generate report
    report_path = os.path.join(OUTPUT_DIR, "step1_report.txt")
    generate_report(results, report_path)


if __name__ == "__main__":
    main()
