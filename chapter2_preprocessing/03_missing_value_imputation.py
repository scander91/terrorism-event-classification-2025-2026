#!/usr/bin/env python3
"""
Step 2: Preprocessing Pipeline Verification
=============================================
Applies Chapter 2's described preprocessing pipeline step-by-step,
tracking sample counts, feature counts, and metrics at each stage.

Steps (from Chapter 2 Algorithm 1):
  Phase 1: Feature Assessment (>90% missing → drop)
  Phase 2: Imputation (skewness-based mean/median, mode for categorical)
  Phase 3: Redundancy Elimination (correlation > 0.85)
  Phase 4: Feature Engineering (create new categorical features)
  Phase 5: VIF Removal (VIF > 10)

Depends on: dataset_config.json, step1_raw_profile.json
Output: step2_pipeline_results.json, step2_report.txt
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/TerrorismNER_Project")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ch2_verification_results")
CONFIG_FILE = os.path.join(OUTPUT_DIR, "dataset_config.json")

# Chapter 2 parameters
MISSING_THRESHOLD = 90.0   # Drop features with >90% missing
SKEWNESS_THRESHOLD = 2.0   # |skew| > 2 → median, else mean
CORR_THRESHOLD = 0.85      # Drop one of pair if |r| > 0.85
VIF_THRESHOLD = 10.0       # Drop if VIF > 10

# Chapter 2 claimed values at each stage (Table 4)
CLAIMED_STAGES = {
    "initial": {"numerical": None, "categorical": None, "text": None, "total": 135},
    "after_missing_elimination": {"numerical": None, "categorical": None, "text": None, "total": None},
    "after_redundancy": {"numerical": 44, "categorical": 13, "text": 15, "total": 72},
    "after_feature_engineering": {"numerical": 40, "categorical": 28, "text": 15, "total": 83},
    "final_after_vif": {"numerical": 30, "categorical": 28, "text": 15, "total": 73},
}


def load_data():
    """Load raw GTD dataset using config."""
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    raw_path = config.get("raw_gtd_path")
    if not raw_path:
        print("ERROR: raw_gtd_path not set in config.")
        sys.exit(1)

    print(f"Loading: {raw_path}")
    if raw_path.endswith(".csv"):
        df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    elif raw_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(raw_path)
    elif raw_path.endswith(".parquet"):
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_pickle(raw_path)

    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def classify_features(df):
    """Classify columns into numerical, categorical, text."""
    numerical, categorical, text = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical.append(col)
        elif pd.api.types.is_object_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                nunique = non_null.nunique()
                if avg_len > 50 or (nunique > 1000 and avg_len > 20):
                    text.append(col)
                else:
                    categorical.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return numerical, categorical, text


def snapshot(df, stage_name, numerical, categorical, text, details=None):
    """Take a snapshot of current dataset state."""
    n_samples, n_features = df.shape
    missing_cells = int(df.isnull().sum().sum())
    total_cells = n_samples * n_features
    features_with_missing = int((df.isnull().sum() > 0).sum())

    snap = {
        "stage": stage_name,
        "samples": n_samples,
        "total_features": n_features,
        "numerical": len(numerical),
        "categorical": len(categorical),
        "text": len(text),
        "feature_count_check": len(numerical) + len(categorical) + len(text),
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "cell_missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0,
        "features_with_missing": features_with_missing,
        "feature_prevalence_pct": round(features_with_missing / n_features * 100, 2) if n_features > 0 else 0,
        "complete_rows": int(df.dropna().shape[0]),
        "complete_rows_pct": round(df.dropna().shape[0] / n_samples * 100, 2) if n_samples > 0 else 0,
        "numerical_cols": list(numerical),
        "categorical_cols": list(categorical),
        "text_cols": list(text),
    }
    if details:
        snap["details"] = details
    return snap


def phase1_feature_elimination(df, numerical, categorical, text, threshold=90):
    """Phase 1: Remove features with >threshold% missing values."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 1: Feature Elimination (>{threshold}% missing)")
    print(f"{'─'*60}")

    n_samples = len(df)
    missing_pcts = (df.isnull().sum() / n_samples * 100)

    # Identify features to drop
    to_drop = missing_pcts[missing_pcts > threshold].sort_values(ascending=False)
    kept = missing_pcts[missing_pcts <= threshold]

    print(f"  Features with >{threshold}% missing: {len(to_drop)}")
    if len(to_drop) > 0:
        print(f"  Dropping:")
        for col, pct in to_drop.items():
            ftype = "NUM" if col in numerical else ("CAT" if col in categorical else "TXT")
            print(f"    [{ftype}] {col}: {pct:.1f}% missing")

    # Drop them
    df_out = df.drop(columns=to_drop.index.tolist())

    # Reclassify remaining
    num_out = [c for c in numerical if c in df_out.columns]
    cat_out = [c for c in categorical if c in df_out.columns]
    txt_out = [c for c in text if c in df_out.columns]

    details = {
        "threshold": threshold,
        "dropped_features": {col: round(float(pct), 2) for col, pct in to_drop.items()},
        "n_dropped": len(to_drop),
    }

    print(f"  Before: {df.shape[1]} features → After: {df_out.shape[1]} features")
    return df_out, num_out, cat_out, txt_out, details


def phase2_imputation(df, numerical, categorical, text, skew_threshold=2.0):
    """Phase 2: Impute missing values (skewness-based for numerical, mode for categorical)."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 2: Missing Value Imputation")
    print(f"{'─'*60}")

    df_out = df.copy()
    imputation_log = {}

    # Numerical: skewness-based
    for col in numerical:
        if df_out[col].isnull().any():
            non_null = df_out[col].dropna()
            if len(non_null) > 2:
                skew = float(non_null.skew())
                if abs(skew) > skew_threshold:
                    fill_val = float(non_null.median())
                    method = "median"
                else:
                    fill_val = float(non_null.mean())
                    method = "mean"

                n_imputed = int(df_out[col].isnull().sum())
                df_out[col].fillna(fill_val, inplace=True)
                imputation_log[col] = {
                    "type": "numerical",
                    "method": method,
                    "skewness": round(skew, 4),
                    "fill_value": round(fill_val, 4),
                    "n_imputed": n_imputed,
                }

    # Categorical: mode imputation
    for col in categorical:
        if df_out[col].isnull().any():
            mode_val = df_out[col].mode()
            if len(mode_val) > 0:
                fill_val = mode_val.iloc[0]
                n_imputed = int(df_out[col].isnull().sum())
                df_out[col].fillna(fill_val, inplace=True)
                imputation_log[col] = {
                    "type": "categorical",
                    "method": "mode",
                    "fill_value": str(fill_val),
                    "n_imputed": n_imputed,
                }

    # Text: leave as-is or fill with empty string
    for col in text:
        if df_out[col].isnull().any():
            n_imputed = int(df_out[col].isnull().sum())
            df_out[col].fillna("", inplace=True)
            imputation_log[col] = {
                "type": "text",
                "method": "empty_string",
                "n_imputed": n_imputed,
            }

    print(f"  Imputed {len(imputation_log)} features")
    print(f"  Numerical (mean): {sum(1 for v in imputation_log.values() if v.get('method') == 'mean')}")
    print(f"  Numerical (median): {sum(1 for v in imputation_log.values() if v.get('method') == 'median')}")
    print(f"  Categorical (mode): {sum(1 for v in imputation_log.values() if v.get('method') == 'mode')}")

    remaining_missing = int(df_out.isnull().sum().sum())
    print(f"  Remaining missing cells after imputation: {remaining_missing:,}")

    return df_out, imputation_log


def phase3_redundancy_elimination(df, numerical, categorical, text, corr_threshold=0.85):
    """Phase 3: Remove redundant features based on correlation."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 3: Redundancy Elimination (|r| > {corr_threshold})")
    print(f"{'─'*60}")

    # Compute correlation on numerical features only
    usable = [c for c in numerical if c in df.columns and df[c].std() > 0]
    if len(usable) < 2:
        print("  Not enough numerical features for correlation analysis")
        return df, numerical, categorical, text, {}

    corr_matrix = df[usable].corr().abs()

    # Find pairs and greedily remove
    to_remove = set()
    pairs_found = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                col_i, col_j = usable[i], usable[j]
                pairs_found.append({
                    "feature_1": col_i,
                    "feature_2": col_j,
                    "correlation": round(float(corr_matrix.iloc[i, j]), 4),
                })
                # Remove the one with more missing in original (or second one)
                if col_j not in to_remove and col_i not in to_remove:
                    # Remove the one with higher mean correlation to others
                    mean_corr_i = corr_matrix[col_i].mean()
                    mean_corr_j = corr_matrix[col_j].mean()
                    victim = col_j if mean_corr_j >= mean_corr_i else col_i
                    to_remove.add(victim)

    print(f"  High-correlation pairs found: {len(pairs_found)}")
    print(f"  Features to remove: {len(to_remove)}")
    for feat in sorted(to_remove):
        print(f"    Removing: {feat}")

    df_out = df.drop(columns=list(to_remove), errors="ignore")
    num_out = [c for c in numerical if c not in to_remove]
    cat_out = [c for c in categorical if c not in to_remove]
    txt_out = [c for c in text if c not in to_remove]

    details = {
        "threshold": corr_threshold,
        "pairs_found": pairs_found,
        "removed_features": list(to_remove),
        "n_removed": len(to_remove),
    }

    print(f"  Numerical: {len(numerical)} → {len(num_out)}")
    return df_out, num_out, cat_out, txt_out, details


def phase4_feature_engineering(df, numerical, categorical, text):
    """Phase 4: Feature engineering — create new categorical features from numerical.
    
    Chapter 2 says: 15 new categorical features created, 4 numerical dropped.
    We simulate typical GTD feature engineering here.
    """
    print(f"\n{'─'*60}")
    print(f"  PHASE 4: Feature Engineering")
    print(f"{'─'*60}")

    df_out = df.copy()
    new_categorical = []
    dropped_numerical = []

    # Common GTD feature engineering patterns:
    # 1. Binary indicators from numerical columns
    binary_candidates = {
        "nkill": "has_fatalities",
        "nwound": "has_wounded",
        "nkillter": "has_perpetrator_deaths",
        "nwoundte": "has_perpetrator_wounded",
        "property": "has_property_damage",
        "nhostkid": "has_hostages",
        "ransom": "has_ransom",
        "ndays": "extended_incident",
    }

    for src_col, new_col in binary_candidates.items():
        if src_col in df_out.columns:
            df_out[new_col] = (df_out[src_col] > 0).astype(int)
            new_categorical.append(new_col)

    # 2. Decade/period from year
    if "iyear" in df_out.columns:
        df_out["decade"] = (df_out["iyear"] // 10 * 10).astype(str) + "s"
        new_categorical.append("decade")

    # 3. Attack severity categories
    if "nkill" in df_out.columns:
        df_out["severity_category"] = pd.cut(
            df_out["nkill"].fillna(0),
            bins=[-1, 0, 1, 5, 20, 100, float("inf")],
            labels=["none", "single", "few", "moderate", "mass", "extreme"],
        ).astype(str)
        new_categorical.append("severity_category")

    # 4. Geographic region grouping
    if "region" in df_out.columns:
        df_out["region_group"] = df_out["region"].map(lambda x: "high" if x in [6, 10, 11] else "other")
        new_categorical.append("region_group")

    # 5. Success indicator categories
    if "success" in df_out.columns:
        new_categorical.append("success")  # Already exists but re-classify

    # 6. Time-based features
    if "imonth" in df_out.columns:
        df_out["quarter"] = ((df_out["imonth"] - 1) // 3 + 1).astype(str)
        new_categorical.append("quarter")

    if "iday" in df_out.columns:
        # Day part (early/mid/late month)
        df_out["month_part"] = pd.cut(
            df_out["iday"].clip(1, 31),
            bins=[0, 10, 20, 31],
            labels=["early", "mid", "late"]
        ).astype(str)
        new_categorical.append("month_part")

    # Track what was created vs what was dropped
    # Note: We do NOT actually drop numerical features here since the chapter
    # doesn't explain the 44→40 drop. We just log what COULD be dropped.
    potential_drops = []
    for col in numerical:
        if col in binary_candidates:
            potential_drops.append(col)

    cat_out = categorical + new_categorical
    num_out = numerical  # Keep all for now — the unexplained drop needs investigation

    details = {
        "new_categorical_features": new_categorical,
        "n_new_categorical": len(new_categorical),
        "potential_numerical_drops": potential_drops,
        "NOTE": "Chapter claims 44→40 numerical (4 dropped) but never explains which 4. "
                "This script does NOT drop them. The unexplained gap needs investigation.",
    }

    print(f"  Created {len(new_categorical)} new categorical features")
    print(f"  New features: {new_categorical}")
    print(f"  ⚠️  Chapter claims 44→40 numerical features here (4 dropped)")
    print(f"     but never explains which 4 are removed. Flagging for review.")

    return df_out, num_out, cat_out, text, details


def phase5_vif_removal(df, numerical, categorical, text, vif_threshold=10.0):
    """Phase 5: Remove features with VIF > threshold."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 5: VIF-Based Feature Removal (VIF > {vif_threshold})")
    print(f"{'─'*60}")

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Only use numerical features that are in the dataframe and have variance
    usable = [c for c in numerical if c in df.columns]
    usable = [c for c in usable if df[c].std() > 0 and df[c].notna().sum() > 10]

    if len(usable) < 2:
        print("  Not enough numerical features for VIF analysis")
        return df, numerical, categorical, text, {}

    # Prepare data: drop rows with NaN in numerical columns, standardize
    df_vif = df[usable].dropna()

    if len(df_vif) < 10:
        print("  Not enough complete rows for VIF computation")
        return df, numerical, categorical, text, {}

    # Standardize to avoid numerical issues
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_vif), columns=usable)

    # Iterative VIF removal
    removed = []
    vif_history = []
    current_features = list(usable)
    iteration = 0

    while True:
        iteration += 1
        if len(current_features) < 2:
            break

        X_current = X[current_features]

        # Compute VIF for each feature
        vif_values = {}
        for i, col in enumerate(current_features):
            try:
                vif = variance_inflation_factor(X_current.values, i)
                vif_values[col] = round(float(vif), 2)
            except Exception:
                vif_values[col] = float("inf")

        # Record this iteration
        vif_history.append({
            "iteration": iteration,
            "n_features": len(current_features),
            "vif_values": dict(sorted(vif_values.items(), key=lambda x: x[1], reverse=True)),
        })

        # Find max VIF
        max_vif_col = max(vif_values, key=vif_values.get)
        max_vif_val = vif_values[max_vif_col]

        if max_vif_val <= vif_threshold:
            print(f"  Iteration {iteration}: All VIFs ≤ {vif_threshold}. Done.")
            break

        print(f"  Iteration {iteration}: Removing '{max_vif_col}' (VIF = {max_vif_val:.2f})")
        removed.append({"feature": max_vif_col, "vif": max_vif_val, "iteration": iteration})
        current_features.remove(max_vif_col)

    # Apply removal
    remove_cols = [r["feature"] for r in removed]
    df_out = df.drop(columns=remove_cols, errors="ignore")
    num_out = [c for c in numerical if c not in remove_cols]

    details = {
        "vif_threshold": vif_threshold,
        "removed_features": removed,
        "n_removed": len(removed),
        "claimed_n_removed": 10,
        "final_vif_values": vif_history[-1]["vif_values"] if vif_history else {},
        "vif_history": vif_history,
    }

    print(f"\n  Total removed: {len(removed)} (Chapter claims: 10)")
    print(f"  Numerical features: {len(numerical)} → {len(num_out)}")
    if len(removed) != 10:
        print(f"  ⚠️  MISMATCH: Removed {len(removed)} features, chapter claims 10")

    return df_out, num_out, categorical, text, details


def compute_kl_divergence(original_series, imputed_series, n_bins=50):
    """Compute KL-divergence between original and imputed distributions."""
    # Only compare non-null original values with the full imputed column
    orig = original_series.dropna().values
    imp = imputed_series.values

    if len(orig) < 10 or len(imp) < 10:
        return None

    # Create common bin edges
    combined = np.concatenate([orig, imp])
    bin_edges = np.histogram_bin_edges(combined, bins=n_bins)

    # Compute histograms (as probability distributions)
    p, _ = np.histogram(orig, bins=bin_edges, density=True)
    q, _ = np.histogram(imp, bins=bin_edges, density=True)

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    p = p + eps
    q = q + eps

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # KL-divergence
    kl = float(np.sum(p * np.log(p / q)))
    return round(kl, 6)


def compute_post_imputation_kl(df_before, df_after, numerical_cols):
    """Compute KL-divergence for all imputed numerical features."""
    print(f"\n{'─'*60}")
    print(f"  KL-DIVERGENCE: Before vs After Imputation")
    print(f"{'─'*60}")

    results = {}
    for col in numerical_cols:
        if col in df_before.columns and col in df_after.columns:
            if df_before[col].isnull().any():
                kl = compute_kl_divergence(df_before[col], df_after[col])
                if kl is not None:
                    results[col] = {
                        "kl_divergence": kl,
                        "n_original_non_null": int(df_before[col].notna().sum()),
                        "n_after": int(df_after[col].notna().sum()),
                        "n_imputed": int(df_before[col].isnull().sum()),
                    }

    if results:
        print(f"  {'Feature':<35} {'KL-Div':>10} {'N Imputed':>12}")
        print(f"  {'-'*35} {'-'*10} {'-'*12}")
        for col, info in sorted(results.items(), key=lambda x: x[1]["kl_divergence"], reverse=True):
            print(f"  {col:<35} {info['kl_divergence']:>10.6f} {info['n_imputed']:>12,}")

    return results


def generate_report(snapshots, all_details, kl_results, report_path):
    """Generate the final pipeline report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  CHAPTER 2 VERIFICATION — STEP 2: PREPROCESSING PIPELINE")
    lines.append(f"  Generated: {datetime.now().isoformat()}")
    lines.append("=" * 78)

    # Stage-by-stage comparison
    lines.append("\n── PIPELINE STAGE TRACKING (compare with Table 4) ──────────────────")
    lines.append(f"\n  {'Stage':<35} {'Samples':>8} {'Num':>5} {'Cat':>5} {'Txt':>5} {'Total':>6} {'Missing%':>9}")
    lines.append(f"  {'-'*35} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*9}")

    for snap in snapshots:
        lines.append(f"  {snap['stage']:<35} {snap['samples']:>8,} {snap['numerical']:>5} "
                     f"{snap['categorical']:>5} {snap['text']:>5} {snap['total_features']:>6} "
                     f"{snap['cell_missing_pct']:>8.2f}%")

    # Comparison with claimed Table 4
    lines.append(f"\n── COMPARISON WITH CHAPTER 2 TABLE 4 ───────────────────────────────")
    stage_map = {
        "0_initial": "initial",
        "1_after_elimination": "after_missing_elimination",
        "3_after_redundancy": "after_redundancy",
        "4_after_engineering": "after_feature_engineering",
        "5_after_vif": "final_after_vif",
    }

    for snap in snapshots:
        stage_key = stage_map.get(snap["stage"])
        if stage_key and stage_key in CLAIMED_STAGES:
            claimed = CLAIMED_STAGES[stage_key]
            lines.append(f"\n  Stage: {snap['stage']}")
            for metric in ["numerical", "categorical", "text", "total"]:
                claimed_val = claimed.get(metric if metric != "total" else metric)
                if metric == "total":
                    actual_val = snap["total_features"]
                else:
                    actual_val = snap[metric]
                if claimed_val is not None:
                    match = "✅" if claimed_val == actual_val else "❌"
                    lines.append(f"    {metric:<15} claimed: {claimed_val:>5}  actual: {actual_val:>5}  {match}")

    # Data retention
    lines.append(f"\n── DATA RETENTION ANALYSIS ─────────────────────────────────────────")
    initial = snapshots[0]["samples"]
    final = snapshots[-1]["samples"]
    retention = final / initial * 100 if initial > 0 else 0
    lines.append(f"  Initial samples:       {initial:,}")
    lines.append(f"  Final samples:         {final:,}")
    lines.append(f"  Retention:             {retention:.1f}%")
    lines.append(f"  Chapter claims:        89.7%")
    lines.append(f"  Computed (180706/208688): {180706/208688*100:.1f}%")
    if abs(retention - 89.7) > 0.5:
        lines.append(f"  ⚠️  MISMATCH: Actual {retention:.1f}% ≠ claimed 89.7%")

    # Missing rate after preprocessing
    lines.append(f"\n── POST-PREPROCESSING MISSING RATES ───────────────────────────────")
    final_snap = snapshots[-1]
    imputed_snap = [s for s in snapshots if "imputation" in s["stage"]]
    if imputed_snap:
        imp = imputed_snap[0]
        lines.append(f"  After imputation:")
        lines.append(f"    Cell-level missing: {imp['cell_missing_pct']}%")
        lines.append(f"    Feature prevalence: {imp['feature_prevalence_pct']}%")
        lines.append(f"    Complete rows:      {imp['complete_rows_pct']}%")

    lines.append(f"  Final (after all steps):")
    lines.append(f"    Cell-level missing: {final_snap['cell_missing_pct']}%")
    lines.append(f"    Feature prevalence: {final_snap['feature_prevalence_pct']}%")
    lines.append(f"    Complete rows:      {final_snap['complete_rows_pct']}%")
    lines.append(f"  Chapter claims: missing rate = 4.8%, complete cases = 89.7%")

    # VIF details
    lines.append(f"\n── VIF REMOVAL DETAILS ─────────────────────────────────────────────")
    vif_details = all_details.get("phase5", {})
    removed = vif_details.get("removed_features", [])
    if removed:
        lines.append(f"  {'Feature':<35} {'VIF':>10} {'Iteration':>10}")
        lines.append(f"  {'-'*35} {'-'*10} {'-'*10}")
        for r in removed:
            lines.append(f"  {r['feature']:<35} {r['vif']:>10.2f} {r['iteration']:>10}")
        lines.append(f"\n  Total removed: {len(removed)} (Chapter claims: 10)")

    # KL-divergence
    if kl_results:
        lines.append(f"\n── KL-DIVERGENCE (Distribution Preservation) ──────────────────────")
        lines.append(f"  Chapter claims D_KL < ε but never specifies ε or reports values.")
        lines.append(f"\n  {'Feature':<35} {'KL-Divergence':>15} {'N Imputed':>12}")
        lines.append(f"  {'-'*35} {'-'*15} {'-'*12}")
        for col, info in sorted(kl_results.items(), key=lambda x: x[1]["kl_divergence"], reverse=True):
            lines.append(f"  {col:<35} {info['kl_divergence']:>15.6f} {info['n_imputed']:>12,}")
        avg_kl = np.mean([v["kl_divergence"] for v in kl_results.values()])
        max_kl = max(v["kl_divergence"] for v in kl_results.values())
        lines.append(f"\n  Average KL-divergence: {avg_kl:.6f}")
        lines.append(f"  Maximum KL-divergence: {max_kl:.6f}")

    lines.append(f"\n{'='*78}")
    lines.append("  NEXT: Run step3_generate_final_numbers.py for thesis-ready values")
    lines.append(f"{'='*78}\n")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n{'='*60}")
    print(report)
    return report


def main():
    print(f"{'='*70}")
    print(f"  STEP 2: PREPROCESSING PIPELINE VERIFICATION")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    df = load_data()
    numerical, categorical, text = classify_features(df)

    snapshots = []
    all_details = {}

    # Initial snapshot
    snapshots.append(snapshot(df, "0_initial", numerical, categorical, text))
    print(f"\nInitial: {df.shape[0]:,} samples × {df.shape[1]} features "
          f"(Num:{len(numerical)}, Cat:{len(categorical)}, Txt:{len(text)})")

    # Save pre-imputation data for KL-divergence
    df_pre_imputation = df.copy()

    # ── Phase 1: Feature elimination ──
    df, numerical, categorical, text, det = phase1_feature_elimination(
        df, numerical, categorical, text, MISSING_THRESHOLD
    )
    all_details["phase1"] = det
    snapshots.append(snapshot(df, "1_after_elimination", numerical, categorical, text, det))

    # ── Phase 2: Imputation ──
    df, imputation_log = phase2_imputation(df, numerical, categorical, text, SKEWNESS_THRESHOLD)
    all_details["phase2"] = {"imputation_log": imputation_log}
    snapshots.append(snapshot(df, "2_after_imputation", numerical, categorical, text))

    # ── KL-divergence ──
    kl_results = compute_post_imputation_kl(df_pre_imputation, df, numerical)
    all_details["kl_divergence"] = kl_results

    # ── Phase 3: Redundancy elimination ──
    df, numerical, categorical, text, det = phase3_redundancy_elimination(
        df, numerical, categorical, text, CORR_THRESHOLD
    )
    all_details["phase3"] = det
    snapshots.append(snapshot(df, "3_after_redundancy", numerical, categorical, text, det))

    # ── Phase 4: Feature engineering ──
    df, numerical, categorical, text, det = phase4_feature_engineering(
        df, numerical, categorical, text
    )
    all_details["phase4"] = det
    snapshots.append(snapshot(df, "4_after_engineering", numerical, categorical, text, det))

    # ── Phase 5: VIF removal ──
    try:
        df, numerical, categorical, text, det = phase5_vif_removal(
            df, numerical, categorical, text, VIF_THRESHOLD
        )
        all_details["phase5"] = det
    except ImportError:
        print("\n  ⚠️  statsmodels not installed. Install with: pip install statsmodels")
        print("  Skipping VIF analysis.")
        all_details["phase5"] = {"error": "statsmodels not installed"}

    snapshots.append(snapshot(df, "5_after_vif", numerical, categorical, text))

    # ── Save results ──
    results = {
        "timestamp": datetime.now().isoformat(),
        "snapshots": snapshots,
        "details": {k: v for k, v in all_details.items() if k != "kl_divergence"},
        "kl_divergence": kl_results,
        "imputation_log": all_details.get("phase2", {}).get("imputation_log", {}),
    }

    # Serialize (handle non-serializable types)
    json_path = os.path.join(OUTPUT_DIR, "step2_pipeline_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # Generate report
    report_path = os.path.join(OUTPUT_DIR, "step2_report.txt")
    generate_report(snapshots, all_details, kl_results, report_path)


if __name__ == "__main__":
    main()
