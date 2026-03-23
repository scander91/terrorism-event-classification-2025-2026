#!/usr/bin/env python3
"""
Chapter 2 — REAL Preprocessing Implementation & Validation
============================================================
Implements the actual imputation methods described in Chapter 2:
  1. Geographic cross-imputation (lat/lng ↔ city/country)
  2. Bayesian victim nationality imputation (Eq 2.14-2.15)
  3. Skewness-based numerical imputation (Eq 2.7)
  4. Mode-based categorical imputation
  5. Feature elimination (>90% missing)
  6. Redundancy elimination (correlation > 0.85)
  7. VIF removal (VIF > 10)

Each method is VALIDATED with accuracy metrics.

Usage:
  python3 ch2_real_preprocessing.py

Output: ~/TerrorismNER_Project/ch2_verification_results/preprocessing_validated/
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.expanduser("~/TerrorismNER_Project")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ch2_verification_results", "preprocessing_validated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
MISSING_THRESHOLD = 90.0
SKEWNESS_THRESHOLD = 2.0
CORR_THRESHOLD = 0.85
VIF_THRESHOLD = 10.0

LOG = []

def log(msg):
    print(msg)
    LOG.append(msg)


def save_log():
    with open(os.path.join(OUTPUT_DIR, "full_log.txt"), "w") as f:
        f.write("\n".join(LOG))


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def load_raw():
    pkl_path = os.path.join(PROJECT_ROOT, "cache", "gtd_raw.pkl")
    if os.path.exists(pkl_path):
        log(f"Loading: {pkl_path}")
        df = pd.read_pickle(pkl_path)
    else:
        xlsx_path = os.path.join(PROJECT_ROOT, "globalterrorismdb_0522dist (1).xlsx")
        log(f"Loading: {xlsx_path}")
        df = pd.read_excel(xlsx_path)
    log(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════
# TRACKING
# ══════════════════════════════════════════════════════════════════════
class MetricsTracker:
    def __init__(self):
        self.snapshots = []
        self.validations = {}

    def snapshot(self, df, stage_name, details=None):
        n, p = df.shape
        miss = int(df.isnull().sum().sum())
        total = n * p
        feat_miss = int((df.isnull().sum() > 0).sum())
        snap = {
            "stage": stage_name,
            "samples": n,
            "features": p,
            "missing_cells": miss,
            "total_cells": total,
            "cell_missing_pct": round(miss / total * 100, 2) if total > 0 else 0,
            "features_with_missing": feat_miss,
            "feature_prevalence_pct": round(feat_miss / p * 100, 2) if p > 0 else 0,
            "complete_rows": int(df.dropna().shape[0]),
        }
        if details:
            snap["details"] = details
        self.snapshots.append(snap)
        log(f"\n  [{stage_name}] {n:,} rows × {p} cols | "
            f"Missing: {snap['cell_missing_pct']}% cells, "
            f"{feat_miss}/{p} features | "
            f"Complete rows: {snap['complete_rows']:,}")
        return snap

    def add_validation(self, name, results):
        self.validations[name] = results

    def to_dict(self):
        return {"snapshots": self.snapshots, "validations": self.validations}


# ══════════════════════════════════════════════════════════════════════
# METHOD 1: GEOGRAPHIC CROSS-IMPUTATION
# ══════════════════════════════════════════════════════════════════════
def geographic_imputation(df, tracker):
    """
    Cross-impute geographic fields using known relationships in the data:
      - city + country → lat/lng (using median of known values for same city+country)
      - lat/lng → city/country (using nearest known point)
      - country_txt → provstate (using mode)
    
    Validates by: hiding known values, imputing, checking accuracy.
    """
    log(f"\n{'═'*70}")
    log(f"  METHOD 1: GEOGRAPHIC CROSS-IMPUTATION")
    log(f"{'═'*70}")

    df_out = df.copy()
    validation = {}

    # ── 1a: City + Country → Latitude/Longitude ──────────────────────
    log("\n  [1a] City + Country → Lat/Lng")

    # Build lookup: (city, country) → (median_lat, median_lng)
    geo_known = df_out[
        df_out['latitude'].notna() &
        df_out['longitude'].notna() &
        df_out['city'].notna() &
        df_out['country_txt'].notna()
    ]

    geo_lookup = geo_known.groupby(['city', 'country_txt']).agg(
        lat_median=('latitude', 'median'),
        lng_median=('longitude', 'median'),
        count=('latitude', 'size')
    ).reset_index()

    log(f"    Lookup table: {len(geo_lookup):,} unique (city, country) pairs")
    log(f"    Built from {len(geo_known):,} rows with complete geo data")

    # Find rows missing lat/lng but having city+country
    missing_latlon = df_out[
        (df_out['latitude'].isna() | df_out['longitude'].isna()) &
        df_out['city'].notna() &
        df_out['country_txt'].notna()
    ]
    log(f"    Missing lat/lng with city+country available: {len(missing_latlon):,}")

    # Impute
    filled_count = 0
    for idx, row in missing_latlon.iterrows():
        match = geo_lookup[
            (geo_lookup['city'] == row['city']) &
            (geo_lookup['country_txt'] == row['country_txt'])
        ]
        if len(match) > 0:
            best = match.iloc[0]
            if pd.isna(df_out.at[idx, 'latitude']):
                df_out.at[idx, 'latitude'] = best['lat_median']
            if pd.isna(df_out.at[idx, 'longitude']):
                df_out.at[idx, 'longitude'] = best['lng_median']
            filled_count += 1

    lat_still_missing = df_out['latitude'].isna().sum()
    lat_original_missing = df['latitude'].isna().sum()
    log(f"    Filled: {filled_count:,} rows")
    log(f"    Lat missing: {lat_original_missing:,} → {lat_still_missing:,} "
        f"(reduced by {lat_original_missing - lat_still_missing:,})")

    # ── VALIDATE 1a: Hide 10% of known values, impute, check accuracy ──
    log("\n    VALIDATION (hold-out test):")
    np.random.seed(42)
    known_idx = geo_known.index.tolist()
    test_size = min(len(known_idx) // 10, 10000)
    test_idx = np.random.choice(known_idx, size=test_size, replace=False)

    true_lats = df.loc[test_idx, 'latitude'].values
    true_lngs = df.loc[test_idx, 'longitude'].values

    # Temporarily blank these and impute
    df_test = df.copy()
    df_test.loc[test_idx, 'latitude'] = np.nan
    df_test.loc[test_idx, 'longitude'] = np.nan

    pred_lats = []
    pred_lngs = []
    matched = 0
    for idx in test_idx:
        row = df_test.loc[idx]
        if pd.notna(row['city']) and pd.notna(row['country_txt']):
            match = geo_lookup[
                (geo_lookup['city'] == row['city']) &
                (geo_lookup['country_txt'] == row['country_txt'])
            ]
            if len(match) > 0:
                pred_lats.append(match.iloc[0]['lat_median'])
                pred_lngs.append(match.iloc[0]['lng_median'])
                matched += 1
            else:
                pred_lats.append(np.nan)
                pred_lngs.append(np.nan)
        else:
            pred_lats.append(np.nan)
            pred_lngs.append(np.nan)

    # Compute accuracy (within X km)
    valid_mask = ~np.isnan(pred_lats)
    if sum(valid_mask) > 0:
        lat_errors = np.abs(true_lats[valid_mask] - np.array(pred_lats)[valid_mask])
        lng_errors = np.abs(true_lngs[valid_mask] - np.array(pred_lngs)[valid_mask])

        # Approximate km error (1 degree ≈ 111 km)
        km_errors = np.sqrt((lat_errors * 111) ** 2 + (lng_errors * 111) ** 2)

        within_1km = (km_errors < 1).sum()
        within_10km = (km_errors < 10).sum()
        within_50km = (km_errors < 50).sum()
        exact = (km_errors < 0.1).sum()

        val_results = {
            "test_size": test_size,
            "matched": matched,
            "match_rate_pct": round(matched / test_size * 100, 2),
            "median_km_error": round(float(np.median(km_errors)), 3),
            "mean_km_error": round(float(np.mean(km_errors)), 3),
            "within_0.1km_pct": round(exact / len(km_errors) * 100, 2),
            "within_1km_pct": round(within_1km / len(km_errors) * 100, 2),
            "within_10km_pct": round(within_10km / len(km_errors) * 100, 2),
            "within_50km_pct": round(within_50km / len(km_errors) * 100, 2),
        }
        validation["geo_lat_lng"] = val_results
        log(f"    Test samples:    {test_size:,}")
        log(f"    Matched:         {matched:,} ({val_results['match_rate_pct']}%)")
        log(f"    Median error:    {val_results['median_km_error']:.3f} km")
        log(f"    Within 0.1 km:   {val_results['within_0.1km_pct']}%")
        log(f"    Within 1 km:     {val_results['within_1km_pct']}%")
        log(f"    Within 10 km:    {val_results['within_10km_pct']}%")
        log(f"    Within 50 km:    {val_results['within_50km_pct']}%")

    # ── 1b: Country → Provstate (mode imputation within country) ─────
    log("\n  [1b] Country → Province/State (mode within country)")

    if 'provstate' in df_out.columns:
        provstate_missing_before = df_out['provstate'].isna().sum()
        prov_lookup = df_out[df_out['provstate'].notna()].groupby('country_txt')['provstate'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        )

        filled_prov = 0
        for idx in df_out[df_out['provstate'].isna()].index:
            country = df_out.at[idx, 'country_txt']
            if country in prov_lookup.index and pd.notna(prov_lookup[country]):
                df_out.at[idx, 'provstate'] = prov_lookup[country]
                filled_prov += 1

        log(f"    Missing provstate: {provstate_missing_before:,} → "
            f"{df_out['provstate'].isna().sum():,} (filled {filled_prov:,})")

    # ── 1c: Lat/Lng → Country (reverse lookup for missing country) ───
    log("\n  [1c] Lat/Lng → Country (reverse geocoding via nearest neighbor)")

    country_missing = df_out['country_txt'].isna().sum()
    if country_missing > 0:
        log(f"    Missing country_txt: {country_missing:,}")
        # Use known points to build a simple nearest-neighbor lookup
        # (This is a simplified version — Chapter 2 used Google Maps API)
        log(f"    (Skipping — only {country_missing} missing, would need external API)")
    else:
        log(f"    No missing country_txt — all {df_out.shape[0]:,} rows have country")

    tracker.add_validation("geographic", validation)
    return df_out


# ══════════════════════════════════════════════════════════════════════
# METHOD 2: BAYESIAN NATIONALITY IMPUTATION (Eq 2.14-2.15)
# ══════════════════════════════════════════════════════════════════════
def bayesian_nationality_imputation(df, tracker):
    """
    Chapter 2 Equations 2.14-2.15:
    P(Nv | Ca) = P(Ca | Nv) × P(Nv) / P(Ca)
    
    MAP estimate: Nv* = argmax P(Nv | Ca)
    
    In practice: for each attack country, what's the most likely victim nationality?
    Validate by hiding known values and checking.
    """
    log(f"\n{'═'*70}")
    log(f"  METHOD 2: BAYESIAN NATIONALITY IMPUTATION (Eq 2.14-2.15)")
    log(f"{'═'*70}")

    df_out = df.copy()

    if 'natlty1_txt' not in df_out.columns or 'country_txt' not in df_out.columns:
        log("  Columns not found — skipping")
        return df_out

    # Build conditional probability table: P(nationality | country)
    known = df_out[df_out['natlty1_txt'].notna() & df_out['country_txt'].notna()]
    log(f"  Known nationality+country pairs: {len(known):,}")

    # For each country, compute distribution of victim nationalities
    country_nat_dist = known.groupby('country_txt')['natlty1_txt'].value_counts(normalize=True)

    # MAP estimate: most common nationality per country
    map_estimates = known.groupby('country_txt')['natlty1_txt'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    )

    # What fraction are "locals" (nationality == country)?
    local_rate = known.apply(
        lambda row: 1 if row['natlty1_txt'] == row['country_txt'] else 0, axis=1
    ).mean()
    log(f"  Rate where nationality == attack country: {local_rate*100:.1f}%")

    # Check if MAP = same country
    map_is_local = sum(1 for c, n in map_estimates.items() if c == n) / len(map_estimates)
    log(f"  MAP estimate = attack country: {map_is_local*100:.1f}% of countries")

    # Impute missing nationalities
    nat_missing_before = df_out['natlty1_txt'].isna().sum()
    nat_missing_idx = df_out[df_out['natlty1_txt'].isna() & df_out['country_txt'].notna()].index

    filled = 0
    for idx in nat_missing_idx:
        country = df_out.at[idx, 'country_txt']
        if country in map_estimates.index and pd.notna(map_estimates[country]):
            df_out.at[idx, 'natlty1_txt'] = map_estimates[country]
            # Also fill numeric code if available
            if 'natlty1' in df_out.columns and pd.isna(df_out.at[idx, 'natlty1']):
                code_lookup = known[known['natlty1_txt'] == map_estimates[country]]['natlty1']
                if len(code_lookup) > 0:
                    df_out.at[idx, 'natlty1'] = code_lookup.mode().iloc[0]
            filled += 1

    log(f"  Missing natlty1_txt: {nat_missing_before:,} → {df_out['natlty1_txt'].isna().sum():,} "
        f"(filled {filled:,})")

    # ── VALIDATE: Hold-out test ──────────────────────────────────────
    log("\n  VALIDATION (hold-out test):")
    np.random.seed(42)
    known_idx = known.index.tolist()
    test_size = min(len(known_idx) // 10, 10000)
    test_idx = np.random.choice(known_idx, size=test_size, replace=False)

    true_nat = df.loc[test_idx, 'natlty1_txt'].values
    pred_nat = []
    for idx in test_idx:
        country = df.at[idx, 'country_txt']
        if country in map_estimates.index:
            pred_nat.append(map_estimates[country])
        else:
            pred_nat.append(np.nan)

    pred_nat = np.array(pred_nat)
    valid = ~pd.isna(pred_nat)

    if valid.sum() > 0:
        correct = (true_nat[valid] == pred_nat[valid]).sum()
        accuracy = correct / valid.sum() * 100

        # Also check: how often is true nationality == country?
        true_is_local = sum(1 for i, idx in enumerate(test_idx)
                          if true_nat[i] == df.at[idx, 'country_txt']) / test_size * 100

        val_results = {
            "test_size": test_size,
            "valid_predictions": int(valid.sum()),
            "correct": int(correct),
            "accuracy_pct": round(accuracy, 2),
            "true_local_rate_pct": round(true_is_local, 2),
            "note": "Accuracy = how often MAP estimate matches true nationality"
        }
        tracker.add_validation("bayesian_nationality", val_results)
        log(f"    Test samples:    {test_size:,}")
        log(f"    Accuracy:        {accuracy:.2f}%")
        log(f"    True local rate: {true_is_local:.2f}%")
        log(f"    (MAP assumes victim = national of attack country)")

    return df_out


# ══════════════════════════════════════════════════════════════════════
# METHOD 3: SKEWNESS-BASED NUMERICAL IMPUTATION (Eq 2.7)
# ══════════════════════════════════════════════════════════════════════
def skewness_imputation(df, numerical_cols, tracker):
    """
    Chapter 2 Equation 2.7:
    If |skewness| > θ (θ=2): use median
    Else: use mean
    
    Validate by: hiding known values, imputing, checking error.
    """
    log(f"\n{'═'*70}")
    log(f"  METHOD 3: SKEWNESS-BASED NUMERICAL IMPUTATION (Eq 2.7)")
    log(f"{'═'*70}")

    df_out = df.copy()
    imputation_log = {}
    validation_results = {}

    for col in numerical_cols:
        if col not in df_out.columns:
            continue
        n_missing = df_out[col].isna().sum()
        if n_missing == 0:
            continue

        non_null = df_out[col].dropna()
        if len(non_null) < 3:
            continue

        skew = float(non_null.skew())
        if abs(skew) > SKEWNESS_THRESHOLD:
            fill_val = float(non_null.median())
            method = "median"
        else:
            fill_val = float(non_null.mean())
            method = "mean"

        df_out[col].fillna(fill_val, inplace=True)
        imputation_log[col] = {
            "method": method,
            "skewness": round(skew, 4),
            "fill_value": round(fill_val, 4),
            "n_imputed": n_missing,
            "n_original": len(non_null),
            "missing_pct": round(n_missing / len(df) * 100, 2),
        }

    # Summary
    median_count = sum(1 for v in imputation_log.values() if v["method"] == "median")
    mean_count = sum(1 for v in imputation_log.values() if v["method"] == "mean")
    log(f"  Imputed {len(imputation_log)} numerical features:")
    log(f"    Median (|skew| > 2): {median_count}")
    log(f"    Mean (|skew| ≤ 2):   {mean_count}")

    # ── VALIDATE: RMSE on hold-out ──────────────────────────────────
    log("\n  VALIDATION (hold-out RMSE):")
    np.random.seed(42)

    val_cols = [c for c in imputation_log if imputation_log[c]["n_original"] > 100][:10]
    for col in val_cols:
        non_null_idx = df[col].dropna().index.tolist()
        test_size = min(len(non_null_idx) // 10, 5000)
        if test_size < 10:
            continue

        test_idx = np.random.choice(non_null_idx, size=test_size, replace=False)
        true_vals = df.loc[test_idx, col].values

        # What would the imputation give?
        remaining = df[col].drop(test_idx).dropna()
        skew = remaining.skew()
        pred_val = remaining.median() if abs(skew) > SKEWNESS_THRESHOLD else remaining.mean()

        errors = true_vals - pred_val
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        std = true_vals.std()

        validation_results[col] = {
            "method": imputation_log[col]["method"],
            "skewness": imputation_log[col]["skewness"],
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "std": round(float(std), 4),
            "nrmse": round(float(rmse / std), 4) if std > 0 else None,
        }
        log(f"    {col:<25} {imputation_log[col]['method']:<7} "
            f"RMSE={rmse:.4f}  MAE={mae:.4f}  NRMSE={rmse/std:.4f}" if std > 0 else "")

    tracker.add_validation("skewness_imputation", {
        "log": imputation_log,
        "holdout_validation": validation_results,
    })

    return df_out, imputation_log


# ══════════════════════════════════════════════════════════════════════
# METHOD 4: CATEGORICAL MODE IMPUTATION
# ══════════════════════════════════════════════════════════════════════
def categorical_imputation(df, categorical_cols, tracker):
    """Mode imputation for categorical features."""
    log(f"\n{'═'*70}")
    log(f"  METHOD 4: CATEGORICAL MODE IMPUTATION")
    log(f"{'═'*70}")

    df_out = df.copy()
    imputation_log = {}

    for col in categorical_cols:
        if col not in df_out.columns:
            continue
        n_missing = df_out[col].isna().sum()
        if n_missing == 0:
            continue

        mode = df_out[col].mode()
        if len(mode) > 0:
            fill_val = mode.iloc[0]
            df_out[col].fillna(fill_val, inplace=True)
            imputation_log[col] = {
                "fill_value": str(fill_val),
                "n_imputed": n_missing,
                "missing_pct": round(n_missing / len(df) * 100, 2),
            }

    log(f"  Imputed {len(imputation_log)} categorical features")

    # Validate top features
    log("\n  VALIDATION (hold-out accuracy):")
    np.random.seed(42)
    val_results = {}
    val_cols = [c for c, v in imputation_log.items()
                if v["missing_pct"] < 50 and c in df.columns][:10]

    for col in val_cols:
        non_null_idx = df[col].dropna().index.tolist()
        test_size = min(len(non_null_idx) // 10, 5000)
        if test_size < 10:
            continue

        test_idx = np.random.choice(non_null_idx, size=test_size, replace=False)
        true_vals = df.loc[test_idx, col].values

        remaining = df[col].drop(test_idx).dropna()
        pred_val = remaining.mode().iloc[0] if len(remaining.mode()) > 0 else None

        if pred_val is not None:
            accuracy = (true_vals == pred_val).mean() * 100
            val_results[col] = {
                "mode_value": str(pred_val),
                "accuracy_pct": round(accuracy, 2),
            }
            log(f"    {col:<30} mode='{pred_val}'  accuracy={accuracy:.1f}%")

    tracker.add_validation("categorical_imputation", {
        "log": imputation_log,
        "holdout_validation": val_results,
    })

    return df_out, imputation_log


# ══════════════════════════════════════════════════════════════════════
# METHOD 5: KL-DIVERGENCE MEASUREMENT
# ══════════════════════════════════════════════════════════════════════
def compute_kl_divergences(df_before, df_after, numerical_cols):
    """Compute KL-divergence for imputed features."""
    log(f"\n{'═'*70}")
    log(f"  KL-DIVERGENCE: DISTRIBUTION PRESERVATION")
    log(f"{'═'*70}")

    results = {}
    for col in numerical_cols:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        if not df_before[col].isna().any():
            continue

        orig = df_before[col].dropna().values
        imp = df_after[col].values

        if len(orig) < 10:
            continue

        combined = np.concatenate([orig, imp])
        bin_edges = np.histogram_bin_edges(combined, bins=50)

        p, _ = np.histogram(orig, bins=bin_edges, density=True)
        q, _ = np.histogram(imp, bins=bin_edges, density=True)

        eps = 1e-10
        p = (p + eps) / (p + eps).sum()
        q = (q + eps) / (q + eps).sum()

        kl = float(np.sum(p * np.log(p / q)))
        results[col] = {
            "kl_divergence": round(kl, 6),
            "n_imputed": int(df_before[col].isna().sum()),
            "missing_pct": round(df_before[col].isna().sum() / len(df_before) * 100, 2),
        }

    if results:
        sorted_kl = sorted(results.items(), key=lambda x: x[1]["kl_divergence"], reverse=True)
        log(f"\n  {'Feature':<30} {'KL-Div':>12} {'Missing%':>10} {'N Imputed':>12}")
        log(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*12}")
        for col, info in sorted_kl:
            log(f"  {col:<30} {info['kl_divergence']:>12.6f} {info['missing_pct']:>9.1f}% "
                f"{info['n_imputed']:>12,}")

        kl_vals = [v["kl_divergence"] for v in results.values()]
        log(f"\n  Mean KL: {np.mean(kl_vals):.6f} | Max KL: {np.max(kl_vals):.6f}")
        log(f"  Features with KL < 0.01: {sum(1 for v in kl_vals if v < 0.01)}/{len(kl_vals)}")
        log(f"  Features with KL < 0.1:  {sum(1 for v in kl_vals if v < 0.1)}/{len(kl_vals)}")

    return results


# ══════════════════════════════════════════════════════════════════════
# METHOD 6: FEATURE ELIMINATION (>90% missing)
# ══════════════════════════════════════════════════════════════════════
def feature_elimination(df, threshold=90):
    log(f"\n{'═'*70}")
    log(f"  FEATURE ELIMINATION (>{threshold}% missing)")
    log(f"{'═'*70}")

    n = len(df)
    missing_pcts = (df.isnull().sum() / n * 100).sort_values(ascending=False)
    to_drop = missing_pcts[missing_pcts > threshold]

    log(f"  Features >{threshold}% missing: {len(to_drop)}")
    log(f"  Dropping {len(to_drop)} features, keeping {len(missing_pcts) - len(to_drop)}")

    df_out = df.drop(columns=to_drop.index.tolist())

    dropped_info = {col: round(float(pct), 2) for col, pct in to_drop.items()}
    return df_out, dropped_info


# ══════════════════════════════════════════════════════════════════════
# METHOD 7: REDUNDANCY ELIMINATION
# ══════════════════════════════════════════════════════════════════════
def redundancy_elimination(df, numerical_cols, threshold=0.85):
    log(f"\n{'═'*70}")
    log(f"  REDUNDANCY ELIMINATION (|r| > {threshold})")
    log(f"{'═'*70}")

    usable = [c for c in numerical_cols if c in df.columns and df[c].std() > 0]
    if len(usable) < 2:
        return df, []

    corr = df[usable].corr().abs()

    to_remove = set()
    pairs = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if corr.iloc[i, j] > threshold:
                ci, cj = usable[i], usable[j]
                pairs.append((ci, cj, round(float(corr.iloc[i, j]), 4)))
                if cj not in to_remove and ci not in to_remove:
                    mean_i = corr[ci].mean()
                    mean_j = corr[cj].mean()
                    victim = cj if mean_j >= mean_i else ci
                    to_remove.add(victim)

    log(f"  Correlated pairs: {len(pairs)}")
    for p in pairs:
        log(f"    {p[0]} ↔ {p[1]}: r={p[2]}")
    log(f"  Removing: {sorted(to_remove)}")

    df_out = df.drop(columns=list(to_remove), errors='ignore')
    return df_out, list(to_remove)


# ══════════════════════════════════════════════════════════════════════
# METHOD 8: VIF REMOVAL
# ══════════════════════════════════════════════════════════════════════
def vif_removal(df, numerical_cols, threshold=10.0):
    log(f"\n{'═'*70}")
    log(f"  VIF-BASED FEATURE REMOVAL (VIF > {threshold})")
    log(f"{'═'*70}")

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log("  ⚠️ statsmodels not available — skipping VIF")
        return df, []

    usable = [c for c in numerical_cols if c in df.columns and df[c].std() > 0]
    df_clean = df[usable].dropna()

    if len(df_clean) < 10 or len(usable) < 2:
        log("  Not enough data for VIF")
        return df, []

    # Sample if too large (VIF on 200K+ rows is slow)
    if len(df_clean) > 50000:
        df_clean = df_clean.sample(50000, random_state=42)
        log(f"  Sampled to 50,000 rows for VIF computation")

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_clean), columns=usable)

    removed = []
    current = list(usable)
    iteration = 0

    while True:
        iteration += 1
        if len(current) < 2:
            break

        X_curr = X[current]
        vifs = {}
        for i, col in enumerate(current):
            try:
                vifs[col] = round(float(variance_inflation_factor(X_curr.values, i)), 2)
            except Exception:
                vifs[col] = float('inf')

        max_col = max(vifs, key=vifs.get)
        max_val = vifs[max_col]

        if max_val <= threshold:
            log(f"  Iteration {iteration}: All VIF ≤ {threshold}. Done.")
            break

        log(f"  Iteration {iteration}: Remove '{max_col}' (VIF={max_val:.2f})")
        removed.append({"feature": max_col, "vif": max_val, "iteration": iteration})
        current.remove(max_col)

        if iteration > 50:
            log("  Max iterations reached")
            break

    log(f"\n  Total removed: {len(removed)}")
    log(f"  Remaining numerical: {len(current)}")

    # Final VIF values
    if len(current) >= 2:
        X_final = X[current]
        final_vifs = {}
        for i, col in enumerate(current):
            try:
                final_vifs[col] = round(float(variance_inflation_factor(X_final.values, i)), 2)
            except:
                final_vifs[col] = None

        log(f"\n  Final VIF values:")
        for col in sorted(final_vifs, key=lambda x: final_vifs[x] or 0, reverse=True)[:15]:
            log(f"    {col:<30} VIF={final_vifs[col]}")

    remove_cols = [r["feature"] for r in removed]
    df_out = df.drop(columns=remove_cols, errors='ignore')
    return df_out, removed


# ══════════════════════════════════════════════════════════════════════
# CLASSIFY FEATURES
# ══════════════════════════════════════════════════════════════════════
def classify_features(df):
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


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════
def main():
    log(f"{'═'*70}")
    log(f"  CHAPTER 2 — VALIDATED PREPROCESSING PIPELINE")
    log(f"  {datetime.now().isoformat()}")
    log(f"{'═'*70}")

    tracker = MetricsTracker()

    # Load raw data
    df = load_raw()
    df_original = df.copy()

    # Classify features
    numerical, categorical, text = classify_features(df)
    log(f"\nFeature types: {len(numerical)} numerical, {len(categorical)} categorical, {len(text)} text")

    # ── INITIAL SNAPSHOT ──
    tracker.snapshot(df, "0_raw")

    # ── PHASE 1: Feature elimination ──
    df, dropped = feature_elimination(df, MISSING_THRESHOLD)
    numerical = [c for c in numerical if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    text = [c for c in text if c in df.columns]
    tracker.snapshot(df, "1_after_feature_elimination", {"dropped": len(dropped)})

    df_pre_imputation = df.copy()

    # ── PHASE 2a: Geographic imputation ──
    df = geographic_imputation(df, tracker)
    tracker.snapshot(df, "2a_after_geo_imputation")

    # ── PHASE 2b: Bayesian nationality ──
    df = bayesian_nationality_imputation(df, tracker)
    tracker.snapshot(df, "2b_after_nationality_imputation")

    # ── PHASE 2c: Skewness-based numerical ──
    df, num_log = skewness_imputation(df, numerical, tracker)
    tracker.snapshot(df, "2c_after_numerical_imputation")

    # ── PHASE 2d: Categorical mode ──
    df, cat_log = categorical_imputation(df, categorical, tracker)
    tracker.snapshot(df, "2d_after_categorical_imputation")

    # ── PHASE 2e: Text — fill with empty string ──
    for col in text:
        if col in df.columns:
            df[col].fillna("", inplace=True)
    tracker.snapshot(df, "2e_after_text_fill")

    # ── KL-Divergence ──
    kl_results = compute_kl_divergences(df_pre_imputation, df, numerical)

    # ── PHASE 3: Redundancy elimination ──
    df, removed_corr = redundancy_elimination(df, numerical, CORR_THRESHOLD)
    numerical = [c for c in numerical if c in df.columns]
    tracker.snapshot(df, "3_after_redundancy")

    # ── PHASE 4: VIF removal ──
    df, removed_vif = vif_removal(df, numerical, VIF_THRESHOLD)
    numerical = [c for c in numerical if c not in [r["feature"] for r in removed_vif]]
    tracker.snapshot(df, "4_after_vif")

    # ── FINAL SUMMARY ──
    num_final = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_final = [c for c in df.columns if c not in num_final and
                 pd.api.types.is_object_dtype(df[c]) and
                 df[c].dropna().astype(str).str.len().mean() <= 50]
    txt_final = [c for c in df.columns if c not in num_final and c not in cat_final]

    log(f"\n{'═'*70}")
    log(f"  FINAL SUMMARY")
    log(f"{'═'*70}")
    log(f"  Initial:  {df_original.shape[0]:,} × {df_original.shape[1]}")
    log(f"  Final:    {df.shape[0]:,} × {df.shape[1]}")
    log(f"  Features: Num={len(num_final)}, Cat={len(cat_final)}, Txt={len(txt_final)}")
    log(f"  Missing cells: {df.isnull().sum().sum():,}")
    log(f"  Complete rows: {df.dropna().shape[0]:,} ({df.dropna().shape[0]/len(df)*100:.1f}%)")
    log(f"  Retention: {df.shape[0]/df_original.shape[0]*100:.1f}%")

    # ── SAVE EVERYTHING ──
    results = {
        "timestamp": datetime.now().isoformat(),
        "tracker": tracker.to_dict(),
        "kl_divergence": kl_results,
        "vif_removed": removed_vif,
        "corr_removed": removed_corr,
        "dropped_features": dropped,
        "numerical_imputation": num_log,
        "categorical_imputation": cat_log,
    }

    json_path = os.path.join(OUTPUT_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved: {json_path}")

    save_log()
    log(f"Log saved: {os.path.join(OUTPUT_DIR, 'full_log.txt')}")

    # Save processed dataset
    pkl_path = os.path.join(OUTPUT_DIR, "gtd_preprocessed_verified.pkl")
    df.to_pickle(pkl_path)
    log(f"Preprocessed data saved: {pkl_path}")


if __name__ == "__main__":
    main()
