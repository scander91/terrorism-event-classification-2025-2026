#!/usr/bin/env python3
"""
Step 0: Dataset Discovery
=========================
Finds the GTD dataset file(s) in the TerrorismNER_Project directory
and reports basic info so subsequent scripts know the correct path.

Run: python step0_discover_dataset.py
Output: dataset_config.json (used by all subsequent scripts)
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/TerrorismNER_Project")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ch2_verification_results")
CONFIG_FILE = os.path.join(OUTPUT_DIR, "dataset_config.json")

# Common GTD filename patterns
GTD_PATTERNS = [
    "**/globalterrorism*.csv", "**/GTD*.csv", "**/gtd*.csv",
    "**/globalterrorism*.xlsx", "**/GTD*.xlsx", "**/gtd*.xlsx",
    "**/globalterrorism*.xls", "**/GTD*.xls",
    "**/terrorism*.csv", "**/terror*.csv",
    "**/*208688*", "**/*208_688*",  # Sample count in filename
    "**/data/*.csv", "**/dataset/*.csv", "**/datasets/*.csv",
    "**/raw/*.csv", "**/raw_data/*.csv",
]

def find_datasets():
    """Search for potential GTD dataset files."""
    print(f"{'='*70}")
    print(f"  STEP 0: DATASET DISCOVERY")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    if not os.path.isdir(PROJECT_ROOT):
        print(f"ERROR: Project directory not found: {PROJECT_ROOT}")
        print("Please update PROJECT_ROOT at the top of this script.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all CSV and Excel files
    print("[1] Scanning for all data files...\n")
    all_data_files = []
    for ext in ["*.csv", "*.xlsx", "*.xls", "*.tsv", "*.parquet", "*.pkl", "*.h5"]:
        for f in Path(PROJECT_ROOT).rglob(ext):
            # Skip venv, __pycache__, .git, node_modules
            path_str = str(f)
            if any(skip in path_str for skip in [
                "terrorner_env", "__pycache__", ".git", "node_modules",
                "ch2_verification_results"
            ]):
                continue
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
                all_data_files.append({
                    "path": str(f),
                    "name": f.name,
                    "size_mb": round(size_mb, 2),
                    "extension": f.suffix,
                })
            except OSError:
                pass

    # Sort by size (largest first - GTD is likely the biggest)
    all_data_files.sort(key=lambda x: x["size_mb"], reverse=True)

    print(f"Found {len(all_data_files)} data files:\n")
    print(f"{'File':<60} {'Size (MB)':>10} {'Extension':>10}")
    print("-" * 82)
    for f in all_data_files[:30]:  # Show top 30
        name = f["path"].replace(PROJECT_ROOT + "/", "")
        if len(name) > 58:
            name = "..." + name[-55:]
        print(f"{name:<60} {f['size_mb']:>10.2f} {f['extension']:>10}")

    if len(all_data_files) > 30:
        print(f"  ... and {len(all_data_files) - 30} more files")

    # Try to identify the raw GTD file
    print(f"\n\n[2] Identifying potential RAW GTD dataset...\n")
    candidates = []
    for f in all_data_files:
        score = 0
        name_lower = f["name"].lower()
        path_lower = f["path"].lower()

        # Name-based scoring
        if "globalterrorism" in name_lower or "gtd" in name_lower:
            score += 10
        if "raw" in path_lower:
            score += 5
        if "original" in path_lower or "original" in name_lower:
            score += 3
        if "processed" in name_lower or "clean" in name_lower:
            score -= 5
        if "final" in name_lower:
            score -= 3

        # Size-based scoring (GTD with 208K rows should be large, ~100-300MB)
        if 50 < f["size_mb"] < 500:
            score += 8
        elif 10 < f["size_mb"] < 50:
            score += 4
        elif f["size_mb"] > 500:
            score += 2

        # Extension scoring
        if f["extension"] in [".csv", ".xlsx"]:
            score += 2

        if score > 0:
            f["score"] = score
            candidates.append(f)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if candidates:
        print("Candidate GTD files (ranked by likelihood):\n")
        print(f"{'Score':>5} {'File':<55} {'Size (MB)':>10}")
        print("-" * 72)
        for c in candidates[:10]:
            name = c["path"].replace(PROJECT_ROOT + "/", "")
            if len(name) > 53:
                name = "..." + name[-50:]
            print(f"{c['score']:>5} {name:<55} {c['size_mb']:>10.2f}")
    else:
        print("WARNING: No obvious GTD candidate files found!")

    # Try to quick-probe the top candidate
    print(f"\n\n[3] Quick-probing top candidates...\n")
    probed = []
    for c in candidates[:5]:
        try:
            import pandas as pd
            path = c["path"]
            if path.endswith(".csv"):
                # Just read first few rows and column count
                df_head = pd.read_csv(path, nrows=5, encoding="latin-1", low_memory=False)
                nrows_approx = sum(1 for _ in open(path, encoding="latin-1")) - 1
            elif path.endswith((".xlsx", ".xls")):
                df_head = pd.read_excel(path, nrows=5)
                nrows_approx = "unknown (Excel)"
            else:
                continue

            ncols = len(df_head.columns)
            info = {
                "path": path,
                "columns": ncols,
                "approx_rows": nrows_approx,
                "size_mb": c["size_mb"],
                "score": c["score"],
                "column_names_sample": list(df_head.columns[:20]),
            }
            probed.append(info)

            print(f"  File: {path}")
            print(f"  Columns: {ncols}, Approx rows: {nrows_approx}")
            print(f"  Sample columns: {list(df_head.columns[:10])}")

            # Check for GTD-specific columns
            gtd_markers = ["eventid", "iyear", "imonth", "iday", "country_txt",
                          "region_txt", "attacktype1_txt", "targtype1_txt",
                          "gname", "nkill", "nwound", "summary"]
            found_markers = [m for m in gtd_markers if m in
                           [col.lower().strip() for col in df_head.columns]]
            if found_markers:
                print(f"  ✅ GTD markers found: {found_markers}")
                info["is_gtd"] = True
                info["gtd_markers"] = found_markers
            else:
                print(f"  ❌ No GTD column markers found")
                info["is_gtd"] = False

            print()
        except Exception as e:
            print(f"  Could not probe {c['path']}: {e}\n")

    # Also look for any preprocessed/final dataset
    print(f"\n[4] Looking for preprocessed/final dataset...\n")
    processed_candidates = []
    for f in all_data_files:
        name_lower = f["name"].lower()
        path_lower = f["path"].lower()
        if any(kw in name_lower or kw in path_lower for kw in
               ["processed", "clean", "final", "preprocessed", "imputed"]):
            processed_candidates.append(f)
            print(f"  Found: {f['path']} ({f['size_mb']:.2f} MB)")

    # Also check for pickle/parquet versions
    for ext in ["*.pkl", "*.pickle", "*.parquet", "*.h5", "*.hdf5"]:
        for f in Path(PROJECT_ROOT).rglob(ext):
            path_str = str(f)
            if "terrorner_env" not in path_str and "ch2_verification" not in path_str:
                size = f.stat().st_size / (1024 * 1024)
                print(f"  Binary data file: {path_str} ({size:.2f} MB)")

    # Find existing preprocessing scripts
    print(f"\n\n[5] Looking for existing preprocessing code...\n")
    preprocessing_scripts = []
    for ext in ["*.py", "*.ipynb"]:
        for f in Path(PROJECT_ROOT).rglob(ext):
            path_str = str(f)
            if "terrorner_env" in path_str or "__pycache__" in path_str:
                continue
            name_lower = f.name.lower()
            if any(kw in name_lower for kw in
                   ["preprocess", "preprocessing", "clean", "impute", "eda",
                    "feature_eng", "data_prep", "prepare", "chapter2", "ch2"]):
                preprocessing_scripts.append(str(f))
                print(f"  Found: {path_str}")

    if not preprocessing_scripts:
        # Show all Python files for reference
        print("  No preprocessing-specific scripts found. All Python files:")
        for f in Path(PROJECT_ROOT).rglob("*.py"):
            path_str = str(f)
            if "terrorner_env" not in path_str and "__pycache__" not in path_str:
                print(f"    {path_str}")

    # Save configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "project_root": PROJECT_ROOT,
        "output_dir": OUTPUT_DIR,
        "all_data_files": all_data_files[:20],
        "gtd_candidates": probed,
        "processed_candidates": [{"path": f["path"], "size_mb": f["size_mb"]}
                                  for f in processed_candidates],
        "preprocessing_scripts": preprocessing_scripts,
        "raw_gtd_path": None,  # Will be set after user confirms
        "processed_gtd_path": None,
    }

    # Auto-select if we found a clear GTD file
    gtd_files = [p for p in probed if p.get("is_gtd")]
    if gtd_files:
        # Pick the one with most rows (likely the raw file)
        best = max(gtd_files, key=lambda x: x.get("approx_rows", 0)
                   if isinstance(x.get("approx_rows"), int) else 0)
        config["raw_gtd_path"] = best["path"]
        print(f"\n{'='*70}")
        print(f"  AUTO-DETECTED RAW GTD: {best['path']}")
        print(f"  Columns: {best['columns']}, Rows: ~{best['approx_rows']}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("  ⚠️  Could not auto-detect GTD file.")
        print("  Please edit dataset_config.json and set 'raw_gtd_path' manually.")
        print(f"{'='*70}")

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nConfig saved to: {CONFIG_FILE}")
    print(f"\n{'='*70}")
    print("NEXT STEP: Review the config file, then run:")
    print("  python step1_raw_data_profile.py")
    print(f"{'='*70}")

    return config


if __name__ == "__main__":
    find_datasets()
