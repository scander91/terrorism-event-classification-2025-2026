#!/usr/bin/env python3
"""
=============================================================================
Script 01: GTD Preprocessing
=============================================================================
Loads the cached GTD pkl (209,706 records, 135 columns), extracts records
with text summaries for translation and records without for synthesis.

Input:  checkpoints/gtd_full.pkl (confirmed on apl12)
Output: outputs/01_gtd_translatable.jsonl     (~143K records with summaries)
        outputs/01_gtd_synthesis_only.jsonl    (~66K records without)
        outputs/01_gtd_stats.json              (statistics)
=============================================================================
"""

import pandas as pd
import json
import os
import sys
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config():
    try:
        import yaml
        with open("configs/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        log.warning("Config not found, using defaults.")
        return None


def load_gtd(config):
    """Load GTD from cached pkl (fast) or fall back to xlsx."""
    pkl_path = None
    xlsx_path = None

    if config:
        pkl_path = config["paths"].get("gtd_pkl")
        xlsx_path = config["paths"].get("gtd_xlsx")

    # Try pkl first (instant load vs. ~60s for xlsx)
    if pkl_path and Path(pkl_path).exists():
        log.info(f"Loading GTD from pkl: {pkl_path}")
        df = pd.read_pickle(pkl_path)
    elif xlsx_path and Path(xlsx_path).exists():
        log.info(f"Loading GTD from xlsx: {xlsx_path}")
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    else:
        log.error("GTD file not found. Check configs/config.yaml paths.")
        sys.exit(1)

    log.info(f"GTD loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def extract_records(df, min_summary_len=20):
    """Extract structured records from GTD dataframe.
    
    Confirmed columns from apl12 inspection:
    - summary: single column, 143586 non-null, avg 296 chars
    - attacktype1_txt: 9 unique (text labels already present)
    - weaptype1_txt: 12 unique
    - targtype1_txt: 22 unique
    - All label columns have 209706 non-null (no missing)
    """
    records = []
    stats = Counter()

    # Verify expected columns exist
    required = ["eventid", "summary", "attacktype1_txt", "weaptype1_txt", 
                "targtype1_txt", "attacktype1", "weaptype1", "targtype1",
                "iyear", "country_txt", "region_txt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Missing expected columns: {missing}")
        log.info(f"Available columns: {list(df.columns)[:30]}")
        sys.exit(1)

    for idx, row in df.iterrows():
        stats["total"] += 1

        # --- Summary text ---
        summary = ""
        raw = row.get("summary", "")
        if pd.notna(raw) and isinstance(raw, str) and len(raw.strip()) >= min_summary_len:
            summary = raw.strip()

        has_summary = bool(summary)
        if has_summary:
            stats["with_summary"] += 1
        else:
            stats["no_summary"] += 1

        # --- Labels (text) ---
        attack_txt = str(row["attacktype1_txt"]).strip() if pd.notna(row["attacktype1_txt"]) else "Unknown"
        weapon_txt = str(row["weaptype1_txt"]).strip() if pd.notna(row["weaptype1_txt"]) else "Unknown"
        target_txt = str(row["targtype1_txt"]).strip() if pd.notna(row["targtype1_txt"]) else "Unknown"

        # --- Label IDs (numeric) ---
        attack_id = int(row["attacktype1"]) if pd.notna(row["attacktype1"]) else None
        weapon_id = int(row["weaptype1"]) if pd.notna(row["weaptype1"]) else None
        target_id = int(row["targtype1"]) if pd.notna(row["targtype1"]) else None

        # --- Metadata ---
        year = int(row["iyear"]) if pd.notna(row["iyear"]) else None
        country = str(row["country_txt"]).strip() if pd.notna(row["country_txt"]) else ""
        region = str(row["region_txt"]).strip() if pd.notna(row["region_txt"]) else ""

        record = {
            "id": str(int(row["eventid"])) if pd.notna(row["eventid"]) else str(idx),
            "summary_en": summary,
            "has_summary": has_summary,
            "summary_len": len(summary),
            "attack_type": attack_txt,
            "weapon_type": weapon_txt,
            "target_type": target_txt,
            "attack_type_id": attack_id,
            "weapon_type_id": weapon_id,
            "target_type_id": target_id,
            "year": year,
            "country": country,
            "region": region,
            "source": "gtd",
        }

        records.append(record)
        stats["valid"] += 1

    return records, stats


def compute_distributions(records, filter_has_summary=True):
    """Compute label distributions."""
    subset = [r for r in records if r["has_summary"]] if filter_has_summary else records
    dist = {}
    for field in ["attack_type", "weapon_type", "target_type"]:
        counts = Counter(r[field] for r in subset)
        dist[field] = dict(counts.most_common())
    return dist


def compute_summary_stats(records):
    """Compute summary text statistics."""
    with_summary = [r for r in records if r["has_summary"]]
    lengths = [r["summary_len"] for r in with_summary]
    if not lengths:
        return {}
    return {
        "count": len(lengths),
        "mean_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "median_chars": sorted(lengths)[len(lengths) // 2],
    }


def main():
    config = load_config()
    min_len = 20
    if config:
        min_len = config.get("preprocessing", {}).get("min_summary_length", 20)

    output_dir = "outputs"
    if config:
        output_dir = config.get("paths", {}).get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load GTD
    df = load_gtd(config)

    # Extract records
    log.info("Extracting records...")
    records, stats = extract_records(df, min_summary_len=min_len)

    # Split
    translatable = [r for r in records if r["has_summary"]]
    synthesis_only = [r for r in records if not r["has_summary"]]

    # Statistics
    dist_translatable = compute_distributions(records, filter_has_summary=True)
    dist_all = compute_distributions(records, filter_has_summary=False)
    summary_stats = compute_summary_stats(records)

    # Log results
    log.info(f"\n{'='*60}")
    log.info(f"GTD PREPROCESSING RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  Total GTD records     : {stats['total']:,}")
    log.info(f"  Valid records         : {stats['valid']:,}")
    log.info(f"  With summary (→ translate) : {stats['with_summary']:,} ({100*stats['with_summary']/stats['total']:.1f}%)")
    log.info(f"  Without summary (→ synth)  : {stats['no_summary']:,} ({100*stats['no_summary']/stats['total']:.1f}%)")
    log.info(f"  Avg summary length    : {summary_stats.get('mean_chars', 0):.0f} chars")
    log.info(f"{'='*60}")

    log.info("\nLabel distributions (translatable records):")
    for field, counts in dist_translatable.items():
        log.info(f"\n  {field} ({len(counts)} classes):")
        for label, count in counts.items():
            pct = 100 * count / len(translatable)
            log.info(f"    {label}: {count:,} ({pct:.1f}%)")

    # Save outputs
    for filepath, data in [
        (f"{output_dir}/01_gtd_translatable.jsonl", translatable),
        (f"{output_dir}/01_gtd_synthesis_only.jsonl", synthesis_only),
    ]:
        with open(filepath, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"\nSaved: {filepath} ({len(data):,} records)")

    # Save statistics
    stats_out = {
        "extraction_stats": dict(stats),
        "summary_text_stats": summary_stats,
        "label_distributions_translatable": dist_translatable,
        "label_distributions_all": dist_all,
        "total_translatable": len(translatable),
        "total_synthesis_only": len(synthesis_only),
        "summary_rate": round(stats["with_summary"] / max(stats["total"], 1), 4),
    }
    stats_path = f"{output_dir}/01_gtd_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    log.info(f"Saved: {stats_path}")

    log.info(f"\n✓ Script 01 complete. {len(translatable):,} records ready for NLLB translation.")


if __name__ == "__main__":
    main()
