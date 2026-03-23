#!/usr/bin/env python3
"""
=============================================================================
Script 05: Build GTD Evaluation Dataset (Direct from NLLB translations)
=============================================================================
Builds stratified train/test splits for 3 GTD classification tasks directly
from Script 02 output. Skips refinement (03) and validation (04) steps.

Input:  outputs/02_nllb_translations.jsonl
Output: outputs/gtd_eval_attack.json    (9 classes)
        outputs/gtd_eval_weapon.json    (12 classes)
        outputs/gtd_eval_target.json    (22 classes)
        outputs/gtd_eval_stats.json     (summary statistics)

Each output JSON contains:
  - test:  80 stratified samples (for evaluation)
  - train: remaining samples (ICE selection pool)
  - label_map: class name -> integer mapping
  - task_info: metadata
=============================================================================
"""

import json
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────

INPUT_PATH = "outputs/02_nllb_translations.jsonl"
OUTPUT_DIR = "outputs"
TEST_SIZE = 80          # test samples per task
MIN_CLASS_SAMPLES = 2   # minimum samples to include a class

# Task definitions: field name, task label, number of expected classes
TASKS = {
    "attack": {
        "field": "attack_type",
        "name": "GTD-Attack",
        "expected_classes": 9,
    },
    "weapon": {
        "field": "weapon_type",
        "name": "GTD-Weapon",
        "expected_classes": 12,
    },
    "target": {
        "field": "target_type",
        "name": "GTD-Target",
        "expected_classes": 22,
    },
}


def load_records(path):
    """Load all translated records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Use refined if available, else NLLB translation
            text_ar = rec.get("refined_ar", rec.get("translation_ar", ""))
            if not text_ar or len(text_ar.strip()) < 10:
                continue
            records.append(rec)
    return records


def build_task_dataset(records, task_key, task_config, test_size=80):
    """Build stratified train/test split for one task."""
    field = task_config["field"]
    task_name = task_config["name"]

    # Group records by label
    by_label = defaultdict(list)
    skipped = 0
    for rec in records:
        label = rec.get(field, "").strip()
        if not label:
            skipped += 1
            continue
        by_label[label].append(rec)

    # Filter classes with too few samples
    valid_labels = {k: v for k, v in by_label.items() if len(v) >= MIN_CLASS_SAMPLES}
    removed_labels = {k: len(v) for k, v in by_label.items() if len(v) < MIN_CLASS_SAMPLES}

    if removed_labels:
        log.warning(f"  {task_name}: Removed {len(removed_labels)} classes with <{MIN_CLASS_SAMPLES} samples: {removed_labels}")

    n_classes = len(valid_labels)
    total_records = sum(len(v) for v in valid_labels.values())
    log.info(f"  {task_name}: {n_classes} classes, {total_records:,} total records")

    # Stratified sampling for test set
    # Proportional allocation: each class gets test samples proportional to its size
    # but at least 1 sample per class, and total = test_size
    class_sizes = {k: len(v) for k, v in valid_labels.items()}
    total = sum(class_sizes.values())

    # First pass: proportional allocation (floor)
    test_alloc = {}
    for label, size in class_sizes.items():
        proportion = size / total
        alloc = max(1, int(proportion * test_size))
        # Don't take more than half of any class
        alloc = min(alloc, size // 2)
        alloc = max(1, alloc)  # at least 1
        test_alloc[label] = alloc

    # Adjust to hit exact test_size
    current_total = sum(test_alloc.values())
    if current_total < test_size:
        # Add more from largest classes
        sorted_labels = sorted(class_sizes.keys(), key=lambda k: class_sizes[k], reverse=True)
        i = 0
        while current_total < test_size:
            label = sorted_labels[i % len(sorted_labels)]
            if test_alloc[label] < class_sizes[label] // 2:
                test_alloc[label] += 1
                current_total += 1
            i += 1
            if i > test_size * 10:  # safety
                break
    elif current_total > test_size:
        # Remove from smallest classes (but keep at least 1)
        sorted_labels = sorted(class_sizes.keys(), key=lambda k: class_sizes[k])
        i = 0
        while current_total > test_size:
            label = sorted_labels[i % len(sorted_labels)]
            if test_alloc[label] > 1:
                test_alloc[label] -= 1
                current_total -= 1
            i += 1
            if i > test_size * 10:
                break

    log.info(f"  {task_name}: Test allocation = {dict(sorted(test_alloc.items(), key=lambda x: -x[1]))}")

    # Sample test and train sets
    test_records = []
    train_records = []
    label_map = {label: idx for idx, label in enumerate(sorted(valid_labels.keys()))}

    for label, recs in valid_labels.items():
        random.shuffle(recs)
        n_test = test_alloc.get(label, 1)
        test_recs = recs[:n_test]
        train_recs = recs[n_test:]

        for r in test_recs:
            test_records.append({
                "id": r["id"],
                "text_ar": r.get("refined_ar", r.get("translation_ar", "")),
                "text_en": r.get("summary_en", ""),
                "label": label,
                "label_id": label_map[label],
            })
        for r in train_recs:
            train_records.append({
                "id": r["id"],
                "text_ar": r.get("refined_ar", r.get("translation_ar", "")),
                "text_en": r.get("summary_en", ""),
                "label": label,
                "label_id": label_map[label],
            })

    random.shuffle(test_records)
    random.shuffle(train_records)

    # Verify
    test_dist = Counter(r["label"] for r in test_records)
    train_dist = Counter(r["label"] for r in train_records)

    return {
        "task": task_name,
        "task_key": task_key,
        "n_classes": n_classes,
        "label_map": label_map,
        "test": test_records,
        "train": train_records,
        "test_distribution": dict(test_dist),
        "train_distribution": dict(train_dist),
        "task_info": {
            "test_size": len(test_records),
            "train_size": len(train_records),
            "n_classes": n_classes,
            "labels": sorted(valid_labels.keys()),
        }
    }


def main():
    if not Path(INPUT_PATH).exists():
        log.error(f"Input not found: {INPUT_PATH}")
        return

    log.info("Loading translated records...")
    records = load_records(INPUT_PATH)
    log.info(f"Loaded {len(records):,} valid records")

    all_stats = {}

    for task_key, task_config in TASKS.items():
        log.info(f"\nBuilding {task_config['name']}...")
        dataset = build_task_dataset(records, task_key, task_config, TEST_SIZE)

        # Save
        out_path = f"{OUTPUT_DIR}/gtd_eval_{task_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        log.info(f"  Saved: {out_path}")
        log.info(f"  Test: {len(dataset['test'])} | Train: {len(dataset['train'])} | Classes: {dataset['n_classes']}")

        all_stats[task_key] = {
            "name": task_config["name"],
            "n_classes": dataset["n_classes"],
            "test_size": len(dataset["test"]),
            "train_size": len(dataset["train"]),
            "test_distribution": dataset["test_distribution"],
        }

    # Save summary stats
    stats_path = f"{OUTPUT_DIR}/gtd_eval_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    log.info(f"\n{'='*60}")
    log.info("GTD EVALUATION DATASET READY")
    log.info(f"{'='*60}")
    for k, s in all_stats.items():
        log.info(f"  {s['name']}: {s['n_classes']} classes | test={s['test_size']} | train={s['train_size']:,}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
