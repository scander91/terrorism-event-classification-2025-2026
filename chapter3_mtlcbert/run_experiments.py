#!/usr/bin/env python3
"""
Run All Chapter 3 Experiments
================================
Reproduces all MTL-CBERT experimental results across 5 random seeds
with and without data augmentation.

Usage:
    python3 run_experiments.py
    python3 run_experiments.py --seeds 0 1 2
    python3 run_experiments.py --augmented
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from config import *


def run_command(cmd, desc=""):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all MTL-CBERT experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--augmented", action="store_true",
                        help="Run with augmented data")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only evaluate")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Starting experiments at {start_time.isoformat()}")
    print(f"Seeds: {args.seeds}")
    print(f"Augmented: {args.augmented}")

    results = {}

    if not args.skip_training:
        # Train across all seeds
        for seed in args.seeds:
            cmd = [sys.executable, "train.py", "--seed", str(seed)]
            if args.augmented:
                cmd.append("--augmented")

            desc = f"Training seed={seed}" + (" (augmented)" if args.augmented else "")
            success = run_command(cmd, desc)
            results[f"train_seed{seed}"] = success

            if not success:
                print(f"  WARNING: Training failed for seed {seed}")

    # Aggregate evaluation
    print("\n" + "=" * 60)
    print("  Aggregating Results")
    print("=" * 60)

    success = run_command(
        [sys.executable, "evaluate.py", "--results_dir", str(OUTPUT_DIR)],
        "Evaluation aggregation",
    )
    results["evaluation"] = success

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  All Experiments Complete")
    print(f"  Elapsed: {elapsed}")
    print(f"{'='*60}")

    for task, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {task}")

    # Save run log
    log = {
        "timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "seeds": args.seeds,
        "augmented": args.augmented,
        "results": results,
    }
    log_path = OUTPUT_DIR / "experiment_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
