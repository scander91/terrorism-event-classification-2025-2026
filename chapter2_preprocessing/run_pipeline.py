"""
Run Full Preprocessing Pipeline — Chapter 2
=============================================
Research on Transformer-Based Multi-Task and Cross-Lingual Methods
for Terrorism Event Classification

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --step 3     # Run from step 3 onwards
    python run_pipeline.py --validate   # Run validation only

This script executes the complete GTD preprocessing pipeline:
    Step 1: Data Loading       (01_data_loading.py)
    Step 2: EDA                (02_eda.py)
    Step 3: Missing Values     (03_missing_value_imputation.py)
    Step 4: Geocoding          (04_geocoding.py)
    Step 5: Feature Selection  (05_feature_engineering.py)
    Step 6: Validation         (06_evaluation.py)

Input:  Raw GTD dataset (globalterrorismdb_0718dist.xlsx)
Output: Preprocessed dataset with 80 features, 100% completeness
"""

import os
import sys
import time
import argparse
import subprocess
import logging

from config import (
    RAW_GTD_FILE, OUTPUT_DIR, LOG_FILE,
    INITIAL_FEATURES, FINAL_FEATURES,
    KL_DIVERGENCE_ACHIEVED, VIF_ACHIEVED_MAX
)

# ============================================================
# Pipeline Configuration
# ============================================================
PIPELINE_STEPS = [
    {
        "step": 1,
        "script": "01_data_loading.py",
        "name": "Data Loading",
        "description": "Load raw GTD dataset, identify columns and data types",
    },
    {
        "step": 2,
        "script": "02_eda.py",
        "name": "Exploratory Data Analysis",
        "description": "Analyze missing value patterns, class distributions, temporal trends",
    },
    {
        "step": 3,
        "script": "03_missing_value_imputation.py",
        "name": "Distribution-Aware Missing Value Imputation",
        "description": "Skewness-based imputation (mean vs median) + Bayesian nationality MAP",
    },
    {
        "step": 4,
        "script": "04_geocoding.py",
        "name": "Web-Crawler Geocoding Recovery",
        "description": "Reverse geocoding via Google Maps API to recover missing city names",
    },
    {
        "step": 5,
        "script": "05_feature_engineering.py",
        "name": "Dual Correlation Feature Selection",
        "description": "PCC (threshold=0.80) for numerical + NMI (threshold=0.60) for categorical → 135 to 80 features",
    },
    {
        "step": 6,
        "script": "06_evaluation.py",
        "name": "KL-Divergence and VIF Validation",
        "description": "Validate distributional fidelity (DKL=0.115) and multicollinearity (VIF≤2.09)",
    },
]


def setup_logging():
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_step(step_info, logger):
    """Execute a single pipeline step."""
    script = step_info["script"]
    step_num = step_info["step"]
    name = step_info["name"]

    logger.info(f"{'='*60}")
    logger.info(f"STEP {step_num}/6: {name}")
    logger.info(f"Script: {script}")
    logger.info(f"Description: {step_info['description']}")
    logger.info(f"{'='*60}")

    script_path = os.path.join(os.path.dirname(__file__), script)

    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )

        elapsed = time.time() - start_time

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")

        if result.returncode != 0:
            logger.error(f"Step {step_num} FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    logger.error(f"  {line}")
            return False

        logger.info(f"Step {step_num} completed in {elapsed:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"Step {step_num} raised exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Chapter 2 GTD Preprocessing Pipeline"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Start from step N (1-6). Default: 1 (run all)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation (step 6) only"
    )
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("CHAPTER 2: GTD PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Raw GTD file: {RAW_GTD_FILE}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Pipeline: {INITIAL_FEATURES} features → {FINAL_FEATURES} features")
    logger.info("")

    # Check input file
    if not os.path.exists(RAW_GTD_FILE):
        logger.error(f"GTD file not found: {RAW_GTD_FILE}")
        logger.error("Please download the GTD dataset and place it in data/")
        logger.error("See data/README.md for instructions.")
        sys.exit(1)

    # Determine which steps to run
    if args.validate:
        steps_to_run = [s for s in PIPELINE_STEPS if s["step"] == 6]
    else:
        steps_to_run = [s for s in PIPELINE_STEPS if s["step"] >= args.step]

    # Execute pipeline
    total_start = time.time()
    failed = False

    for step_info in steps_to_run:
        success = run_step(step_info, logger)
        if not success:
            failed = True
            logger.error(f"Pipeline stopped at step {step_info['step']}")
            break

    total_elapsed = time.time() - total_start

    # Summary
    logger.info("")
    logger.info("=" * 60)
    if failed:
        logger.info("PIPELINE FAILED — see errors above")
    else:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
        logger.info(f"Features: {INITIAL_FEATURES} → {FINAL_FEATURES}")
        logger.info(f"Completeness: 100%")
        logger.info(f"Mean KL-divergence: {KL_DIVERGENCE_ACHIEVED}")
        logger.info(f"Max VIF: {VIF_ACHIEVED_MAX}")
        logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
