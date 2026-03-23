#!/bin/bash
# ================================================================
# Chapter 2 Verification — Background Runner
# ================================================================
# Usage:
#   chmod +x run_verification.sh
#   nohup ./run_verification.sh > ch2_verify.log 2>&1 &
#
# Or simply:
#   bash run_verification.sh
#
# Monitor progress:
#   tail -f ~/TerrorismNER_Project/ch2_verification_results/run.log
# ================================================================

set -e  # Exit on error

PROJECT_DIR="$HOME/TerrorismNER_Project"
SCRIPT_DIR="$PROJECT_DIR/ch2_verification"
OUTPUT_DIR="$PROJECT_DIR/ch2_verification_results"
LOG_FILE="$OUTPUT_DIR/run.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp function
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1" | tee -a "$LOG_FILE"
}

# ── Start ──────────────────────────────────────────────────────
log "================================================================"
log "  CHAPTER 2 VERIFICATION — STARTING"
log "  Project: $PROJECT_DIR"
log "  Scripts: $SCRIPT_DIR"
log "  Output:  $OUTPUT_DIR"
log "================================================================"

# ── Activate environment ───────────────────────────────────────
log ""
log "Activating virtual environment..."

if [ -d "$PROJECT_DIR/terrorner_env" ]; then
    source "$PROJECT_DIR/terrorner_env/bin/activate"
    log "  Activated: terrorner_env"
elif [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    log "  Activated: venv"
else
    log "  WARNING: No virtual environment found. Using system Python."
fi

log "  Python: $(which python3)"
log "  Version: $(python3 --version)"

# ── Check dependencies ─────────────────────────────────────────
log ""
log "Checking dependencies..."

python3 -c "import pandas; print(f'  pandas {pandas.__version__}')" 2>&1 | tee -a "$LOG_FILE" || {
    log "  ERROR: pandas not installed. Installing..."
    pip install pandas
}

python3 -c "import numpy; print(f'  numpy {numpy.__version__}')" 2>&1 | tee -a "$LOG_FILE" || {
    log "  ERROR: numpy not installed. Installing..."
    pip install numpy
}

python3 -c "import scipy; print(f'  scipy {scipy.__version__}')" 2>&1 | tee -a "$LOG_FILE" || {
    log "  WARNING: scipy not installed. Installing..."
    pip install scipy
}

python3 -c "import sklearn; print(f'  sklearn {sklearn.__version__}')" 2>&1 | tee -a "$LOG_FILE" || {
    log "  WARNING: sklearn not installed. Installing..."
    pip install scikit-learn
}

python3 -c "import statsmodels; print(f'  statsmodels {statsmodels.__version__}')" 2>&1 | tee -a "$LOG_FILE" || {
    log "  WARNING: statsmodels not installed. Installing..."
    pip install statsmodels
}

# ── Step 0: Discover dataset ──────────────────────────────────
log ""
log "================================================================"
log "  STEP 0: DATASET DISCOVERY"
log "================================================================"

cd "$SCRIPT_DIR"
python3 step0_discover_dataset.py 2>&1 | tee -a "$LOG_FILE"

# Check if config was created and has a path
CONFIG_FILE="$OUTPUT_DIR/dataset_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    log "ERROR: dataset_config.json not created!"
    exit 1
fi

RAW_PATH=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('raw_gtd_path',''))")
if [ -z "$RAW_PATH" ] || [ "$RAW_PATH" = "None" ]; then
    log ""
    log "================================================================"
    log "  ⚠️  MANUAL ACTION REQUIRED!"
    log "  Dataset could not be auto-detected."
    log "  Please edit: $CONFIG_FILE"
    log "  Set 'raw_gtd_path' to the path of your raw GTD CSV/Excel file."
    log "  Then re-run this script."
    log "================================================================"
    exit 1
fi

log "  Auto-detected dataset: $RAW_PATH"

# ── Step 1: Raw data profiling ────────────────────────────────
log ""
log "================================================================"
log "  STEP 1: RAW DATA PROFILING"
log "================================================================"

python3 step1_raw_data_profile.py 2>&1 | tee -a "$LOG_FILE"

STEP1_OK=$?
if [ $STEP1_OK -ne 0 ]; then
    log "ERROR: Step 1 failed!"
    exit 1
fi

# ── Step 2: Preprocessing pipeline ────────────────────────────
log ""
log "================================================================"
log "  STEP 2: PREPROCESSING PIPELINE VERIFICATION"
log "================================================================"

python3 step2_preprocessing_pipeline.py 2>&1 | tee -a "$LOG_FILE"

STEP2_OK=$?
if [ $STEP2_OK -ne 0 ]; then
    log "ERROR: Step 2 failed!"
    exit 1
fi

# ── Step 3: Generate final numbers ────────────────────────────
log ""
log "================================================================"
log "  STEP 3: FINAL NUMBERS & LATEX FIXES"
log "================================================================"

python3 step3_generate_final_numbers.py 2>&1 | tee -a "$LOG_FILE"

# ── Done ──────────────────────────────────────────────────────
log ""
log "================================================================"
log "  ✅ ALL STEPS COMPLETE"
log "================================================================"
log ""
log "  Results in: $OUTPUT_DIR"
log ""
log "  Key files:"
log "    step1_report.txt          — Raw data profile vs Chapter 2 claims"
log "    step2_report.txt          — Full pipeline tracking"
log "    step3_final_report.txt    — Master claimed vs actual comparison"
log "    step3_latex_fixes.tex     — Specific LaTeX corrections"
log ""
log "  JSON data (for further analysis):"
log "    step1_raw_profile.json"
log "    step2_pipeline_results.json"
log ""

# List output files
ls -la "$OUTPUT_DIR"/ 2>&1 | tee -a "$LOG_FILE"

log ""
log "Done at $(timestamp)"
