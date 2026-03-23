"""
Configuration for Chapter 2: GTD Preprocessing Pipeline
========================================================
Research on Transformer-Based Multi-Task and Cross-Lingual Methods
for Terrorism Event Classification

All thresholds and parameters correspond to values reported in
Chapter 2, Sections 2.3–2.4 of the dissertation.
"""

import os

# ============================================================
# File Paths
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_GTD_FILE = os.path.join(DATA_DIR, "globalterrorismdb_0718dist.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output files
PROCESSED_OUTPUT = os.path.join(OUTPUT_DIR, "gtd_preprocessed.csv")
FEATURE_REPORT = os.path.join(OUTPUT_DIR, "feature_selection_report.csv")
VALIDATION_REPORT = os.path.join(OUTPUT_DIR, "validation_report.csv")

# ============================================================
# Dataset Parameters (Section 2.2)
# ============================================================
GTD_DATE_RANGE = (1970, 2020)       # GTD coverage period
INITIAL_FEATURES = 135               # Raw GTD feature count
FINAL_FEATURES = 80                  # After selection pipeline
TOTAL_RECORDS = 209706               # Total incidents in GTD

# ============================================================
# Missing Value Imputation (Section 2.3.2)
# ============================================================
# Distribution-aware strategy: use skewness to decide mean vs median
SKEWNESS_THRESHOLD = 2.0
#   skewness < 2.0  →  symmetric  →  use MEAN
#   skewness >= 2.0 →  skewed     →  use MEDIAN
# This preserves original distributions unlike blind mean imputation.

# Bayesian nationality imputation (Section 2.3.3)
# Uses MAP (Maximum A Posteriori) estimation based on
# P(nationality | country, region) priors from GTD.
NATIONALITY_MAP_CONFIDENCE = 0.9161  # 91.61% validation accuracy

# ============================================================
# Geocoding Recovery (Section 2.3.4)
# ============================================================
# Reverse geocoding via Google Maps API to recover missing city names
GEOCODING_API = "Google Maps Geocoding API"
GEOCODING_RECOVERY_RATE = 0.8004     # 80.04% of missing cities recovered
GEOCODING_BATCH_SIZE = 100           # API calls per batch
GEOCODING_DELAY_SEC = 0.1            # Delay between API calls

# ============================================================
# Feature Selection (Section 2.3.5)
# ============================================================
# Dual correlation analysis: PCC for numerical, NMI for categorical

# Pearson Correlation Coefficient threshold
PCC_THRESHOLD = 0.80
# Feature pairs with |PCC| > 0.80 are considered redundant;
# remove the feature with lower correlation to the target.

# Normalized Mutual Information threshold
NMI_THRESHOLD = 0.60
# Categorical feature pairs with NMI > 0.60 are considered redundant.
# NMI captures non-linear dependencies that Pearson cannot detect.

# ============================================================
# Validation Thresholds (Section 2.4)
# ============================================================
# KL-Divergence: measures distributional distortion after imputation
KL_DIVERGENCE_THRESHOLD = 0.20      # DKL < 0.20 = minimal distortion
KL_DIVERGENCE_ACHIEVED = 0.115      # Actual mean DKL across all features

# Variance Inflation Factor: checks multicollinearity after selection
VIF_THRESHOLD = 5.0                  # Standard threshold for VIF
VIF_ACHIEVED_MAX = 2.09             # Maximum VIF across all 80 features

# ============================================================
# Classification Tasks (used by downstream chapters)
# ============================================================
CLASSIFICATION_TARGETS = {
    "attack_type": {
        "column": "attacktype1_txt",
        "num_classes": 9,
        "description": "Type of terrorist attack"
    },
    "weapon_type": {
        "column": "weaptype1_txt",
        "num_classes": 12,
        "description": "Weapon used in the attack"
    },
    "target_type": {
        "column": "targtype1_txt",
        "num_classes": 22,
        "description": "Target of the attack"
    },
    "perpetrator_group": {
        "column": "gname",
        "num_classes": None,  # varies by scale (10/20/50/100)
        "description": "Responsible terrorist organization"
    }
}

# ============================================================
# Text Feature Columns
# ============================================================
TEXT_COLUMNS = [
    "summary",          # Incident summary (English narrative)
    "motive",           # Motive description
    "addnotes",         # Additional notes
    "scite1",           # Source citation 1
    "scite2",           # Source citation 2
    "scite3",           # Source citation 3
]

# ============================================================
# Categorical Feature Columns (for NMI analysis)
# ============================================================
CATEGORICAL_COLUMNS = [
    "country_txt", "region_txt", "provstate", "city",
    "attacktype1_txt", "targtype1_txt", "targsubtype1_txt",
    "weaptype1_txt", "weapsubtype1_txt", "gname",
    "natlty1_txt", "dbsource", "claimmode_txt",
]

# ============================================================
# Numerical Feature Columns (for PCC analysis)
# ============================================================
NUMERICAL_COLUMNS = [
    "iyear", "imonth", "iday", "latitude", "longitude",
    "nkill", "nkillus", "nkillter", "nwound", "nwoundus",
    "nwoundte", "property", "propextent", "nhostkid",
    "ndays", "nreleased", "ransom", "ransomamt",
    "success", "suicide", "individual", "extended",
    "multiple", "vicinity",
]

# ============================================================
# Logging
# ============================================================
VERBOSE = True
LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline.log")
