# Chapter 2: Advanced Data Preprocessing and Feature Engineering for the GTD

## Overview

This chapter implements a systematic preprocessing pipeline for the Global Terrorism Database (GTD) that transforms 209,706 raw incidents with 135 features into an analysis-ready dataset of 80 features with 100% data completeness.

## Pipeline Steps

```
Raw GTD (135 features, 56% missing) 
    → Step 1: Data Loading & Cleaning
    → Step 2: Exploratory Data Analysis  
    → Step 3: Distribution-Aware Missing Value Imputation
    → Step 4: Web-Crawler Geocoding (80.04% city name recovery)
    → Step 5: Feature Engineering (PCC, NMI correlation analysis)
    → Step 6: Validation (KL-divergence, VIF)
    → Clean Dataset (80 features, 0% missing)
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`
- `scikit-learn >= 1.2.0`
- `scipy >= 1.9.0`
- `matplotlib >= 3.6.0`
- `seaborn >= 0.12.0`
- `geopy >= 2.3.0` (for geocoding)
- `requests >= 2.28.0`

## Data Access

1. Request GTD data from: https://www.start.umd.edu/gtd/contact/
2. Place the downloaded CSV file in `data/` directory
3. Rename it to `globalterrorismdb.csv`

## Usage

### Run the full pipeline:
```bash
python run_pipeline.py
```

### Run individual steps:
```bash
python 01_data_loading.py          # Load and initial cleaning
python 02_eda.py                   # Exploratory analysis + figures
python 03_missing_value_imputation.py  # Imputation (mean/median/geocoding)
python 04_geocoding.py             # Web-crawler city name recovery
python 05_feature_engineering.py   # PCC/NMI analysis + feature selection
python 06_evaluation.py            # KL-divergence + VIF validation
```

## Configuration

Edit `config.py` to set paths and parameters:

```python
# File paths
RAW_DATA_PATH = "data/globalterrorismdb.csv"
OUTPUT_PATH = "data/gtd_preprocessed.csv"

# Imputation parameters
PCC_THRESHOLD = 0.80        # Pearson correlation for numerical features
NMI_THRESHOLD = 0.60        # Normalized MI for categorical features
COSINE_SIM_THRESHOLD = 0.85 # For semantic similarity filtering

# Geocoding
GOOGLE_MAPS_API_KEY = ""    # Required for geocoding step
```

## Output

The pipeline produces:
- `data/gtd_preprocessed.csv` — Clean dataset (80 features, 209,706 rows)
- `results/eda_figures/` — EDA visualizations (Figures 2.3–2.9)
- `results/correlation_matrices/` — PCC and NMI heatmaps (Figures 2.10–2.11)
- `results/geocoding_results/` — Geographic imputation maps (Figures 2.12–2.14)
- `results/validation_report.txt` — KL-divergence and VIF statistics

## Key Results

| Metric | Value |
|--------|-------|
| Features reduced | 135 → 80 |
| Data completeness | 100% |
| City name recovery | 80.04% |
| Nationality imputation accuracy | 91.61% |
| Mean KL-divergence | 0.115 |
| Max VIF | 2.09 |
