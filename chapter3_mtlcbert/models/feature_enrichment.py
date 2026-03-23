#!/usr/bin/env python3
"""
Feature Enrichment Module
==========================
Converts structured GTD attributes into natural language text
using template functions. Attributes are selected based on
Cramér's V association with attack type (threshold > 0.25).

Templates:
  φ(region)     → "The attack occurred in {region}."
  φ(weaptype1)  → "The weapon used was {weaptype1}."
  φ(targtype1)  → "The target was {targtype1}."
  φ(gname)      → "The attack was carried out by {gname}."

The enriched representation concatenates the original text summary
with all template-generated sentences, separated by [SEP].
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


TEMPLATES = {
    "region_txt": "The attack occurred in {value}.",
    "weaptype1_txt": "The weapon used was {value}.",
    "targtype1_txt": "The target was {value}.",
    "gname": "The attack was carried out by {value}.",
    "country_txt": "The attack took place in {value}.",
    "provstate": "The province or state was {value}.",
    "city": "The attack happened in {value}.",
    "natlty1_txt": "The nationality of the target was {value}.",
}


def cramers_v(x, y):
    """Compute Cramér's V between two categorical variables."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, c = confusion.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def select_enrichment_attributes(df, target_col="attacktype1_txt", threshold=0.25):
    """Select attributes with Cramér's V > threshold."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    categorical_cols = [c for c in categorical_cols if c != target_col]

    selected = []
    for col in categorical_cols:
        mask = df[col].notna() & df[target_col].notna()
        if mask.sum() < 100:
            continue
        try:
            v = cramers_v(df.loc[mask, col], df.loc[mask, target_col])
            if v > threshold:
                selected.append((col, v))
        except Exception:
            continue

    selected.sort(key=lambda x: x[1], reverse=True)
    return selected


def enrich_text(row, attributes, summary_col="summary"):
    """
    Build enriched text representation for a single incident.

    T_i^enriched = T_i ⊕ [SEP] ⊕ φ(region) ⊕ φ(weaptype1) ⊕ φ(targtype1) ⊕ φ(gname)
    """
    parts = []

    summary = row.get(summary_col, "")
    if pd.notna(summary) and str(summary).strip():
        parts.append(str(summary).strip())

    for attr in attributes:
        value = row.get(attr, "")
        if pd.notna(value) and str(value).strip() and str(value).lower() != "unknown":
            template = TEMPLATES.get(attr, "The {attr} is {value}.")
            text = template.format(value=str(value).strip(), attr=attr)
            parts.append(text)

    return " [SEP] ".join(parts) if parts else ""


def enrich_dataset(df, attributes=None, summary_col="summary", threshold=0.25):
    """
    Apply feature enrichment to entire dataset.

    Args:
        df: DataFrame with GTD incidents
        attributes: List of column names to use (auto-selected if None)
        summary_col: Name of the text summary column
        threshold: Cramér's V threshold for auto-selection

    Returns:
        Series of enriched text strings
    """
    if attributes is None:
        selected = select_enrichment_attributes(df, threshold=threshold)
        attributes = [col for col, _ in selected]

    enriched = df.apply(lambda row: enrich_text(row, attributes, summary_col), axis=1)
    return enriched
