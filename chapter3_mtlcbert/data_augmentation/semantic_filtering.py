#!/usr/bin/env python3
"""
Semantic Similarity Filtering
================================
Filters augmented paraphrases by computing cosine similarity
between original and generated text using Sentence-BERT.

Threshold: cosine similarity ≥ 0.85

Paraphrases below the threshold are discarded as they may
have drifted too far from the original meaning.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def compute_similarity(original_texts, generated_texts,
                       model_name="all-MiniLM-L6-v2", batch_size=64):
    """
    Compute pairwise cosine similarity between original and generated texts.

    Args:
        original_texts: List of original incident descriptions
        generated_texts: List of generated paraphrases
        model_name: Sentence-BERT model for embedding
        batch_size: Encoding batch size

    Returns:
        similarities: Array of cosine similarity scores
    """
    model = SentenceTransformer(model_name)

    orig_embs = model.encode(original_texts, batch_size=batch_size,
                             show_progress_bar=True, normalize_embeddings=True)
    gen_embs = model.encode(generated_texts, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)

    similarities = np.sum(orig_embs * gen_embs, axis=1)
    return similarities


def filter_augmented_data(original_df, augmented_df,
                          text_col="enriched_text",
                          threshold=0.85,
                          model_name="all-MiniLM-L6-v2"):
    """
    Filter augmented samples by semantic similarity to originals.

    Args:
        original_df: DataFrame with original samples
        augmented_df: DataFrame with augmented samples (must have
                      same index mapping or a source_idx column)
        text_col: Column containing text
        threshold: Minimum cosine similarity to keep
        model_name: Sentence-BERT model

    Returns:
        filtered_df: Augmented samples passing the threshold
        stats: Filtering statistics
    """
    if augmented_df.empty:
        return augmented_df, {"total": 0, "kept": 0, "removed": 0}

    gen_texts = augmented_df[text_col].tolist()

    if "source_idx" in augmented_df.columns:
        orig_texts = [original_df.loc[idx, text_col]
                      for idx in augmented_df["source_idx"]]
    else:
        orig_sample = original_df.sample(n=len(gen_texts), replace=True,
                                         random_state=42)
        orig_texts = orig_sample[text_col].tolist()

    similarities = compute_similarity(orig_texts, gen_texts, model_name)
    mask = similarities >= threshold

    filtered_df = augmented_df[mask].copy()
    filtered_df["similarity_score"] = similarities[mask]

    stats = {
        "total": len(augmented_df),
        "kept": int(mask.sum()),
        "removed": int((~mask).sum()),
        "mean_similarity": float(similarities.mean()),
        "min_similarity_kept": float(similarities[mask].min()) if mask.any() else 0,
        "threshold": threshold,
    }

    print(f"  Filtering: {stats['total']} → {stats['kept']} "
          f"(removed {stats['removed']}, mean sim={stats['mean_similarity']:.3f})")

    return filtered_df, stats
