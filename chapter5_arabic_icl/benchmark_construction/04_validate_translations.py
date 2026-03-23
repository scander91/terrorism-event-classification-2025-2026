#!/usr/bin/env python3
"""
=============================================================================
Script 04: Translation Validation
=============================================================================
Multi-metric validation of Arabic translations:
  1. Back-translation BLEU (Ar→En via NLLB, compare with original)
  2. Pseudo-perplexity (AraBERT masked LM coherence)
  3. Duplicate detection (TF-IDF cosine similarity)
  4. Length ratio check (Ar/En character ratio)
  5. Final glossary term rate

Input:  outputs/03_refined_translations.jsonl
Output: outputs/04_validated_translations.jsonl  (accepted)
        outputs/04_rejected.jsonl                (filtered out)
        outputs/04_validation_stats.json

GPU: RTX 2080 Ti — NLLB for back-translation + AraBERT for perplexity
Estimated time: ~3-4 hours for 143K records
=============================================================================
"""

import json
import os
import sys
import logging
import time
import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer as BertTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config():
    try:
        import yaml
        with open("configs/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Metric 1: Back-Translation BLEU
# ===========================================================================

def compute_bleu_simple(reference, hypothesis):
    """Compute simple sentence-level BLEU (1-4 gram) without nltk dependency."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # N-gram precisions
    precisions = []
    for n in range(1, 5):
        ref_ngrams = Counter()
        hyp_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams[tuple(ref_tokens[i:i+n])] += 1
        for i in range(len(hyp_tokens) - n + 1):
            hyp_ngrams[tuple(hyp_tokens[i:i+n])] += 1

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total if total > 0 else 0)

    # Geometric mean with smoothing
    log_avg = 0
    for p in precisions:
        log_avg += math.log(max(p, 1e-10)) / 4
    bleu = bp * math.exp(log_avg)

    return round(bleu, 4)


def back_translate_batch(texts_ar, model, tokenizer, device="cuda"):
    """Back-translate Arabic → English using NLLB."""
    tokenizer.src_lang = "arb_Arab"
    inputs = tokenizer(texts_ar, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=512,
            num_beams=4,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ===========================================================================
# Metric 2: Pseudo-Perplexity (AraBERT)
# ===========================================================================

def compute_pseudo_perplexity(text, model, tokenizer, device="cuda", max_length=512):
    """Compute pseudo-perplexity using masked language model.
    
    For each token, mask it and compute NLL. Average gives pseudo-PPL.
    Approximation: sample 20% of tokens for efficiency.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].squeeze()

    if len(input_ids) < 3:
        return 999.0  # Too short to evaluate

    # Sample 20% of positions (skip [CLS] and [SEP])
    positions = list(range(1, len(input_ids) - 1))
    sample_size = max(1, len(positions) // 5)
    sampled = sorted(np.random.choice(positions, size=sample_size, replace=False))

    total_nll = 0
    count = 0

    for pos in sampled:
        masked_input = input_ids.clone().unsqueeze(0)
        original_token = masked_input[0, pos].item()
        masked_input[0, pos] = tokenizer.mask_token_id

        if device == "cuda":
            masked_input = masked_input.to(device)

        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits[0, pos]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            nll = -log_probs[original_token].item()

        total_nll += nll
        count += 1

    avg_nll = total_nll / max(count, 1)
    ppl = math.exp(min(avg_nll, 20))  # Cap to avoid overflow
    return round(ppl, 2)


# ===========================================================================
# Metric 3: Duplicate Detection
# ===========================================================================

def detect_duplicates(texts, threshold=0.92):
    """Detect near-duplicate translations using TF-IDF cosine similarity.
    
    Returns set of indices that are duplicates (keeping first occurrence).
    Uses sklearn if available, falls back to simple Jaccard.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Process in chunks to manage memory
        chunk_size = 10000
        duplicate_indices = set()

        for start in range(0, len(texts), chunk_size):
            end = min(start + chunk_size, len(texts))
            chunk = texts[start:end]

            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
            tfidf = vectorizer.fit_transform(chunk)
            sim_matrix = cosine_similarity(tfidf)

            for i in range(len(chunk)):
                for j in range(i + 1, len(chunk)):
                    if sim_matrix[i, j] >= threshold:
                        duplicate_indices.add(start + j)  # Mark later one as dup

        return duplicate_indices

    except ImportError:
        log.warning("sklearn not available. Using simple Jaccard for dedup.")
        duplicate_indices = set()
        # Simple character trigram Jaccard (much slower, only for fallback)
        def trigrams(text):
            return set(text[i:i+3] for i in range(len(text)-2))

        seen = []
        for i, text in enumerate(texts):
            tg = trigrams(text)
            for j, prev_tg in seen:
                if len(tg & prev_tg) / max(len(tg | prev_tg), 1) >= threshold:
                    duplicate_indices.add(i)
                    break
            seen.append((i, tg))
            if len(seen) > 1000:
                seen = seen[-500:]  # Sliding window

        return duplicate_indices


# ===========================================================================
# Metric 4: Length Ratio
# ===========================================================================

def check_length_ratio(en_text, ar_text, min_ratio=0.3, max_ratio=3.0):
    """Check if Arabic/English character length ratio is reasonable."""
    en_len = max(len(en_text), 1)
    ar_len = len(ar_text)
    ratio = ar_len / en_len
    return min_ratio <= ratio <= max_ratio, round(ratio, 3)


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main():
    config = load_config()

    # Thresholds
    min_bt_bleu = 0.25
    min_term_rate = 0.80
    max_perplexity = 150.0
    max_dup_sim = 0.92
    min_len_ratio = 0.3
    max_len_ratio = 3.0

    if config:
        v = config.get("validation", {})
        min_bt_bleu = v.get("min_back_translation_bleu", min_bt_bleu)
        min_term_rate = v.get("min_term_success_rate", min_term_rate)
        max_perplexity = v.get("max_perplexity", max_perplexity)
        max_dup_sim = v.get("max_duplicate_similarity", max_dup_sim)
        lr = v.get("length_ratio_range", [min_len_ratio, max_len_ratio])
        min_len_ratio, max_len_ratio = lr[0], lr[1]

    output_dir = config["paths"]["output_dir"] if config else "outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load input
    input_path = f"{output_dir}/03_refined_translations.jsonl"
    if not Path(input_path).exists():
        log.error(f"Input not found: {input_path}. Run script 03 first.")
        sys.exit(1)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    log.info(f"Loaded {len(records):,} records for validation")

    # --- Load models ---
    # NLLB for back-translation
    nllb_name = "facebook/nllb-200-distilled-1.3B"
    if config:
        nllb_name = config.get("validation", {}).get("backtranslation_model", nllb_name)

    log.info(f"Loading NLLB for back-translation: {nllb_name}")
    nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        nllb_name, torch_dtype=torch.float16
    )
    if device == "cuda":
        nllb_model = nllb_model.to(device)
    nllb_model.eval()

    # AraBERT for perplexity
    bert_name = "aubmindlab/bert-base-arabertv02"
    if config:
        bert_name = config.get("validation", {}).get("perplexity_model", bert_name)

    log.info(f"Loading AraBERT for perplexity: {bert_name}")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_name)
    if device == "cuda":
        bert_model = bert_model.to(device)
    bert_model.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024**3)
        log.info(f"VRAM usage (both models): {used:.1f} GB")

    # --- Validation Loop ---
    start_time = time.time()
    stats = Counter()
    rejection_reasons = Counter()

    validated = []
    rejected = []

    batch_size = 16  # For back-translation batching

    # Step 1: Back-translation BLEU (batched)
    log.info("Step 1/4: Back-translation BLEU...")
    bt_bleus = []
    ar_texts = [r.get("refined_ar", r.get("translation_ar", "")) for r in records]

    for i in range(0, len(ar_texts), batch_size):
        batch_ar = ar_texts[i:i + batch_size]
        batch_en_orig = [records[j]["summary_en"] for j in range(i, min(i + batch_size, len(records)))]

        try:
            back_translations = back_translate_batch(batch_ar, nllb_model, nllb_tokenizer, device)
            for en_orig, bt_en in zip(batch_en_orig, back_translations):
                bleu = compute_bleu_simple(en_orig, bt_en)
                bt_bleus.append(bleu)
        except Exception as e:
            log.warning(f"Back-translation error at batch {i}: {e}")
            bt_bleus.extend([0.0] * len(batch_ar))

        if (i // batch_size) % 200 == 0:
            log.info(f"  BT progress: {i:,}/{len(records):,}")

    # Free NLLB VRAM
    del nllb_model
    torch.cuda.empty_cache()

    # Step 2: Pseudo-perplexity (sampled)
    log.info("Step 2/4: Pseudo-perplexity (AraBERT)...")
    perplexities = []
    for i, rec in enumerate(records):
        ar_text = rec.get("refined_ar", rec.get("translation_ar", ""))
        try:
            ppl = compute_pseudo_perplexity(ar_text, bert_model, bert_tokenizer, device)
        except:
            ppl = 999.0
        perplexities.append(ppl)

        if (i + 1) % 5000 == 0:
            log.info(f"  PPL progress: {i+1:,}/{len(records):,}")

    # Free AraBERT VRAM
    del bert_model
    torch.cuda.empty_cache()

    # Step 3: Duplicate detection
    log.info("Step 3/4: Duplicate detection...")
    duplicate_indices = detect_duplicates(ar_texts, threshold=max_dup_sim)
    log.info(f"  Found {len(duplicate_indices):,} duplicates")

    # Step 4: Compile results and filter
    log.info("Step 4/4: Applying quality filters...")
    for i, rec in enumerate(records):
        ar_text = rec.get("refined_ar", rec.get("translation_ar", ""))
        term_rate = rec.get("final_term_rate", rec.get("glossary_term_rate", 1.0))
        bt_bleu = bt_bleus[i] if i < len(bt_bleus) else 0.0
        ppl = perplexities[i] if i < len(perplexities) else 999.0
        is_dup = i in duplicate_indices
        len_ok, len_ratio = check_length_ratio(rec["summary_en"], ar_text, min_len_ratio, max_len_ratio)

        # Add metrics to record
        rec["bt_bleu"] = bt_bleu
        rec["perplexity"] = ppl
        rec["is_duplicate"] = is_dup
        rec["length_ratio"] = len_ratio

        # Apply filters
        reasons = []
        if bt_bleu < min_bt_bleu:
            reasons.append("low_bt_bleu")
        if term_rate < min_term_rate and rec.get("glossary_terms_found", 0) > 0:
            reasons.append("low_term_rate")
        if ppl > max_perplexity:
            reasons.append("high_perplexity")
        if is_dup:
            reasons.append("duplicate")
        if not len_ok:
            reasons.append("bad_length_ratio")
        if len(ar_text.strip()) < 10:
            reasons.append("too_short")

        if reasons:
            rec["rejection_reasons"] = reasons
            rejected.append(rec)
            for r in reasons:
                rejection_reasons[r] += 1
            stats["rejected"] += 1
        else:
            validated.append(rec)
            stats["accepted"] += 1

    # Save outputs
    out_valid = f"{output_dir}/04_validated_translations.jsonl"
    out_reject = f"{output_dir}/04_rejected.jsonl"

    for filepath, data in [(out_valid, validated), (out_reject, rejected)]:
        with open(filepath, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time

    # Compute metric summaries
    valid_bleus = [r["bt_bleu"] for r in validated]
    valid_ppls = [r["perplexity"] for r in validated if r["perplexity"] < 500]
    valid_terms = [r.get("final_term_rate", 1.0) for r in validated]

    log.info(f"\n{'='*60}")
    log.info(f"VALIDATION RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  Total records         : {len(records):,}")
    log.info(f"  Accepted              : {stats['accepted']:,} ({100*stats['accepted']/len(records):.1f}%)")
    log.info(f"  Rejected              : {stats['rejected']:,} ({100*stats['rejected']/len(records):.1f}%)")
    log.info(f"  Duplicates removed    : {len(duplicate_indices):,}")
    log.info(f"")
    log.info(f"  Rejection breakdown:")
    for reason, count in rejection_reasons.most_common():
        log.info(f"    {reason}: {count:,}")
    log.info(f"")
    log.info(f"  Accepted metrics (mean):")
    log.info(f"    BT-BLEU      : {np.mean(valid_bleus):.4f}")
    log.info(f"    Perplexity   : {np.mean(valid_ppls):.1f}")
    log.info(f"    Term rate    : {np.mean(valid_terms):.4f}")
    log.info(f"  Time: {elapsed/3600:.1f} hours")
    log.info(f"{'='*60}")

    # Save stats
    with open(f"{output_dir}/04_validation_stats.json", "w") as f:
        json.dump({
            "total": len(records),
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "rejection_breakdown": dict(rejection_reasons),
            "metrics_accepted": {
                "mean_bt_bleu": round(np.mean(valid_bleus), 4) if valid_bleus else 0,
                "mean_perplexity": round(np.mean(valid_ppls), 2) if valid_ppls else 0,
                "mean_term_rate": round(np.mean(valid_terms), 4) if valid_terms else 0,
            },
            "elapsed_hours": round(elapsed / 3600, 2),
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n✓ Script 04 complete. {stats['accepted']:,} validated translations ready.")
    log.info(f"  Next: Run script 05 for synthetic data generation.")


if __name__ == "__main__":
    main()
