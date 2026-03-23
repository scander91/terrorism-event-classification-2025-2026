#!/usr/bin/env python3
"""
=============================================================================
Script 02: NLLB-200 Base Translation (En → Ar)
=============================================================================
Translates GTD English summaries to Arabic using NLLB-200-distilled-1.3B.
Tracks glossary term presence and flags records needing refinement.

Input:  outputs/01_gtd_translatable.jsonl
Output: outputs/02_nllb_translations.jsonl
        outputs/02_translation_stats.json

GPU: RTX 2080 Ti (11GB) — NLLB-1.3B uses ~3GB in fp16
Estimated time: ~4-6 hours for 143K records (batch_size=8)
=============================================================================
"""

import json
import os
import sys
import logging
import time
import re
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config():
    try:
        import yaml
        with open("configs/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        return None


def load_glossary(glossary_path="glossary/terrorism_glossary.json"):
    """Load bilingual glossary for term tracking."""
    if not Path(glossary_path).exists():
        log.warning(f"Glossary not found: {glossary_path}. Skipping term tracking.")
        return {}

    with open(glossary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build lookup: English term → list of acceptable Arabic terms
    en_to_ar = {}
    for category, terms in data["terms"].items():
        for entry in terms:
            if "en" in entry and "ar" in entry:
                en_lower = entry["en"].lower()
                ar_terms = [entry["ar"]]
                ar_terms.extend(entry.get("variants", []))
                if "coded_ar" in entry:
                    ar_terms.append(entry["coded_ar"])
                if "dialectal" in entry:
                    ar_terms.append(entry["dialectal"])
                if "msa" in entry:
                    ar_terms.append(entry["msa"])
                en_to_ar[en_lower] = list(set(ar_terms))

    log.info(f"Loaded glossary: {len(en_to_ar)} English terms → Arabic mappings")
    return en_to_ar


def check_glossary_terms(source_en, translation_ar, glossary):
    """Check how many glossary terms from source appear in translation."""
    if not glossary:
        return 1.0, [], []

    source_lower = source_en.lower()
    found_en = []
    matched_ar = []
    missing_ar = []

    for en_term, ar_options in glossary.items():
        if en_term in source_lower:
            found_en.append(en_term)
            # Check if ANY acceptable Arabic rendering appears
            found = False
            for ar_term in ar_options:
                if ar_term in translation_ar:
                    matched_ar.append(ar_term)
                    found = True
                    break
            if not found:
                missing_ar.append(en_term)

    if not found_en:
        return 1.0, [], []  # No glossary terms in source → perfect score

    term_rate = len(matched_ar) / len(found_en)
    return term_rate, found_en, missing_ar


def select_model(config):
    """Select appropriate NLLB model based on available VRAM."""
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f} GB")
    else:
        vram_gb = 0
        log.warning("No GPU detected. Using CPU (will be very slow).")

    model_name = "/home/macierz/mohabdal/TerrorismNER_Project/models/nllb-1.3b"  # Default: ~3GB fp16
    if config:
        if vram_gb >= 14:
            model_name = config.get("translation", {}).get("model_large", model_name)
            log.info(f"Sufficient VRAM for large model: {model_name}")
        else:
            model_name = config.get("translation", {}).get("model", model_name)
    
    return model_name


def load_model(model_name, device="cuda", dtype="float16"):
    """Load NLLB model and tokenizer."""
    log.info(f"Loading model: {model_name}")
    log.info(f"Device: {device}, dtype: {dtype}")

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)

    model.eval()

    # Log VRAM usage
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info(f"VRAM after model load: {used:.1f}/{total:.1f} GB")

    return model, tokenizer


def translate_batch(texts, model, tokenizer, src_lang, tgt_lang, 
                    max_length=512, num_beams=5, device="cuda"):
    """Translate a batch of English texts to Arabic."""
    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return translations


def main():
    config = load_config()

    # Parameters
    src_lang = "eng_Latn"
    tgt_lang = "arb_Arab"
    batch_size = 8
    max_length = 512
    num_beams = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config:
        t = config.get("translation", {})
        src_lang = t.get("src_lang", src_lang)
        tgt_lang = t.get("tgt_lang", tgt_lang)
        batch_size = t.get("batch_size", batch_size)
        max_length = t.get("max_length", max_length)
        num_beams = t.get("num_beams", num_beams)

    output_dir = config["paths"]["output_dir"] if config else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load input
    input_path = f"{output_dir}/01_gtd_translatable.jsonl"
    if not Path(input_path).exists():
        log.error(f"Input not found: {input_path}. Run script 01 first.")
        sys.exit(1)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    log.info(f"Loaded {len(records):,} records for translation")

    # Load glossary
    glossary_path = config["paths"]["glossary"] if config else "glossary/terrorism_glossary.json"
    glossary = load_glossary(glossary_path)

    # Select and load model
    model_name = select_model(config)
    model, tokenizer = load_model(model_name, device=device)

    # Resume support: check for partial output
    output_path = f"{output_dir}/02_nllb_translations.jsonl"
    completed_ids = set()
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["id"])
                except:
                    pass
        log.info(f"Resuming: {len(completed_ids):,} already translated")

    remaining = [r for r in records if r["id"] not in completed_ids]
    log.info(f"Remaining to translate: {len(remaining):,}")

    if not remaining:
        log.info("All records already translated. Skipping.")
        return

    # Translate in batches
    stats = Counter()
    term_rates = []
    start_time = time.time()

    outfile = open(output_path, "a", encoding="utf-8")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        texts = [r["summary_en"] for r in batch]

        try:
            translations = translate_batch(
                texts, model, tokenizer, src_lang, tgt_lang,
                max_length=max_length, num_beams=num_beams, device=device
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning(f"OOM at batch {i}. Reducing batch size and retrying...")
                torch.cuda.empty_cache()
                # Retry one by one
                translations = []
                for t in texts:
                    try:
                        tr = translate_batch(
                            [t], model, tokenizer, src_lang, tgt_lang,
                            max_length=max_length, num_beams=num_beams, device=device
                        )
                        translations.append(tr[0])
                    except:
                        translations.append("")
                        stats["oom_failures"] += 1
            else:
                raise

        for rec, trans in zip(batch, translations):
            # Check glossary terms
            term_rate, found_terms, missing_terms = check_glossary_terms(
                rec["summary_en"], trans, glossary
            )
            term_rates.append(term_rate)

            needs_refinement = term_rate < 0.8 and len(found_terms) > 0

            output_rec = {
                **rec,
                "translation_ar": trans,
                "translation_model": model_name,
                "glossary_term_rate": round(term_rate, 4),
                "glossary_terms_found": len(found_terms),
                "glossary_terms_missing": missing_terms,
                "needs_refinement": needs_refinement,
            }

            outfile.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
            stats["translated"] += 1
            if needs_refinement:
                stats["needs_refinement"] += 1

        # Progress
        elapsed = time.time() - start_time
        done = i + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else 0
        if (i // batch_size) % 100 == 0:
            log.info(
                f"Progress: {done:,}/{len(remaining):,} "
                f"({100*done/len(remaining):.1f}%) | "
                f"Rate: {rate:.1f} rec/s | "
                f"ETA: {eta/3600:.1f}h | "
                f"Need refinement: {stats['needs_refinement']:,}"
            )

    outfile.close()

    # Final stats
    elapsed_total = time.time() - start_time
    avg_term_rate = sum(term_rates) / len(term_rates) if term_rates else 0

    log.info(f"\n{'='*60}")
    log.info(f"TRANSLATION COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Records translated    : {stats['translated']:,}")
    log.info(f"  OOM failures          : {stats.get('oom_failures', 0):,}")
    log.info(f"  Avg glossary term rate : {avg_term_rate:.2%}")
    log.info(f"  Need refinement       : {stats['needs_refinement']:,} ({100*stats['needs_refinement']/max(stats['translated'],1):.1f}%)")
    log.info(f"  Total time            : {elapsed_total/3600:.1f} hours")
    log.info(f"  Output: {output_path}")
    log.info(f"{'='*60}")

    # Save stats
    stats_out = {
        "model": model_name,
        "total_translated": stats["translated"],
        "needs_refinement": stats["needs_refinement"],
        "avg_glossary_term_rate": round(avg_term_rate, 4),
        "oom_failures": stats.get("oom_failures", 0),
        "elapsed_hours": round(elapsed_total / 3600, 2),
        "batch_size": batch_size,
    }
    with open(f"{output_dir}/02_translation_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    log.info(f"✓ Script 02 complete. Run script 03 for terminology refinement.")


if __name__ == "__main__":
    main()
