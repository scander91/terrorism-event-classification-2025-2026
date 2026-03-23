#!/usr/bin/env python3
"""
=============================================================================
Script 03: Terminology-Constrained LLM Refinement
=============================================================================
Refines NLLB translations using AceGPT-7B-chat with glossary-augmented
prompting. Only processes records flagged as needs_refinement (term_rate < 0.8).

Follows WMT 2023-2025 translate-then-refine best practice.

Input:  outputs/02_nllb_translations.jsonl
Output: outputs/03_refined_translations.jsonl
        outputs/03_refinement_stats.json

GPU: RTX 2080 Ti (11GB) — AceGPT-7B in 8-bit uses ~5GB
Estimated time: ~8-12 hours for ~20-30K records
=============================================================================
"""

import json
import os
import sys
import logging
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    """Load glossary for prompt construction."""
    if not Path(glossary_path).exists():
        return {}
    with open(glossary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    en_to_ar = {}
    for category, terms in data["terms"].items():
        for entry in terms:
            if "en" in entry and "ar" in entry:
                en_to_ar[entry["en"].lower()] = entry["ar"]
    return en_to_ar


def check_term_rate(source_en, text_ar, glossary):
    """Compute glossary term success rate."""
    if not glossary:
        return 1.0

    source_lower = source_en.lower()
    found = 0
    matched = 0

    for en_term, ar_term in glossary.items():
        if en_term in source_lower:
            found += 1
            if ar_term in text_ar:
                matched += 1

    if found == 0:
        return 1.0
    return matched / found


def build_refinement_prompt(source_en, nllb_ar, missing_terms, glossary, 
                            attack_type="", weapon_type="", target_type=""):
    """Build Arabic refinement prompt following translate-then-refine paradigm."""

    # Extract relevant glossary pairs for this record
    glossary_pairs = []
    source_lower = source_en.lower()
    for en_term, ar_term in glossary.items():
        if en_term in source_lower:
            glossary_pairs.append(f"  - {en_term} → {ar_term}")

    glossary_str = "\n".join(glossary_pairs) if glossary_pairs else "  (لا توجد مصطلحات)"

    missing_str = ""
    if missing_terms:
        missing_items = []
        for en_term in missing_terms:
            ar = glossary.get(en_term.lower(), "")
            if ar:
                missing_items.append(f"⚠ {en_term} → {ar}")
        missing_str = "\nالمصطلحات الناقصة في الترجمة الحالية:\n" + "\n".join(missing_items)

    # Context from GTD labels
    context = ""
    if attack_type or weapon_type or target_type:
        context = f"\nسياق الحادث: نوع الهجوم: {attack_type} | نوع السلاح: {weapon_type} | نوع الهدف: {target_type}"

    prompt = f"""أنت مترجم متخصص في النصوص الأمنية والعسكرية. مهمتك تحسين الترجمة الآلية التالية مع ضمان استخدام المصطلحات الصحيحة.

النص الإنجليزي الأصلي:
{source_en}

الترجمة الآلية الحالية:
{nllb_ar}
{context}

قاموس المصطلحات المطلوبة:
{glossary_str}
{missing_str}

التعليمات:
1. أعد صياغة الترجمة بالعربية الفصحى الحديثة
2. تأكد من استخدام جميع المصطلحات من القاموس أعلاه
3. لا تغير المحتوى الواقعي
4. حافظ على الأسلوب الإخباري الموضوعي

الترجمة المحسنة:"""

    return prompt


def load_llm(model_name, device="cuda"):
    """Load LLM with 8-bit quantization for 11GB VRAM."""
    log.info(f"Loading LLM: {model_name} (8-bit quantization)")

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memoryory / (1024**3)
        log.info(f"VRAM after model load: {used:.1f}/{total:.1f} GB")

    return model, tokenizer


def generate_refinement(prompt, model, tokenizer, max_new_tokens=512, temperature=0.3):
    """Generate refined translation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    refined = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up: remove any re-generated prompt parts
    for marker in ["النص الإنجليزي", "الترجمة الآلية", "التعليمات:", "قاموس المصطلحات"]:
        if marker in refined:
            refined = refined[:refined.index(marker)].strip()

    return refined


def main():
    config = load_config()

    # Parameters
    model_name = "FreedomIntelligence/AceGPT-7B-chat"
    max_new_tokens = 512
    temperature = 0.3

    if config:
        r = config.get("refinement", {})
        model_name = r.get("model", model_name)
        max_new_tokens = r.get("max_new_tokens", max_new_tokens)
        temperature = r.get("temperature", temperature)

    output_dir = config["paths"]["output_dir"] if config else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load input
    input_path = f"{output_dir}/02_nllb_translations.jsonl"
    if not Path(input_path).exists():
        log.error(f"Input not found: {input_path}. Run script 02 first.")
        sys.exit(1)

    all_records = []
    needs_refinement = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            all_records.append(rec)
            if rec.get("needs_refinement", False):
                needs_refinement.append(rec)

    log.info(f"Loaded {len(all_records):,} total records")
    log.info(f"Records needing refinement: {len(needs_refinement):,}")

    if not needs_refinement:
        log.info("No records need refinement. Copying all to output.")
        output_path = f"{output_dir}/03_refined_translations.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                rec["refined_ar"] = rec["translation_ar"]
                rec["refinement_applied"] = False
                rec["final_term_rate"] = rec.get("glossary_term_rate", 1.0)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return

    # Load glossary and model
    glossary_path = config["paths"]["glossary"] if config else "glossary/terrorism_glossary.json"
    glossary = load_glossary(glossary_path)
    model, tokenizer = load_llm(model_name)

    # Resume support
    output_path = f"{output_dir}/03_refined_translations.jsonl"
    completed_ids = set()
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["id"])
                except:
                    pass
        log.info(f"Resuming: {len(completed_ids):,} already processed")

    # Process refinements
    stats = Counter()
    start_time = time.time()

    # First, write non-refinement records
    outfile = open(output_path, "a" if completed_ids else "w", encoding="utf-8")

    if not completed_ids:  # Fresh start — write pass-through records first
        for rec in all_records:
            if not rec.get("needs_refinement", False):
                rec["refined_ar"] = rec["translation_ar"]
                rec["refinement_applied"] = False
                rec["final_term_rate"] = rec.get("glossary_term_rate", 1.0)
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["passed_through"] += 1

    # Now refine flagged records
    remaining = [r for r in needs_refinement if r["id"] not in completed_ids]
    log.info(f"Remaining to refine: {len(remaining):,}")

    for i, rec in enumerate(remaining):
        try:
            prompt = build_refinement_prompt(
                source_en=rec["summary_en"],
                nllb_ar=rec["translation_ar"],
                missing_terms=rec.get("glossary_terms_missing", []),
                glossary=glossary,
                attack_type=rec.get("attack_type", ""),
                weapon_type=rec.get("weapon_type", ""),
                target_type=rec.get("target_type", ""),
            )

            refined = generate_refinement(prompt, model, tokenizer, 
                                          max_new_tokens, temperature)

            # Validate: only keep if term rate improved or maintained
            new_term_rate = check_term_rate(rec["summary_en"], refined, glossary)
            old_term_rate = rec.get("glossary_term_rate", 0)

            if new_term_rate >= old_term_rate and len(refined) > 10:
                rec["refined_ar"] = refined
                rec["refinement_applied"] = True
                rec["final_term_rate"] = round(new_term_rate, 4)
                stats["improved"] += 1
            else:
                # Keep original if refinement didn't help
                rec["refined_ar"] = rec["translation_ar"]
                rec["refinement_applied"] = False
                rec["final_term_rate"] = round(old_term_rate, 4)
                stats["kept_original"] += 1

        except Exception as e:
            log.warning(f"Error refining record {rec['id']}: {e}")
            rec["refined_ar"] = rec["translation_ar"]
            rec["refinement_applied"] = False
            rec["final_term_rate"] = rec.get("glossary_term_rate", 0)
            stats["errors"] += 1

        outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["processed"] += 1

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            log.info(
                f"Refinement: {i+1:,}/{len(remaining):,} | "
                f"Improved: {stats['improved']:,} | "
                f"Kept original: {stats['kept_original']:,} | "
                f"ETA: {eta/3600:.1f}h"
            )

    outfile.close()

    elapsed_total = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"REFINEMENT COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Processed            : {stats['processed']:,}")
    log.info(f"  Improved by LLM      : {stats['improved']:,}")
    log.info(f"  Kept NLLB original   : {stats['kept_original']:,}")
    log.info(f"  Errors               : {stats.get('errors', 0):,}")
    log.info(f"  Passed through       : {stats.get('passed_through', 0):,}")
    log.info(f"  Total time           : {elapsed_total/3600:.1f} hours")
    log.info(f"{'='*60}")

    # Save stats
    with open(f"{output_dir}/03_refinement_stats.json", "w") as f:
        json.dump({
            "model": model_name,
            "total_refined": stats["processed"],
            "improved": stats["improved"],
            "kept_original": stats["kept_original"],
            "errors": stats.get("errors", 0),
            "elapsed_hours": round(elapsed_total / 3600, 2),
        }, f, indent=2)

    log.info(f"✓ Script 03 complete. Run script 04 for validation.")


if __name__ == "__main__":
    main()
