#!/usr/bin/env python3
"""
Cross-Domain ICL Evaluation on Arabic NLP Benchmarks
======================================================
Evaluates in-context learning performance of LLMs on 11 Arabic
NLP benchmarks spanning dialect identification, sentiment analysis,
and content moderation.

Models: DeepSeek-R1-Distill-Qwen-7B, Qwen3-8B
Method: Likelihood-based prediction with 4-bit NF4 quantization
Prompts: 3 variants (P1: Arabic task-specific, P2: Arabic expert, P3: English)
Shots: k ∈ {5, 8, 10}
Selection: Random (R=5 runs) and Embed Top-k (deterministic)

Usage:
    python3 02_run_cross_domain.py --model deepseek --dataset nadi
    python3 02_run_cross_domain.py --model qwen3 --dataset all
    python3 02_run_cross_domain.py --all

Output:
    results/cross_domain/<dataset>_<model>_results.json
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen3": "Qwen/Qwen3-8B",
}

DATASETS = {
    "nadi":          {"name": "NADI 2020",     "task": "dialect_identification", "classes": 21,
                      "data_dir": "data/nadi_2020",     "text_field": "text", "label_field": "label"},
    "madar":         {"name": "MADAR-26",      "task": "dialect_identification", "classes": 26,
                      "data_dir": "data/madar_26",      "text_field": "text", "label_field": "label"},
    "astd":          {"name": "ASTD",          "task": "sentiment_analysis",     "classes": 4,
                      "data_dir": "data/astd",          "text_field": "text", "label_field": "sentiment"},
    "arsas":         {"name": "ArSAS",         "task": "sentiment_analysis",     "classes": 4,
                      "data_dir": "data/arsas",         "text_field": "text", "label_field": "sentiment"},
    "asad":          {"name": "ASAD",          "task": "sentiment_analysis",     "classes": 3,
                      "data_dir": "data/asad",          "text_field": "text", "label_field": "label"},
    "labr":          {"name": "LABR",          "task": "sentiment_analysis",     "classes": 5,
                      "data_dir": "data/labr",          "text_field": "review", "label_field": "rating"},
    "semeval2016":   {"name": "SemEval-2016",  "task": "sentiment_analysis",     "classes": 3,
                      "data_dir": "data/semeval_2016",  "text_field": "text", "label_field": "sentiment"},
    "alomari":       {"name": "Alomari 2017",  "task": "sentiment_analysis",     "classes": 3,
                      "data_dir": "data/alomari_2017",  "text_field": "text", "label_field": "sentiment"},
    "arsentiment":   {"name": "ArSentiment",   "task": "sentiment_analysis",     "classes": 3,
                      "data_dir": "data/arsentiment",   "text_field": "text", "label_field": "label"},
    "osact4":        {"name": "OSACT4",        "task": "content_moderation",     "classes": 2,
                      "data_dir": "data/osact4",        "text_field": "text", "label_field": "label"},
    "adult_content": {"name": "Adult Content", "task": "content_moderation",     "classes": 2,
                      "data_dir": "data/adult_content", "text_field": "text", "label_field": "label"},
}

SHOT_LEVELS = [5, 8, 10]
NUM_RUNS = 5
TEMPERATURE = 0.1
MAX_TEST_SAMPLES = 500
SELECTION_METHODS = ["random", "topk"]

# Prompt templates: P1 = Arabic task-specific, P2 = Arabic expert, P3 = English
PROMPT_TEMPLATES = {
    "P1": {
        "dialect_identification": "صنّف اللهجة العربية للنص التالي.\nالنص: {text}\nالتصنيف:",
        "sentiment_analysis":    "حدد المشاعر في النص التالي.\nالنص: {text}\nالمشاعر:",
        "content_moderation":    "هل يحتوي النص التالي على محتوى مسيء؟\nالنص: {text}\nالتصنيف:",
    },
    "P2": {
        "dialect_identification": "أنت خبير في اللهجات العربية. حدد لهجة النص:\nالنص: {text}\nاللهجة:",
        "sentiment_analysis":    "أنت محلل مشاعر. صنّف مشاعر النص:\nالنص: {text}\nالمشاعر:",
        "content_moderation":    "أنت مراقب محتوى. هل النص التالي مسيء أم لا؟\nالنص: {text}\nالحكم:",
    },
    "P3": {
        "dialect_identification": "Identify which Arabic dialect the following text belongs to.\nText: {text}\nDialect:",
        "sentiment_analysis":    "Classify the sentiment of the following Arabic text.\nText: {text}\nSentiment:",
        "content_moderation":    "Determine if the following Arabic text contains offensive content.\nText: {text}\nLabel:",
    },
}


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_data(data_dir, split="test"):
    """Load a dataset split from JSON, JSONL, or CSV."""
    for ext in [".json", ".jsonl", ".csv", ".tsv"]:
        fpath = os.path.join(data_dir, f"{split}{ext}")
        if not os.path.exists(fpath):
            continue
        if ext == ".json":
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        elif ext == ".jsonl":
            data = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        elif ext in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            return pd.read_csv(fpath, sep=sep).to_dict("records")

    # Check repo subdirectory
    repo_dir = os.path.join(data_dir, "repo")
    if os.path.exists(repo_dir):
        for root, _, files in os.walk(repo_dir):
            for f in files:
                if split in f.lower() and f.endswith((".json", ".csv")):
                    print(f"  Found candidate: {os.path.join(root, f)}")

    print(f"  WARNING: No {split} data found in {data_dir}")
    return None


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_model(model_key):
    """Load LLM with 4-bit NF4 quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_name = MODELS[model_key]
    print(f"Loading {model_name} (4-bit NF4)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ══════════════════════════════════════════════════════════════
# DEMONSTRATION SELECTION
# ══════════════════════════════════════════════════════════════

def select_random(train_data, k, rng):
    """Randomly select k demonstrations."""
    idx = rng.choice(len(train_data), size=min(k, len(train_data)), replace=False)
    return [train_data[i] for i in idx]


def select_topk(train_data, query_text, k, embedder, text_field="text"):
    """Select k most similar demonstrations via embedding cosine similarity."""
    from sentence_transformers import SentenceTransformer

    if embedder is None:
        embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    query_emb = embedder.encode([query_text])
    texts = [d.get(text_field, d.get("review", "")) for d in train_data]
    train_embs = embedder.encode(texts)

    sims = np.dot(train_embs, query_emb.T).flatten()
    top_idx = np.argsort(sims)[-k:][::-1]

    return [train_data[i] for i in top_idx], embedder


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ══════════════════════════════════════════════════════════════

def build_prompt(dataset_key, prompt_key, test_sample, demos=None):
    """Build prompt with optional few-shot demonstrations."""
    cfg = DATASETS[dataset_key]
    template = PROMPT_TEMPLATES[prompt_key][cfg["task"]]
    tf, lf = cfg["text_field"], cfg["label_field"]

    parts = []
    if demos:
        for d in demos:
            parts.append(template.format(text=d.get(tf, "")) + f" {d.get(lf, '')}")
        parts.append("")

    parts.append(template.format(text=test_sample.get(tf, "")))
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════

def predict_likelihood(model, tokenizer, prompt, labels):
    """Pick the label with the highest average log-probability."""
    import torch

    best_label, best_score = None, float("-inf")

    for label in labels:
        inputs = tokenizer(prompt + f" {label}", return_tensors="pt",
                          truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        label_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        if not label_ids:
            continue

        n = len(label_ids)
        log_probs = torch.log_softmax(logits[0, -n - 1:-1, :], dim=-1)
        score = sum(log_probs[i, tid].item() for i, tid in enumerate(label_ids)) / n

        if score > best_score:
            best_score, best_label = score, label

    return best_label


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_dataset(model, tokenizer, dataset_key, model_key):
    """Run full evaluation on one dataset across all configurations."""
    cfg = DATASETS[dataset_key]
    base = os.path.join(os.path.dirname(__file__), cfg["data_dir"])

    print(f"\n{'='*60}")
    print(f"Evaluating: {cfg['name']} with {model_key}")
    print(f"{'='*60}")

    test_data = load_data(base, "test")
    if test_data is None:
        return None

    train_data = load_data(base, "train") or test_data
    lf = cfg["label_field"]
    labels = sorted({str(d[lf]) for d in test_data + train_data if lf in d})
    print(f"  Labels ({len(labels)}): {labels[:10]}{'...' if len(labels)>10 else ''}")

    if len(test_data) > MAX_TEST_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(test_data), MAX_TEST_SAMPLES, replace=False)
        test_data = [test_data[i] for i in idx]
    print(f"  Test samples: {len(test_data)}")

    all_results = []
    embedder = None

    for prompt_key in ["P1", "P2", "P3"]:
        for shot in SHOT_LEVELS:
            for selection in SELECTION_METHODS:
                n_runs = NUM_RUNS if selection == "random" else 1

                for run in range(n_runs):
                    run_id = f"{prompt_key}_s{shot}_{selection}_r{run}"
                    print(f"  {run_id}...", end=" ", flush=True)

                    rng = np.random.RandomState(run)
                    correct, total = 0, 0

                    for sample in test_data:
                        true = str(sample.get(lf, ""))

                        if selection == "random":
                            demos = select_random(train_data, shot, rng)
                        else:
                            tf = cfg["text_field"]
                            demos, embedder = select_topk(
                                train_data, sample.get(tf, ""), shot, embedder, tf)

                        prompt = build_prompt(dataset_key, prompt_key, sample, demos)
                        pred = predict_likelihood(model, tokenizer, prompt, labels)

                        correct += int(pred == true)
                        total += 1

                    acc = correct / total if total else 0
                    print(f"{acc:.4f}")

                    all_results.append({
                        "dataset": dataset_key,
                        "model": model_key,
                        "prompt": prompt_key,
                        "shots": shot,
                        "selection": selection,
                        "run": run,
                        "accuracy": acc,
                        "correct": correct,
                        "total": total,
                        "timestamp": datetime.now().isoformat(),
                    })

    return all_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain ICL evaluation on Arabic NLP benchmarks")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="deepseek")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--all", action="store_true",
                       help="Run all models on all datasets")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "cross_domain")
    os.makedirs(results_dir, exist_ok=True)

    model_keys = list(MODELS.keys()) if args.all else [args.model]
    ds_keys = list(DATASETS.keys()) if args.dataset == "all" or args.all else [args.dataset]

    for mk in model_keys:
        model, tokenizer = load_model(mk)

        for dk in ds_keys:
            results = evaluate_dataset(model, tokenizer, dk, mk)
            if results:
                out = os.path.join(results_dir, f"{dk}_{mk}_results.json")
                with open(out, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved: {out}")

        del model, tokenizer
        import torch
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
