#!/usr/bin/env python3
"""
=============================================================================
GTD Arabic Terrorism Classification — ICL Prompt Scoring Experiments
=============================================================================
Chapter 5 experiments: Systematic evaluation of prompt strategies for
Arabic terrorism event classification using in-context learning.

Configuration:
  - 3 Tasks:      GTD-Attack (9cls), GTD-Weapon (12cls), GTD-Target (22cls)
  - 5 Prompts:    Basic, CoT, Expert-Role, Scenario, Plan-and-Solve
  - 2 Models:     DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct
  - 3 Shot levels: 0, 5, 10
  - 2 Selection:  Random, Top-k (LaBSE)
  - 80 test samples per task (stratified)
  - 3 independent runs per random configuration
  - 4-bit NF4 quantization

Total configurations: 3 tasks × 5 prompts × 2 models × (1 zero-shot + 2 shots × 2 selection) = 150
Total passes: 150 × 3 runs = 450 (but Top-k is deterministic → ~300 unique)

Hardware: apl13 — GPU 0,1: RTX 8000 (48GB), GPU 2: RTX 5000 (16GB)
Strategy: Run 2 models in parallel on GPU 0 and GPU 1
Estimated time: ~11 hours

Usage:
  python gtd_icl_experiments.py                    # Run all
  python gtd_icl_experiments.py --model deepseek   # Single model
  python gtd_icl_experiments.py --gpu 0            # Specific GPU
  python gtd_icl_experiments.py --resume           # Resume from checkpoint
=============================================================================
"""

import json
import os
import sys
import time
import logging
import argparse
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gtd_experiments.log"),
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(os.path.expanduser("~/TerrorismNER_Research"))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "gtd_icl"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "gtd_icl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODELS = {
    'deepseek': '/tmp/deepseek_local',
    'qwen': '/tmp/qwen_local',
}

TASKS = {
    'attack': {'file': 'gtd_eval_attack.json', 'name': 'GTD-Attack'},
    'weapon': {'file': 'gtd_eval_weapon.json', 'name': 'GTD-Weapon'},
    'target': {'file': 'gtd_eval_target.json', 'name': 'GTD-Target'},
}

SHOT_CONFIGS = [0, 5, 10]
SELECTION_METHODS = ['random', 'topk']
NUM_RUNS = 3
SEED_BASE = 42
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1

# =============================================================================
# PROMPT TEMPLATES (5 strategies)
# =============================================================================

def get_prompt_templates(task_key, labels_str):
    """Return 5 prompt templates for a given task."""

    task_descriptions = {
        'attack': ('attack methodology', 'type of attack'),
        'weapon': ('weapon used', 'type of weapon'),
        'target': ('target of the attack', 'type of target'),
    }
    desc, type_phrase = task_descriptions[task_key]

    templates = {
        'basic': {
            'name': 'Basic',
            'system': '',
            'instruction': f"""صنّف النص التالي إلى أحد التصنيفات التالية لـ{desc}:
{labels_str}

{{examples}}النص: {{text}}
التصنيف:""",
        },

        'cot': {
            'name': 'Chain-of-Thought',
            'system': '',
            'instruction': f"""صنّف {type_phrase} في النص التالي. فكّر خطوة بخطوة:
1. اقرأ وصف الحادث بعناية
2. حدد الأدلة الرئيسية المتعلقة بـ{desc}
3. اختر التصنيف الأنسب من: {labels_str}

{{examples}}النص: {{text}}
التحليل خطوة بخطوة:""",
        },

        'expert': {
            'name': 'Expert-Role',
            'system': '',
            'instruction': f"""أنت محلل أمني متخصص في تحليل الحوادث الإرهابية. بناءً على خبرتك، صنّف {type_phrase} في الحادث التالي.

التصنيفات المتاحة: {labels_str}

{{examples}}الحادث: {{text}}
تصنيف المحلل:""",
        },

        'scenario': {
            'name': 'Scenario-Based',
            'system': '',
            'instruction': f"""سياق: أنت تعمل في مركز تحليل الحوادث الأمنية. وردك تقرير عن حادث إرهابي ومطلوب تصنيف {desc}.

التصنيفات: {labels_str}

{{examples}}التقرير: {{text}}
التصنيف المطلوب ({type_phrase}):""",
        },

        'plan_solve': {
            'name': 'Plan-and-Solve',
            'system': '',
            'instruction': f"""المهمة: تصنيف {desc} في النص التالي.

الخطة:
- أولاً: استخرج الكلمات المفتاحية المتعلقة بـ{desc}
- ثانياً: قارن مع التصنيفات المتاحة: {labels_str}
- ثالثاً: اختر التصنيف الأدق

{{examples}}النص: {{text}}
الكلمات المفتاحية:
التصنيف النهائي:""",
        },
    }

    return templates


def format_examples(template_text, examples, task_key):
    """Format few-shot examples into the template."""
    if not examples:
        return template_text.replace("{examples}", "")

    task_descriptions = {
        'attack': 'التصنيف',
        'weapon': 'التصنيف',
        'target': 'التصنيف',
    }
    label_prefix = task_descriptions[task_key]

    examples_str = ""
    for ex in examples:
        examples_str += f"النص: {ex['text_ar']}\n{label_prefix}: {ex['label']}\n\n"

    return template_text.replace("{examples}", examples_str)


# =============================================================================
# LaBSE TOP-K RETRIEVAL
# =============================================================================

class LaBSERetriever:
    """LaBSE-based semantic similarity retrieval for ICE selection."""

    def __init__(self, device='cpu'):
        from sentence_transformers import SentenceTransformer
        log.info("Loading LaBSE for Top-k retrieval...")
        self.model = SentenceTransformer('/tmp/miniLM_snap/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d', device=device)
        self.train_embeddings = None
        self.train_data = None
        log.info("LaBSE loaded.")

    def index(self, train_data):
        """Pre-compute embeddings for all training examples."""
        self.train_data = train_data
        texts = [r['text_ar'] for r in train_data]
        log.info(f"  Encoding {len(texts):,} training examples with LaBSE...")
        self.train_embeddings = self.model.encode(
            texts, batch_size=256, show_progress_bar=False,
            normalize_embeddings=True
        )
        log.info(f"  LaBSE indexing complete. Shape: {self.train_embeddings.shape}")

    def retrieve(self, query_text, k=10):
        """Retrieve top-k most similar training examples."""
        query_emb = self.model.encode(
            [query_text], normalize_embeddings=True
        )
        similarities = query_emb @ self.train_embeddings.T
        top_indices = np.argsort(similarities[0])[::-1][:k]
        return [self.train_data[i] for i in top_indices]


# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================

def load_model(model_name, gpu_id=0):
    """Load model with 4-bit NF4 quantization."""
    log.info(f"Loading {model_name} on GPU {gpu_id} (4-bit NF4)...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.eval()

    used = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    log.info(f"  Model loaded. VRAM: {used:.1f} GB on GPU {gpu_id}")
    return model, tokenizer


def generate_prediction(prompt, model, tokenizer, max_new_tokens=64, temperature=0.1):
    """Generate prediction using the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def match_label(response, valid_labels):
    """Match generated response to a valid label."""
    response_clean = response.strip().split('\n')[0].strip()

    # Exact match
    for label in valid_labels:
        if label.lower() == response_clean.lower():
            return label

    # Substring match (response contains the label)
    for label in valid_labels:
        if label.lower() in response_clean.lower():
            return label

    # Label contains response
    for label in valid_labels:
        if response_clean.lower() in label.lower() and len(response_clean) > 3:
            return label

    return None  # No match — prediction failure


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_config(
    model, tokenizer, model_key,
    task_key, task_data,
    prompt_key, prompt_template,
    n_shots, selection_method,
    run_id, retriever=None,
):
    """Run a single experimental configuration."""
    test_data = task_data['test']
    train_data = task_data['train']
    valid_labels = task_data['task_info']['labels']
    labels_str = ", ".join(valid_labels)

    predictions = []
    gold_labels = []
    failures = 0

    for sample in test_data:
        # Select ICEs
        if n_shots == 0:
            examples = []
        elif selection_method == 'random':
            rng = random.Random(SEED_BASE + run_id * 1000 + hash(sample['id']) % 10000)
            examples = rng.sample(train_data, min(n_shots, len(train_data)))
        elif selection_method == 'topk':
            examples = retriever.retrieve(sample['text_ar'], k=n_shots)
        else:
            examples = []

        # Build prompt
        template_text = prompt_template['instruction']
        template_filled = template_text.replace("{text}", sample['text_ar'])
        template_filled = template_filled.replace(f"{{{labels_str}}}", labels_str)
        template_filled = format_examples(template_filled, examples, task_key)

        # Generate
        response = generate_prediction(template_filled, model, tokenizer,
                                       MAX_NEW_TOKENS, TEMPERATURE)

        # Match label
        predicted = match_label(response, valid_labels)
        if predicted is None:
            failures += 1
            predicted = valid_labels[0]  # Default fallback

        predictions.append(predicted)
        gold_labels.append(sample['label'])

    # Compute metrics
    acc = accuracy_score(gold_labels, predictions)
    f1_macro = f1_score(gold_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(gold_labels, predictions, average='weighted', zero_division=0)

    result = {
        'model': model_key,
        'task': task_key,
        'task_name': TASKS[task_key]['name'],
        'prompt': prompt_key,
        'prompt_name': prompt_template['name'],
        'n_shots': n_shots,
        'selection': selection_method,
        'run_id': run_id,
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'failures': failures,
        'failure_rate': round(failures / len(test_data), 4),
        'n_test': len(test_data),
        'predictions': predictions,
        'gold_labels': gold_labels,
        'timestamp': datetime.now().isoformat(),
    }

    return result


def get_config_id(model_key, task_key, prompt_key, n_shots, selection, run_id):
    """Unique identifier for a configuration."""
    return f"{model_key}_{task_key}_{prompt_key}_s{n_shots}_{selection}_r{run_id}"


def load_completed(results_path):
    """Load already-completed configuration IDs."""
    completed = set()
    if results_path.exists():
        with open(results_path, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cid = get_config_id(
                        r['model'], r['task'], r['prompt'],
                        r['n_shots'], r['selection'], r['run_id']
                    )
                    completed.add(cid)
                except:
                    pass
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['deepseek', 'qwen'], default=None,
                        help='Run single model (default: both)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID (default: 0 for deepseek, 1 for qwen)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--task', choices=['attack', 'weapon', 'target'], default=None,
                        help='Run single task')
    args = parser.parse_args()

    # Determine which models to run
    if args.model:
        models_to_run = {args.model: LLM_MODELS[args.model]}
        gpu_map = {args.model: args.gpu if args.gpu is not None else 0}
    else:
        models_to_run = LLM_MODELS
        gpu_map = {'deepseek': 0, 'qwen': 1}

    tasks_to_run = {args.task: TASKS[args.task]} if args.task else TASKS

    # Results file (JSONL for append-friendly resume)
    results_path = RESULTS_DIR / "gtd_all_results.jsonl"

    # Load completed configs for resume
    completed = load_completed(results_path) if args.resume else set()
    if completed:
        log.info(f"Resuming: {len(completed)} configurations already completed")

    # Load task data
    log.info("Loading task datasets...")
    task_datasets = {}
    for task_key, task_info in tasks_to_run.items():
        fpath = DATA_DIR / task_info['file']
        with open(fpath, 'r') as f:
            task_datasets[task_key] = json.load(f)
        n_cls = task_datasets[task_key]['task_info']['n_classes']
        n_test = len(task_datasets[task_key]['test'])
        n_train = len(task_datasets[task_key]['train'])
        log.info(f"  {task_info['name']}: {n_cls} classes, test={n_test}, train={n_train:,}")

    # Initialize LaBSE retriever (shared across models, runs on CPU)
    retriever = LaBSERetriever(device='cpu')

    # Build all configurations
    all_configs = []
    for model_key in models_to_run:
        for task_key in tasks_to_run:
            labels = task_datasets[task_key]['task_info']['labels']
            labels_str = ", ".join(labels)
            templates = get_prompt_templates(task_key, labels_str)

            for prompt_key, prompt_template in templates.items():
                for n_shots in SHOT_CONFIGS:
                    if n_shots == 0:
                        # Zero-shot: single selection method, multiple runs
                        for run_id in range(NUM_RUNS):
                            cid = get_config_id(model_key, task_key, prompt_key, 0, 'none', run_id)
                            if cid not in completed:
                                all_configs.append({
                                    'model_key': model_key,
                                    'task_key': task_key,
                                    'prompt_key': prompt_key,
                                    'prompt_template': prompt_template,
                                    'n_shots': 0,
                                    'selection': 'none',
                                    'run_id': run_id,
                                })
                    else:
                        for selection in SELECTION_METHODS:
                            n_runs = NUM_RUNS if selection == 'random' else 1
                            for run_id in range(n_runs):
                                cid = get_config_id(model_key, task_key, prompt_key, n_shots, selection, run_id)
                                if cid not in completed:
                                    all_configs.append({
                                        'model_key': model_key,
                                        'task_key': task_key,
                                        'prompt_key': prompt_key,
                                        'prompt_template': prompt_template,
                                        'n_shots': n_shots,
                                        'selection': selection,
                                        'run_id': run_id,
                                    })

    log.info(f"\nTotal configurations to run: {len(all_configs)}")

    # Group configs by model for efficient processing
    configs_by_model = defaultdict(list)
    for cfg in all_configs:
        configs_by_model[cfg['model_key']].append(cfg)

    total_done = 0
    total_configs = len(all_configs)
    start_time = time.time()

    for model_key, configs in configs_by_model.items():
        gpu_id = gpu_map.get(model_key, 0)
        model_name = models_to_run[model_key]

        log.info(f"\n{'='*70}")
        log.info(f"MODEL: {model_key} ({model_name}) on GPU {gpu_id}")
        log.info(f"Configurations: {len(configs)}")
        log.info(f"{'='*70}")

        # Load model
        model, tokenizer = load_model(model_name, gpu_id)

        # Index LaBSE for each task (only once per task)
        indexed_tasks = set()

        # Sort configs: group by task for LaBSE efficiency
        configs.sort(key=lambda c: (c['task_key'], c['prompt_key'], c['n_shots']))

        for i, cfg in enumerate(configs):
            task_key = cfg['task_key']
            task_data = task_datasets[task_key]

            # Index LaBSE if needed for this task
            if task_key not in indexed_tasks:
                retriever.index(task_data['train'])
                indexed_tasks.add(task_key)

            config_id = get_config_id(
                cfg['model_key'], task_key, cfg['prompt_key'],
                cfg['n_shots'], cfg['selection'], cfg['run_id']
            )

            log.info(
                f"[{total_done+1}/{total_configs}] {config_id} "
                f"({cfg['prompt_template']['name']}, {cfg['n_shots']}-shot, {cfg['selection']})"
            )

            t0 = time.time()
            result = run_single_config(
                model=model,
                tokenizer=tokenizer,
                model_key=model_key,
                task_key=task_key,
                task_data=task_data,
                prompt_key=cfg['prompt_key'],
                prompt_template=cfg['prompt_template'],
                n_shots=cfg['n_shots'],
                selection_method=cfg['selection'],
                run_id=cfg['run_id'],
                retriever=retriever if cfg['selection'] == 'topk' else None,
            )
            elapsed = time.time() - t0

            # Remove large fields before saving
            result_save = {k: v for k, v in result.items()
                          if k not in ('predictions', 'gold_labels')}
            result_save['elapsed_sec'] = round(elapsed, 1)

            # Save predictions separately for detailed analysis
            pred_save = {
                'config_id': config_id,
                'predictions': result['predictions'],
                'gold_labels': result['gold_labels'],
            }

            # Append to results
            with open(results_path, 'a') as f:
                f.write(json.dumps(result_save, ensure_ascii=False) + '\n')

            # Save predictions
            pred_path = RESULTS_DIR / "predictions"
            pred_path.mkdir(exist_ok=True)
            with open(pred_path / f"{config_id}.json", 'w') as f:
                json.dump(pred_save, f, ensure_ascii=False)

            total_done += 1
            total_elapsed = time.time() - start_time
            avg_per_config = total_elapsed / total_done
            eta = avg_per_config * (total_configs - total_done)

            log.info(
                f"  Acc={result['accuracy']:.3f} F1={result['f1_macro']:.3f} "
                f"Fail={result['failures']}/{result['n_test']} "
                f"Time={elapsed:.0f}s | ETA={eta/3600:.1f}h"
            )

        # Free model memory
        del model, tokenizer
        torch.cuda.empty_cache()
        log.info(f"Model {model_key} unloaded. GPU memory freed.")

    # Final summary
    total_elapsed = time.time() - start_time
    log.info(f"\n{'='*70}")
    log.info(f"ALL EXPERIMENTS COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"  Total configurations: {total_done}")
    log.info(f"  Total time: {total_elapsed/3600:.1f} hours")
    log.info(f"  Results: {results_path}")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
