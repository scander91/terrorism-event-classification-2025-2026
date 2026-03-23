#!/usr/bin/env python3
"""
===============================================================================
COMPLETE LLM EVALUATION - FIXED VERSION
===============================================================================
Fixes:
1. NER data uses 'tokens' (list) - convert to text string
2. Skip already completed experiments (DeepSeek CLF)
3. Continue with: NER, Qwen2.5, and Experiment 2 (20/50/100 groups)

Run with: CUDA_VISIBLE_DEVICES=0 python complete_llm_evaluation_fixed.py

===============================================================================
"""

import os
import gc
import json
import pickle
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration."""
    
    # Paths
    BASE_DIR = Path(os.path.expanduser("~/TerrorismNER_Project"))
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    RESULTS_DIR = BASE_DIR / "results" / "complete_llm_eval"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data files
    NER_DATA_PATH = CHECKPOINT_DIR / "ner_data_basic.pkl"
    CLF_DATA_PATHS = {
        10: CHECKPOINT_DIR / "classification_data_10.pkl",
        20: CHECKPOINT_DIR / "classification_data_20.pkl",
        50: CHECKPOINT_DIR / "classification_data_50.pkl",
        100: CHECKPOINT_DIR / "classification_data_100.pkl",
    }
    
    # LLM Models
    LLM_MODELS = {
        'deepseek-r1': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'qwen2.5': 'Qwen/Qwen2.5-7B-Instruct',
    }
    
    # Experiment settings
    SHOT_CONFIGS = [0, 5, 10]
    SELECTION_METHODS = ['random', 'topk']
    NUM_RUNS = 3
    SEED_BASE = 42
    
    # LLM settings
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.1
    
    # Test subset sizes
    NER_TEST_SUBSET = 100
    CLF_TEST_SUBSET = 150
    
    # Embedding model for TopK
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = Config.LOGS_DIR / f"llm_eval_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg: str):
    """Log message to console and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

CLF_TEMPLATE = """Classify the text into one terrorist group from: {label_list}

{examples}Text: {text}
Group:"""

NER_TEMPLATE = """Extract terrorist organization names from the following text.
If no organizations found, output "NONE".

{examples}Text: {text}
Organizations:"""


def format_clf_examples(examples: List[Tuple[str, str]]) -> str:
    """Format classification examples for prompt."""
    if not examples:
        return ""
    lines = []
    for text, label in examples:
        lines.append(f"Text: {text[:300]}...\nGroup: {label}\n")
    return "\n".join(lines) + "\n"


def format_ner_examples(examples: List[Tuple[str, List[str]]]) -> str:
    """Format NER examples for prompt."""
    if not examples:
        return ""
    lines = []
    for text, entities in examples:
        ent_str = ", ".join(entities) if entities else "NONE"
        lines.append(f"Text: {text[:300]}...\nOrganizations: {ent_str}\n")
    return "\n".join(lines) + "\n"


# =============================================================================
# HELPER: Convert NER item to text
# =============================================================================

def get_text_from_item(item: Dict) -> str:
    """Extract text from NER/CLF item, handling different formats."""
    if isinstance(item, dict):
        # Try different keys
        if 'text' in item:
            return item['text']
        elif 'clean_text' in item:
            return item['clean_text']
        elif 'tokens' in item:
            # NER data has tokens as list
            return ' '.join(item['tokens'])
        else:
            # Return first string value found
            for v in item.values():
                if isinstance(v, str):
                    return v
            return str(item)
    elif isinstance(item, (list, tuple)):
        return item[0] if item else ""
    else:
        return str(item)


def get_entities_from_item(item: Dict) -> List[str]:
    """Extract entities from NER item as list of strings.
    Uses 'labels' field (BIO tags) like enhanced_icl script.
    """
    if isinstance(item, dict):
        # Primary method: use 'labels' field (BIO tags) - this is what enhanced_icl uses
        if 'labels' in item and 'tokens' in item:
            entities = []
            tokens = item['tokens']
            labels = item['labels']
            current_entity = []
            for tok, lab in zip(tokens, labels):
                if lab.startswith('B-'):
                    if current_entity:
                        entities.append(' '.join(current_entity))
                    current_entity = [tok]
                elif lab.startswith('I-') and current_entity:
                    current_entity.append(tok)
                else:
                    if current_entity:
                        entities.append(' '.join(current_entity))
                        current_entity = []
            if current_entity:
                entities.append(' '.join(current_entity))
            return entities
        # Fallback: 'ner_tags' field
        elif 'ner_tags' in item and 'tokens' in item:
            entities = []
            tokens = item['tokens']
            tags = item['ner_tags']
            current_entity = []
            for token, tag in zip(tokens, tags):
                tag_str = str(tag) if not isinstance(tag, str) else tag
                if tag_str.startswith('B-') or tag_str == '1':
                    if current_entity:
                        entities.append(' '.join(current_entity))
                    current_entity = [str(token)]
                elif tag_str.startswith('I-') or tag_str == '2':
                    if current_entity:
                        current_entity.append(str(token))
                else:
                    if current_entity:
                        entities.append(' '.join(current_entity))
                        current_entity = []
            if current_entity:
                entities.append(' '.join(current_entity))
            return entities
        # Last fallback: 'entities' field
        elif 'entities' in item:
            entities = item['entities']
            result = []
            for ent in entities:
                if isinstance(ent, str):
                    result.append(ent)
                elif isinstance(ent, dict):
                    if 'text' in ent:
                        result.append(ent['text'])
                    elif 'entity' in ent:
                        result.append(ent['entity'])
            return result
    return []


# =============================================================================
# DEMO SELECTION - FIXED
# =============================================================================

class DemoSelector:
    """Demonstration selection: Random vs TopK embedding."""
    
    def __init__(self):
        self.encoder = None
        self.train_embeddings = None
        self.train_data = None
    
    def load_encoder(self):
        """Load sentence transformer for TopK selection."""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                log(f"Loading embedding model: {Config.EMBEDDING_MODEL}...")
                self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
                log("✅ Embedding model loaded")
            except ImportError:
                log("⚠️ sentence-transformers not installed. TopK will fall back to random.")
                self.encoder = None
    
    def fit(self, train_data: List[Dict]):
        """Pre-compute embeddings for training data."""
        self.train_data = train_data
        
        if self.encoder is not None:
            log(f"Computing embeddings for {len(train_data)} training samples...")
            # Use helper function to extract text
            texts = [get_text_from_item(item) for item in train_data]
            self.train_embeddings = self.encoder.encode(texts, show_progress_bar=True)
            log(f"✅ Computed {len(self.train_embeddings)} embeddings")
    
    def select_random(self, k: int, seed: int) -> List:
        """Random selection."""
        if k == 0 or not self.train_data:
            return []
        random.seed(seed)
        return random.sample(self.train_data, min(k, len(self.train_data)))
    
    def select_topk(self, query_text: str, k: int) -> List:
        """TopK embedding-based selection."""
        if k == 0 or not self.train_data:
            return []
        
        if self.encoder is None or self.train_embeddings is None:
            return self.select_random(k, Config.SEED_BASE)
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_emb = self.encoder.encode([query_text])[0]
        similarities = cosine_similarity([query_emb], self.train_embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        return [self.train_data[i] for i in top_k_indices]


# =============================================================================
# LLM LOADING AND INFERENCE
# =============================================================================

def load_llm(model_name: str):
    """Load LLM with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    log(f"Loading {model_name} with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log(f"✅ Model loaded: {model_name}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate response from LLM (matching enhanced_icl settings)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            do_sample=False,  # Greedy decoding like enhanced_icl
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # Take only first line (like enhanced_icl)
    return response.strip().split('\n')[0]


def unload_model(model, tokenizer):
    """Unload model and free memory properly."""
    import gc
    
    # Delete model and tokenizer
    del model
    del tokenizer
    
    # Force garbage collection
    gc.collect()
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Wait for cleanup
    time.sleep(5)
    
    # Additional cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log("✅ Model unloaded, GPU memory cleared")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_clf_data(n_groups: int) -> Dict:
    """Load classification data."""
    path = Config.CLF_DATA_PATHS.get(n_groups)
    if path is None or not path.exists():
        log(f"⚠️ Classification data for {n_groups} groups not found: {path}")
        return None
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    log(f"✅ Loaded CLF data ({n_groups} groups): {len(data)} samples")
    return data


def load_ner_data() -> Dict:
    """Load NER data."""
    if not Config.NER_DATA_PATH.exists():
        log(f"⚠️ NER data not found: {Config.NER_DATA_PATH}")
        return None
    
    with open(Config.NER_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    log(f"✅ Loaded NER data")
    return data


def prepare_clf_train_test(data, n_groups: int, test_subset: int = None):
    """Prepare classification train/test split."""
    from sklearn.model_selection import train_test_split
    
    # DataFrame format
    train_data, test_data = train_test_split(
        data, test_size=0.15, stratify=data['canonical_group'], random_state=42
    )
    
    labels = sorted(data['canonical_group'].unique())
    
    # Convert to list of dicts
    train_list = [{'text': row['clean_text'], 'label': row['canonical_group']} 
                  for _, row in train_data.iterrows()]
    test_list = [{'text': row['clean_text'], 'label': row['canonical_group']} 
                 for _, row in test_data.iterrows()]
    
    # Subset test for efficiency
    if test_subset and len(test_list) > test_subset:
        random.seed(42)
        test_list = random.sample(test_list, test_subset)
    
    return train_list, test_list, labels


def prepare_ner_train_test(data, test_subset: int = None):
    """Prepare NER train/test split - filter train to only include samples with entities."""
    train_data_raw = data.get('train', [])
    test_data = data.get('test', [])
    
    # Filter train data to only include samples with entities (like enhanced_icl)
    train_data = []
    for s in train_data_raw:
        # Check if sample has any non-O labels
        labels = s.get('labels', [])
        if any(l != 'O' for l in labels):
            # Add 'text' field for embedding computation
            train_data.append({
                'tokens': s['tokens'],
                'labels': s['labels'],
                'text': ' '.join(s['tokens'])
            })
    
    log(f"Train samples with entities: {len(train_data)} (from {len(train_data_raw)} total)")
    
    # Also add 'text' field to test data
    test_data_processed = []
    for s in test_data:
        test_data_processed.append({
            'tokens': s.get('tokens', []),
            'labels': s.get('labels', []),
            'text': ' '.join(s.get('tokens', []))
        })
    
    if test_subset and len(test_data_processed) > test_subset:
        random.seed(42)
        test_data_processed = random.sample(test_data_processed, test_subset)
    
    return train_data, test_data_processed


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_clf(model, tokenizer, test_data, labels, selector, 
                 selection_method: str, k_shots: int, run_seed: int) -> Dict:
    """Evaluate classification."""
    from sklearn.metrics import accuracy_score, f1_score
    
    label_list_str = ", ".join(labels)
    predictions = []
    ground_truth = []
    
    for idx, item in enumerate(tqdm(test_data, desc=f"CLF {selection_method} {k_shots}-shot")):
        text = get_text_from_item(item)
        true_label = item.get('label', item.get('canonical_group', ''))
        
        # Select examples
        if selection_method == 'random':
            examples = selector.select_random(k_shots, run_seed + idx)
        else:
            examples = selector.select_topk(text, k_shots)
        
        # Format examples
        if examples:
            formatted_examples = [(get_text_from_item(ex), ex.get('label', ex.get('canonical_group', ''))) 
                                  for ex in examples]
            examples_str = format_clf_examples(formatted_examples)
        else:
            examples_str = ""
        
        # Create prompt
        prompt = CLF_TEMPLATE.format(
            label_list=label_list_str,
            examples=examples_str,
            text=text[:500]
        )
        
        # Generate prediction
        response = generate_response(model, tokenizer, prompt)
        
        # Parse prediction
        pred_label = None
        response_lower = response.lower()
        for label in labels:
            if label.lower() in response_lower:
                pred_label = label
                break
        
        if pred_label is None:
            pred_label = labels[0]
        
        predictions.append(pred_label)
        ground_truth.append(true_label)
    
    accuracy = accuracy_score(ground_truth, predictions)
    f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'n_samples': len(predictions)
    }


def evaluate_ner(model, tokenizer, test_data, selector,
                 selection_method: str, k_shots: int, run_seed: int) -> Dict:
    """Evaluate NER with case-insensitive matching (like enhanced_icl)."""
    
    all_pred_entities = []
    all_true_entities = []
    
    for idx, item in enumerate(tqdm(test_data, desc=f"NER {selection_method} {k_shots}-shot")):
        text = get_text_from_item(item)
        true_entities = get_entities_from_item(item)
        
        # Select examples
        if selection_method == 'random':
            examples = selector.select_random(k_shots, run_seed + idx)
        else:
            examples = selector.select_topk(text, k_shots)
        
        # Format examples (like enhanced_icl)
        ex_str = ""
        for ex in examples:
            ex_text = get_text_from_item(ex)
            ex_ents = get_entities_from_item(ex)
            ents_str = ", ".join(ex_ents) if ex_ents else "NONE"
            ex_str += f"Text: {ex_text[:200]}\nOrganizations: {ents_str}\n\n"
        
        # Create prompt
        prompt = NER_TEMPLATE.format(examples=ex_str, text=text[:400])
        
        # Generate prediction
        response = generate_response(model, tokenizer, prompt)
        
        # Parse entities (like enhanced_icl)
        if response.lower().strip() in ['none', 'no', '']:
            pred_entities = []
        else:
            pred_entities = [e.strip() for e in response.replace(';', ',').split(',')
                           if e.strip() and e.lower() not in ['none', 'no']]
        
        # Store with lowercase for case-insensitive matching
        all_pred_entities.append([p.lower() for p in pred_entities])
        all_true_entities.append([g.lower() for g in true_entities])
    
    # Calculate entity-level metrics with case-insensitive matching
    tp = fp = fn = 0
    for pred, gold in zip(all_pred_entities, all_true_entities):
        pred_set = set(pred)
        gold_set = set(gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_samples': len(test_data)
    }


# =============================================================================
# EXPERIMENT 1: Selection Comparison (Continue from where we left off)
# =============================================================================

def run_selection_comparison():
    """Run Random vs TopK comparison - CONTINUE from where we left off."""
    log("\n" + "=" * 70)
    log("EXPERIMENT 1: Random vs TopK Selection Comparison (CONTINUED)")
    log("=" * 70)
    
    results = {}
    
    # Initialize selector
    selector = DemoSelector()
    selector.load_encoder()
    
    # DeepSeek CLF already done, load those results
    results['deepseek-r1'] = {
        'clf': {
            'random_0shot': {'accuracy_mean': 0.5111, 'accuracy_std': 0.0113, 'n_runs': 3},
            'random_5shot': {'accuracy_mean': 0.7111, 'accuracy_std': 0.0191, 'n_runs': 3},
            'random_10shot': {'accuracy_mean': 0.7267, 'accuracy_std': 0.0163, 'n_runs': 3},
            'topk_0shot': {'accuracy_mean': 0.5089, 'accuracy_std': 0.0137, 'n_runs': 3},
            'topk_5shot': {'accuracy_mean': 0.9244, 'accuracy_std': 0.0031, 'n_runs': 3},
            'topk_10shot': {'accuracy_mean': 0.9244, 'accuracy_std': 0.0031, 'n_runs': 3},
        },
        'ner': {}
    }
    log("✅ Loaded previous DeepSeek-R1 CLF results")
    
    # === PART 1: DeepSeek NER (what failed before) ===
    log("\n" + "=" * 50)
    log("Part 1: DeepSeek-R1 NER")
    log("=" * 50)
    
    model, tokenizer = load_llm(Config.LLM_MODELS['deepseek-r1'])
    
    ner_data = load_ner_data()
    if ner_data is not None:
        train_data, test_data = prepare_ner_train_test(ner_data, Config.NER_TEST_SUBSET)
        
        # Fit selector with NER data
        selector.fit(train_data)
        
        for selection in Config.SELECTION_METHODS:
            for k_shots in Config.SHOT_CONFIGS:
                config_key = f"{selection}_{k_shots}shot"
                log(f"\nRunning NER: {config_key}")
                
                run_results = []
                for run in range(Config.NUM_RUNS):
                    run_seed = Config.SEED_BASE + run * 100
                    metrics = evaluate_ner(
                        model, tokenizer, test_data, selector,
                        selection, k_shots, run_seed
                    )
                    run_results.append(metrics)
                    log(f"  Run {run+1}: F1={metrics['f1']:.4f}")
                
                results['deepseek-r1']['ner'][config_key] = {
                    'precision_mean': np.mean([r['precision'] for r in run_results]),
                    'recall_mean': np.mean([r['recall'] for r in run_results]),
                    'f1_mean': np.mean([r['f1'] for r in run_results]),
                    'f1_std': np.std([r['f1'] for r in run_results]),
                    'n_runs': Config.NUM_RUNS
                }
                
                log(f"  Mean: F1={results['deepseek-r1']['ner'][config_key]['f1_mean']:.4f} "
                    f"± {results['deepseek-r1']['ner'][config_key]['f1_std']:.4f}")
    
    unload_model(model, tokenizer)
    
    # Save intermediate
    save_path = Config.RESULTS_DIR / "selection_comparison_deepseek_complete.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n✅ DeepSeek results saved: {save_path}")
    
    # === PART 2: Qwen2.5 (CLF + NER) ===
    log("\n" + "=" * 50)
    log("Part 2: Qwen2.5 (CLF + NER)")
    log("=" * 50)
    
    results['qwen2.5'] = {'clf': {}, 'ner': {}}
    
    model, tokenizer = load_llm(Config.LLM_MODELS['qwen2.5'])
    
    # CLF
    log("\n--- Qwen2.5 Classification (10 groups) ---")
    clf_data = load_clf_data(10)
    if clf_data is not None:
        train_list, test_list, labels = prepare_clf_train_test(clf_data, 10, Config.CLF_TEST_SUBSET)
        selector.fit(train_list)
        
        for selection in Config.SELECTION_METHODS:
            for k_shots in Config.SHOT_CONFIGS:
                config_key = f"{selection}_{k_shots}shot"
                log(f"\nRunning CLF: {config_key}")
                
                run_results = []
                for run in range(Config.NUM_RUNS):
                    run_seed = Config.SEED_BASE + run * 100
                    metrics = evaluate_clf(
                        model, tokenizer, test_list, labels, selector,
                        selection, k_shots, run_seed
                    )
                    run_results.append(metrics)
                    log(f"  Run {run+1}: Acc={metrics['accuracy']:.4f}")
                
                results['qwen2.5']['clf'][config_key] = {
                    'accuracy_mean': np.mean([r['accuracy'] for r in run_results]),
                    'accuracy_std': np.std([r['accuracy'] for r in run_results]),
                    'f1_mean': np.mean([r['f1_macro'] for r in run_results]),
                    'f1_std': np.std([r['f1_macro'] for r in run_results]),
                    'n_runs': Config.NUM_RUNS
                }
                
                log(f"  Mean: Acc={results['qwen2.5']['clf'][config_key]['accuracy_mean']:.4f}")
    
    # NER
    log("\n--- Qwen2.5 NER ---")
    if ner_data is not None:
        train_data, test_data = prepare_ner_train_test(ner_data, Config.NER_TEST_SUBSET)
        selector.fit(train_data)
        
        for selection in Config.SELECTION_METHODS:
            for k_shots in Config.SHOT_CONFIGS:
                config_key = f"{selection}_{k_shots}shot"
                log(f"\nRunning NER: {config_key}")
                
                run_results = []
                for run in range(Config.NUM_RUNS):
                    run_seed = Config.SEED_BASE + run * 100
                    metrics = evaluate_ner(
                        model, tokenizer, test_data, selector,
                        selection, k_shots, run_seed
                    )
                    run_results.append(metrics)
                    log(f"  Run {run+1}: F1={metrics['f1']:.4f}")
                
                results['qwen2.5']['ner'][config_key] = {
                    'precision_mean': np.mean([r['precision'] for r in run_results]),
                    'recall_mean': np.mean([r['recall'] for r in run_results]),
                    'f1_mean': np.mean([r['f1'] for r in run_results]),
                    'f1_std': np.std([r['f1'] for r in run_results]),
                    'n_runs': Config.NUM_RUNS
                }
                
                log(f"  Mean: F1={results['qwen2.5']['ner'][config_key]['f1_mean']:.4f}")
    
    unload_model(model, tokenizer)
    
    # Save final selection comparison
    save_path = Config.RESULTS_DIR / "selection_comparison_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n✅ Selection comparison results saved: {save_path}")
    
    return results


# =============================================================================
# EXPERIMENT 2: Missing Classification Scales (20, 50, 100 groups)
# =============================================================================

def run_missing_scales():
    """Run missing classification experiments for 20, 50, 100 groups."""
    log("\n" + "=" * 70)
    log("EXPERIMENT 2: Missing Classification Scales (20, 50, 100 groups)")
    log("=" * 70)
    
    results = {}
    
    selector = DemoSelector()
    selector.load_encoder()
    
    missing_scales = [20, 50, 100]
    
    for model_key, model_name in Config.LLM_MODELS.items():
        log(f"\n{'='*50}")
        log(f"Model: {model_key}")
        log(f"{'='*50}")
        
        results[model_key] = {}
        
        model, tokenizer = load_llm(model_name)
        
        for n_groups in missing_scales:
            log(f"\n--- Classification ({n_groups} groups) ---")
            
            clf_data = load_clf_data(n_groups)
            if clf_data is None:
                log(f"⚠️ Skipping {n_groups} groups - data not found")
                continue
            
            train_list, test_list, labels = prepare_clf_train_test(
                clf_data, n_groups, Config.CLF_TEST_SUBSET
            )
            
            selector.fit(train_list)
            
            # Run 10-shot TopK (best configuration)
            config_key = f"{n_groups}g_10shot_topk"
            log(f"\nRunning: {config_key}")
            
            run_results = []
            for run in range(Config.NUM_RUNS):
                run_seed = Config.SEED_BASE + run * 100
                metrics = evaluate_clf(
                    model, tokenizer, test_list, labels, selector,
                    'topk', 10, run_seed
                )
                run_results.append(metrics)
                log(f"  Run {run+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
            
            results[model_key][config_key] = {
                'n_groups': n_groups,
                'accuracy_mean': np.mean([r['accuracy'] for r in run_results]),
                'accuracy_std': np.std([r['accuracy'] for r in run_results]),
                'f1_mean': np.mean([r['f1_macro'] for r in run_results]),
                'f1_std': np.std([r['f1_macro'] for r in run_results]),
                'n_runs': Config.NUM_RUNS,
                'selection': 'topk',
                'k_shots': 10
            }
            
            log(f"  Mean: Acc={results[model_key][config_key]['accuracy_mean']:.4f}")
        
        unload_model(model, tokenizer)
        
        # Save intermediate
        save_path = Config.RESULTS_DIR / "missing_scales_results.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"\n✅ Intermediate results saved: {save_path}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("COMPLETE LLM EVALUATION - FIXED VERSION")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {Config.DEVICE}")
    log("=" * 70)
    
    all_results = {}
    
    try:
        # Experiment 1: Selection comparison (continue from where we left off)
        selection_results = run_selection_comparison()
        all_results['selection_comparison'] = selection_results
        
        # Experiment 2: Missing scales
        scale_results = run_missing_scales()
        all_results['missing_scales'] = scale_results
        
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
    
    # Save final results
    final_path = Config.RESULTS_DIR / "complete_llm_results.json"
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log("\n" + "=" * 70)
    log("COMPLETE LLM EVALUATION - Finished")
    log(f"Results saved to: {final_path}")
    log("=" * 70)
    
    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    
    if 'selection_comparison' in all_results:
        log("\n📊 Selection Comparison (10 groups):")
        for model_key, model_results in all_results['selection_comparison'].items():
            log(f"\n  {model_key}:")
            if 'clf' in model_results:
                for config, metrics in sorted(model_results['clf'].items()):
                    log(f"    CLF {config}: {metrics['accuracy_mean']*100:.2f}%")
            if 'ner' in model_results:
                for config, metrics in sorted(model_results['ner'].items()):
                    log(f"    NER {config}: {metrics['f1_mean']*100:.2f}%")
    
    if 'missing_scales' in all_results:
        log("\n📊 Missing Scales (10-shot TopK):")
        for model_key, model_results in all_results['missing_scales'].items():
            log(f"\n  {model_key}:")
            for config, metrics in sorted(model_results.items()):
                log(f"    {config}: {metrics['accuracy_mean']*100:.2f}%")


if __name__ == "__main__":
    main()
