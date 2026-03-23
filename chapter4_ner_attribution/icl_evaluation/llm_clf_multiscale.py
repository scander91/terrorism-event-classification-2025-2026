#!/usr/bin/env python3
"""
LLM Classification on 50, 100 Groups (RETRY)
With proper GPU memory clearing between runs
"""

import sys
import os
import gc
import json
import random
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from utils import *

def clear_all_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    time.sleep(2)  # Give GPU time to release memory
    print("✅ GPU memory cleared aggressively")

def load_llm_fresh(model_name):
    """Load LLM with fresh GPU state."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    # Clear memory before loading
    clear_all_memory()
    
    print(f"Loading {model_name} (4-bit)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    if 'deepseek' in model_name.lower():
        model_id = "deepseek-ai/deepseek-llm-7b-chat"
    else:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto", 
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Added for better memory management
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Loaded {model_name}")
    return model, tokenizer

def create_prompt(text, examples, groups, n_shots=10):
    ex_str = ""
    for t, l in examples[:n_shots]:
        ex_str += f"Text: {t[:200]}\nGroup: {l}\n\n"
    
    # Truncate group list if too long
    g_str = ", ".join(groups[:25]) if len(groups) > 25 else ", ".join(groups)
    if len(groups) > 25:
        g_str += f"... ({len(groups)} total groups)"
    
    return f"""Classify the text into one terrorist group from: {g_str}

Examples:
{ex_str}Text: {text[:350]}
Group:"""

def run_single_experiment(model_name, num_groups, max_samples=150):
    """Run a single LLM experiment with fresh GPU state."""
    
    set_seed(42)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - {num_groups} Groups")
    print(f"{'='*60}")
    
    # Load data
    data = load_pickle(CLF_DATA_PATHS[num_groups])
    groups = sorted(data['canonical_group'].unique())
    
    # Split
    _, test = train_test_split(data, test_size=0.15, stratify=data['canonical_group'], random_state=42)
    train_data = data[~data.index.isin(test.index)]
    
    if len(test) > max_samples:
        test = test.sample(max_samples, random_state=42)
    
    print(f"Groups: {len(groups)}, Test: {len(test)}")
    
    # Examples
    examples = list(zip(train_data['clean_text'].tolist(), train_data['canonical_group'].tolist()))
    random.shuffle(examples)
    
    # Load model with fresh GPU
    model, tokenizer = load_llm_fresh(model_name)
    
    preds = []
    trues = []
    times = []
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        prompt = create_prompt(row['clean_text'], examples, groups)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, temperature=0.1, 
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        times.append((time.time() - t0) * 1000)
        
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Extract prediction
        if "Group:" in gen:
            pred = gen.split("Group:")[-1].strip().split("\n")[0].strip()
        else:
            pred = gen.strip().split("\n")[-1].strip()
        
        # Match to group
        matched = None
        for g in groups:
            if g.lower() in pred.lower() or pred.lower() in g.lower():
                matched = g
                break
        if not matched:
            matched = groups[0]
        
        preds.append(matched)
        trues.append(row['canonical_group'])
    
    # Metrics
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    
    results = {
        'model': model_name, 'num_groups': num_groups, 'num_shots': 10,
        'test_samples': len(test), 'accuracy': float(acc), 'macro_f1': float(f1),
        'avg_inference_time_ms': float(np.mean(times)),
        'timestamp': datetime.now().isoformat()
    }
    
    save_json(results, RESULTS_DIR / f'llm_{model_name}_{num_groups}g_10shot.json')
    
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Macro F1: {f1*100:.2f}%")
    print(f"  Avg Time: {np.mean(times):.0f}ms")
    
    # IMPORTANT: Delete model and clear memory
    del model, tokenizer
    clear_all_memory()
    
    return results

def main():
    print("\n" + "="*70)
    print("LLM CLASSIFICATION - RETRY 50 & 100 GROUPS")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Clear GPU before starting
    clear_all_memory()
    
    all_results = {}
    
    # Run ONE experiment at a time with full memory clearing
    experiments = [
        ('qwen', 50),
        ('deepseek', 50),
        ('qwen', 100),
        ('deepseek', 100),
    ]
    
    for model_name, num_groups in experiments:
        try:
            print(f"\n>>> Starting {model_name} on {num_groups} groups...")
            res = run_single_experiment(model_name, num_groups)
            all_results[f'{model_name}_{num_groups}'] = res
            
            # Save after each experiment
            save_json(all_results, RESULTS_DIR / 'llm_clf_50_100_results.json')
            
            # Extra memory clearing between experiments
            clear_all_memory()
            time.sleep(5)  # Extra wait time
            
        except Exception as e:
            print(f"❌ ERROR {model_name} {num_groups}g: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'{model_name}_{num_groups}'] = {'error': str(e)}
            clear_all_memory()
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for k, v in all_results.items():
        if 'error' not in v:
            print(f"{v['model']:<10} {v['num_groups']:<5}g: {v['accuracy']*100:.2f}% acc, {v['macro_f1']*100:.2f}% F1")
        else:
            print(f"{k}: FAILED - {v['error'][:50]}")

if __name__ == "__main__":
    main()
