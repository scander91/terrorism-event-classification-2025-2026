#!/usr/bin/env python3
"""
Enhanced ICL Evaluation - Based on Paper Methodology
- 3 Prompt Templates
- Random + Embedding Top-k Selection  
- Multi-run (5 runs with mean±std)
- Shots: 0, 5, 8, 10
"""

import os, gc, json, pickle, random, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - VERIFIED PATHS
# =============================================================================
BASE_DIR = Path(os.path.expanduser("~/TerrorismNER_Research"))
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "enhanced_icl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Models
LLM_MODELS = {
    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
}

# Settings
SHOT_CONFIGS = [0, 5, 8, 10]
NUM_RUNS = 5
SEED_BASE = 42
MAX_NEW_TOKENS = 64
CLF_TEST_SUBSET = 500
NER_TEST_SUBSET = 300

# =============================================================================
# PROMPT TEMPLATES (3 different styles)
# =============================================================================
CLF_TEMPLATES = {
    'template_1': """Classify the text into one terrorist group from: {labels}

{examples}Text: {text}
Group:""",

    'template_2': """Based on the attack description, determine which terrorist organization is responsible.
Options: {labels}

{examples}Description: {text}
Organization:""",

    'template_3': """Task: Attack Attribution
Groups: {labels}

{examples}Input: {text}
Output:"""
}

NER_TEMPLATES = {
    'template_1': """Extract terrorist organization names from the text.
Output names separated by commas. If none, output NONE.

{examples}Text: {text}
Organizations:""",

    'template_2': """Identify all terrorist groups mentioned below.

{examples}Text: {text}
Groups found:""",

    'template_3': """NER Task - Extract terrorist organization names.

{examples}Input: {text}
Entities:"""
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_results(data, filename):
    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    log(f"Saved: {path}")

# =============================================================================
# LLM LOADING
# =============================================================================
def load_llm(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    log(f"Loading {model_name}...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log(f"Model loaded!")
    return model, tokenizer

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip().split('\n')[0]

# =============================================================================
# DEMO SELECTION
# =============================================================================
class DemoSelector:
    def __init__(self, train_data, text_key='text', use_embeddings=False):
        self.train_data = train_data
        self.text_key = text_key
        self.use_embeddings = use_embeddings
        self.embeddings = None
        
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                log("Loading embedding model...")
                encoder = SentenceTransformer('all-MiniLM-L6-v2')
                texts = [d[text_key] for d in train_data]
                self.embeddings = encoder.encode(texts, show_progress_bar=True)
                self.encoder = encoder
                log("Embeddings computed!")
            except Exception as e:
                log(f"Embedding error: {e}, using random only")
                self.use_embeddings = False
    
    def select_random(self, k, seed):
        if k == 0: return []
        random.seed(seed)
        return random.sample(self.train_data, min(k, len(self.train_data)))
    
    def select_topk(self, query_text, k):
        if k == 0: return []
        if not self.use_embeddings:
            return self.select_random(k, SEED_BASE)
        
        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = self.encoder.encode([query_text])[0]
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        top_idx = sims.argsort()[-k:][::-1]
        return [self.train_data[i] for i in top_idx]

# =============================================================================
# CLASSIFICATION EVALUATION
# =============================================================================
def run_classification():
    log("=" * 60)
    log("CLASSIFICATION EXPERIMENTS")
    log("=" * 60)
    
    # Load data
    clf_data = pickle.load(open(CHECKPOINT_DIR / "classification_data_10.pkl", "rb"))
    log(f"Loaded {len(clf_data)} samples")
    
    # Split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(clf_data, test_size=0.15, 
                                          stratify=clf_data['canonical_group'], 
                                          random_state=SEED_BASE)
    
    test_df = test_df.sample(min(CLF_TEST_SUBSET, len(test_df)), random_state=42)
    log(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Prepare
    groups = sorted(clf_data['canonical_group'].unique())
    label2id = {g: i for i, g in enumerate(groups)}
    labels_str = ", ".join(groups)
    
    train_list = [{'text': row['clean_text'], 'label': row['canonical_group']} 
                  for _, row in train_df.iterrows()]
    
    # Selectors
    random_selector = DemoSelector(train_list, 'text', use_embeddings=False)
    topk_selector = DemoSelector(train_list, 'text', use_embeddings=True)
    
    all_results = {}
    
    for model_key, model_path in LLM_MODELS.items():
        log(f"\n{'='*50}")
        log(f"Model: {model_key}")
        log(f"{'='*50}")
        
        try:
            model, tokenizer = load_llm(model_path)
            
            for tmpl_name, tmpl in CLF_TEMPLATES.items():
                for k_shots in SHOT_CONFIGS:
                    for sel_name, selector in [('random', random_selector), ('topk', topk_selector)]:
                        
                        config = f"{model_key}_{tmpl_name}_{k_shots}shot_{sel_name}"
                        log(f"Running: {config}")
                        
                        run_accs = []
                        for run in range(NUM_RUNS):
                            seed = SEED_BASE + run
                            set_seed(seed)
                            
                            correct = 0
                            for _, row in tqdm(test_df.iterrows(), total=len(test_df), 
                                              desc=f"Run {run+1}/{NUM_RUNS}"):
                                text = row['clean_text']
                                gold = row['canonical_group']
                                
                                # Get demos
                                if sel_name == 'random':
                                    demos = selector.select_random(k_shots, seed)
                                else:
                                    demos = selector.select_topk(text, k_shots)
                                
                                # Format examples
                                ex_str = ""
                                for d in demos:
                                    ex_str += f"Text: {d['text'][:200]}\nGroup: {d['label']}\n\n"
                                
                                # Build prompt
                                prompt = tmpl.format(labels=labels_str, examples=ex_str, text=text[:400])
                                
                                # Predict
                                response = generate(model, tokenizer, prompt)
                                
                                # Match
                                pred = None
                                for g in groups:
                                    if g.lower() in response.lower():
                                        pred = g
                                        break
                                
                                if pred == gold:
                                    correct += 1
                            
                            acc = correct / len(test_df)
                            run_accs.append(acc)
                            log(f"  Run {run+1}: {acc*100:.2f}%")
                        
                        all_results[config] = {
                            'accuracy_mean': round(np.mean(run_accs), 4),
                            'accuracy_std': round(np.std(run_accs), 4),
                            'runs': [round(a, 4) for a in run_accs]
                        }
                        
                        save_results(all_results, 'clf_results_progress.json')
            
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            log(f"ERROR: {e}")
            all_results[f'{model_key}_error'] = str(e)
    
    save_results(all_results, 'clf_results_final.json')
    return all_results

# =============================================================================
# NER EVALUATION
# =============================================================================
def extract_entities(sample):
    entities = []
    current = []
    for tok, lab in zip(sample['tokens'], sample['labels']):
        if lab.startswith('B-'):
            if current: entities.append(' '.join(current))
            current = [tok]
        elif lab.startswith('I-') and current:
            current.append(tok)
        else:
            if current: entities.append(' '.join(current))
            current = []
    if current: entities.append(' '.join(current))
    return entities

def run_ner():
    log("=" * 60)
    log("NER EXPERIMENTS")
    log("=" * 60)
    
    # Load data
    ner_data = pickle.load(open(CHECKPOINT_DIR / "ner_data_basic.pkl", "rb"))
    train_data = ner_data['train']
    test_data = ner_data['test']
    
    log(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Subset
    random.seed(SEED_BASE)
    test_subset = random.sample(test_data, min(NER_TEST_SUBSET, len(test_data)))
    log(f"Using {len(test_subset)} test samples")
    
    # Prepare train with text
    train_with_text = []
    for s in train_data:
        if any(l != 'O' for l in s['labels']):  # Has entities
            train_with_text.append({
                'tokens': s['tokens'],
                'labels': s['labels'],
                'text': ' '.join(s['tokens'])
            })
    log(f"Train samples with entities: {len(train_with_text)}")
    
    # Selectors
    random_selector = DemoSelector(train_with_text, 'text', use_embeddings=False)
    topk_selector = DemoSelector(train_with_text, 'text', use_embeddings=True)
    
    all_results = {}
    
    for model_key, model_path in LLM_MODELS.items():
        log(f"\n{'='*50}")
        log(f"Model: {model_key}")
        log(f"{'='*50}")
        
        try:
            model, tokenizer = load_llm(model_path)
            
            for tmpl_name, tmpl in NER_TEMPLATES.items():
                for k_shots in SHOT_CONFIGS:
                    for sel_name, selector in [('random', random_selector), ('topk', topk_selector)]:
                        
                        config = f"{model_key}_{tmpl_name}_{k_shots}shot_{sel_name}"
                        log(f"Running: {config}")
                        
                        run_f1s = []
                        for run in range(NUM_RUNS):
                            seed = SEED_BASE + run
                            set_seed(seed)
                            
                            all_pred, all_gold = [], []
                            
                            for sample in tqdm(test_subset, desc=f"Run {run+1}/{NUM_RUNS}"):
                                text = ' '.join(sample['tokens'])
                                gold = extract_entities(sample)
                                all_gold.append(gold)
                                
                                # Get demos
                                if sel_name == 'random':
                                    demos = selector.select_random(k_shots, seed)
                                else:
                                    demos = selector.select_topk(text, k_shots)
                                
                                # Format examples
                                ex_str = ""
                                for d in demos:
                                    d_ents = extract_entities(d)
                                    ents_str = ", ".join(d_ents) if d_ents else "NONE"
                                    ex_str += f"Text: {d['text'][:200]}\nOrganizations: {ents_str}\n\n"
                                
                                # Build prompt
                                prompt = tmpl.format(examples=ex_str, text=text[:400])
                                
                                # Predict
                                response = generate(model, tokenizer, prompt)
                                
                                # Parse
                                if response.lower().strip() in ['none', 'no', '']:
                                    pred = []
                                else:
                                    pred = [e.strip() for e in response.replace(';', ',').split(',') 
                                           if e.strip() and e.lower() not in ['none', 'no']]
                                all_pred.append(pred)
                            
                            # Calculate F1
                            tp, fp, fn = 0, 0, 0
                            for pred, gold in zip(all_pred, all_gold):
                                pred_set = set(p.lower() for p in pred)
                                gold_set = set(g.lower() for g in gold)
                                tp += len(pred_set & gold_set)
                                fp += len(pred_set - gold_set)
                                fn += len(gold_set - pred_set)
                            
                            p = tp / (tp + fp) if (tp + fp) > 0 else 0
                            r = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                            
                            run_f1s.append(f1)
                            log(f"  Run {run+1}: F1={f1*100:.2f}%")
                        
                        all_results[config] = {
                            'f1_mean': round(np.mean(run_f1s), 4),
                            'f1_std': round(np.std(run_f1s), 4),
                            'runs': [round(f, 4) for f in run_f1s]
                        }
                        
                        save_results(all_results, 'ner_results_progress.json')
            
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            log(f"ERROR: {e}")
            all_results[f'{model_key}_error'] = str(e)
    
    save_results(all_results, 'ner_results_final.json')
    return all_results

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    log("=" * 60)
    log("ENHANCED ICL EVALUATION")
    log(f"Started: {datetime.now()}")
    log("=" * 60)
    
    start = time.time()
    
    # Install sentence-transformers if needed
    try:
        import sentence_transformers
    except:
        log("Installing sentence-transformers...")
        os.system("pip install sentence-transformers --quiet")
    
    # Run experiments
    clf_results = run_classification()
    ner_results = run_ner()
    
    # Summary
    elapsed = (time.time() - start) / 3600
    log("=" * 60)
    log(f"ALL DONE! Time: {elapsed:.2f} hours")
    log(f"Results saved to: {RESULTS_DIR}")
    log("=" * 60)
