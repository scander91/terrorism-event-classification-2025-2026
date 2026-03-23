#!/usr/bin/env python3
"""
UTILITY FUNCTIONS FOR PHASE 2 EXPERIMENTS
"""

import os
import json
import pickle
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to {seed}")

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved pickle: {path}")

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    data = convert(data)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved JSON: {path}")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def log_experiment_start(name):
    print("\n" + "="*70)
    print(f"EXPERIMENT: {name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def log_experiment_end(name, results=None):
    print("\n" + "="*70)
    print(f"COMPLETED: {name}")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if results:
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    print("="*70 + "\n")

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU")
    return device

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU memory cleared")

def calculate_classification_metrics(y_true, y_pred, y_prob=None, num_classes=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = [float(f) for f in per_class_f1]
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    if y_prob is not None and num_classes is not None:
        try:
            if num_classes == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except Exception as e:
            print(f"⚠️ Could not calculate ROC AUC: {e}")
    return metrics

def verify_data_files(config):
    print("\n" + "="*50)
    print("DATA VERIFICATION")
    print("="*50)
    all_good = True
    if config.NER_DATA_PATH.exists():
        data = load_pickle(config.NER_DATA_PATH)
        print(f"✅ NER Data: {config.NER_DATA_PATH}")
        print(f"   Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")
    else:
        print(f"❌ NER Data: {config.NER_DATA_PATH} NOT FOUND")
        all_good = False
    for groups, path in config.CLF_DATA_PATHS.items():
        if path.exists():
            data = load_pickle(path)
            print(f"✅ CLF {groups} groups: {len(data)} samples")
        else:
            print(f"❌ CLF {groups} groups: {path} NOT FOUND")
            all_good = False
    print("="*50)
    return all_good

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
