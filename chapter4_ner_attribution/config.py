#!/usr/bin/env python3
"""
CONFIGURATION FILE FOR PHASE 2 EXPERIMENTS
"""

from pathlib import Path

# PATHS
BASE_DIR = Path('/home/macierz/mohabdal/TerrorismNER_Project')
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
RESULTS_DIR = BASE_DIR / 'results' / 'phase2'
FIGURES_DIR = BASE_DIR / 'figures' / 'phase2'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories
for dir_path in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# DATA FILES
NER_DATA_PATH = CHECKPOINT_DIR / 'ner_data_basic.pkl'
CLF_DATA_PATHS = {
    10: CHECKPOINT_DIR / 'classification_data_10.pkl',
    20: CHECKPOINT_DIR / 'classification_data_20.pkl',
    50: CHECKPOINT_DIR / 'classification_data_50.pkl',
    100: CHECKPOINT_DIR / 'classification_data_100.pkl',
}

# MODELS
MODELS = {
    'roberta': {'name': 'roberta-base', 'type': 'RobertaForSequenceClassification'},
    'bert': {'name': 'bert-base-uncased', 'type': 'BertForSequenceClassification'},
    'distilbert': {'name': 'distilbert-base-uncased', 'type': 'DistilBertForSequenceClassification'},
    'conflibert': {'name': 'snowood1/ConfliBERT-scr-uncased', 'type': 'BertForSequenceClassification'},
}

# TRAINING CONFIG
TRAINING_CONFIG = {
    'seed': 42, 'max_length': 256, 'batch_size': 8,
    'gradient_accumulation_steps': 2, 'learning_rate': 2e-5,
    'num_epochs': 4, 'warmup_ratio': 0.1, 'weight_decay': 0.01,
    'early_stopping_patience': 2, 'fp16': True,
    'test_size': 0.15, 'val_size': 0.15,
}

NER_CONFIG = {
    'seed': 42, 'max_length': 128, 'batch_size': 16,
    'learning_rate': 2e-5, 'num_epochs': 5,
    'warmup_ratio': 0.1, 'weight_decay': 0.01,
}

# EXPERIMENTS TO RUN
CLF_EXPERIMENTS = [
    ('bert', 10), ('bert', 50), ('bert', 100),
    ('distilbert', 20), ('distilbert', 50), ('distilbert', 100),
    ('conflibert', 10), ('conflibert', 20), ('conflibert', 50), ('conflibert', 100),
]

NER_EXPERIMENTS = ['conflibert']

# EXISTING RESULTS
EXISTING_CLF_RESULTS = {
    ('roberta', 10): {'accuracy': 0.996, 'macro_f1': 0.996},
    ('roberta', 20): {'accuracy': 0.9907, 'macro_f1': 0.9907},
    ('roberta', 50): {'accuracy': 0.987, 'macro_f1': 0.9838},
    ('roberta', 100): {'accuracy': 0.9852, 'macro_f1': 0.9813},
    ('bert', 20): {'accuracy': 0.9883, 'macro_f1': 0.9883},
    ('distilbert', 10): {'accuracy': 0.9907, 'macro_f1': 0.9907},
}

EXISTING_NER_RESULTS = {
    'roberta': {'precision': 0.9821, 'recall': 0.9826, 'f1': 0.9823},
    'bert': {'precision': 0.9820, 'recall': 0.9824, 'f1': 0.9822},
    'distilbert': {'precision': 0.9799, 'recall': 0.9803, 'f1': 0.9801},
    'spacy_trained': {'precision': 0.91, 'recall': 0.91, 'f1': 0.9107},
}

EXISTING_LLM_CLF_RESULTS = {
    'deepseek_0shot': {'accuracy': 0.1787, 'macro_f1': 0.1326},
    'deepseek_5shot': {'accuracy': 0.1453, 'macro_f1': 0.0963},
    'deepseek_10shot': {'accuracy': 0.2427, 'macro_f1': 0.2237},
    'qwen_0shot': {'accuracy': 0.8, 'macro_f1': 0.7519},
    'qwen_5shot': {'accuracy': 0.808, 'macro_f1': 0.7546},
    'qwen_10shot': {'accuracy': 0.8107, 'macro_f1': 0.7593},
}

EXISTING_LLM_NER_RESULTS = {
    'deepseek_0shot': {'precision': 0.0512, 'recall': 0.118, 'f1': 0.0714},
    'deepseek_10shot': {'precision': 0.2481, 'recall': 0.266, 'f1': 0.2568},
    'qwen_0shot': {'precision': 0.199, 'recall': 0.209, 'f1': 0.2039},
    'qwen_10shot': {'precision': 0.2291, 'recall': 0.298, 'f1': 0.259},
}

EXISTING_EFFICIENCY = {
    'distilbert': {'parameters_m': 67.0, 'inference_time_ms': 9.17},
    'roberta': {'parameters_m': 124.7, 'inference_time_ms': 18.32},
    'bert': {'parameters_m': 109.5, 'inference_time_ms': 17.98},
    'spacy': {'parameters_m': 1.0, 'inference_time_ms': 6.81},
    'deepseek': {'parameters_b': 7, 'inference_time_ms': 4245.16},
    'qwen': {'parameters_b': 8, 'inference_time_ms': 4250.74},
}
