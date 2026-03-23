#!/usr/bin/env python3
"""
EXPERIMENT: Classification Training (BERT, DistilBERT, ConfliBERT)
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from utils import *

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), truncation=True,
            max_length=self.max_length, padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_classification(model_key, num_groups):
    set_seed(TRAINING_CONFIG['seed'])
    model_config = MODELS[model_key]
    model_name = model_config['name']
    
    print(f"\n{'='*60}")
    print(f"Training {model_key.upper()} on {num_groups} groups")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    data = load_pickle(CLF_DATA_PATHS[num_groups])
    print(f"Loaded {len(data)} samples")
    
    groups = sorted(data['canonical_group'].unique())
    label2id = {g: i for i, g in enumerate(groups)}
    id2label = {i: g for g, i in label2id.items()}
    num_classes = len(groups)
    print(f"Number of classes: {num_classes}")
    
    train_val, test = train_test_split(
        data, test_size=TRAINING_CONFIG['test_size'],
        stratify=data['canonical_group'], random_state=TRAINING_CONFIG['seed']
    )
    train, val = train_test_split(
        train_val, test_size=TRAINING_CONFIG['val_size'] / (1 - TRAINING_CONFIG['test_size']),
        stratify=train_val['canonical_group'], random_state=TRAINING_CONFIG['seed']
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    train_texts = train['clean_text'].tolist()
    train_labels = [label2id[g] for g in train['canonical_group'].tolist()]
    val_texts = val['clean_text'].tolist()
    val_labels = [label2id[g] for g in val['canonical_group'].tolist()]
    test_texts = test['clean_text'].tolist()
    test_labels = [label2id[g] for g in test['canonical_group'].tolist()]
    
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes, id2label=id2label, label2id=label2id
    )
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, TRAINING_CONFIG['max_length'])
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, TRAINING_CONFIG['max_length'])
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, TRAINING_CONFIG['max_length'])
    
    output_dir = RESULTS_DIR / f'{model_key}_clf_{num_groups}_groups'
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAINING_CONFIG['num_epochs'],
        per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['batch_size'] * 2,
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        fp16=TRAINING_CONFIG['fp16'],
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=1, logging_steps=50, report_to='none',
        seed=TRAINING_CONFIG['seed'], dataloader_num_workers=0,
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights, model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'])]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    
    metrics = calculate_classification_metrics(test_labels, preds, probs, num_classes)
    metrics['model'] = model_key
    metrics['num_groups'] = num_groups
    metrics['train_samples'] = len(train)
    metrics['test_samples'] = len(test)
    metrics['num_classes'] = num_classes
    
    predictions_data = {
        'test_labels': test_labels, 'predictions': preds.tolist(),
        'probabilities': probs.tolist(), 'label2id': label2id, 'id2label': id2label,
    }
    
    save_json(metrics, RESULTS_DIR / f'{model_key}_clf_{num_groups}_groups_results.json')
    save_pickle(predictions_data, RESULTS_DIR / f'{model_key}_clf_{num_groups}_groups_predictions.pkl')
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_key.upper()} on {num_groups} groups")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"{'='*60}\n")
    
    del model, trainer
    clear_gpu_memory()
    return metrics

def run_all_classification_experiments():
    log_experiment_start("Classification Training (All Missing)")
    all_results = {}
    for model_key, num_groups in CLF_EXPERIMENTS:
        try:
            print(f"\n>>> Starting: {model_key} on {num_groups} groups")
            results = train_classification(model_key, num_groups)
            all_results[f'{model_key}_{num_groups}'] = results
            save_json(all_results, RESULTS_DIR / 'all_clf_results_intermediate.json')
        except Exception as e:
            print(f"❌ ERROR in {model_key} {num_groups} groups: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'{model_key}_{num_groups}'] = {'error': str(e)}
    save_json(all_results, RESULTS_DIR / 'all_clf_results_final.json')
    log_experiment_end("Classification Training", all_results)
    return all_results

if __name__ == "__main__":
    import config
    if not verify_data_files(config):
        print("❌ Data verification failed!")
        sys.exit(1)
    run_all_classification_experiments()
