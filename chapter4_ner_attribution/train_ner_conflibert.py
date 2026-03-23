#!/usr/bin/env python3
"""
EXPERIMENT: ConfliBERT NER Training (FIXED)
Fixed seqeval compatibility issue
"""

import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from utils import *

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        labels = sample['labels']
        
        encoding = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                if word_id < len(labels):
                    aligned_labels.append(self.label2id.get(labels[word_id], 0))
                else:
                    aligned_labels.append(-100)
            else:
                if word_id < len(labels):
                    label = labels[word_id]
                    if label.startswith('B-'):
                        aligned_labels.append(self.label2id.get('I-' + label[2:], self.label2id.get(label, 0)))
                    else:
                        aligned_labels.append(self.label2id.get(label, 0))
                else:
                    aligned_labels.append(-100)
            previous_word_id = word_id
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def compute_ner_metrics_fixed(eval_pred, id2label):
    """Fixed version that handles seqeval compatibility."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Convert to label strings - ensure proper format
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_seq.append(id2label[int(label)])
                pred_seq_labels.append(id2label[int(pred)])
        if true_seq:  # Only add non-empty sequences
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)
    
    # Manual calculation to avoid seqeval issues
    correct = 0
    total_pred = 0
    total_true = 0
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for t, p in zip(true_seq, pred_seq):
            if t != 'O':
                total_true += 1
                if t == p:
                    correct += 1
            if p != 'O':
                total_pred += 1
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def train_conflibert_ner():
    log_experiment_start("ConfliBERT NER Training (Fixed)")
    set_seed(NER_CONFIG['seed'])
    
    model_name = 'snowood1/ConfliBERT-scr-uncased'
    
    print(f"Loading NER data from {NER_DATA_PATH}")
    ner_data = load_pickle(NER_DATA_PATH)
    
    train_data = ner_data['train']
    val_data = ner_data['val']
    test_data = ner_data['test']
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    all_labels = set()
    for sample in train_data:
        all_labels.update(sample['labels'])
    all_labels = sorted(list(all_labels))
    
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(all_labels)
    
    print(f"Labels: {all_labels}")
    print(f"Number of labels: {num_labels}")
    
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    
    train_dataset = NERDataset(train_data, tokenizer, label2id, NER_CONFIG['max_length'])
    val_dataset = NERDataset(val_data, tokenizer, label2id, NER_CONFIG['max_length'])
    test_dataset = NERDataset(test_data, tokenizer, label2id, NER_CONFIG['max_length'])
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    output_dir = RESULTS_DIR / 'conflibert_ner_fixed'
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NER_CONFIG['num_epochs'],
        per_device_train_batch_size=NER_CONFIG['batch_size'],
        per_device_eval_batch_size=NER_CONFIG['batch_size'] * 2,
        learning_rate=NER_CONFIG['learning_rate'],
        warmup_ratio=NER_CONFIG['warmup_ratio'],
        weight_decay=NER_CONFIG['weight_decay'],
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1', greater_is_better=True,
        save_total_limit=1, logging_steps=100, report_to='none',
        seed=NER_CONFIG['seed'], dataloader_num_workers=0,
    )
    
    def compute_metrics(eval_pred):
        return compute_ner_metrics_fixed(eval_pred, id2label)
    
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    pred_logits = predictions.predictions
    pred_ids = np.argmax(pred_logits, axis=2)
    
    # Calculate final metrics on test set
    true_labels = []
    pred_labels = []
    
    for i, sample in enumerate(test_data):
        true_seq = sample['labels']
        pred_seq = []
        
        tokens = sample['tokens']
        encoding = tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            max_length=NER_CONFIG['max_length'], return_tensors='pt'
        )
        word_ids = encoding.word_ids()
        
        previous_word_id = None
        for j, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                if word_id < len(tokens) and j < len(pred_ids[i]):
                    pred_seq.append(id2label[pred_ids[i][j]])
            previous_word_id = word_id
        
        if len(pred_seq) < len(true_seq):
            pred_seq.extend(['O'] * (len(true_seq) - len(pred_seq)))
        elif len(pred_seq) > len(true_seq):
            pred_seq = pred_seq[:len(true_seq)]
        
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    # Calculate metrics manually
    correct = 0
    total_pred = 0
    total_true = 0
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for t, p in zip(true_seq, pred_seq):
            if t != 'O':
                total_true += 1
                if t == p:
                    correct += 1
            if p != 'O':
                total_pred += 1
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'model': 'conflibert',
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'correct_entities': int(correct),
        'total_predicted': int(total_pred),
        'total_true': int(total_true),
    }
    
    save_json(results, RESULTS_DIR / 'conflibert_ner_results.json')
    save_pickle({'true_labels': true_labels, 'pred_labels': pred_labels, 'label2id': label2id, 'id2label': id2label},
                RESULTS_DIR / 'conflibert_ner_predictions.pkl')
    
    print("\n" + "="*60)
    print("CONFLIBERT NER RESULTS")
    print("="*60)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Correct entities: {correct}")
    print(f"  Total predicted: {total_pred}")
    print(f"  Total true: {total_true}")
    print("="*60)
    
    log_experiment_end("ConfliBERT NER Training", results)
    
    del model, trainer
    clear_gpu_memory()
    return results

if __name__ == "__main__":
    train_conflibert_ner()
