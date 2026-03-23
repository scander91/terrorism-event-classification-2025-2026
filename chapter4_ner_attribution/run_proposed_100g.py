#!/usr/bin/env python3
"""
===============================================================================
Run Missing Proposed Model (RoBERTa+BiLSTM+Attention) on 100 Groups
===============================================================================
This script runs the Proposed classification model on 100 groups.

Estimated Runtime: ~30-45 minutes
Run with: CUDA_VISIBLE_DEVICES=0 python run_proposed_100g.py

Author: PhD Research - Terrorism Intelligence Framework
===============================================================================
"""

import os
import gc
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    BASE_DIR = Path(os.path.expanduser("~/TerrorismNER_Project"))
    RESULTS_DIR = BASE_DIR / "results"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    # Data
    CLF_DATA_100 = CHECKPOINT_DIR / "classification_data_100.pkl"
    
    # Training
    MODEL_NAME = 'roberta-base'
    MAX_LEN = 256
    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 2e-5
    LSTM_HIDDEN = 256
    DROPOUT = 0.1
    SEED = 42
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# PROPOSED MODEL: RoBERTa + BiLSTM + Attention
# =============================================================================

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attn_weights


class ProposedModel(nn.Module):
    def __init__(self, model_name, num_classes, lstm_hidden=256, dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.bilstm = nn.LSTM(
            hidden_size,
            lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = AttentionLayer(lstm_hidden * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask):
        # Get encoder output
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        encoder_output = self.dropout(encoder_output)
        
        # BiLSTM
        lstm_output, _ = self.bilstm(encoder_output)
        
        # Attention
        context, _ = self.attention(lstm_output, attention_mask)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_proposed_100g():
    print("=" * 70)
    print("PROPOSED MODEL (RoBERTa+BiLSTM+Attention) - 100 Groups")
    print(f"Device: {Config.DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    set_seed(Config.SEED)
    start_time = time.time()
    
    # Load data
    print("\n1. Loading data...")
    if not Config.CLF_DATA_100.exists():
        print(f"❌ Data not found: {Config.CLF_DATA_100}")
        print("Trying alternative path...")
        
        # Try loading from pickle in checkpoints
        alt_path = Config.BASE_DIR / "data" / "classification_100.pkl"
        if alt_path.exists():
            with open(alt_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"❌ No data found. Please ensure classification_data_100.pkl exists.")
            return None
    else:
        with open(Config.CLF_DATA_100, 'rb') as f:
            data = pickle.load(f)
    
    print(f"   Data loaded: {len(data) if hasattr(data, '__len__') else 'dict format'}")
    
    # Prepare data
    print("\n2. Preparing train/val/test splits...")
    
    if isinstance(data, dict):
        if 'train' in data:
            train_texts = [d['text'] for d in data['train']]
            train_labels = [d['label'] for d in data['train']]
            test_texts = [d['text'] for d in data['test']]
            test_labels = [d['label'] for d in data['test']]
            val_texts = [d['text'] for d in data.get('val', data['test'][:len(data['test'])//2])]
            val_labels = [d['label'] for d in data.get('val', data['test'][:len(data['test'])//2])]
            labels = sorted(set(train_labels))
        else:
            # DataFrame-like dict
            texts = data['clean_text'].tolist() if hasattr(data['clean_text'], 'tolist') else list(data['clean_text'])
            label_col = data['canonical_group'].tolist() if hasattr(data['canonical_group'], 'tolist') else list(data['canonical_group'])
            labels = sorted(set(label_col))
            label2id = {l: i for i, l in enumerate(labels)}
            encoded_labels = [label2id[l] for l in label_col]
            
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                texts, encoded_labels, test_size=0.3, stratify=encoded_labels, random_state=Config.SEED
            )
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=Config.SEED
            )
    else:
        # DataFrame
        texts = data['clean_text'].tolist()
        label_col = data['canonical_group'].tolist()
        labels = sorted(set(label_col))
        label2id = {l: i for i, l in enumerate(labels)}
        encoded_labels = [label2id[l] for l in label_col]
        
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, encoded_labels, test_size=0.3, stratify=encoded_labels, random_state=Config.SEED
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=Config.SEED
        )
    
    num_classes = len(labels) if isinstance(labels, list) else len(set(train_labels))
    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    print(f"   Number of classes: {num_classes}")
    
    # Tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, Config.MAX_LEN)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, Config.MAX_LEN)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, Config.MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Model
    print("\n4. Initializing model...")
    model = ProposedModel(
        Config.MODEL_NAME,
        num_classes,
        lstm_hidden=Config.LSTM_HIDDEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("\n5. Training...")
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels_batch = batch['label'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels_batch.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = accuracy_score(train_true, train_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels_batch = batch['label'].to(Config.DEVICE)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels_batch.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        
        print(f"   Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    print("\n6. Evaluating on test set...")
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels_batch = batch['label'].to(Config.DEVICE)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels_batch.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_true, test_preds)
    f1_macro = f1_score(test_true, test_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    precision = precision_score(test_true, test_preds, average='macro', zero_division=0)
    recall = recall_score(test_true, test_preds, average='macro', zero_division=0)
    
    training_time = time.time() - start_time
    
    results = {
        'model': 'Proposed(RoBERTa+BiLSTM+Attn)',
        'n_groups': 100,
        'accuracy': round(accuracy, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'training_time': round(training_time, 2),
        'best_val_acc': round(best_val_acc, 4)
    }
    
    print("\n" + "=" * 70)
    print("RESULTS - Proposed(RoBERTa+BiLSTM+Attn) on 100 Groups")
    print("=" * 70)
    print(f"   Accuracy:    {accuracy*100:.2f}%")
    print(f"   F1 Macro:    {f1_macro*100:.2f}%")
    print(f"   F1 Weighted: {f1_weighted*100:.2f}%")
    print(f"   Precision:   {precision*100:.2f}%")
    print(f"   Recall:      {recall*100:.2f}%")
    print(f"   Training Time: {training_time/60:.1f} minutes")
    print("=" * 70)
    
    # Save results
    results_path = Config.RESULTS_DIR / "proposed_100g_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Also append to classification_results.json if it exists
    clf_results_path = Config.RESULTS_DIR / "classification_results.json"
    if clf_results_path.exists():
        with open(clf_results_path, 'r') as f:
            all_results = json.load(f)
        all_results.append(results)
        with open(clf_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Appended to: {clf_results_path}")
    
    return results


if __name__ == "__main__":
    results = train_proposed_100g()
