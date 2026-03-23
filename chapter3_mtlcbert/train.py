#!/usr/bin/env python3
"""
MTL-CBERT Training Pipeline
==============================
Trains the MTL-CBERT model for terrorism attack type classification.

Training procedure:
  1. Load preprocessed GTD data
  2. Apply feature enrichment (attribute-to-text)
  3. Build heterogeneous event graph with temporal decay edges
  4. Initialize ConfliBERT encoder + graph module + gated fusion
  5. Train with Focal Loss (γ=2.0) and AdamW optimizer
  6. Evaluate on held-out test set across 5 random seeds

Usage:
    python3 train.py --seed 0
    python3 train.py --seed 0 --augmented
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import *
from models import MTLCBERT, FocalLoss
from models.feature_enrichment import enrich_dataset
from models.graph_learning import TemporalEdgeBuilder
from models.conflibert_encoder import ConfliBERTEncoder

warnings.filterwarnings("ignore")


class GTDDataset(Dataset):
    """Dataset for tokenized GTD incidents."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data(data_path, augmented_path=None):
    """Load GTD preprocessed data and optional augmented data."""
    df = pd.read_csv(data_path)

    if augmented_path and os.path.exists(augmented_path):
        aug_df = pd.read_csv(augmented_path)
        df = pd.concat([df, aug_df], ignore_index=True)
        print(f"  Loaded {len(aug_df)} augmented samples")

    return df


def prepare_labels(df, label_col="attacktype1_txt"):
    """Encode attack type labels as integers."""
    label_map = {name: i for i, name in enumerate(ATTACK_TYPES)}
    labels = df[label_col].map(label_map)
    valid = labels.notna()
    return labels[valid].astype(int).values, label_map, valid


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights."""
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "macro_precision": precision_score(all_labels, all_preds, average="macro"),
        "macro_recall": recall_score(all_labels, all_preds, average="macro"),
        "mcc": matthews_corrcoef(all_labels, all_preds),
    }

    return metrics, all_preds, all_labels


def train(args):
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading data...")
    df = load_data(
        args.data_path,
        args.augmented_path if args.augmented else None,
    )

    # Feature enrichment
    print("Applying feature enrichment...")
    df["enriched_text"] = enrich_dataset(df)

    # Prepare labels
    labels, label_map, valid_mask = prepare_labels(df)
    texts = df.loc[valid_mask, "enriched_text"].values

    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=TEST_RATIO, random_state=args.seed, stratify=labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=VAL_RATIO / (1 - TEST_RATIO),
        random_state=args.seed, stratify=train_labels)

    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Tokenizer and datasets
    tokenizer = ConfliBERTEncoder.get_tokenizer(CONFLIBERT_MODEL)
    train_ds = GTDDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
    val_ds = GTDDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH)
    test_ds = GTDDataset(test_texts, test_labels, tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)

    # Model
    model = MTLCBERT(
        num_classes=NUM_CLASSES,
        conflibert_model=CONFLIBERT_MODEL,
        graph_hidden_dim=GRAPH_HIDDEN_DIM,
        graph_output_dim=GRAPH_EMBED_DIM,
        fusion_dim=FUSION_DIM,
        dropout=DROPOUT,
    ).to(device)

    # Loss with class weights
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
    criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=class_weights)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps)

    # Training loop
    best_val_f1 = 0
    best_model_path = CHECKPOINT_DIR / f"mtl_cbert_seed{args.seed}.pt"

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, GRADIENT_CLIP)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"  Epoch {epoch+1}/{EPOCHS}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)

    # Test evaluation
    model.load_state_dict(torch.load(best_model_path))
    test_metrics, test_preds, test_true = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results (seed={args.seed}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")

    # Per-class report
    report = classification_report(
        test_true, test_preds, target_names=ATTACK_TYPES, output_dict=True)

    # Save results
    results = {
        "seed": args.seed,
        "augmented": args.augmented,
        "test_metrics": test_metrics,
        "per_class": report,
        "config": {
            "model": CONFLIBERT_MODEL,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "focal_gamma": FOCAL_LOSS_GAMMA,
            "graph_dim": GRAPH_EMBED_DIM,
            "fusion_dim": FUSION_DIM,
        },
    }

    results_path = OUTPUT_DIR / f"results_seed{args.seed}{'_aug' if args.augmented else ''}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MTL-CBERT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default=str(DATA_DIR / "gtd_preprocessed.csv"))
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--augmented_path", type=str,
                        default=str(DATA_DIR / "augmented_samples.csv"))
    args = parser.parse_args()
    train(args)
