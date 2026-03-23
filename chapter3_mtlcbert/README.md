# Chapter 3: Graph-Augmented BERT-Based Framework for Multi-Class Classification of Terrorism Events

## Overview

This chapter introduces MTL-CBERT, a multi-task learning framework that fuses ConfliBERT text embeddings with heterogeneous graph representations for terrorism attack type classification across nine categories.

## Architecture

```
Input: GTD incident (text summary + structured attributes)
    → Feature Enrichment: Attribute-to-text templates (Cramér's V > 0.25)
    → ConfliBERT Encoder: Domain-specific contextual embeddings (768-dim)
    → Graph Learning: Heterogeneous graph with temporal decay (λ=0.01)
    → Gated Fusion: Adaptive text (R^768) + graph (R^256) combination
    → Focal Loss (γ=2.0) + inverse-frequency class weighting
    → Output: 9-class attack type prediction
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch >= 2.0.0`
- `transformers >= 4.35.0`
- `torch-geometric >= 2.3.0`
- `networkx >= 3.0`
- `scikit-learn >= 1.2.0`

## Usage

### Train MTL-CBERT:
```bash
python train.py --config config.py
```

### Run all experiments (Tables 3.7–3.13):
```bash
python run_experiments.py
```

### Run with data augmentation:
```bash
python data_augmentation/llm_augmentation.py   # Generate paraphrases
python data_augmentation/semantic_filtering.py  # Filter by cosine sim ≥ 0.85
python train.py --augmented
```

## Configuration

Edit `config.py`:
```python
CONFLIBERT_MODEL = "snowood1/ConfliBERT-scr-uncased"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 15
FOCAL_LOSS_GAMMA = 2.0
TEMPORAL_DECAY_LAMBDA = 0.01
SEEDS = [0, 1, 2, 3, 4]
```

## Key Results

| Model | Macro F1 (%) |
|-------|-------------|
| XGBoost (non-textual) | 87.5 |
| ConfliBERT (text only) | 95.2 |
| MTL-CBERT (full) | **98.5 ± 0.3** |
| ConfLLaMA + Prompting | 96.8 |
