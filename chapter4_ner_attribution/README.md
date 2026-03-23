# Chapter 4: A Transformer-Based Framework for Terrorist Organization Recognition and Classification

## Overview

This chapter proposes a unified framework for two complementary intelligence tasks: Named Entity Recognition (NER) for terrorist organization extraction and attack attribution classification.

## Architecture

### NER: RoBERTa-BiLSTM-CRF-Gazetteer
```
Input: GTD incident text
    → RoBERTa Encoder (768-dim contextual embeddings)
    → Gazetteer Injection (100+ organizations, projected to 64-dim)
    → BiLSTM (2 layers, 512-dim output)
    → CRF Layer (valid BIO sequence prediction)
    → Output: Organization entity spans
```

### Classification: ConfliBERT + Focal Loss
```
Input: GTD incident text
    → ConfliBERT Encoder (768-dim)
    → Classification Head + Focal Loss (γ=2.0)
    → Output: Responsible organization (10/20/50/100 groups)
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch >= 2.0.0`
- `transformers >= 4.35.0`
- `torchcrf >= 1.1.0`
- `seqeval >= 1.2.2`
- `bitsandbytes >= 0.41.0` (for LLM evaluation)

## Usage

### Build datasets:
```bash
cd data_preparation
python build_ner_corpus.py          # 67,386 BIO-tagged samples
python build_classification.py      # Multi-scale (10/20/50/100 groups)
```

### Train NER:
```bash
python train_ner.py
```

### Train Classification:
```bash
python train_classification.py --num_groups 10
python train_classification.py --num_groups 20
python train_classification.py --num_groups 50
python train_classification.py --num_groups 100
```

### Run LLM comparison:
```bash
cd icl_evaluation
python run_icl.py --task ner --model deepseek --shots 10
python run_icl.py --task classification --model qwen --shots 10
```

### Run all experiments:
```bash
python run_experiments.py
```

## Key Results

### NER
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| RoBERTa (fine-tuned) | 97.81 | 97.25 | 97.53 |
| **RoBERTa-BiLSTM-CRF-Gaz** | **99.45** | **99.01** | **99.23** |
| DeepSeek-R1 (10-shot) | 70.12 | 67.88 | 69.00 |

### Classification (Accuracy %)
| Model | 10 groups | 50 groups | 100 groups |
|-------|-----------|-----------|------------|
| ConfliBERT + FL | **99.73** | 99.02 | **98.21** |
| DeepSeek-R1 (10-shot) | 92.44 | 89.15 | 86.36 |
