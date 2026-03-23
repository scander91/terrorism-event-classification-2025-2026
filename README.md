# Research on Transformer-Based Multi-Task and Cross-Lingual Methods for Terrorism Event Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the code and data for the doctoral dissertation:

> **Research on Transformer-Based Multi-Task and Cross-Lingual Methods for Terrorism Event Classification**  
> Mohammed Abdalsalam, Wuhan University of Technology, 2026  

## Overview

This dissertation develops an integrated computational framework for terrorism event analysis through four interconnected contributions:

| Chapter | Contribution | Key Result |
|---------|-------------|------------|
| **Ch.2** | GTD Preprocessing Pipeline | 135 → 80 features, 100% completeness |
| **Ch.3** | MTL-CBERT: Graph-augmented attack classification | 98.5% macro F1 |
| **Ch.4** | Unified NER & attack attribution | 99.23% NER F1; 98.21% attribution |
| **Ch.5** | Arabic terrorism benchmark & ICL evaluation | 85.0% attack type accuracy |

## Repository Structure

```
terrorism-event-classification/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Global dependencies
├── .gitignore                         # Git ignore rules
│
├── chapter2_preprocessing/            # Ch.2: GTD Preprocessing Pipeline
│   ├── README.md                      # Chapter-specific instructions
│   ├── requirements.txt               # Chapter dependencies
│   ├── config.py                      # Configuration and file paths
│   ├── 01_data_loading.py             # Load raw GTD data
│   ├── 02_eda.py                      # Exploratory data analysis
│   ├── 03_missing_value_imputation.py # Distribution-aware imputation
│   ├── 04_geocoding.py                # Web-crawler geocoding
│   ├── 05_feature_engineering.py      # Feature selection (PCC, NMI)
│   ├── 06_evaluation.py               # KL-divergence, VIF validation
│   ├── run_pipeline.py                # Run full pipeline end-to-end
│   └── data/                          # Data directory (see instructions)
│       └── README.md                  # How to obtain GTD data
│
├── chapter3_mtlcbert/                 # Ch.3: MTL-CBERT Framework
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── models/
│   │   ├── feature_enrichment.py      # Attribute-to-text templates
│   │   ├── conflibert_encoder.py      # ConfliBERT with self-attention
│   │   ├── graph_learning.py          # Heterogeneous graph + temporal decay
│   │   ├── gated_fusion.py            # Gated fusion mechanism
│   │   └── mtl_cbert.py               # Full MTL-CBERT model
│   ├── data_augmentation/
│   │   ├── llm_augmentation.py        # GPT-4 Turbo paraphrase generation
│   │   └── semantic_filtering.py      # Cosine similarity filtering
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation + significance tests
│   └── run_experiments.py             # Reproduce all Ch.3 experiments
│
├── chapter4_ner_attribution/          # Ch.4: NER + Classification
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── models/
│   │   ├── roberta_bilstm_crf.py      # RoBERTa-BiLSTM-CRF
│   │   ├── gazetteer.py               # Terrorism-specific gazetteer
│   │   ├── ner_model.py               # Full NER model
│   │   └── attribution_model.py       # ConfliBERT + Focal Loss
│   ├── data_preparation/
│   │   ├── build_ner_corpus.py        # Build BIO-tagged NER dataset
│   │   └── build_classification.py    # Multi-scale classification data
│   ├── icl_evaluation/
│   │   ├── prompt_templates.py        # ICL prompt templates
│   │   └── run_icl.py                 # LLM in-context learning
│   ├── train_ner.py
│   ├── train_classification.py
│   └── run_experiments.py
│
├── chapter5_arabic_icl/               # Ch.5: Arabic Benchmark + ICL
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── benchmark_construction/
│   │   ├── translate_gtd.py           # NLLB-200 translation
│   │   ├── glossary.json              # 87-term bilingual glossary
│   │   ├── quality_validation.py      # BLEU, PPL, deduplication
│   │   └── build_benchmark.py         # Final benchmark assembly
│   ├── icl_evaluation/
│   │   ├── prompt_templates.py        # 5 prompt strategies
│   │   ├── demonstration_selection.py # Random + Top-k selection
│   │   ├── run_evaluation.py          # Run 330 configurations
│   │   └── scoring_framework.py       # Multi-dimensional scoring
│   ├── cross_domain/
│   │   ├── arabic_benchmarks.py       # 11 Arabic NLP benchmarks
│   │   └── run_cross_domain.py        # Cross-domain validation
│   └── run_experiments.py
│
├── shared/                            # Shared utilities across chapters
│   ├── metrics.py                     # Accuracy, F1, MCC, etc.
│   ├── visualization.py               # Plotting utilities
│   ├── statistical_tests.py           # Bootstrap test, Cohen's d
│   └── utils.py                       # Common utilities
│
└── results/                           # Saved results and figures
    ├── chapter2/
    ├── chapter3/
    ├── chapter4/
    └── chapter5/
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/terrorism-event-classification.git
cd terrorism-event-classification
```

### 2. Install dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 3. Obtain the GTD dataset
The Global Terrorism Database requires a registration request:
- Visit: https://www.start.umd.edu/gtd/contact/
- Submit a data access request
- Place the downloaded file in `chapter2_preprocessing/data/`

### 4. Run experiments

```bash
# Chapter 2: Preprocessing
cd chapter2_preprocessing
python run_pipeline.py

# Chapter 3: MTL-CBERT
cd chapter3_mtlcbert
python run_experiments.py

# Chapter 4: NER + Attribution
cd chapter4_ner_attribution
python run_experiments.py

# Chapter 5: Arabic ICL
cd chapter5_arabic_icl
python run_experiments.py
```

## Hardware Requirements

All GPU experiments were conducted on a server with 2× NVIDIA Quadro RTX 8000 (48 GB VRAM each) and 1× NVIDIA Quadro RTX 5000 (16 GB VRAM), CUDA 12.2.

| Chapter | Task | GPU VRAM | Estimated Runtime |
|---------|------|----------|-------------------|
| Ch.2 | Preprocessing pipeline | CPU only | ~30 min |
| Ch.3 | MTL-CBERT training (1 seed) | 16 GB+ | ~4–5 hours |
| Ch.3 | MTL-CBERT full (5 seeds × 2 conditions) | 16 GB+ | ~50 hours |
| Ch.4 | NER training (RoBERTa-BiLSTM-CRF) | 12 GB+ | ~3 hours |
| Ch.4 | Multi-scale classification (4 scales) | 12 GB+ | ~6 hours |
| Ch.4 | LLM ICL evaluation (DeepSeek-7B, Qwen-7B) | 24 GB+ | ~8 hours |
| Ch.5 | NLLB-200 translation (3.3B params) | 16 GB+ | ~6 hours |
| Ch.5 | LLM ICL evaluation (330 configs, 4-bit NF4) | 2× 24 GB+ | ~13.4 hours |
| Ch.5 | Cross-domain evaluation (11 benchmarks) | 24 GB+ | ~20 hours |

**Minimum requirements:** A single GPU with 24 GB VRAM (e.g., RTX 3090/4090) can run all experiments sequentially. Chapters 3 and 4 (fine-tuning) require 12–16 GB. Chapters 4 and 5 (LLM inference with 7B models at 4-bit quantization) require 24 GB+. Dual-GPU setups enable parallel evaluation of the 330 ICL configurations.

## Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| GTD (1970–2020) | University of Maryland | [Request access](https://www.start.umd.edu/gtd/contact/) |
| SF Crime Data | DataSF | [Public download](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783) |
| LA Crime Data | LA Open Data | [Public download](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) |
| Arabic NLP benchmarks | Various | See `chapter5_arabic_icl/README.md` |

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{abdalsalam2026terrorism,
  title={Research on Transformer-Based Multi-Task and Cross-Lingual 
         Methods for Terrorism Event Classification},
  author={Abdalsalam, Mohammed},
  year={2026},
  school={Wuhan University of Technology}
}
```

### Related publications:
```bibtex
@article{abdalsalam2024terrorism,
  title={Terrorism Attack Classification Using Machine Learning: 
         The Effectiveness of Using Textual Features Extracted from GTD Dataset},
  author={Abdalsalam, M. and Dahou, A. and Kryvinska, N.},
  journal={CMES-Computer Modeling in Engineering \& Sciences},
  volume={138},
  number={2},
  year={2024}
}

@article{abdalsalam2024group,
  title={Terrorism group prediction using feature combination and 
         BiGRU with self-attention mechanism},
  author={Abdalsalam, M. and Dahou, A. and Kryvinska, N.},
  journal={PeerJ Computer Science},
  volume={10},
  pages={e2252},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Mohammed Abdalsalam** — Wuhan University of Technology
