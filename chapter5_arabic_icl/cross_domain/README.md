# Cross-Domain Validation on Arabic NLP Benchmarks

## Overview

Evaluates generalizability of ICL prompt-design findings across 11 established Arabic NLP benchmarks spanning three task categories.

## Datasets

| Dataset | Task | Classes | Access |
|---------|------|---------|--------|
| NADI 2020 | Dialect ID | 21 | [GitHub](https://github.com/Wikipedia-based-ADI) |
| MADAR-26 | Dialect ID | 26 | [Request](https://camel.abudhabi.nyu.edu/madar-shared-task-2019/) |
| ASTD | Sentiment | 4 | [GitHub](https://github.com/mahmoudnabil/ASTD) |
| ArSAS | Sentiment | 4 | Request from authors |
| ASAD | Sentiment | 3 | [GitHub](https://github.com/basharalhafni/ASAD) |
| LABR | Sentiment | 5 | [GitHub](https://github.com/mohamedadaly/LABR) |
| SemEval-2016 | Sentiment | 3 | [SemEval](https://alt.qcri.org/semeval2016/task7/) |
| Alomari 2017 | Sentiment | 3 | Request from authors |
| ArSentiment | Sentiment | 3 | Aggregated |
| OSACT4 | Moderation | 2 | [Shared task](https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/) |
| Adult Content | Moderation | 2 | Request from authors |

## Usage

```bash
# Step 1: Download available datasets
python3 01_download_benchmarks.py

# Step 2: Run evaluation
python3 02_run_cross_domain.py --model deepseek --dataset all
python3 02_run_cross_domain.py --model qwen3 --dataset all

# Step 3: Aggregate results
python3 03_aggregate_results.py
```

## Evaluation Protocol

- **Prediction:** Likelihood-based (label with highest log-probability)
- **Prompts:** P1 (Arabic task-specific), P2 (Arabic expert-role), P3 (English)
- **Shot levels:** k ∈ {5, 8, 10}
- **Selection:** Random (R=5 independent runs) and Embed Top-k (deterministic)
- **Models:** DeepSeek-R1-Distill-Qwen-7B, Qwen3-8B (4-bit NF4 quantization)
- **Embedding:** paraphrase-multilingual-MiniLM-L12-v2 for Top-k retrieval
