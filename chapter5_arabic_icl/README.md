# Chapter 5: Multi-Dimensional Arabic Terrorism Event Classification using Retrieval-Augmented In-Context Learning

## Overview

This chapter constructs the first multi-dimensional Arabic terrorism classification benchmark from GTD English records and conducts a systematic evaluation of 330 in-context learning (ICL) configurations.

## Pipeline

```
GTD English Records
    → Step 1: NLLB-200 Neural Machine Translation
    → Step 2: Glossary-Constrained Refinement (87-term bilingual glossary)
    → Step 3: Quality Validation (BLEU ≥ 0.25, PPL ≤ 150, deduplication)
    → Arabic Terrorism Benchmark (3 sub-tasks)
    
Arabic Benchmark
    → Step 4: ICL Evaluation (330 configurations)
        - 5 prompt strategies × 3 shot levels × 2 selection methods × 2 LLMs × 3 tasks
    → Step 5: Multi-Dimensional Scoring (Performance, Similarity, Consistency)
    → Step 6: Cross-Domain Validation (11 Arabic NLP benchmarks)
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch >= 2.0.0`
- `transformers >= 4.35.0`
- `bitsandbytes >= 0.41.0` (for 4-bit quantization)
- `sentence-transformers >= 2.2.0`
- `sacrebleu >= 2.3.0`
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`

## Models Used

| Model | Parameters | Quantization | VRAM |
|-------|-----------|--------------|------|
| DeepSeek-R1-Distill-Qwen-7B | 7B | 4-bit NF4 | ~5.2 GB |
| Qwen2.5-7B-Instruct | 7B | 4-bit NF4 | ~5.2 GB |
| Qwen3-8B (cross-domain) | 8.2B | 4-bit NF4 | ~5.5 GB |
| NLLB-200 (translation) | 3.3B | FP16 | ~7 GB |

## Usage

### Build the Arabic benchmark:
```bash
cd benchmark_construction
python translate_gtd.py            # NLLB-200 translation
python quality_validation.py       # BLEU, PPL, deduplication
python build_benchmark.py          # Assemble final benchmark
```

### Run ICL evaluation (330 configs):
```bash
cd icl_evaluation
python run_evaluation.py --model deepseek --task attack
python run_evaluation.py --model qwen --task attack
python run_evaluation.py --model deepseek --task weapon
python run_evaluation.py --model qwen --task weapon
python run_evaluation.py --model deepseek --task target
python run_evaluation.py --model qwen --task target
```

### Run all experiments:
```bash
python run_experiments.py          # Runs everything end-to-end
```

### Cross-domain validation:
```bash
cd cross_domain
python run_cross_domain.py         # 11 Arabic NLP benchmarks
```

## Configuration

Edit `config.py`:
```python
# Models
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NLLB_MODEL = "facebook/nllb-200-3.3B"

# ICL Parameters
SHOT_LEVELS = [0, 5, 10]
PROMPT_STRATEGIES = ["basic", "cot", "expert_role", "scenario", "plan_solve"]
SELECTION_METHODS = ["random", "topk"]
NUM_RUNS = 3               # Independent runs for random selection
TEMPERATURE = 0.1

# Quality thresholds
MIN_BLEU = 0.25
MAX_PPL = 150
```

## Benchmark Statistics

| Sub-task | Classes | Train | Test |
|----------|---------|-------|------|
| GTD-Attack | 9 | 179,016 | 19,891 |
| GTD-Weapon | 12 | 179,016 | 19,891 |
| GTD-Target | 22 | 179,016 | 19,891 |

## Key Results

| Finding | Detail |
|---------|--------|
| Best prompt strategy | Basic consistently outperforms CoT, Expert-Role |
| Top-k vs Random | +11–32 pp accuracy improvement |
| CoT failure rate | 85–100% under string-matching evaluation |
| Best accuracy (attack) | 85.0% (Qwen2.5, Basic, 10-shot, Top-k) |
| Best accuracy (weapon) | 83.8% (Qwen2.5, Basic, 10-shot, Top-k) |
| Best accuracy (target) | 81.2% (Qwen2.5, Basic, 10-shot, Top-k) |
