#!/usr/bin/env python3
"""
Configuration for MTL-CBERT training and evaluation.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
FIGURES_DIR = BASE_DIR / "figures"

for d in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────
CONFLIBERT_MODEL = "snowood1/ConfliBERT-scr-uncased"
TEXT_EMBED_DIM = 768
GRAPH_EMBED_DIM = 256
FUSION_DIM = 512
NUM_CLASSES = 9
DROPOUT = 0.3

# ── Graph ──────────────────────────────────────────────────────
TEMPORAL_DECAY_LAMBDA = 0.01
GRAPH_HIDDEN_DIM = 256
GRAPH_NUM_HEADS = 4
GRAPH_LAYERS = 2
CRAMERS_V_THRESHOLD = 0.25

# ── Training ───────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 15
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 256
FOCAL_LOSS_GAMMA = 2.0
GRADIENT_CLIP = 1.0

# ── Augmentation ───────────────────────────────────────────────
AUGMENTATION_MODEL = "gpt-4-turbo"
COSINE_SIM_THRESHOLD = 0.85
MINORITY_CLASSES = ["Hijacking", "Hostage Taking (Barricade Incident)", "Unarmed Assault"]
AUGMENTATION_SAMPLES_PER_CLASS = 2000

# ── Evaluation ─────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
TEST_RATIO = 0.1
VAL_RATIO = 0.1

# ── Attack type labels ─────────────────────────────────────────
ATTACK_TYPES = [
    "Armed Assault",
    "Assassination",
    "Bombing/Explosion",
    "Facility/Infrastructure Attack",
    "Hijacking",
    "Hostage Taking (Barricade Incident)",
    "Hostage Taking (Kidnapping)",
    "Unarmed Assault",
    "Unknown",
]

# ── Feature enrichment templates (Cramér's V > 0.25) ──────────
ENRICHMENT_ATTRIBUTES = [
    "region",
    "weaptype1",
    "targtype1",
    "gname",
]
