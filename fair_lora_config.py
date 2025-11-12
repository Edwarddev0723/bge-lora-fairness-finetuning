"""
Project-wide configuration for Fair LoRA fine-tuning.

This file is imported by src/fair_data_loader.py and src/fair_lora_model.py.
It centralizes LoRA, dataset, and fairness-related knobs so the notebook
can simply import and run without redefining constants.
"""

from pathlib import Path
import torch

# Base model and tokenizer
BASE_MODEL = "BAAI/bge-large-en-v1.5"

# Dataset location (updated to use re-split dataset with ensured positive labels in each split)
# We keep an ORIGINAL_DATASET_PATH for auto-resplitting if the re-split directory is missing.
ORIGINAL_DATASET_PATH = Path("./data/processed/processed_resume_dataset").resolve()
DATASET_PATH = Path("./data/processed/processed_resume_dataset_resplit").resolve()

# Tokenization / sequence
MAX_LENGTH = 512

# LoRA configuration (matches user request)
USE_LORA = True
USE_ADVERSARIAL_DEBIASING = True
USE_MULTITASK = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
# Target modules per user request
LORA_TARGET_MODULES = ["query", "key", "value"]

# Device (prioritize CUDA, then Apple Metal MPS, else CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Training / batching
BATCH_SIZE = 16
# macOS/MPS often benefits from fewer workers; adjust dynamically
if DEVICE.type == "mps":
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 4
VAL_SPLIT = 0.1
SEED = 42

# Sensitive attribute handling
MASK_SENSITIVE_ATTRS = True
REPLACE_WITH_NEUTRAL = True  # replace school mentions with [SCHOOL]

# School categories present in the dataset. If your dataset differs,
# update the list accordingly. Order defines ids used by dataloader.
SCHOOL_CATEGORIES = [
    "top_school",
    "non_top_school",
    "no_school_mentioned",
]

# Optional manual weights for reweighting sampler (category -> weight multiplier)
USE_REWEIGHTING = True  # legacy weighted sampler (inverse frequency * prior)
USE_GROUP_BATCH_SAMPLER = False  # new sampler guaranteeing ≥1 minority joint group per batch
CREATE_BALANCED_VAL = True       # build a balanced validation loader (downsample per joint group)
FAIRNESS_REG_LAMBDA = 0.05       # prediction gap regularizer strength (max-min group prob)
SAMPLE_WEIGHTS = {
    "top_school": 1.0,
    "non_top_school": 1.5,           # upweight minority if needed
    "no_school_mentioned": 1.2,
}

# Adversarial / fairness hyperparameters
ADVERSARIAL_LAMBDA = 0.5      # how strongly to penalize sensitive info leakage
FAIRNESS_LAMBDA = 0.3         # weight of fairness regularization
MULTITASK_LAMBDA = 0.2        # weight of auxiliary attribute classifier
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 2e-5
NUM_LABELS = 2

# Model head sizes for FairLoRAModel
EMBEDDING_DIM = 1024              # BGE-large hidden size
ADVERSARIAL_HIDDEN_DIM = 256
ADVERSARIAL_NUM_LAYERS = 2
ADVERSARIAL_DROPOUT = 0.3
ATTRIBUTE_CLASSIFIER_HIDDEN_DIM = 256
ATTRIBUTE_CLASSIFIER_DROPOUT = 0.3

# Window-based model selection parameters
WINDOW_SELECTION_START = 7
WINDOW_SELECTION_END = 12
WINDOW_FAIRNESS_THRESHOLD = 0.12

# Temperature scaling and metrics toggles
TEMPERATURE_CALIBRATION = True
AUC_ENABLED = True
