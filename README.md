# BGE LoRA Fairness Fine-Tuning

Fairness-aware fine-tuning of BAAI/bge-large-en-v1.5 using LoRA (PEFT) with adversarial debiasing, multi-task learning, and fairness regularization to reduce group disparities while keeping strong utility.

## Highlights

- Base model: BAAI/bge-large-en-v1.5 with LoRA adapters (PEFT)
- Fairness techniques: adversarial debiasing, attribute multi-task head, demographic parity and equalized-odds regularizers
- Stability and selection: temperature scaling, tuned-threshold validation, balanced validation option, window-based best-checkpoint selection
- Reproducible reports: epoch CSV metrics, JSON training history, fairness reports and counterfactual analyses

## Repository layout

- `src/`
  - `fair_lora_model.py` — FairLoRAModel (BGE + LoRA + adversarial + multi-task)
  - `fair_trainer.py` — training loop with fairness objectives, metrics, and checkpointing
  - `fairness_metrics.py` — demographic parity, equalized odds, predictive parity, calibration, and report utils
  - plus classic modules (`data_loader.py`, `preprocessing.py`, `utils.py`, etc.)
- `configs/` — legacy config snippets; current training uses `fair_lora_config.py`
- `fair_lora_config.py` — centralized knobs: base model, LoRA, batching, fairness lambdas, device (CUDA/MPS/CPU)
- `data/`
  - `processed/processed_resume_dataset_resplit/` — default dataset splits used by notebooks and training
  - `eval/` — scored outputs and counterfactual pairs for analysis
- `models/`
  - `fair_adversarial/` — training outputs: checkpoints, best models, metrics CSV, history JSON
  - `peft_adapters/fairness_lora/` — exported LoRA adapter (PEFT) ready to load or publish
- `notebooks/` — end-to-end workflow: exploration (01), training (02), fairness evaluation (03), inference demo (04), counterfactual eval (05), publish to HF (06)
- `reports/` — figures and `fairness_metrics/fairness_evaluation.json`
- `scripts/` — utilities like `export_fair_lora_to_hf.py` (recommended for publishing)
- `tests/` — focused unit tests for preprocessing, model, and fairness metrics

## Installation

```bash
git clone <your-repo-url>
cd bge-lora-fairness-finetuning
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (to push to Hugging Face Hub later):

```bash
huggingface-cli login
```

## Data

This repo assumes a preprocessed resume/job dataset under:

- `data/processed/processed_resume_dataset_resplit/` (train/validation/test with `dataset_dict.json`)

Preparation and exploratory analysis are captured in:

- `data/processed/data_prepare.ipynb`
- `01_data_exploration.ipynb`

If you use your own data, update paths or logic in your notebook and, if needed, in `fair_lora_config.py`.

## Quickstart

Most users should run the notebooks:

1) `01_data_exploration.ipynb` — dataset overview and sanity checks
2) `02_model_training.ipynb` — fairness-aware LoRA training (writes to `models/fair_adversarial/`)
3) `03_fairness_evaluation.ipynb` — utility and fairness metrics, plots, and JSON reports

For a minimal programmatic example (Python), you can follow this outline using the core APIs:

```python
from src.fair_lora_model import FairLoRAModel
from src.fair_trainer import FairTrainer
from fair_lora_config import *

# Build model
model = FairLoRAModel(base_model_name=BASE_MODEL, use_lora=USE_LORA, use_adversarial=USE_ADVERSARIAL_DEBIASING, use_multitask=USE_MULTITASK, num_labels=NUM_LABELS)

# Build your DataLoaders (see notebooks for the project’s dataloader that yields
# dict batches with keys: input_ids, attention_mask, label, school_category, is_top_school)
train_loader = ...
val_loader = ...

trainer = FairTrainer(
    model,
    train_loader,
    val_loader,
    device=str(DEVICE),
    learning_rate=LEARNING_RATE,
    adversarial_lambda=ADVERSARIAL_LAMBDA,
    fairness_lambda=FAIRNESS_LAMBDA,
    multitask_lambda=MULTITASK_LAMBDA,
    fairness_reg_lambda=FAIRNESS_REG_LAMBDA,
    save_dir=Path("models/fair_adversarial")
)

history = trainer.train(
    num_epochs=15,
    early_stopping_patience=5,
    warmup_steps=500,
    window_selection_start=WINDOW_SELECTION_START,
    window_selection_end=WINDOW_SELECTION_END,
    window_fairness_threshold=WINDOW_FAIRNESS_THRESHOLD
)
```

Outputs include checkpoints per epoch, `best_model.pt`, `best_fairness_model.pt`, `best_util_model.pt`, a window-selected checkpoint (`window_best_model.pt`), `epoch_metrics.csv`, and `training_history.json` under `models/fair_adversarial/`.

## Inference

Load the PEFT adapter locally (from `models/peft_adapters/fairness_lora/`) or from the Hub:

```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch, torch.nn.functional as F

BASE = "BAAI/bge-large-en-v1.5"
ADAPTER = "models/peft_adapters/fairness_lora"  # or "<username>/<repo>" on the Hub

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModel.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

def encode(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**enc)
        emb = F.normalize(out.last_hidden_state.mean(dim=1), p=2, dim=1)
    return emb

q = "Software engineer with Python experience."
j = "Hiring backend Python developer."
cos = (encode(q) * encode(j)).sum(dim=1).item()
print({"cosine": cos})
```

See `notebooks/04_inference_demo.ipynb` for a more complete demo and thresholding examples.

## Evaluation and Fairness Reports

- During training, validation metrics are logged to `models/fair_adversarial/epoch_metrics.csv` and `training_history.json`.
- Post-training notebooks generate richer analyses:
  - `03_fairness_evaluation.ipynb`
  - `05_scoring_and_counterfactual_eval.ipynb` (uses `data/eval/*`)
- Aggregated fairness outputs are saved under `reports/fairness_metrics/` (e.g., `fairness_evaluation.json`).

Key metrics your pipeline monitors:

- Utility: accuracy, ROC-AUC, PR-AUC
- Fairness: demographic parity diff, equalized odds (TPR/FPR) diffs, predictive parity diff
- Overall fairness score (lower is better), both raw and balanced-validation variants

## Export LoRA adapter to Hugging Face Hub

Use the dedicated script to extract and upload only the adapter weights:

```bash
python scripts/export_fair_lora_to_hf.py \
  --checkpoint models/fair_adversarial/best_fairness_model.pt \
  --adapter-dir models/lora_adapters/fairness_lora \
  --repo-id <your-username/fair-resume-matcher-lora> \
  --push
```

Flags:

- `--private` to create a private repo
- `--base-model` if you need to override the base model name

After upload, consumers can load via `PeftModel.from_pretrained(base, "<user/repo>")`.

## Configuration tips (fair_lora_config.py)

- Device is auto-selected (CUDA, then Apple Metal MPS, else CPU)
- LoRA hyperparameters: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES`
- Fairness knobs: `ADVERSARIAL_LAMBDA`, `FAIRNESS_LAMBDA`, `MULTITASK_LAMBDA`, `FAIRNESS_REG_LAMBDA`
- Selection window: `WINDOW_SELECTION_START/END`, threshold `WINDOW_FAIRNESS_THRESHOLD`
- Sensitive groups: `SCHOOL_CATEGORIES` and attribute masking controls

## Testing

Run unit tests (optionally after creating/activating your venv):

```bash
pytest -q
```

## Troubleshooting

- macOS: MPS (Apple Silicon) is supported and auto-detected; reduce `BATCH_SIZE` if you hit memory limits.
- CUDA OOM: lower `BATCH_SIZE`, `MAX_LENGTH`, or use gradient accumulation; ensure only the last encoder layers and LoRA adapters are trainable (the trainer and model handle this by default).
- Hugging Face auth: run `huggingface-cli login` before `--push` exporting.
- Outdated scripts: prefer notebooks and the `FairLoRAModel`/`FairTrainer` APIs. `scripts/inference.py` and some legacy configs are kept for reference and may not reflect the latest pipeline.

## License

MIT License.