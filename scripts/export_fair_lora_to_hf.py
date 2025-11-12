#!/usr/bin/env python
"""
Export the fair LoRA adapter and push to Hugging Face Hub.

This script will:
1) Rebuild the FairLoRAModel with the configured base model
2) Load your training checkpoint (e.g., models/fair_adversarial/best_fairness_model.pt)
3) Extract and save only the LoRA adapter (PEFT) weights
4) Create a lightweight model card (README.md)
5) Optionally create a repo and upload the adapter folder to Hugging Face Hub

Usage (requires `huggingface_hub` login beforehand via CLI or env HF_TOKEN):

  python scripts/export_fair_lora_to_hf.py \
      --checkpoint models/fair_adversarial/best_fairness_model.pt \
      --repo-id <your-username/fair-resume-matcher-lora> \
      --adapter-dir models/lora_adapters/fairness_lora \
      --push

Optional flags:
  --private             Create a private repo on the Hub
  --adapter-name NAME   Logical adapter name displayed in usage snippet (default: fairness-lora)
  --base-model NAME     Override base model name (default from fair_lora_config or BAAI/bge-large-en-v1.5)

After upload, users can load the adapter with:

  from transformers import AutoModel, AutoTokenizer
  from peft import PeftModel

  base = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
  model = PeftModel.from_pretrained(base, "<your-username/fair-resume-matcher-lora>")

"""

import argparse
import json
from pathlib import Path
import sys

import torch

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fair_lora_config import BASE_MODEL as CONFIG_BASE_MODEL
except Exception:
    CONFIG_BASE_MODEL = "BAAI/bge-large-en-v1.5"

from src.fair_lora_model import FairLoRAModel
from typing import Optional, Dict, Any


def build_model(base_model_name: str) -> FairLoRAModel:
    # Build with LoRA enabled; other heads present but we only export adapter
    model = FairLoRAModel(base_model_name=base_model_name, use_lora=True, use_adversarial=True, use_multitask=True, num_labels=2)
    return model


def load_checkpoint_into_model(model: FairLoRAModel, ckpt_path: Path, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", None)
    if state is None and isinstance(ckpt, dict):
        state = ckpt
    if state is None:
        raise RuntimeError(f"No model_state_dict found in checkpoint: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return missing, unexpected


def write_model_card(adapter_dir: Path, repo_id: str, base_model_name: str, adapter_name: str, metrics: Optional[Dict[str, Any]] = None):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Use a regular (non-f) triple-quoted string and format placeholders to avoid f-string brace issues
    card_template = """
# {repo_id}

Fairness-aware LoRA adapter for resume–job matching built on top of `{base_model_name}`.

This adapter was trained with adversarial debiasing and multi-task objectives to reduce group disparities while maintaining utility.

## Usage

```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

BASE = "{base_model_name}"
ADAPTER = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModel.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# Encode two texts and compute cosine
import torch
import torch.nn.functional as F

def encode(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**enc)
        hidden = out.last_hidden_state
        emb = F.normalize(hidden.mean(dim=1), p=2, dim=1)
    return emb

r = "Software engineer with Python experience."
j = "Hiring backend Python developer."
cos = (encode(r) * encode(j)).sum(dim=1).item()
prob = torch.sigmoid(torch.tensor(cos)).item()
print({{"cosine": cos, "prob": prob}})
```

## Training notes
- Base model: `{base_model_name}`
- Technique: LoRA + adversarial debiasing + multi-task attribute prediction
- Objective: Lower demographic parity / equalized odds gaps while preserving accuracy/AUC

## Metrics (summary)
{metrics_block}

## License
- Please ensure the license of the base model `{base_model_name}` allows derivative adapters.
- Provide your dataset and usage terms accordingly.
"""
    metrics_block = json.dumps(metrics or {}, indent=2)
    card = card_template.format(
        repo_id=repo_id,
        base_model_name=base_model_name,
        metrics_block=metrics_block,
    )
    (adapter_dir / "README.md").write_text(card.strip() + "\n", encoding="utf-8")


def maybe_collect_metrics(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        out = {}
        # Best fairness (min) and best AUC (max)
        if 'fairness_score' in df.columns:
            out['min_fairness_score'] = float(df['fairness_score'].min())
        if 'val_auc' in df.columns:
            out['max_auc'] = float(df['val_auc'].max())
        if 'val_pr_auc' in df.columns:
            out['max_pr_auc'] = float(df['val_pr_auc'].max())
        if 'acc_thr_0_5' in df.columns:
            out['max_acc_thr_0_5'] = float(df['acc_thr_0_5'].max())
        return out
    except Exception:
        return {}


def push_to_hub(adapter_dir: Path, repo_id: str, private: bool = False):
    from huggingface_hub import HfApi, create_repo, upload_folder
    api = HfApi()
    # Create the repo if not exists
    create_repo(repo_id, private=private, exist_ok=True)
    # Upload the entire adapter directory
    upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_dir),
        path_in_repo=".",
        commit_message="Upload fairness LoRA adapter"
    )
    print(f"✓ Uploaded adapter to https://huggingface.co/{repo_id}")


def main():
    p = argparse.ArgumentParser(description="Export fair LoRA adapter and push to Hugging Face Hub")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint .pt")
    p.add_argument("--adapter-dir", type=str, default="models/lora_adapters/fairness_lora", help="Local output dir for adapter")
    p.add_argument("--repo-id", type=str, help="Hugging Face repo id, e.g., username/repo-name")
    p.add_argument("--adapter-name", type=str, default="fairness-lora", help="Adapter display name in docs")
    p.add_argument("--base-model", type=str, default=CONFIG_BASE_MODEL, help="Base model name (if not using config)")
    p.add_argument("--push", action="store_true", help="If set, push to Hugging Face Hub after export")
    p.add_argument("--private", action="store_true", help="Create a private repo on Hub")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build model and load weights
    model = build_model(args.base_model)
    load_checkpoint_into_model(model, ckpt_path)

    # 2) Save only the PEFT adapter
    #   model.base_model is a PEFT-wrapped model; use its .save_pretrained
    try:
        model.base_model.save_pretrained(out_dir)
        print(f"✓ Saved PEFT adapter to: {out_dir}")
    except Exception as e:
        print(f"Failed to save adapter: {e}")
        sys.exit(1)

    # 3) Add a model card
    metrics = maybe_collect_metrics(Path("models/fair_adversarial/epoch_metrics.csv"))
    repo_id = args.repo_id or ""
    write_model_card(out_dir, repo_id or out_dir.name, args.base_model, args.adapter_name, metrics)

    # 4) Optionally push to Hub
    if args.push:
        if not repo_id:
            print("--push specified but --repo-id is missing. Aborting upload.")
            sys.exit(2)
        push_to_hub(out_dir, repo_id, private=args.private)

    print("Done.")


if __name__ == "__main__":
    main()
