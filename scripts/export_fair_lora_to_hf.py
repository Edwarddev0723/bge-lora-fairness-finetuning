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
    metrics = metrics or {}
    # Extract common metrics if present
    min_fair = metrics.get("min_fairness_score")
    max_auc = metrics.get("max_auc")
    max_pr_auc = metrics.get("max_pr_auc")
    max_acc = metrics.get("max_acc_thr_0_5")
    fairness_report = metrics.get("fairness_report")

    # YAML front matter for Hub rendering
    front_matter = """---
tags:
  - lora
  - peft
  - fairness
  - resume-matching
  - retrieval
  - sentence-similarity
library_name: peft
base_model: {base_model}
pipeline_tag: sentence-similarity
language:
  - en
---
""".format(base_model=base_model_name)

    # Richer model card with sections
    card_template = """
{front_matter}
# {title}

Fairness-aware LoRA adapter for resume–job matching built on top of `{base_model_name}`.

This adapter was trained with adversarial debiasing and multi-task objectives to reduce group disparities while maintaining utility.

## Model Summary
- Base model: `{base_model_name}`
- Adapter type: LoRA (PEFT)
- Task: Resume–job text similarity (cosine over mean-pooled, L2-normalized embeddings; optional sigmoid for probability)
- Intended audience: Researchers and practitioners exploring fairness-aware matching

## Quick Start

```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch, torch.nn.functional as F

BASE = "{base_model_name}"
ADAPTER = "{adapter_or_repo}"  # replace with your Hub repo id or local path

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModel.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

def encode(text: str):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**enc)
        emb = F.normalize(out.last_hidden_state.mean(dim=1), p=2, dim=1)
    return emb

r = "Software engineer with Python experience."
j = "Hiring backend Python developer."
cos = (encode(r) * encode(j)).sum(dim=1).item()
prob = torch.sigmoid(torch.tensor(cos)).item()
print({{"cosine": cos, "prob": prob}})
```

## Training & Data (overview)
- Approach: LoRA fine-tuning with adversarial debiasing + multi-task attribute prediction.
- See repository for data preparation pipelines under `data/processed/` and training code in `src/fair_trainer.py`.

## Evaluation (high level)
Key utility and fairness metrics (if available in `models/fair_adversarial/epoch_metrics.csv` and reports):

| Metric | Value |
|---|---|
| Max AUC | {max_auc_str} |
| Max PR-AUC | {max_pr_auc_str} |
| Max Acc@thr=0.5 | {max_acc_str} |
| Min Fairness Score (lower is better) | {min_fair_str} |

Additional fairness and consistency analyses (when available):
- Demographic Parity gap at fixed acceptance rate
- Top-K exposure per job
- Local consistency (nearest-neighbor stability)
- Counterfactual shift and flip rate

See `notebooks/05_scoring_and_counterfactual_eval.ipynb` for the full evaluation and plots.

{fairness_block}

## Intended Use & Limitations
Intended for research and educational use in fairness-aware matching. Not a substitute for human oversight in hiring decisions.

Limitations:
- Residual bias may persist; results depend on data coverage and definitions of fairness.
- Thresholds on cosine/sigmoid affect selection rates and fairness gaps.
- The adapter specializes the base model for this domain and may not generalize to unrelated tasks.

## How to Cite / Acknowledge
Please cite this repository and the base model `{base_model_name}` if you use this adapter in your work.

## License & Usage
- Ensure the license of `{base_model_name}` permits derivative adapters for your use case.
- Review any dataset terms relevant to your deployment context.
"""

    def fmt(v):
        return "—" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))

    fairness_block = ""
    if isinstance(fairness_report, dict) and fairness_report:
        try:
            fairness_block = "\n### Fairness report (excerpt)\n\n```json\n" + json.dumps(fairness_report, indent=2) + "\n```\n"
        except Exception:
            fairness_block = "\n### Fairness report (available)\n\nSee reports in the repository for details.\n"

    card = card_template.format(
        front_matter=front_matter,
        title=repo_id or adapter_name,
        base_model_name=base_model_name,
        adapter_or_repo=repo_id or adapter_name,
        max_auc_str=fmt(max_auc),
        max_pr_auc_str=fmt(max_pr_auc),
        max_acc_str=fmt(max_acc),
        min_fair_str=fmt(min_fair),
        fairness_block=fairness_block,
    )

    (adapter_dir / "README.md").write_text(card.strip() + "\n", encoding="utf-8")


def maybe_collect_metrics(csv_path: Path) -> dict:
    out: Dict[str, Any] = {}
    # CSV metrics
    if csv_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            if 'fairness_score' in df.columns:
                out['min_fairness_score'] = float(df['fairness_score'].min())
            if 'val_auc' in df.columns:
                out['max_auc'] = float(df['val_auc'].max())
            if 'val_pr_auc' in df.columns:
                out['max_pr_auc'] = float(df['val_pr_auc'].max())
            if 'acc_thr_0_5' in df.columns:
                out['max_acc_thr_0_5'] = float(df['acc_thr_0_5'].max())
        except Exception:
            pass

    # Optional fairness evaluation JSON
    fairness_json = Path("reports/fairness_metrics/fairness_evaluation.json")
    if fairness_json.exists():
        try:
            out['fairness_report'] = json.loads(fairness_json.read_text(encoding='utf-8'))
        except Exception:
            pass
    return out


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
