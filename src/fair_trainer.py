"""FairTrainer
Enhanced training loop for fairness-aware LoRA fine-tuning.

New features added:
1. Step-based lambda ramp (adv & multitask) instead of epoch piecewise.
2. Prediction gap fairness regularizer (group probability max-min).
3. ROC AUC & PR AUC metrics in validation.
4. Persistent temperature scaling (calibrate once; reuse for tuned threshold search).
5. Unified CSV logging of epoch metrics & dynamic lambdas.
6. Window-based checkpoint selection (epochs 7–12, fairness < threshold).
7. Hooks for balanced validation loader (optional) used for fairness & early stopping stability.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path as _Path
import csv

try:
    from fair_lora_config import *
except ImportError:
    # Fallback defaults
    ADVERSARIAL_LAMBDA = 0.5
    FAIRNESS_LAMBDA = 0.3
    MULTITASK_LAMBDA = 0.2
    SAVE_DIR = Path("./models/fair_adversarial")

from src.fair_lora_model import FairLoRAModel, gradient_reversal
from src.fairness_metrics import FairnessMetrics, FairnessLoss


class FairTrainer:
    """
    Trainer for fair LoRA fine-tuning with multiple fairness objectives.
    
    Features:
    - Adversarial debiasing to remove sensitive information from embeddings
    - Fairness regularization for demographic parity / equalized odds
    - Multi-task learning with attribute classification
    - Fairness-aware early stopping
    """
    
    def __init__(
        self,
        model: FairLoRAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        balanced_val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 2e-5,
        adversarial_lambda: float = 0.5,
        fairness_lambda: float = 0.3,
        multitask_lambda: float = 0.2,
        fairness_reg_lambda: float = 0.0,
        max_grad_norm: float = 1.0,
        save_dir: Path = Path("./models/fair_adversarial"),
        temperature_calibration: bool = True,
        auc_enabled: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: FairLoRAModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            adversarial_lambda: Weight for adversarial loss
            fairness_lambda: Weight for fairness regularization
            multitask_lambda: Weight for multi-task loss
            max_grad_norm: Max gradient norm for clipping
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.balanced_val_loader = balanced_val_loader  # Optional artificially balanced validation loader
        self.device = device
        
        self.adversarial_lambda = adversarial_lambda
        self.fairness_lambda = fairness_lambda
        self.multitask_lambda = multitask_lambda
        self.fairness_reg_lambda = fairness_reg_lambda
        self.max_grad_norm = max_grad_norm
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizers with grouped LRs + LLRD
        # Param groups:
        # - Heads/projections: 2e-4
        # - LoRA adapters: 1e-4
        # - Base last4 attention (non-LoRA): start 3e-5 with 0.85 layer-wise decay
        self.learning_rate = learning_rate
        param_groups = []
        heads_patterns = ['main_projection', 'adv_projection', 'attr_projection', 'classifier']
        lora_group = []
        heads_group = []
        base_llrd_groups = {}
        # Determine encoder depth for LLRD
        base_prefix = None
        n_layers = None
        if hasattr(model.base_model, 'encoder') and hasattr(model.base_model.encoder, 'layer'):
            n_layers = len(model.base_model.encoder.layer)
            base_prefix = 'encoder.layer.'
        elif hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'encoder') and hasattr(model.base_model.model.encoder, 'layer'):
            n_layers = len(model.base_model.model.encoder.layer)
            base_prefix = 'model.encoder.layer.'
        # Helper to get layer index
        def get_layer_idx(name: str):
            if base_prefix and base_prefix in name:
                try:
                    after = name.split(base_prefix, 1)[1]
                    return int(after.split('.', 1)[0])
                except Exception:
                    return None
            return None
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'adversary' in n:
                continue  # separate optimizer below
            # LoRA params
            if 'lora_' in n:
                lora_group.append(p)
                continue
            # Heads/projections
            if any(hp in n for hp in heads_patterns):
                heads_group.append(p)
                continue
            # Base last4 with LLRD (attention base weights marked requires_grad by model)
            idx = get_layer_idx(n)
            if n_layers is not None and idx is not None:
                # far from top -> deeper -> stronger decay; top layer = n_layers-1
                dist = (n_layers - 1) - idx
                lr = 3e-5 * (0.85 ** max(0, dist))
                base_llrd_groups.setdefault(lr, []).append(p)
            else:
                # Fallback: treat as base low-lr
                base_llrd_groups.setdefault(3e-5, []).append(p)
        if heads_group:
            param_groups.append({'params': heads_group, 'lr': 2e-4})
        if lora_group:
            param_groups.append({'params': lora_group, 'lr': 1e-4})
        for lr, params in base_llrd_groups.items():
            if params:
                param_groups.append({'params': params, 'lr': lr})
        if not param_groups:
            # Fallback to all trainable
            main_params = [p for n, p in model.named_parameters() if ('adversary' not in n) and p.requires_grad]
            param_groups = [{'params': main_params, 'lr': self.learning_rate}]
        self.main_optimizer = torch.optim.AdamW(param_groups)
        
        # Adversarial discriminator parameters
        if model.use_adversarial:
            adv_params = list(model.adversary.parameters())
            self.adv_optimizer = torch.optim.AdamW(adv_params, lr=self.learning_rate * 2)
        else:
            self.adv_optimizer = None
        
        # Loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.attribute_loss_fn = nn.CrossEntropyLoss()
        self.fairness_loss_fn = FairnessLoss(lambda_fairness=fairness_lambda)
        # Small L2 on group gap as a fuse
        self.gap_l2_lambda = 0.03
        
        # Metrics
        self.fairness_metrics = FairnessMetrics()
        # Temperature scaling
        self.temperature_calibration = temperature_calibration
        self.temperature = 1.0
        self.temperature_fixed = False
        self.auc_enabled = auc_enabled
        # Dynamic adversarial lambda cap control
        self._adv_cap_active = False
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auc': [],
            'val_pr_auc': [],
            'fairness_scores': [],
            'fairness_scores_raw': [],
            'fairness_scores_balanced': [],
            'fairness_reg_gaps': [],
            'adv_lambdas': [],
            'multitask_lambdas': [],
            'best_thresholds': [],
            'temperature': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'best_fairness_score': float('inf'),
            'window_best_epoch': None
        }
        # CSV logging setup
        self.metrics_log_path = self.save_dir / "epoch_metrics.csv"
        if not self.metrics_log_path.exists():
            with open(self.metrics_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch','train_loss','train_acc','val_loss','val_acc','acc_thr_0_5','val_auc','val_pr_auc',
                    'fairness_score','fairness_score_raw','fairness_score_balanced',
                    'fairness_reg_gap','adv_lambda','multitask_lambda','best_threshold','temperature'])
    
    def train_epoch(self,
                    epoch: int,
                    adv_lambda: Optional[float] = None,
                    multitask_lambda: Optional[float] = None,
                    fair_lambda: Optional[float] = None,
                    scheduler=None,
                    global_step_start: int = 0,
                    warmup_steps: int = 0,
                    total_steps: int = 0) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_class_loss = 0
        total_adv_loss = 0
        total_fairness_loss = 0
        total_multitask_loss = 0
        
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        fairness_reg_gap_accum = 0.0
        current_step = global_step_start
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            school_categories = batch['school_category'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                sensitive_attr=school_categories,
                return_embeddings=True
            )
            logits = outputs['logits']
            # Head-specific embeddings (main / adv / attr)
            # Avoid using Python 'or' with tensors (ambiguous truth value). Use explicit None checks.
            embeddings_main = outputs.get('embeddings_main')
            if embeddings_main is None:
                embeddings_main = outputs.get('embeddings')  # fallback for older model interface

            embeddings_adv = outputs.get('embeddings_adv')
            if embeddings_adv is None:
                embeddings_adv = embeddings_main

            embeddings_attr = outputs.get('embeddings_attr')
            if embeddings_attr is None:
                embeddings_attr = embeddings_main
            
            # Classification loss
            class_loss = self.classification_loss_fn(logits, labels)
            
            # Adversarial loss (if enabled)
            adv_loss = torch.tensor(0.0, device=self.device)
            if self.model.use_adversarial and self.adv_optimizer is not None and embeddings_adv is not None:
                # 1) Update adversary to best predict group from detached embeddings
                adv_logits_detached = self.model.adversary(embeddings_adv.detach())
                adv_loss_detached = self.classification_loss_fn(adv_logits_detached, school_categories)
                self.adv_optimizer.zero_grad()
                adv_loss_detached.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.adversary.parameters(), self.max_grad_norm
                )
                self.adv_optimizer.step()

                # 2) For main model, reverse gradients to discourage group leakage
                adv_logits_main = self.model.adversary(gradient_reversal(embeddings_adv, alpha=1.0))
                adv_loss = self.classification_loss_fn(adv_logits_main, school_categories)
            
            # Fairness loss (demographic parity / equalized odds)
            fairness_loss = self.fairness_loss_fn.demographic_parity_loss(
                logits, school_categories
            )
            fairness_loss += self.fairness_loss_fn.equalized_odds_loss(
                logits, labels, school_categories
            )
            
            # Multi-task loss (if enabled)
            multitask_loss = torch.tensor(0.0, device=self.device)
            if self.model.use_multitask and embeddings_attr is not None:
                attr_logits = self.model.attribute_classifier(embeddings_attr)
                multitask_loss = self.attribute_loss_fn(attr_logits, school_categories)
            
            # Resolve dynamic lambdas (warmup)
            # Step-based ramp (linear 0->target over warmup_steps)
            ramp_factor = 1.0
            if warmup_steps > 0:
                ramp_factor = min(1.0, current_step / float(warmup_steps))
            adv_target_lmb = self.adversarial_lambda if adv_lambda is None else adv_lambda
            mt_target_lmb = self.multitask_lambda if multitask_lambda is None else multitask_lambda
            adv_lmb = adv_target_lmb * ramp_factor
            mt_lmb = mt_target_lmb * ramp_factor
            fair_lmb = self.fairness_lambda if fair_lambda is None else fair_lambda

            # Total loss
            # Note: adversarial loss is already used to update discriminator
            # For main model, we use -adv_loss to encourage invariant representations
            # With GRL applied above, we ADD the adversarial loss here.
            # Fairness regularizer: prediction gap across groups (prob positive)
            # Compute group probability gap and apply two regularizers: linear (if enabled) and small L2 fuse
            fairness_reg_loss = torch.tensor(0.0, device=self.device)
            gap_l2_loss = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                probs_pos = torch.softmax(logits, dim=1)[:, 1]
                gaps = []
                # group by school_categories
                for g in torch.unique(school_categories):
                    mask = school_categories == g
                    if mask.sum() > 0:
                        gaps.append(probs_pos[mask].mean())
                if len(gaps) >= 2:
                    gap_val = torch.stack(gaps)
                    fairness_reg_gap = (gap_val.max() - gap_val.min()).detach()
                    if self.fairness_reg_lambda > 0.0:
                        fairness_reg_loss = fairness_reg_gap * self.fairness_reg_lambda
                    gap_l2_loss = (fairness_reg_gap ** 2) * self.gap_l2_lambda
                    fairness_reg_gap_accum += float(fairness_reg_gap.item())

            # Apply dynamic adversarial cap once target fairness is met (activated externally)
            if getattr(self, '_adv_cap_active', False):
                adv_lmb = min(max(adv_lmb, 0.3), 0.5)

            total_batch_loss = (class_loss + fair_lmb * fairness_loss + mt_lmb * multitask_loss + adv_lmb * adv_loss + fairness_reg_loss + gap_l2_loss)
            
            # Backward pass for main model
            self.main_optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm
            )
            self.main_optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Accumulate metrics
            total_loss += total_batch_loss.item()
            total_class_loss += class_loss.item()
            total_adv_loss += adv_loss.item()
            total_fairness_loss += fairness_loss.item()
            total_multitask_loss += multitask_loss.item()
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_batch_loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%', 'advλ': f'{adv_lmb:.3f}', 'mtλ': f'{mt_lmb:.3f}'})
            current_step += 1
        
        # Compute epoch metrics
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'class_loss': total_class_loss / num_batches,
            'adv_loss': total_adv_loss / num_batches,
            'fairness_loss': total_fairness_loss / num_batches,
            'multitask_loss': total_multitask_loss / num_batches,
            'fairness_reg_gap': fairness_reg_gap_accum / max(1, num_batches),
            'accuracy': correct / total,
            'final_adv_lambda': adv_lmb,
            'final_multitask_lambda': mt_lmb
        }
        
        return metrics
    
    def validate(self, epoch: int, threshold_search: bool = True, fairness_penalty_weight: float = 0.0, loader: Optional[DataLoader] = None, use_temperature: bool = True) -> Dict[str, Any]:
        """Validate model.

        - Tuned threshold search (F1) on (optionally temperature-scaled) probabilities.
        - Fairness metrics at raw fixed threshold 0.5.
        - Optional ROC AUC & PR AUC.
        - Can pass a custom loader (e.g., balanced validation) for fairness-only assessment.
        """
        self.model.eval()
        eval_loader = loader if loader is not None else self.val_loader
        total_loss = 0.0
        all_labels: list = []
        all_predictions: list = []
        all_probabilities: list = []
        all_logits: list = []
        all_school_categories: list = []
        all_is_top_school: list = []

        with torch.no_grad():
            pbar = tqdm(eval_loader, desc=f"Epoch {epoch} [Val{'-Balanced' if loader is not None else ''}]")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                sc = batch['school_category'].to(self.device)
                ts = batch['is_top_school'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, sensitive_attr=sc)
                logits = outputs['logits']
                loss = self.classification_loss_fn(logits, labels)
                total_loss += float(loss.item())

                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_probabilities.extend(probs[:, 1].cpu().numpy().tolist())
                all_school_categories.extend(sc.cpu().numpy().tolist())
                all_is_top_school.extend(ts.cpu().numpy().tolist())
                all_logits.append(logits.cpu())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        labels_np = np.array(all_labels)
        probs_np = np.array(all_probabilities)
        sc_np = np.array(all_school_categories)
        ts_np = np.array(all_is_top_school)

        logits_np = torch.cat(all_logits, dim=0).numpy() if len(all_logits) else np.zeros((0, self.model.num_labels))
        # Temperature scaling (calibrate once on original validation loader)
        scaled_probs_np = probs_np.copy()
        if threshold_search and self.temperature_calibration and not self.temperature_fixed and loader is None and labels_np.size > 0:
            self._calibrate_temperature(logits_np, labels_np)
        if use_temperature and self.temperature_fixed and logits_np.shape[0] > 0:
            scaled_probs_np = self._apply_temperature(logits_np)

        best_thr = 0.5
        tuned_predictions = np.array(all_predictions)
        if threshold_search and labels_np.size > 0 and probs_np.size > 0:
            from sklearn.metrics import f1_score
            cand = np.linspace(0.1, 0.8, 15)
            best_score = -1e9
            for thr in cand:
                preds_t = (scaled_probs_np >= thr).astype(int)
                f1 = f1_score(labels_np, preds_t, zero_division=0)
                pen = 0.0
                if fairness_penalty_weight > 0.0 and sc_np.size > 0:
                    rates = []
                    for g in np.unique(sc_np):
                        m = (sc_np == g)
                        if m.sum() == 0:
                            continue
                        rates.append(preds_t[m].mean())
                    if len(rates) >= 2:
                        pen -= fairness_penalty_weight * (np.max(rates) - np.min(rates))
                score = f1 + pen
                if score > best_score:
                    best_score = score
                    best_thr = float(thr)
            tuned_predictions = (scaled_probs_np >= best_thr).astype(int)

        # Fairness & raw accuracy strictly at 0.5
        fairness_predictions = (probs_np >= 0.5).astype(int)
        acc_thr_0_5 = float((fairness_predictions == labels_np).mean()) if labels_np.size else 0.0

        # Group counts
        unique_sc, counts_sc = (np.unique(sc_np), np.unique(sc_np, return_counts=True)[1]) if sc_np.size else ([], [])
        unique_ts, counts_ts = (np.unique(ts_np), np.unique(ts_np, return_counts=True)[1]) if ts_np.size else ([], [])
        group_counts_sc = {int(k): int(v) for k, v in zip(list(unique_sc), list(counts_sc))} if len(unique_sc) else {}
        group_counts_ts = {int(k): int(v) for k, v in zip(list(unique_ts), list(counts_ts))} if len(unique_ts) else {}

        fairness_valid_sc = len(unique_sc) >= 2
        fairness_valid_ts = len(unique_ts) >= 2

        def balanced_indices(groups: np.ndarray, seed: int = 42):
            if groups.size == 0:
                return None
            uniq, cnt = np.unique(groups, return_counts=True)
            if len(uniq) < 2:
                return None
            m = int(cnt.min())
            if m == 0:
                return None
            rng = np.random.default_rng(seed)
            idxs = []
            for g in uniq:
                gidx = np.where(groups == g)[0]
                if gidx.size < m:
                    return None
                sel = rng.choice(gidx, size=m, replace=False)
                idxs.append(sel)
            return np.concatenate(idxs)

        # Raw fairness
        fairness_scores_raw = []
        metrics_sc_raw = None
        metrics_ts_raw = None
        if fairness_valid_sc:
            metrics_sc_raw = self.fairness_metrics.compute_all_metrics(labels_np, fairness_predictions, probs_np, sc_np, 'school_category')
            fairness_scores_raw.append(metrics_sc_raw['overall_fairness_score'])
        if fairness_valid_ts:
            metrics_ts_raw = self.fairness_metrics.compute_all_metrics(labels_np, fairness_predictions, probs_np, ts_np, 'is_top_school')
            fairness_scores_raw.append(metrics_ts_raw['overall_fairness_score'])
        fairness_score_raw = float(np.mean(fairness_scores_raw)) if fairness_scores_raw else float('inf')

        # Balanced fairness
        fairness_scores_bal = []
        metrics_sc_bal = None
        metrics_ts_bal = None
        bal_sc = balanced_indices(sc_np, seed=epoch + 123)
        if fairness_valid_sc and bal_sc is not None and bal_sc.size > 0:
            metrics_sc_bal = self.fairness_metrics.compute_all_metrics(labels_np[bal_sc], fairness_predictions[bal_sc], probs_np[bal_sc], sc_np[bal_sc], 'school_category')
            fairness_scores_bal.append(metrics_sc_bal['overall_fairness_score'])
        bal_ts = balanced_indices(ts_np, seed=epoch + 456)
        if fairness_valid_ts and bal_ts is not None and bal_ts.size > 0:
            metrics_ts_bal = self.fairness_metrics.compute_all_metrics(labels_np[bal_ts], fairness_predictions[bal_ts], probs_np[bal_ts], ts_np[bal_ts], 'is_top_school')
            fairness_scores_bal.append(metrics_ts_bal['overall_fairness_score'])
        fairness_score_bal = float(np.mean(fairness_scores_bal)) if fairness_scores_bal else float('inf')
        fairness_score = fairness_score_bal if np.isfinite(fairness_score_bal) else fairness_score_raw

        # Differences (prefer school_category raw)
        dp_diff = float('nan')
        eo_diff = float('nan')
        eo_opp_diff = float('nan')
        if metrics_sc_raw is not None:
            dp_diff = metrics_sc_raw['demographic_parity_difference']
            eo_diff = metrics_sc_raw['equalized_odds_avg_difference']
            eo_opp_diff = metrics_sc_raw['equal_opportunity_difference']
        elif metrics_ts_raw is not None:
            dp_diff = metrics_ts_raw['demographic_parity_difference']
            eo_diff = metrics_ts_raw['equalized_odds_avg_difference']
            eo_opp_diff = metrics_ts_raw['equal_opportunity_difference']

        def group_rates(groups: np.ndarray, preds: np.ndarray, labels: np.ndarray):
            out = {}
            for g in np.unique(groups):
                m = (groups == g)
                if m.sum() == 0:
                    continue
                out[int(g)] = {
                    'count': int(m.sum()),
                    'true_pos_rate': float(labels[m].mean()),
                    'pred_pos_rate_fairness_thr': float(preds[m].mean()),
                    'pred_pos_rate_tuned_thr': float(((scaled_probs_np >= best_thr).astype(int))[m].mean())
                }
            return out

        group_rates_school = group_rates(sc_np, fairness_predictions, labels_np)
        group_rates_top = group_rates(ts_np, fairness_predictions, labels_np)

        # AUC metrics (use scaled probabilities if temperature applied, else raw)
        val_auc = float('nan')
        val_pr_auc = float('nan')
        if self.auc_enabled and labels_np.size > 0 and len(np.unique(labels_np)) > 1:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                val_auc = roc_auc_score(labels_np, scaled_probs_np)
                val_pr_auc = average_precision_score(labels_np, scaled_probs_np)
            except Exception:
                pass

        return {
            'loss': total_loss / max(1, len(eval_loader)),
            'accuracy': float((tuned_predictions == labels_np).mean()) if labels_np.size else 0.0,
            'accuracy_thr_0_5': acc_thr_0_5,
            'auc': val_auc,
            'pr_auc': val_pr_auc,
            'fairness_score': fairness_score,
            'fairness_score_raw': fairness_score_raw,
            'fairness_score_balanced': fairness_score_bal,
            'demographic_parity_diff': dp_diff,
            'equalized_odds_diff': eo_diff,
            'equal_opportunity_diff': eo_opp_diff,
            'fairness_valid_school_category': fairness_valid_sc,
            'fairness_valid_is_top_school': fairness_valid_ts,
            'group_counts_school_category': group_counts_sc,
            'group_counts_is_top_school': group_counts_ts,
            'balanced_group_counts_school_category': {int(k): int(v) for k, v in zip(*np.unique(sc_np[bal_sc], return_counts=True))} if fairness_valid_sc and bal_sc is not None else {},
            'balanced_group_counts_is_top_school': {int(k): int(v) for k, v in zip(*np.unique(ts_np[bal_ts], return_counts=True))} if fairness_valid_ts and bal_ts is not None else {},
            'best_threshold': best_thr,
            'group_rates_school_category': group_rates_school,
            'group_rates_is_top_school': group_rates_top,
            'metrics_school_category_raw': metrics_sc_raw,
            'metrics_is_top_school_raw': metrics_ts_raw,
            'metrics_school_category_balanced': metrics_sc_bal,
            'metrics_is_top_school_balanced': metrics_ts_bal
        }
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
        fairness_weight: float = 0.5,  # retained for compatibility (unused in new combined score)
        test_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        warmup_steps: int = 0,
        adv_lambda_target: Optional[float] = None,
        multitask_lambda_target: Optional[float] = None,
        fairness_lambda_target: Optional[float] = None,
        window_selection_start: int = 7,
        window_selection_end: int = 12,
        window_fairness_threshold: float = 0.12
    ) -> Dict[str, Any]:
        """
        Train model for multiple epochs with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            fairness_weight: Weight for fairness in early stopping criterion (0-1)
                           0 = only loss, 1 = only fairness
        
        Returns:
            Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Adversarial debiasing: {self.model.use_adversarial}")
        print(f"Multi-task learning: {self.model.use_multitask}")
        print("-" * 60)
        
        best_score = float('inf')
        patience_counter = 0
        # Track fairness-for-ES history for normalization
        if 'fairness_for_es' not in self.history:
            self.history['fairness_for_es'] = []

        total_steps = num_epochs * len(self.train_loader)
        global_step = 0
        for epoch in range(1, num_epochs + 1):
            adv_target = adv_lambda_target if adv_lambda_target is not None else self.adversarial_lambda
            mt_target = multitask_lambda_target if multitask_lambda_target is not None else self.multitask_lambda
            fair_target = fairness_lambda_target if fairness_lambda_target is not None else self.fairness_lambda

            # Train epoch with step-based ramp
            train_metrics = self.train_epoch(
                epoch,
                adv_lambda=adv_target,
                multitask_lambda=mt_target,
                fair_lambda=fair_target,
                scheduler=scheduler,
                global_step_start=global_step,
                warmup_steps=warmup_steps,
                total_steps=total_steps
            )
            global_step += len(self.train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['fairness_reg_gaps'].append(train_metrics['fairness_reg_gap'])
            self.history['adv_lambdas'].append(train_metrics['final_adv_lambda'])
            self.history['multitask_lambdas'].append(train_metrics['final_multitask_lambda'])

            # Validate
            val_metrics = self.validate(epoch, threshold_search=True, fairness_penalty_weight=0.0, loader=None, use_temperature=True)
            # Balanced fairness evaluation (loader only for fairness stability)
            balanced_metrics = None
            if self.balanced_val_loader is not None:
                balanced_metrics = self.validate(epoch, threshold_search=False, fairness_penalty_weight=0.0, loader=self.balanced_val_loader, use_temperature=False)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['fairness_scores'].append(val_metrics['fairness_score'])
            self.history['fairness_scores_raw'].append(val_metrics['fairness_score_raw'])
            self.history['fairness_scores_balanced'].append(
                balanced_metrics['fairness_score'] if balanced_metrics is not None else val_metrics['fairness_score_balanced']
            )
            self.history['val_auc'].append(val_metrics.get('auc', float('nan')))
            self.history['val_pr_auc'].append(val_metrics.get('pr_auc', float('nan')))
            self.history['best_thresholds'].append(val_metrics.get('best_threshold', 0.5))
            self.history['temperature'].append(self.temperature)

            # Combined score for early stopping; prefer balanced fairness when available
            MIN_GROUP_COUNT = 20
            sufficient_groups_raw = all([
                min(val_metrics['group_counts_school_category'].values()) >= MIN_GROUP_COUNT if val_metrics['group_counts_school_category'] else False,
                min(val_metrics['group_counts_is_top_school'].values()) >= MIN_GROUP_COUNT if val_metrics['group_counts_is_top_school'] else False
            ])

            # Balanced fairness is defined whenever we can sample at least 1 per group
            has_balanced_sc = bool(val_metrics.get('balanced_group_counts_school_category'))
            has_balanced_ts = bool(val_metrics.get('balanced_group_counts_is_top_school'))
            sufficient_groups_balanced = has_balanced_sc or has_balanced_ts

            if balanced_metrics is not None:
                fairness_for_es = balanced_metrics.get('fairness_score', balanced_metrics.get('fairness_score_balanced'))
            else:
                if sufficient_groups_balanced:
                    fairness_for_es = val_metrics.get('fairness_score_balanced', val_metrics.get('fairness_score'))
                elif sufficient_groups_raw:
                    fairness_for_es = val_metrics.get('fairness_score')
                else:
                    fairness_for_es = None

            # Record fairness_for_es history for normalization
            if fairness_for_es is None or (isinstance(fairness_for_es, float) and not np.isfinite(fairness_for_es)):
                fairness_for_es = val_metrics.get('fairness_score')
            # Ensure scalar float
            if fairness_for_es is None:
                fairness_scalar = float('inf')
            else:
                try:
                    fairness_scalar = float(fairness_for_es)
                except Exception:
                    fairness_scalar = float('inf')
            if not np.isfinite(fairness_scalar):
                fairness_scalar = float('inf')
            self.history['fairness_for_es'].append(fairness_scalar)

            # Multi-objective normalized score (minimize)
            # score = 0.5 * norm(val_loss) + 0.3 * norm(fairness_for_es) + 0.2 * (1 - norm(val_acc))
            losses = self.history['val_loss']
            accs = self.history['val_acc']
            fairs = self.history['fairness_for_es']

            def norm(x, lo, hi):
                if hi is None or lo is None or not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
                    return 0.5
                return float((x - lo) / (hi - lo + 1e-8))

            loss_lo, loss_hi = (np.nanmin(losses), np.nanmax(losses)) if len(losses) else (None, None)
            acc_lo, acc_hi = (np.nanmin(accs), np.nanmax(accs)) if len(accs) else (None, None)
            # Filter out inf for fairness bounds
            fairs_finite = [f for f in fairs if np.isfinite(f)]
            if len(fairs_finite) == 0:
                fair_lo, fair_hi = (None, None)
            else:
                fair_lo, fair_hi = (np.nanmin(fairs_finite), np.nanmax(fairs_finite))

            nl = norm(val_metrics['loss'], loss_lo, loss_hi)
            nf = norm(fairness_scalar if np.isfinite(fairness_scalar) else 0.5, fair_lo, fair_hi)
            na = norm(val_metrics['accuracy'], acc_lo, acc_hi)

            # Adaptive scoring: before target fairness -> emphasize fairness; after -> emphasize utility
            fairness_target_met = np.isfinite(val_metrics.get('fairness_score_raw', np.inf)) and (val_metrics.get('fairness_score_raw', np.inf) < window_fairness_threshold)
            if fairness_target_met:
                # post-target: 0.7*loss + 0.3*fairness
                combined_score = 0.7 * nl + 0.3 * nf
            else:
                # pre-target: 0.5*loss + 0.5*fairness
                combined_score = 0.5 * nl + 0.5 * nf
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Acc (thr=0.5 raw): {val_metrics.get('accuracy_thr_0_5', float('nan')):.4f} | Acc (tuned): {val_metrics['accuracy']:.4f}")
            print(f"Fairness (thr=0.5) - Raw: {val_metrics.get('fairness_score_raw', float('nan')):.4f} | Balanced: {val_metrics.get('fairness_score_balanced', float('nan')):.4f}")
            if balanced_metrics is not None:
                print(f"Balanced Loader Fairness: {balanced_metrics.get('fairness_score', float('nan')):.4f}")
            print(f"                 - Demographic Parity Diff: {val_metrics['demographic_parity_diff']:.4f}")
            print(f"                 - Equalized Odds Diff: {val_metrics['equalized_odds_diff']:.4f}")
            print(f"Group positive rates (school_category): {val_metrics['group_rates_school_category']}")
            print(f"Group positive rates (is_top_school): {val_metrics['group_rates_is_top_school']}")
            print(f"AUC: {val_metrics.get('auc', float('nan')):.4f} | PR-AUC: {val_metrics.get('pr_auc', float('nan')):.4f}")
            print(f"Fairness Reg Gap (train avg): {train_metrics['fairness_reg_gap']:.4f}")
            print(f"Temperature: {self.temperature:.4f}")
            print(f"EarlyStopping score (normalized): {combined_score:.4f}")
            print("-" * 60)
            # Append CSV row
            with open(self.metrics_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_metrics['loss'],
                    train_metrics['accuracy'],
                    val_metrics['loss'],
                    val_metrics['accuracy'],
                    val_metrics.get('accuracy_thr_0_5', float('nan')),
                    val_metrics.get('auc', float('nan')),
                    val_metrics.get('pr_auc', float('nan')),
                    val_metrics['fairness_score'],
                    val_metrics['fairness_score_raw'],
                    val_metrics['fairness_score_balanced'],
                    train_metrics['fairness_reg_gap'],
                    train_metrics['final_adv_lambda'],
                    train_metrics['final_multitask_lambda'],
                    val_metrics.get('best_threshold', 0.5),
                    self.temperature
                ])
            
            # Save checkpoint
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            self.save_checkpoint(checkpoint_path, epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if combined_score < best_score:
                best_score = combined_score
                patience_counter = 0
                
                # Save best model
                best_model_path = self.save_dir / "best_model.pt"
                self.save_checkpoint(best_model_path, epoch, train_metrics, val_metrics)
                
                self.history['best_epoch'] = epoch
                self.history['best_val_loss'] = val_metrics['loss']
                self.history['best_fairness_score'] = val_metrics['fairness_score']
                
                print(f"✓ New best model saved (score: {combined_score:.4f})")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            # Dynamic adversarial lambda cap activation: if fairness achieved but utility still low
            if (not self._adv_cap_active):
                fs_raw = val_metrics.get('fairness_score_raw', np.inf)
                if np.isfinite(fs_raw) and (fs_raw < window_fairness_threshold) and (val_metrics.get('accuracy', 0.0) < 0.56):
                    self._adv_cap_active = True
                    print("[Adv Lambda] Fairness target met with low utility -> clamp adv lambda to [0.3, 0.5] going forward.")

            # Dual checkpoint strategy
            # Save best fairness checkpoint (min fairness among raw/balanced)
            fairness_val = val_metrics.get('fairness_score_balanced', np.inf)
            if not np.isfinite(fairness_val):
                fairness_val = val_metrics.get('fairness_score_raw', np.inf)
            if not hasattr(self, 'best_fairness_value'):
                self.best_fairness_value = float('inf')
            if np.isfinite(fairness_val) and fairness_val < self.best_fairness_value:
                self.best_fairness_value = float(fairness_val)
                fairness_ckpt = self.save_dir / 'best_fairness_model.pt'
                self.save_checkpoint(fairness_ckpt, epoch, train_metrics, val_metrics)
                print(f"[Checkpoint] Saved best_fairness_model.pt (fairness={self.best_fairness_value:.4f})")

            # Save best utility checkpoint (AUC highest under fairness <= threshold)
            if not hasattr(self, 'best_util_value'):
                self.best_util_value = -float('inf')
            auc_val = val_metrics.get('auc', float('nan'))
            fairness_ok = (val_metrics.get('fairness_score_balanced', np.inf) < window_fairness_threshold) or (val_metrics.get('fairness_score_raw', np.inf) < window_fairness_threshold)
            if np.isfinite(auc_val) and fairness_ok and (auc_val > self.best_util_value):
                self.best_util_value = float(auc_val)
                util_ckpt = self.save_dir / 'best_util_model.pt'
                self.save_checkpoint(util_ckpt, epoch, train_metrics, val_metrics)
                print(f"[Checkpoint] Saved best_util_model.pt (AUC={self.best_util_value:.4f}, fairness_ok={fairness_ok})")
        
        # Final evaluation on validation with tuned threshold already done.
        # Optionally, if a separate test loader exists externally, users can apply
        # self.history['best_thresholds'][self.history['best_epoch']-1] for final test.

        # Optional final test evaluation with best-epoch threshold
        if test_loader is not None and len(self.history.get('best_thresholds', [])) > 0:
            best_epoch = int(self.history.get('best_epoch', 1))
            thr_idx = max(0, min(best_epoch - 1, len(self.history['best_thresholds']) - 1))
            best_thr = float(self.history['best_thresholds'][thr_idx])

            self.model.eval()
            t_labels = []
            t_probs = []
            t_pred = []
            t_sc = []
            t_ts = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs['logits']
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    t_probs.extend(probs.cpu().numpy())
                    t_labels.extend(labels.cpu().numpy())
                    t_sc.extend(batch['school_category'].cpu().numpy())
                    t_ts.extend(batch['is_top_school'].cpu().numpy())

            t_labels = np.array(t_labels)
            t_probs = np.array(t_probs)
            t_sc = np.array(t_sc)
            t_ts = np.array(t_ts)
            t_pred = (t_probs >= best_thr).astype(int)

            metrics_sc = self.fairness_metrics.compute_all_metrics(t_labels, t_pred, t_probs, t_sc, 'school_category') if len(np.unique(t_sc))>1 else {}
            metrics_ts = self.fairness_metrics.compute_all_metrics(t_labels, t_pred, t_probs, t_ts, 'is_top_school') if len(np.unique(t_ts))>1 else {}
            acc = float((t_pred == t_labels).mean()) if len(t_labels)>0 else 0.0
            self.history['final_test'] = {
                'threshold_used': best_thr,
                'accuracy': acc,
                'school_category': metrics_sc,
                'is_top_school': metrics_ts
            }

        # Window-based checkpoint selection
        self._select_window_best(window_selection_start, window_selection_end, window_fairness_threshold)

        # Save final training history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"Best epoch: {self.history['best_epoch']}")
        print(f"Best validation loss: {self.history['best_val_loss']:.4f}")
        print(f"Best fairness score: {self.history['best_fairness_score']:.4f}")
        return self.history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'main_optimizer_state_dict': self.main_optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.adv_optimizer is not None:
            checkpoint['adv_optimizer_state_dict'] = self.adv_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
        
        if self.adv_optimizer is not None and 'adv_optimizer_state_dict' in checkpoint:
            self.adv_optimizer.load_state_dict(checkpoint['adv_optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_metrics']['loss']:.4f}")
        print(f"  Val Acc: {checkpoint['val_metrics']['accuracy']:.4f}")

    # -------------------- Helper Methods --------------------
    def _calibrate_temperature(self, logits_np: np.ndarray, labels_np: np.ndarray, max_iter: int = 50):
        """Calibrate temperature once using validation logits.
        Minimizes NLL over temperature T (>0).
        """
        if logits_np.shape[0] == 0 or self.temperature_fixed:
            return
        logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels_np, dtype=torch.long, device=self.device)
        temp = torch.ones(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([temp], lr=0.1, max_iter=25)

        def closure():
            optimizer.zero_grad()
            scaled = logits / temp.clamp(min=1e-4)
            loss = nn.CrossEntropyLoss()(scaled, labels)
            loss.backward()
            return loss
        try:
            for _ in range(max_iter):
                optimizer.step(closure)
        except Exception:
            pass
        self.temperature = float(temp.detach().cpu().clamp(min=1e-4).item())
        self.temperature_fixed = True
        print(f"[Temperature Scaling] Calibrated temperature: {self.temperature:.4f}")

    def _apply_temperature(self, logits_np: np.ndarray) -> np.ndarray:
        scaled = logits_np / max(1e-4, self.temperature)
        # Softmax
        e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return probs[:, 1]  # positive class probability

    def _select_window_best(self, start_epoch: int, end_epoch: int, fairness_thr: float):
        """Select best checkpoint inside epoch window with fairness below threshold."""
        if len(self.history['val_acc']) == 0:
            return
        window_indices = [i for i in range(len(self.history['val_acc'])) if (start_epoch <= i + 1 <= end_epoch)]
        if not window_indices:
            return
        candidates = []
        for i in window_indices:
            fairness_bal = self.history['fairness_scores_balanced'][i] if i < len(self.history['fairness_scores_balanced']) else float('inf')
            if np.isfinite(fairness_bal) and fairness_bal < fairness_thr:
                candidates.append(i)
        if not candidates:
            return
        best_i = max(candidates, key=lambda idx: self.history['val_acc'][idx])
        self.history['window_best_epoch'] = best_i + 1
        src_ckpt = self.save_dir / f"checkpoint_epoch_{best_i + 1}.pt"
        dst_ckpt = self.save_dir / "window_best_model.pt"
        if src_ckpt.exists():
            import shutil
            shutil.copyfile(src_ckpt, dst_ckpt)
            print(f"[Window Selection] Saved window-best model from epoch {best_i + 1} (fairness_bal={self.history['fairness_scores_balanced'][best_i]:.4f}, acc={self.history['val_acc'][best_i]:.4f})")
