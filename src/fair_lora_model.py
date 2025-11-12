"""
Fair LoRA Model Architecture
BGE-large + LoRA + Adversarial Debiasing + Multi-task Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fair_lora_config import *
except ImportError:
    # Fallback defaults if config not found
    EMBEDDING_DIM = 1024
    ADVERSARIAL_HIDDEN_DIM = 256
    ADVERSARIAL_NUM_LAYERS = 2
    ADVERSARIAL_DROPOUT = 0.3
    ATTRIBUTE_CLASSIFIER_HIDDEN_DIM = 256
    ATTRIBUTE_CLASSIFIER_DROPOUT = 0.3
    SCHOOL_CATEGORIES = ["top_school", "non_top_school", "no_school_mentioned"]
    BASE_MODEL = "BAAI/bge-large-en-v1.5"
    USE_LORA = True
    USE_ADVERSARIAL_DEBIASING = True
    USE_MULTITASK = True
    NUM_LABELS = 2
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_TARGET_MODULES = ["query", "key", "value"]
    LORA_DROPOUT = 0.1
    LORA_BIAS = "none"


class AdversarialDiscriminator(nn.Module):
    """
    Adversarial discriminator for sensitive attribute prediction
    Goal: Predict sensitive attribute from embeddings
    Main model tries to fool this discriminator
    """
    
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = ADVERSARIAL_HIDDEN_DIM,
        num_classes: int = 2,  # Binary: top_school vs others
        num_layers: int = ADVERSARIAL_NUM_LAYERS,
        dropout: float = ADVERSARIAL_DROPOUT
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.discriminator(embeddings)


class AttributeClassifier(nn.Module):
    """
    Auxiliary task: Predict sensitive attribute
    Used for multi-task learning to improve fairness
    """
    
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = ATTRIBUTE_CLASSIFIER_HIDDEN_DIM,
        num_classes: int = len(SCHOOL_CATEGORIES),
        dropout: float = ATTRIBUTE_CLASSIFIER_DROPOUT
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class FairLoRAModel(nn.Module):
    """
    Main model: BGE + LoRA + Fairness components
    """
    
    def __init__(
        self,
        base_model_name: str = BASE_MODEL,
        use_lora: bool = USE_LORA,
        use_adversarial: bool = USE_ADVERSARIAL_DEBIASING,
        use_multitask: bool = USE_MULTITASK,
        num_labels: int = NUM_LABELS
    ):
        super().__init__()
        
        self.use_adversarial = use_adversarial
        self.use_multitask = use_multitask
        self.num_labels = num_labels
        
        # Load base model
        print(f"🔧 Loading base model: {base_model_name}")
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32
        )
        
        # Apply LoRA (restricted to last few layers + increased rank)
        if use_lora:
            # Increased rank/alpha for more adaptation capacity
            lora_r = 16
            lora_alpha = 32
            print(f"🔧 Applying LoRA (expanded targets, r={lora_r}, alpha={lora_alpha})...")
            # Expand target modules: include attention output & MLP projections (bert/llama style)
            # We'll probe module names dynamically and keep intersection.
            candidate_targets = [
                'query','key','value',            # Bert style attention
                'q_proj','k_proj','v_proj','o_proj',  # Llama style attention
                'dense','intermediate.dense','output.dense',  # Bert FFN
                'fc1','fc2','up_proj','down_proj','gate_proj'  # Llama / generic MLP
            ]
            discovered = set()
            for name, module in self.base_model.named_modules():
                # Only keep leaf linear layers
                if isinstance(module, nn.Linear):
                    # Extract the last component of name
                    tail = name.split('.')[-1]
                    if tail in candidate_targets:
                        discovered.add(tail)
                    # Also handle full path matches for intermediate.dense/output.dense
                    if any(full in name for full in ['intermediate.dense','output.dense']) and 'dense' not in discovered:
                        discovered.add('dense')  # PEFT expects 'dense'
            if not discovered:
                # Fallback to previous default
                discovered = set(list(LORA_TARGET_MODULES) + ['dense'])
            target_modules = sorted(discovered)
            print("   • LoRA target modules detected:", target_modules)
            print("     (modules not present are ignored silently by PEFT)")
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=LORA_DROPOUT,
                bias=LORA_BIAS,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            # Freeze earlier encoder layers entirely (except LoRA params which remain trainable)
            try:
                # Determine number of encoder layers
                if hasattr(self.base_model, 'encoder') and hasattr(self.base_model.encoder, 'layer'):
                    n_layers = len(self.base_model.encoder.layer)
                    base_prefix = 'encoder.layer.'
                elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'encoder') and hasattr(self.base_model.model.encoder, 'layer'):
                    n_layers = len(self.base_model.model.encoder.layer)
                    base_prefix = 'model.encoder.layer.'
                else:
                    n_layers = 0
                    base_prefix = ''

                keep_start = max(0, n_layers - 4)

                for name, param in self.base_model.named_parameters():
                    # Skip LoRA parameters always trainable
                    if 'lora_' in name:
                        continue
                    if base_prefix and f"{base_prefix}" in name:
                        # Try to extract layer index
                        try:
                            after = name.split(base_prefix, 1)[1]
                            idx = int(after.split('.', 1)[0])
                        except Exception:
                            idx = None
                    else:
                        idx = None

                    if idx is None:
                        # Non-encoder or unknown; default freeze to be conservative
                        param.requires_grad = False
                        continue

                    if idx < keep_start:
                        # Earlier layers: freeze all base weights
                        param.requires_grad = False
                    else:
                        # Last 4 layers: only attention base weights trainable; FFN base frozen
                        if f"{base_prefix}{idx}.attention." in name:
                            param.requires_grad = True
                        elif (f"{base_prefix}{idx}.intermediate." in name) or (f"{base_prefix}{idx}.output." in name):
                            param.requires_grad = False
                        else:
                            # LayerNorms etc. keep as frozen for stability
                            param.requires_grad = False
                print(f"🔒 Frozen encoder base weights except attention in last 4 layers; LoRA adapters remain trainable.")
                # Print a concise summary of trainable parameter groups
                trainable_lora = 0
                trainable_base_last4 = 0
                for n,p in self.base_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if 'lora_' in n:
                        trainable_lora += p.numel()
                    else:
                        trainable_base_last4 += p.numel()
                print(f"📊 Trainable params -> LoRA: {trainable_lora:,} | Base last4(attn): {trainable_base_last4:,} | Total: {trainable_lora+trainable_base_last4:,}")
            except Exception as e:
                print(f"⚠️ Layer freezing skipped due to: {e}")
            self.base_model.print_trainable_parameters()
        
        # Shared frozen projection + head-specific projections
        self.shared_projection = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.ReLU()
        )
        for p in self.shared_projection.parameters():
            p.requires_grad = False

        self.main_projection = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU()
        )
        self.adv_projection = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU()
        )
        self.attr_projection = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        
        # Adversarial discriminator
        if use_adversarial:
            print(f"🔧 Adding adversarial discriminator...")
            self.adversary = AdversarialDiscriminator(
                input_dim=EMBEDDING_DIM,
                hidden_dim=ADVERSARIAL_HIDDEN_DIM,
                num_classes=2  # Binary: top_school vs not
            )
        
        # Attribute classifier for multi-task learning
        if use_multitask:
            print(f"🔧 Adding attribute classifier for multi-task learning...")
            self.attribute_classifier = AttributeClassifier(
                input_dim=EMBEDDING_DIM,
                hidden_dim=ATTRIBUTE_CLASSIFIER_HIDDEN_DIM,
                num_classes=len(SCHOOL_CATEGORIES)
            )
        
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from base model
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, embedding_dim)
        
        # L2 normalize for better similarity computation
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sensitive_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) - classification labels
            sensitive_attr: (batch_size,) - sensitive attribute labels
            return_embeddings: Whether to return embeddings
        
        Returns:
            Dictionary with:
                - logits: Classification logits
                - embeddings: (optional) Sentence embeddings
                - adversarial_logits: (optional) Adversary predictions
                - attribute_logits: (optional) Attribute predictions
        """
        # Get embeddings
        embeddings = self.get_embeddings(input_ids, attention_mask)
        # Shared frozen projection
        shared = self.shared_projection(embeddings)
        # Head-specific features
        main_feat = self.main_projection(shared)
        adv_feat = self.adv_projection(shared)
        attr_feat = self.attr_projection(shared)
        # Classification logits on main head
        logits = self.classifier(main_feat)

        outputs: Dict[str, torch.Tensor] = {'logits': logits}

        if return_embeddings:
            # Backward compatibility: 'embeddings' -> main head feat
            outputs['embeddings'] = main_feat
            outputs['embeddings_main'] = main_feat
            outputs['embeddings_adv'] = adv_feat
            outputs['embeddings_attr'] = attr_feat

        # Adversarial prediction
        if self.use_adversarial and sensitive_attr is not None:
            adversarial_logits = self.adversary(adv_feat.detach())  # Detach to prevent gradient flow
            outputs['adversarial_logits'] = adversarial_logits

        # Attribute prediction (multi-task)
        if self.use_multitask and sensitive_attr is not None:
            attribute_logits = self.attribute_classifier(attr_feat)
            outputs['attribute_logits'] = attribute_logits

        return outputs


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Gradient Reversal Layer
    Forward: identity
    Backward: reverse gradient and multiply by alpha
    """
    # Pylance type hint: ensure Tensor return type
    out = GradientReversalFunction.apply(x, alpha)
    assert isinstance(out, torch.Tensor)
    return out


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer implementation
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def test_model():
    """Test model initialization"""
    print("="*80)
    print("Testing Fair LoRA Model")
    print("="*80)
    
    # Create model
    model = FairLoRAModel()
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, NUM_LABELS, (batch_size,))
    sensitive_attr = torch.randint(0, 2, (batch_size,))
    
    print(f"\n🧪 Test forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            sensitive_attr=sensitive_attr,
            return_embeddings=True
        )
    
    print(f"\n✅ Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable percentage: {trainable_params/total_params*100:.2f}%")
    
    print(f"\n✅ Model test completed!")


if __name__ == "__main__":
    test_model()
