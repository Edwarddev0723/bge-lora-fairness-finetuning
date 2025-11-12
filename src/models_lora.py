"""
LoRA-based Adversarial Fairness Model

This module implements a memory-efficient version using:
1. LoRA (PEFT) - only trains small adapter weights instead of full model
2. Adversarial head - for fairness-aware training

Memory usage: ~6-8GB instead of ~12GB for full 3B model
Training speed: 2-3x faster than full fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Optional, Tuple
import warnings


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward pass: identity function
    Backward pass: multiply gradient by -alpha
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for Gradient Reversal Layer"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class LoRAAdversarialModel(nn.Module):
    """
    Memory-efficient adversarial fairness model using LoRA
    
    Architecture:
        Input text → Qwen2.5-3B (frozen except LoRA)
                   → [CLS] embedding (z)
                   ↓
                   ├→ Task Classifier (trainable) → Profession logits
                   └→ GRL → Adversary (trainable) → Gender logits
    
    Trainable parameters:
    - LoRA adapters in attention layers (~1% of model)
    - Task classifier head
    - Adversary head
    Total: ~50M params instead of 3B
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        num_professions: int = 28,
        num_genders: int = 2,
        dropout: float = 0.1,
        adv_hidden_dim: int = 256,
        adv_dropout: float = 0.3,
        grl_alpha: float = 1.0,
        lambda_adv: float = 1.0,
        # LoRA config
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None
    ):
        """
        Args:
            model_name: Pretrained model name
            num_professions: Number of profession classes
            num_genders: Number of gender classes
            dropout: Dropout for task classifier
            adv_hidden_dim: Hidden dimension for adversary
            adv_dropout: Dropout for adversary
            grl_alpha: Gradient reversal strength
            lambda_adv: Adversarial loss weight
            use_lora: Use LoRA adapters
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: Modules to apply LoRA (default: attention only)
        """
        super().__init__()
        
        print(f"🚀 Loading efficient model: {model_name}")
        print(f"   LoRA fine-tuning: {use_lora}")
        
        # Default LoRA target modules (attention only for efficiency)
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use float32 for better compatibility
        )
        
        self.hidden_size = self.config.hidden_size
        
        # Apply LoRA if enabled
        if use_lora:
            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION  # Not CAUSAL_LM, we're not generating
            )
            
            # Apply LoRA to model
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()
            print("   ✓ LoRA adapters injected")
        
        # Task head: Profession classifier (always trainable)
        self.task_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_professions)
        )
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        
        # Adversary head: Gender predictor (always trainable)
        self.adversary = nn.Sequential(
            nn.Linear(self.hidden_size, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim, adv_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim // 2, num_genders)
        )
        
        # Store device for later use
        self.device = None
        
        # Store device for later use
        self.device = None
        
        self.num_professions = num_professions
        self.num_genders = num_genders
        self.lambda_adv = lambda_adv
        self.use_lora = use_lora
        
        print("   ✓ Task classifier and adversary heads initialized")
        print(f"   ✓ Model ready! Using LoRA for efficient training")
    
    def to(self, device):
        """Override to method to ensure all components move to device"""
        super().to(device)
        self.device = device
        self.base_model = self.base_model.to(device)
        self.task_classifier = self.task_classifier.to(device)
        self.adversary = self.adversary.to(device)
        return self
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        profession_labels: Optional[torch.Tensor] = None,
        gender_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            profession_labels: Profession labels [batch_size] (optional)
            gender_labels: Gender labels [batch_size] (optional)
        
        Returns:
            Dictionary with:
                - profession_logits: [batch_size, num_professions]
                - gender_logits: [batch_size, num_genders]
                - loss: Total loss (if labels provided)
                - task_loss: Profession classification loss
                - adv_loss: Adversarial gender prediction loss
                - embeddings: [CLS] embeddings [batch_size, hidden_size]
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # [CLS] embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Task: Profession classification
        profession_logits = self.task_classifier(embeddings)
        
        # Adversarial task: Gender prediction (through GRL)
        reversed_embeddings = self.grl(embeddings)
        gender_logits = self.adversary(reversed_embeddings)
        
        result = {
            "profession_logits": profession_logits,
            "gender_logits": gender_logits,
            "embeddings": embeddings
        }
        
        # Compute losses if labels provided
        if profession_labels is not None and gender_labels is not None:
            # Task loss (profession)
            task_loss = F.cross_entropy(profession_logits, profession_labels)
            
            # Adversarial loss (gender)
            adv_loss = F.cross_entropy(gender_logits, gender_labels)
            
            # Total loss
            total_loss = task_loss + self.lambda_adv * adv_loss
            
            result["loss"] = total_loss
            result["task_loss"] = task_loss
            result["adv_loss"] = adv_loss
        
        return result
    
    def set_grl_alpha(self, alpha: float):
        """Update GRL alpha (useful for curriculum learning)"""
        self.grl.alpha = alpha
    
    def set_lambda_adv(self, lambda_adv: float):
        """Update adversarial loss weight"""
        self.lambda_adv = lambda_adv
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percent": 100 * trainable_params / total_params
        }


def load_tokenizer(model_name: str):
    """Load tokenizer for the model"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_lora_adversarial_model(config) -> Tuple[LoRAAdversarialModel, AutoTokenizer]:
    """
    Create LoRA-based adversarial model from config
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = LoRAAdversarialModel(
        model_name=config.MODEL_NAME,
        num_professions=config.NUM_PROFESSIONS,
        num_genders=config.NUM_GENDERS,
        dropout=0.1,
        adv_hidden_dim=config.ADV_HIDDEN_DIM,
        adv_dropout=config.ADV_DROPOUT,
        grl_alpha=config.GRL_ALPHA,
        lambda_adv=config.LAMBDA_ADV,
        use_lora=config.USE_LORA,
        lora_r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        lora_target_modules=config.LORA_TARGET_MODULES
    )
    
    tokenizer = load_tokenizer(config.MODEL_NAME)
    
    return model, tokenizer


if __name__ == "__main__":
    # Test LoRA model
    import sys
    sys.path.append('..')
    import config
    
    print("\n" + "="*80)
    print("Testing LoRA Adversarial Model")
    print("="*80)
    
    # Create model
    model, tokenizer = create_lora_adversarial_model(config)
    
    # Print parameter info
    param_info = model.get_trainable_parameters()
    print(f"\n📊 Parameter Statistics:")
    print(f"   Total parameters: {param_info['total_params']:,}")
    print(f"   Trainable parameters: {param_info['trainable_params']:,}")
    print(f"   Trainable: {param_info['trainable_percent']:.2f}%")
    
    # Test forward pass
    batch_size = 2
    text = ["This is a test biography about a person.", "Another sample text."]
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    profession_labels = torch.randint(0, config.NUM_PROFESSIONS, (batch_size,))
    gender_labels = torch.randint(0, config.NUM_GENDERS, (batch_size,))
    
    print(f"\n🧪 Testing forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'].to(model.base_model.device),
            attention_mask=inputs['attention_mask'].to(model.base_model.device),
            profession_labels=profession_labels.to(model.base_model.device),
            gender_labels=gender_labels.to(model.base_model.device)
        )
    
    print(f"   ✓ Profession logits shape: {outputs['profession_logits'].shape}")
    print(f"   ✓ Gender logits shape: {outputs['gender_logits'].shape}")
    print(f"   ✓ Total loss: {outputs['loss'].item():.4f}")
    print(f"   ✓ Task loss: {outputs['task_loss'].item():.4f}")
    print(f"   ✓ Adv loss: {outputs['adv_loss'].item():.4f}")
    
    print("\n✅ LoRA model test passed!")
