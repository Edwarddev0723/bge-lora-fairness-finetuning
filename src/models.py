"""
Model architectures for Fair AI Classification

Implements:
1. Baseline Classifier (standard Qwen2.5-3B for profession classification)
2. Adversarial Fairness Model with Gradient Reversal Layer (GRL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Optional, Tuple


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward pass: identity function
    Backward pass: multiply gradient by -alpha
    
    This forces the feature extractor to learn representations that:
    - Are useful for the main task (profession classification)
    - Are NOT useful for the adversary task (gender prediction)
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
        """
        Args:
            alpha: Strength of gradient reversal (typically 1.0)
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class BaselineClassifier(nn.Module):
    """
    Baseline profession classifier using Qwen2.5-3B
    
    Architecture:
        Input text → Qwen2.5-3B → [CLS] embedding → Classification head → Profession logits
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        num_professions: int = 28,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        """
        Args:
            model_name: Pretrained model name
            num_professions: Number of profession classes
            dropout: Dropout rate for classification head
            freeze_base: Whether to freeze the base model
        """
        super().__init__()
        
        print(f"Loading baseline model: {model_name}")
        print("⏳ First-time download may take 10-30 minutes (6GB model)")
        print("💡 Tip: Model will be cached for future use")
        
        # Load pretrained model with optimizations
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use half precision to reduce memory
            low_cpu_mem_usage=True      # Optimize CPU memory during loading
        )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get hidden size
        self.hidden_size = self.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_professions)
        )
        
        self.num_professions = num_professions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Profession labels [batch_size] (optional)
        
        Returns:
            Dictionary with:
                - logits: Profession logits [batch_size, num_professions]
                - loss: Classification loss (if labels provided)
                - embeddings: [CLS] embeddings [batch_size, hidden_size]
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state's [CLS] token (first token)
        # Shape: [batch_size, hidden_size]
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(embeddings)
        
        result = {
            "logits": logits,
            "embeddings": embeddings
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result


class AdversarialFairnessModel(nn.Module):
    """
    Adversarial model for fairness-aware classification
    
    Architecture:
        Input text → Qwen2.5-3B → [CLS] embedding (z)
                                    ↓
                                    ├→ Classifier → Profession logits
                                    └→ GRL → Adversary → Gender logits
    
    Training objective:
        L = L_task + λ * L_adv
        
        Where:
        - L_task: Profession classification loss
        - L_adv: Gender prediction loss (through GRL)
        - λ: Trade-off parameter (LAMBDA_ADV in config)
        
    The GRL reverses gradients, so the feature extractor learns to:
        1. Predict profession well (minimize L_task)
        2. Hide gender information (maximize L_adv, which minimizes it after gradient reversal)
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
        freeze_base: bool = False
    ):
        """
        Args:
            model_name: Pretrained model name
            num_professions: Number of profession classes
            num_genders: Number of gender classes
            dropout: Dropout for main classifier
            adv_hidden_dim: Hidden dimension for adversary
            adv_dropout: Dropout for adversary
            grl_alpha: Gradient reversal strength
            lambda_adv: Adversarial loss weight
            freeze_base: Whether to freeze base model
        """
        super().__init__()
        
        print(f"Loading adversarial model: {model_name}")
        print("⏳ First-time download may take 10-30 minutes (6GB model)")
        print("💡 Tip: Model will be cached for future use")
        
        # Load pretrained model with optimizations
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use half precision to reduce memory
            low_cpu_mem_usage=True      # Optimize CPU memory during loading
        )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.config.hidden_size
        
        # Main task: Profession classifier
        self.task_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_professions)
        )
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        
        # Adversary: Gender predictor (should fail to predict gender)
        self.adversary = nn.Sequential(
            nn.Linear(self.hidden_size, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim, adv_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim // 2, num_genders)
        )
        
        self.num_professions = num_professions
        self.num_genders = num_genders
        self.lambda_adv = lambda_adv
    
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
                - profession_logits: Profession predictions [batch_size, num_professions]
                - gender_logits: Gender predictions [batch_size, num_genders]
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
        
        # [CLS] embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Main task: Profession classification
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
            # Note: The adversary tries to predict gender, but GRL reverses gradients
            # So the feature extractor learns to hide gender info
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


def create_baseline_model(config) -> Tuple[BaselineClassifier, AutoTokenizer]:
    """
    Create baseline model and tokenizer from config
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = BaselineClassifier(
        model_name=config.MODEL_NAME,
        num_professions=config.NUM_PROFESSIONS,
        dropout=0.1
    )
    
    tokenizer = load_tokenizer(config.MODEL_NAME)
    
    return model, tokenizer


def create_adversarial_model(config) -> Tuple[AdversarialFairnessModel, AutoTokenizer]:
    """
    Create adversarial fairness model and tokenizer from config
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = AdversarialFairnessModel(
        model_name=config.MODEL_NAME,
        num_professions=config.NUM_PROFESSIONS,
        num_genders=config.NUM_GENDERS,
        dropout=0.1,
        adv_hidden_dim=config.ADV_HIDDEN_DIM,
        adv_dropout=config.ADV_DROPOUT,
        grl_alpha=config.GRL_ALPHA,
        lambda_adv=config.LAMBDA_ADV
    )
    
    tokenizer = load_tokenizer(config.MODEL_NAME)
    
    return model, tokenizer


if __name__ == "__main__":
    # Test models
    import sys
    sys.path.append('..')
    import config
    
    print("\n" + "="*80)
    print("Testing Baseline Classifier")
    print("="*80)
    
    # Create dummy input
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.NUM_PROFESSIONS, (batch_size,))
    
    # Test baseline
    baseline_model = BaselineClassifier(
        model_name=config.MODEL_NAME,
        num_professions=config.NUM_PROFESSIONS
    )
    
    outputs = baseline_model(input_ids, attention_mask, labels)
    print(f"Profession logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    print("\n" + "="*80)
    print("Testing Adversarial Model")
    print("="*80)
    
    # Test adversarial
    gender_labels = torch.randint(0, config.NUM_GENDERS, (batch_size,))
    
    adv_model = AdversarialFairnessModel(
        model_name=config.MODEL_NAME,
        num_professions=config.NUM_PROFESSIONS,
        num_genders=config.NUM_GENDERS,
        lambda_adv=config.LAMBDA_ADV
    )
    
    outputs = adv_model(input_ids, attention_mask, labels, gender_labels)
    print(f"Profession logits shape: {outputs['profession_logits'].shape}")
    print(f"Gender logits shape: {outputs['gender_logits'].shape}")
    print(f"Total loss: {outputs['loss'].item():.4f}")
    print(f"Task loss: {outputs['task_loss'].item():.4f}")
    print(f"Adv loss: {outputs['adv_loss'].item():.4f}")
