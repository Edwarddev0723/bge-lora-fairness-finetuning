"""
Training module for Fair AI Classification

Implements training loops for:
1. Baseline classifier
2. Adversarial fairness model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class BiasDataset(Dataset):
    """PyTorch Dataset for Bias in Bios"""
    
    def __init__(
        self,
        texts: list,
        professions: list,
        genders: list,
        tokenizer,
        max_length: int = 256
    ):
        """
        Args:
            texts: List of biography texts
            professions: List of profession labels
            genders: List of gender labels
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.professions = professions
        self.genders = genders
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        profession = self.professions[idx]
        gender = self.genders[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'profession': torch.tensor(profession, dtype=torch.long),
            'gender': torch.tensor(gender, dtype=torch.long)
        }


def create_dataloaders(
    train_ds,
    dev_ds,
    test_ds,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 4
):
    """
    Create DataLoaders from HuggingFace datasets
    
    Args:
        train_ds: Training dataset
        dev_ds: Development dataset
        test_ds: Test dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of workers for DataLoader
    
    Returns:
        Tuple of (train_loader, dev_loader, test_loader)
    """
    # Convert to PyTorch datasets
    train_dataset = BiasDataset(
        texts=[x['hard_text'] for x in train_ds],
        professions=[x['profession'] for x in train_ds],
        genders=[x['gender'] for x in train_ds],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dev_dataset = BiasDataset(
        texts=[x['hard_text'] for x in dev_ds],
        professions=[x['profession'] for x in dev_ds],
        genders=[x['gender'] for x in dev_ds],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = BiasDataset(
        texts=[x['hard_text'] for x in test_ds],
        professions=[x['profession'] for x in test_ds],
        genders=[x['gender'] for x in test_ds],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, dev_loader, test_loader


def train_baseline_model(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    device: str = "cuda",
    save_path: Optional[Path] = None,
    logging_steps: int = 50,
    eval_steps: int = 200
) -> Dict:
    """
    Train baseline classifier
    
    Args:
        model: Baseline model
        train_loader: Training DataLoader
        dev_loader: Development DataLoader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        device: Device to train on
        save_path: Path to save model
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
    
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_accuracy": [],
        "best_dev_accuracy": 0.0,
        "best_epoch": 0
    }
    
    global_step = 0
    best_dev_loss = float('inf')
    
    print(f"\n{'='*80}")
    print(f"Training Baseline Model")
    print(f"{'='*80}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        
        epoch_loss = 0
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['profession'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Evaluation
            if global_step % eval_steps == 0:
                dev_metrics = evaluate_baseline_model(
                    model, dev_loader, device
                )
                print(f"\nStep {global_step} - Dev Loss: {dev_metrics['loss']:.4f}, "
                      f"Dev Acc: {dev_metrics['accuracy']:.4f}")
                
                history["dev_loss"].append(dev_metrics['loss'])
                history["dev_accuracy"].append(dev_metrics['accuracy'])
                
                # Save best model
                if dev_metrics['accuracy'] > history["best_dev_accuracy"]:
                    history["best_dev_accuracy"] = dev_metrics['accuracy']
                    history["best_epoch"] = epoch
                    
                    if save_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'dev_accuracy': dev_metrics['accuracy'],
                        }, save_path / "best_model.pt")
                        print(f"Saved best model to {save_path / 'best_model.pt'}")
                
                model.train()
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_epoch_loss)
        print(f"\nEpoch {epoch + 1} - Avg Train Loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation on Dev Set")
    print("="*80)
    final_metrics = evaluate_baseline_model(model, dev_loader, device)
    print(f"Dev Loss: {final_metrics['loss']:.4f}")
    print(f"Dev Accuracy: {final_metrics['accuracy']:.4f}")
    
    # Save final model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, save_path / "final_model.pt")
        print(f"\nSaved final model to {save_path / 'final_model.pt'}")
    
    return history


def train_adversarial_model(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    device: str = "cuda",
    save_path: Optional[Path] = None,
    logging_steps: int = 50,
    eval_steps: int = 200
) -> Dict:
    """
    Train adversarial fairness model
    
    Args:
        model: Adversarial model
        train_loader: Training DataLoader
        dev_loader: Development DataLoader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        device: Device to train on
        save_path: Path to save model
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
    
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        "train_loss": [],
        "train_task_loss": [],
        "train_adv_loss": [],
        "dev_loss": [],
        "dev_accuracy": [],
        "dev_adversary_accuracy": [],
        "best_dev_accuracy": 0.0,
        "best_epoch": 0
    }
    
    global_step = 0
    
    print(f"\n{'='*80}")
    print(f"Training Adversarial Fairness Model")
    print(f"{'='*80}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Device: {device}")
    print(f"Lambda (adversarial weight): {model.lambda_adv}")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        
        epoch_loss = 0
        epoch_task_loss = 0
        epoch_adv_loss = 0
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            profession_labels = batch['profession'].to(device)
            gender_labels = batch['gender'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                profession_labels=profession_labels,
                gender_labels=gender_labels
            )
            
            loss = outputs['loss']
            task_loss = outputs['task_loss']
            adv_loss = outputs['adv_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item()
            epoch_adv_loss += adv_loss.item()
            global_step += 1
            
            # Logging
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_task = epoch_task_loss / (batch_idx + 1)
                avg_adv = epoch_adv_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'task': f'{avg_task:.4f}',
                    'adv': f'{avg_adv:.4f}'
                })
            
            # Evaluation
            if global_step % eval_steps == 0:
                dev_metrics = evaluate_adversarial_model(
                    model, dev_loader, device
                )
                print(f"\nStep {global_step} - Dev Loss: {dev_metrics['loss']:.4f}, "
                      f"Dev Acc: {dev_metrics['accuracy']:.4f}, "
                      f"Adv Acc: {dev_metrics['adversary_accuracy']:.4f}")
                
                history["dev_loss"].append(dev_metrics['loss'])
                history["dev_accuracy"].append(dev_metrics['accuracy'])
                history["dev_adversary_accuracy"].append(dev_metrics['adversary_accuracy'])
                
                # Save best model (based on task accuracy)
                if dev_metrics['accuracy'] > history["best_dev_accuracy"]:
                    history["best_dev_accuracy"] = dev_metrics['accuracy']
                    history["best_epoch"] = epoch
                    
                    if save_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'dev_accuracy': dev_metrics['accuracy'],
                            'adversary_accuracy': dev_metrics['adversary_accuracy']
                        }, save_path / "best_model.pt")
                        print(f"Saved best model to {save_path / 'best_model.pt'}")
                
                model.train()
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_task_loss = epoch_task_loss / len(train_loader)
        avg_adv_loss = epoch_adv_loss / len(train_loader)
        
        history["train_loss"].append(avg_epoch_loss)
        history["train_task_loss"].append(avg_task_loss)
        history["train_adv_loss"].append(avg_adv_loss)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Avg Task Loss: {avg_task_loss:.4f}")
        print(f"  Avg Adv Loss: {avg_adv_loss:.4f}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation on Dev Set")
    print("="*80)
    final_metrics = evaluate_adversarial_model(model, dev_loader, device)
    print(f"Dev Loss: {final_metrics['loss']:.4f}")
    print(f"Dev Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Adversary Accuracy: {final_metrics['adversary_accuracy']:.4f}")
    print(f"(Lower adversary accuracy = better fairness)")
    
    # Save final model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, save_path / "final_model.pt")
        print(f"\nSaved final model to {save_path / 'final_model.pt'}")
    
    return history


def evaluate_baseline_model(
    model,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate baseline model
    
    Args:
        model: Baseline model
        dataloader: DataLoader
        device: Device
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['profession'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(dataloader)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels
    }


def evaluate_adversarial_model(
    model,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate adversarial model
    
    Args:
        model: Adversarial model
        dataloader: DataLoader
        device: Device
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    all_profession_preds = []
    all_profession_labels = []
    all_gender_preds = []
    all_gender_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            profession_labels = batch['profession'].to(device)
            gender_labels = batch['gender'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                profession_labels=profession_labels,
                gender_labels=gender_labels
            )
            
            total_loss += outputs['loss'].item()
            
            profession_preds = torch.argmax(outputs['profession_logits'], dim=1)
            gender_preds = torch.argmax(outputs['gender_logits'], dim=1)
            
            all_profession_preds.extend(profession_preds.cpu().numpy())
            all_profession_labels.extend(profession_labels.cpu().numpy())
            all_gender_preds.extend(gender_preds.cpu().numpy())
            all_gender_labels.extend(gender_labels.cpu().numpy())
    
    all_profession_preds = np.array(all_profession_preds)
    all_profession_labels = np.array(all_profession_labels)
    all_gender_preds = np.array(all_gender_preds)
    all_gender_labels = np.array(all_gender_labels)
    
    accuracy = (all_profession_preds == all_profession_labels).mean()
    adversary_accuracy = (all_gender_preds == all_gender_labels).mean()
    avg_loss = total_loss / len(dataloader)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "adversary_accuracy": adversary_accuracy,
        "profession_predictions": all_profession_preds,
        "profession_labels": all_profession_labels,
        "gender_predictions": all_gender_preds,
        "gender_labels": all_gender_labels
    }


if __name__ == "__main__":
    print("Training module loaded successfully")
    print("Use train_baseline_model() or train_adversarial_model() to train models")
