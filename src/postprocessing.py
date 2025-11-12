"""
Post-processing module: Greedy reranking for fairness

Implements demographic parity enforcement through prediction reranking
Does not retrain the model - only adjusts final predictions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class GreedyReranker:
    """
    Greedy reranking to balance gender representation in predictions
    
    Strategy:
    1. Get model predictions with confidence scores
    2. For each profession, ensure male/female ratio is balanced
    3. Greedily select predictions to minimize gender imbalance
    
    This is a post-processing step that doesn't require retraining
    """
    
    def __init__(
        self,
        target_balance: float = 0.5,
        top_k: Optional[int] = None
    ):
        """
        Args:
            target_balance: Target ratio of female predictions (0.5 = perfect balance)
            top_k: Only rerank top-k predictions (None = rerank all)
        """
        self.target_balance = target_balance
        self.top_k = top_k
    
    def rerank_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        gender: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Rerank predictions to balance gender representation
        
        Args:
            predictions: Original predicted labels [N]
            probabilities: Prediction probabilities [N, num_classes]
            gender: Gender labels [N]
            true_labels: True labels [N] (optional, for analysis)
        
        Returns:
            Tuple of (reranked_predictions, statistics)
        """
        n_samples = len(predictions)
        num_classes = probabilities.shape[1]
        
        # Initialize with original predictions
        reranked_preds = predictions.copy()
        
        # Group samples by their original prediction
        pred_groups = defaultdict(list)
        for idx in range(n_samples):
            pred = predictions[idx]
            pred_groups[pred].append(idx)
        
        stats = {
            "total_samples": n_samples,
            "reranked_count": 0,
            "gender_balance_before": {},
            "gender_balance_after": {}
        }
        
        # For each prediction class, balance gender
        for pred_class, indices in pred_groups.items():
            if len(indices) < 2:
                continue
            
            # Get genders for this prediction group
            group_genders = gender[indices]
            male_count = (group_genders == 0).sum()
            female_count = (group_genders == 1).sum()
            total = len(indices)
            
            stats["gender_balance_before"][int(pred_class)] = {
                "male": int(male_count),
                "female": int(female_count),
                "female_ratio": float(female_count / total) if total > 0 else 0
            }
            
            # Calculate current imbalance
            current_female_ratio = female_count / total if total > 0 else 0.5
            
            # If already balanced, skip
            if abs(current_female_ratio - self.target_balance) < 0.1:
                stats["gender_balance_after"][int(pred_class)] = stats["gender_balance_before"][int(pred_class)]
                continue
            
            # Determine which gender to reduce
            if current_female_ratio > self.target_balance:
                # Too many females, need to change some female predictions
                over_gender = 1
                target_reduce = female_count - int(total * self.target_balance)
            else:
                # Too many males, need to change some male predictions
                over_gender = 0
                target_reduce = male_count - int(total * (1 - self.target_balance))
            
            # Find candidates to change (those with lower confidence)
            over_gender_indices = [idx for idx in indices if gender[idx] == over_gender]
            
            if len(over_gender_indices) == 0 or target_reduce <= 0:
                stats["gender_balance_after"][int(pred_class)] = stats["gender_balance_before"][int(pred_class)]
                continue
            
            # Sort by confidence (ascending - change least confident first)
            confidences = probabilities[over_gender_indices, pred_class]
            sorted_indices = np.argsort(confidences)
            
            # Change predictions for least confident samples
            num_to_change = min(target_reduce, len(over_gender_indices))
            for i in range(num_to_change):
                idx = over_gender_indices[sorted_indices[i]]
                
                # Find alternative prediction (2nd highest probability)
                probs = probabilities[idx]
                # Set current prediction to 0 to find alternative
                probs_copy = probs.copy()
                probs_copy[pred_class] = 0
                new_pred = np.argmax(probs_copy)
                
                reranked_preds[idx] = new_pred
                stats["reranked_count"] += 1
            
            # Record after-reranking balance
            group_genders_after = gender[indices]
            reranked_for_group = reranked_preds[indices]
            # Count those still predicted as pred_class
            still_pred_mask = (reranked_for_group == pred_class)
            male_after = ((group_genders_after == 0) & still_pred_mask).sum()
            female_after = ((group_genders_after == 1) & still_pred_mask).sum()
            total_after = still_pred_mask.sum()
            
            stats["gender_balance_after"][int(pred_class)] = {
                "male": int(male_after),
                "female": int(female_after),
                "female_ratio": float(female_after / total_after) if total_after > 0 else 0
            }
        
        return reranked_preds, stats
    
    def rerank_by_top_k(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        gender: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Rerank only top-k confident predictions
        
        Args:
            predictions: Original predicted labels [N]
            probabilities: Prediction probabilities [N, num_classes]
            gender: Gender labels [N]
            k: Number of top predictions to rerank
        
        Returns:
            Tuple of (reranked_predictions, statistics)
        """
        # Get confidence scores (max probability per sample)
        confidences = np.max(probabilities, axis=1)
        
        # Get indices of top-k confident predictions
        top_k_indices = np.argsort(confidences)[-k:]
        
        # Create mask
        mask = np.zeros(len(predictions), dtype=bool)
        mask[top_k_indices] = True
        
        # Rerank only top-k
        reranked_preds = predictions.copy()
        reranked_subset, stats = self.rerank_predictions(
            predictions[mask],
            probabilities[mask],
            gender[mask]
        )
        reranked_preds[mask] = reranked_subset
        
        stats["top_k"] = k
        stats["reranked_ratio"] = k / len(predictions)
        
        return reranked_preds, stats


class CalibrationBasedReranker:
    """
    Alternative reranking strategy based on calibration
    
    Adjusts confidence thresholds differently for each gender group
    to achieve fairness
    """
    
    def __init__(self, target_tpr_gap: float = 0.05):
        """
        Args:
            target_tpr_gap: Target TPR gap between genders
        """
        self.target_tpr_gap = target_tpr_gap
    
    def rerank_by_threshold(
        self,
        probabilities: np.ndarray,
        gender: np.ndarray,
        base_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply different confidence thresholds for each gender
        
        Args:
            probabilities: Prediction probabilities [N, num_classes]
            gender: Gender labels [N]
            base_threshold: Base confidence threshold
        
        Returns:
            Tuple of (adjusted_predictions, statistics)
        """
        # Get initial predictions
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate gender-specific thresholds
        # (This is a simplified version - in practice, would optimize these)
        threshold_male = base_threshold
        threshold_female = base_threshold * 0.9  # Slightly lower for female
        
        # Apply thresholds
        adjusted_preds = predictions.copy()
        
        # For males: only keep if confidence > threshold_male
        male_mask = (gender == 0)
        low_conf_males = male_mask & (max_probs < threshold_male)
        
        # For females: only keep if confidence > threshold_female
        female_mask = (gender == 1)
        low_conf_females = female_mask & (max_probs < threshold_female)
        
        # For low confidence, assign to 2nd choice
        for mask in [low_conf_males, low_conf_females]:
            for idx in np.where(mask)[0]:
                probs = probabilities[idx].copy()
                probs[predictions[idx]] = 0  # Remove top choice
                adjusted_preds[idx] = np.argmax(probs)
        
        stats = {
            "threshold_male": threshold_male,
            "threshold_female": threshold_female,
            "adjusted_male_count": int(low_conf_males.sum()),
            "adjusted_female_count": int(low_conf_females.sum())
        }
        
        return adjusted_preds, stats


def apply_postprocessing(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    gender: np.ndarray,
    method: str = "greedy",
    config=None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply post-processing reranking
    
    Args:
        predictions: Original predictions [N]
        probabilities: Prediction probabilities [N, num_classes]
        gender: Gender labels [N]
        method: 'greedy' or 'calibration'
        config: Configuration object (optional)
    
    Returns:
        Tuple of (reranked_predictions, statistics)
    """
    if method == "greedy":
        reranker = GreedyReranker(
            target_balance=0.5,
            top_k=config.RERANK_TOP_K if config else None
        )
        if config and config.RERANK_TOP_K:
            return reranker.rerank_by_top_k(
                predictions, probabilities, gender, k=config.RERANK_TOP_K
            )
        else:
            return reranker.rerank_predictions(
                predictions, probabilities, gender
            )
    
    elif method == "calibration":
        reranker = CalibrationBasedReranker(target_tpr_gap=0.05)
        return reranker.rerank_by_threshold(probabilities, gender)
    
    else:
        raise ValueError(f"Unknown reranking method: {method}")


if __name__ == "__main__":
    # Test reranking
    print("Testing post-processing reranking...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 28
    
    # Simulate biased predictions (more males predicted for certain professions)
    predictions = np.random.randint(0, num_classes, n_samples)
    probabilities = np.random.rand(n_samples, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Introduce bias: males more likely predicted as profession 0
    gender = np.random.randint(0, 2, n_samples)
    male_mask = (gender == 0)
    predictions[male_mask][:200] = 0  # Make many males predicted as prof 0
    
    print(f"\nBefore reranking:")
    prof_0_mask = (predictions == 0)
    prof_0_genders = gender[prof_0_mask]
    print(f"Profession 0 predictions: {prof_0_mask.sum()}")
    print(f"  Male: {(prof_0_genders == 0).sum()}")
    print(f"  Female: {(prof_0_genders == 1).sum()}")
    
    # Apply reranking
    reranker = GreedyReranker(target_balance=0.5)
    reranked_preds, stats = reranker.rerank_predictions(
        predictions, probabilities, gender
    )
    
    print(f"\nAfter reranking:")
    print(f"Reranked {stats['reranked_count']} predictions")
    prof_0_mask_after = (reranked_preds == 0)
    prof_0_genders_after = gender[prof_0_mask_after]
    print(f"Profession 0 predictions: {prof_0_mask_after.sum()}")
    print(f"  Male: {(prof_0_genders_after == 0).sum()}")
    print(f"  Female: {(prof_0_genders_after == 1).sum()}")
    
    if 0 in stats["gender_balance_after"]:
        print(f"\nBalance stats for profession 0:")
        print(f"  Before: {stats['gender_balance_before'][0]}")
        print(f"  After: {stats['gender_balance_after'][0]}")
