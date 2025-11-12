"""
Evaluation metrics for Fair AI Classification

Implements:
1. Standard classification metrics (accuracy, F1)
2. Fairness metrics (TPR gap, demographic parity, equalized odds)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)
from collections import defaultdict


class FairnessMetrics:
    """
    Calculate fairness metrics for binary sensitive attributes
    
    Focuses on gender fairness (male vs female) for profession classification
    """
    
    def __init__(self, num_classes: int = 28):
        """
        Args:
            num_classes: Number of target classes (professions)
        """
        self.num_classes = num_classes
    
    def calculate_per_gender_tpr(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        gender: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate True Positive Rate (Recall) per gender
        
        TPR measures: Of all positive cases, how many did we correctly identify?
        
        Args:
            y_true: True profession labels [N]
            y_pred: Predicted profession labels [N]
            gender: Gender labels [N] (0=male, 1=female)
        
        Returns:
            Dictionary with TPR for each gender
        """
        tpr_by_gender = {}
        
        for g in [0, 1]:  # male, female
            mask = (gender == g)
            if mask.sum() == 0:
                continue
            
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]
            
            # Calculate recall (TPR) for this gender
            # Use macro average across all professions
            tpr = recall_score(y_true_g, y_pred_g, average='macro', zero_division=0)
            
            gender_name = "male" if g == 0 else "female"
            tpr_by_gender[gender_name] = tpr
        
        return tpr_by_gender
    
    def calculate_tpr_gap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        gender: np.ndarray
    ) -> float:
        """
        Calculate TPR gap between genders
        
        TPR Gap = |TPR_male - TPR_female|
        
        Lower is better (0 = perfect equality)
        
        Args:
            y_true: True profession labels
            y_pred: Predicted profession labels
            gender: Gender labels
        
        Returns:
            TPR gap (absolute difference)
        """
        tpr_by_gender = self.calculate_per_gender_tpr(y_true, y_pred, gender)
        
        if len(tpr_by_gender) < 2:
            return 0.0
        
        tpr_male = tpr_by_gender.get("male", 0)
        tpr_female = tpr_by_gender.get("female", 0)
        
        return abs(tpr_male - tpr_female)
    
    def calculate_demographic_parity(
        self,
        y_pred: np.ndarray,
        gender: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate demographic parity difference
        
        Demographic Parity: P(pred=positive | male) ≈ P(pred=positive | female)
        
        For multi-class, we measure the distribution difference of predictions
        
        Args:
            y_pred: Predicted labels
            gender: Gender labels
        
        Returns:
            Dictionary with:
                - dp_diff: Maximum difference in prediction rates across genders
                - pred_rate_male: Prediction rate for males
                - pred_rate_female: Prediction rate for females
        """
        # Calculate prediction distribution per gender
        pred_dist_male = np.zeros(self.num_classes)
        pred_dist_female = np.zeros(self.num_classes)
        
        mask_male = (gender == 0)
        mask_female = (gender == 1)
        
        if mask_male.sum() > 0:
            unique, counts = np.unique(y_pred[mask_male], return_counts=True)
            pred_dist_male[unique] = counts / mask_male.sum()
        
        if mask_female.sum() > 0:
            unique, counts = np.unique(y_pred[mask_female], return_counts=True)
            pred_dist_female[unique] = counts / mask_female.sum()
        
        # Maximum difference across all classes
        dp_diff = np.max(np.abs(pred_dist_male - pred_dist_female))
        
        return {
            "dp_diff": dp_diff,
            "pred_dist_male": pred_dist_male.tolist(),
            "pred_dist_female": pred_dist_female.tolist()
        }
    
    def calculate_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        gender: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Equalized Odds difference
        
        Equalized Odds: TPR and FPR should be equal across genders
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            gender: Gender labels
        
        Returns:
            Dictionary with:
                - eo_diff: Max of TPR and FPR differences
                - tpr_diff: TPR difference
                - fpr_diff: FPR difference
        """
        # Calculate TPR and FPR per gender
        tpr_male, fpr_male = self._calculate_tpr_fpr(
            y_true[gender == 0],
            y_pred[gender == 0]
        )
        
        tpr_female, fpr_female = self._calculate_tpr_fpr(
            y_true[gender == 1],
            y_pred[gender == 1]
        )
        
        tpr_diff = abs(tpr_male - tpr_female)
        fpr_diff = abs(fpr_male - fpr_female)
        eo_diff = max(tpr_diff, fpr_diff)
        
        return {
            "eo_diff": eo_diff,
            "tpr_diff": tpr_diff,
            "fpr_diff": fpr_diff,
            "tpr_male": tpr_male,
            "tpr_female": tpr_female,
            "fpr_male": fpr_male,
            "fpr_female": fpr_female
        }
    
    def _calculate_tpr_fpr(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate TPR and FPR (macro-averaged across classes)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Tuple of (TPR, FPR)
        """
        if len(y_true) == 0:
            return 0.0, 0.0
        
        tpr_list = []
        fpr_list = []
        
        for c in range(self.num_classes):
            # Binary classification for this class
            y_true_binary = (y_true == c).astype(int)
            y_pred_binary = (y_pred == c).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_true_binary,
                y_pred_binary,
                labels=[0, 1]
            ).ravel()
            
            # TPR = TP / (TP + FN)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # FPR = FP / (FP + TN)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Macro average
        return np.mean(tpr_list), np.mean(fpr_list)


class MetricsCalculator:
    """
    Calculate all metrics (classification + fairness)
    """
    
    def __init__(self, num_classes: int = 28):
        """
        Args:
            num_classes: Number of target classes
        """
        self.num_classes = num_classes
        self.fairness_metrics = FairnessMetrics(num_classes)
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
        gender: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            y_true: True labels [N]
            y_pred: Predicted labels [N]
            y_probs: Prediction probabilities [N, num_classes] (optional)
            gender: Gender labels [N] (optional, needed for fairness metrics)
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["weighted_f1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["macro_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["macro_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # AUC (if probabilities provided)
        if y_probs is not None:
            try:
                # One-vs-rest AUC
                metrics["macro_auc"] = roc_auc_score(
                    y_true,
                    y_probs,
                    multi_class='ovr',
                    average='macro'
                )
            except ValueError:
                metrics["macro_auc"] = 0.0
        
        # Fairness metrics (if gender provided)
        if gender is not None:
            # Per-gender TPR
            tpr_by_gender = self.fairness_metrics.calculate_per_gender_tpr(
                y_true, y_pred, gender
            )
            metrics["tpr_male"] = tpr_by_gender.get("male", 0.0)
            metrics["tpr_female"] = tpr_by_gender.get("female", 0.0)
            
            # TPR gap
            metrics["tpr_gap"] = self.fairness_metrics.calculate_tpr_gap(
                y_true, y_pred, gender
            )
            
            # Demographic parity
            dp_results = self.fairness_metrics.calculate_demographic_parity(
                y_pred, gender
            )
            metrics["demographic_parity_diff"] = dp_results["dp_diff"]
            
            # Equalized odds
            eo_results = self.fairness_metrics.calculate_equalized_odds(
                y_true, y_pred, gender
            )
            metrics["equalized_odds_diff"] = eo_results["eo_diff"]
            metrics["tpr_diff"] = eo_results["tpr_diff"]
            metrics["fpr_diff"] = eo_results["fpr_diff"]
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Metrics"):
        """
        Pretty print metrics
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics report
        """
        print("\n" + "="*80)
        print(f"{title}")
        print("="*80)
        
        # Classification metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:         {metrics.get('accuracy', 0):.4f}")
        print(f"  Macro F1:         {metrics.get('macro_f1', 0):.4f}")
        print(f"  Weighted F1:      {metrics.get('weighted_f1', 0):.4f}")
        print(f"  Macro Precision:  {metrics.get('macro_precision', 0):.4f}")
        print(f"  Macro Recall:     {metrics.get('macro_recall', 0):.4f}")
        if "macro_auc" in metrics:
            print(f"  Macro AUC:        {metrics.get('macro_auc', 0):.4f}")
        
        # Fairness metrics
        if "tpr_gap" in metrics:
            print("\nFairness Metrics:")
            print(f"  TPR (Male):       {metrics.get('tpr_male', 0):.4f}")
            print(f"  TPR (Female):     {metrics.get('tpr_female', 0):.4f}")
            print(f"  TPR Gap:          {metrics.get('tpr_gap', 0):.4f} ⬅ Lower is better")
            print(f"  Demographic Parity Diff: {metrics.get('demographic_parity_diff', 0):.4f}")
            print(f"  Equalized Odds Diff:     {metrics.get('equalized_odds_diff', 0):.4f}")
        
        print("="*80)


def compare_models(
    baseline_metrics: Dict[str, float],
    fair_metrics: Dict[str, float],
    reranked_metrics: Dict[str, float]
) -> Dict:
    """
    Compare metrics across three models and generate comparison report
    
    Args:
        baseline_metrics: Metrics from baseline model
        fair_metrics: Metrics from fair adversarial model
        reranked_metrics: Metrics from fair model with reranking
    
    Returns:
        Comparison report dictionary
    """
    report = {
        "baseline": baseline_metrics,
        "fair_adversarial": fair_metrics,
        "fair_reranked": reranked_metrics,
        "improvements": {}
    }
    
    # Calculate improvements
    # Fairness: How much did we reduce TPR gap?
    baseline_gap = baseline_metrics.get("tpr_gap", 0)
    fair_gap = fair_metrics.get("tpr_gap", 0)
    reranked_gap = reranked_metrics.get("tpr_gap", 0)
    
    report["improvements"]["tpr_gap_reduction_fair"] = baseline_gap - fair_gap
    report["improvements"]["tpr_gap_reduction_reranked"] = baseline_gap - reranked_gap
    report["improvements"]["tpr_gap_reduction_pct_fair"] = (
        (baseline_gap - fair_gap) / baseline_gap * 100 if baseline_gap > 0 else 0
    )
    report["improvements"]["tpr_gap_reduction_pct_reranked"] = (
        (baseline_gap - reranked_gap) / baseline_gap * 100 if baseline_gap > 0 else 0
    )
    
    # Accuracy: How much did we sacrifice?
    baseline_acc = baseline_metrics.get("accuracy", 0)
    fair_acc = fair_metrics.get("accuracy", 0)
    reranked_acc = reranked_metrics.get("accuracy", 0)
    
    report["improvements"]["accuracy_drop_fair"] = baseline_acc - fair_acc
    report["improvements"]["accuracy_drop_reranked"] = baseline_acc - reranked_acc
    report["improvements"]["accuracy_drop_pct_fair"] = (
        (baseline_acc - fair_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
    )
    report["improvements"]["accuracy_drop_pct_reranked"] = (
        (baseline_acc - reranked_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
    )
    
    # Summary
    report["summary"] = {
        "best_accuracy": max(baseline_acc, fair_acc, reranked_acc),
        "best_fairness": min(baseline_gap, fair_gap, reranked_gap),
        "best_tradeoff": "reranked" if reranked_gap < fair_gap and reranked_acc > fair_acc else "fair"
    }
    
    return report


def print_comparison_report(report: Dict):
    """
    Pretty print comparison report
    
    Args:
        report: Comparison report from compare_models()
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)
    
    print("\n📊 Accuracy Comparison:")
    print(f"  Baseline:         {report['baseline']['accuracy']:.4f}")
    print(f"  Fair (Adv):       {report['fair_adversarial']['accuracy']:.4f} "
          f"({report['improvements']['accuracy_drop_pct_fair']:+.2f}%)")
    print(f"  Fair + Rerank:    {report['fair_reranked']['accuracy']:.4f} "
          f"({report['improvements']['accuracy_drop_pct_reranked']:+.2f}%)")
    
    print("\n⚖️  Fairness Comparison (TPR Gap):")
    print(f"  Baseline:         {report['baseline']['tpr_gap']:.4f}")
    print(f"  Fair (Adv):       {report['fair_adversarial']['tpr_gap']:.4f} "
          f"({report['improvements']['tpr_gap_reduction_pct_fair']:.1f}% reduction)")
    print(f"  Fair + Rerank:    {report['fair_reranked']['tpr_gap']:.4f} "
          f"({report['improvements']['tpr_gap_reduction_pct_reranked']:.1f}% reduction)")
    
    print("\n🎯 Key Insights:")
    print(f"  • In-processing (adversarial) reduced TPR gap by "
          f"{report['improvements']['tpr_gap_reduction_pct_fair']:.1f}%")
    print(f"  • Post-processing (reranking) reduced TPR gap by "
          f"{report['improvements']['tpr_gap_reduction_pct_reranked']:.1f}%")
    print(f"  • Accuracy cost: {report['improvements']['accuracy_drop_pct_reranked']:.2f}% "
          f"for {report['improvements']['tpr_gap_reduction_pct_reranked']:.1f}% fairness gain")
    
    print("="*80)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics calculation...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 28
    
    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = np.random.randint(0, num_classes, n_samples)
    y_probs = np.random.rand(n_samples, num_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    gender = np.random.randint(0, 2, n_samples)
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes=num_classes)
    metrics = calculator.calculate_all_metrics(
        y_true, y_pred, y_probs, gender
    )
    
    calculator.print_metrics(metrics, "Test Metrics")
