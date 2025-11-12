"""
Fairness metrics for evaluating bias in resume matching models.
Includes demographic parity, equalized odds, and other fairness measures.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class FairnessMetrics:
    """
    Compute comprehensive fairness metrics for binary classification.
    
    Metrics include:
    - Demographic Parity (Statistical Parity)
    - Equalized Odds (TPR and FPR parity)
    - Equal Opportunity (TPR parity)
    - Predictive Parity (PPV parity)
    - Calibration metrics
    """
    
    def __init__(self):
        self.epsilon = 1e-8  # Small constant to avoid division by zero
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attr: np.ndarray,
        sensitive_attr_name: str = "sensitive_attribute"
    ) -> Dict[str, float]:
        """
        Compute all fairness metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Prediction probabilities
            sensitive_attr: Sensitive attribute values (e.g., school category)
            sensitive_attr_name: Name of the sensitive attribute for reporting
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Overall performance metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Get unique groups
        unique_groups = np.unique(sensitive_attr)
        
        # Compute group-wise metrics
        group_metrics = {}
        for group in unique_groups:
            group_mask = sensitive_attr == group
            group_metrics[group] = self._compute_group_metrics(
                y_true[group_mask],
                y_pred[group_mask],
                y_prob[group_mask]
            )
        
        # Demographic Parity (Selection Rate Parity)
        selection_rates = {
            group: group_metrics[group]['selection_rate'] 
            for group in unique_groups
        }
        metrics['demographic_parity_difference'] = self._compute_parity_difference(selection_rates)
        metrics['demographic_parity_ratio'] = self._compute_parity_ratio(selection_rates)
        # Disparate Impact Ratio is commonly defined as the ratio of selection rates between groups
        # Here we report it explicitly (alias of demographic_parity_ratio for clarity)
        metrics['disparate_impact_ratio'] = metrics['demographic_parity_ratio']
        
        # Equalized Odds (TPR and FPR parity)
        tpr_values = {
            group: group_metrics[group]['tpr'] 
            for group in unique_groups
        }
        fpr_values = {
            group: group_metrics[group]['fpr'] 
            for group in unique_groups
        }
        
        metrics['equalized_odds_tpr_difference'] = self._compute_parity_difference(tpr_values)
        metrics['equalized_odds_fpr_difference'] = self._compute_parity_difference(fpr_values)
        metrics['equalized_odds_avg_difference'] = (
            metrics['equalized_odds_tpr_difference'] + 
            metrics['equalized_odds_fpr_difference']
        ) / 2
        
        # Equal Opportunity (TPR parity only)
        metrics['equal_opportunity_difference'] = metrics['equalized_odds_tpr_difference']
        
        # Predictive Parity (PPV/Precision parity)
        ppv_values = {
            group: group_metrics[group]['ppv'] 
            for group in unique_groups
        }
        metrics['predictive_parity_difference'] = self._compute_parity_difference(ppv_values)
        
        # Store group-wise metrics
        for group in unique_groups:
            prefix = f"{sensitive_attr_name}_{group}"
            for metric_name, value in group_metrics[group].items():
                metrics[f'{prefix}_{metric_name}'] = value
        
        # Overall fairness score (lower is better, 0 is perfectly fair)
        metrics['overall_fairness_score'] = (
            abs(metrics['demographic_parity_difference']) +
            abs(metrics['equalized_odds_avg_difference']) +
            abs(metrics['predictive_parity_difference'])
        ) / 3
        
        return metrics
    
    def _compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for a single group."""
        
        if len(y_true) == 0:
            return {
                'selection_rate': 0.0,
                'tpr': 0.0,
                'fpr': 0.0,
                'tnr': 0.0,
                'fnr': 0.0,
                'ppv': 0.0,
                'npv': 0.0,
                'accuracy': 0.0,
                'count': 0
            }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Selection rate (positive prediction rate)
        selection_rate = np.mean(y_pred)
        
        # True Positive Rate (Sensitivity, Recall)
        tpr = tp / (tp + fn + self.epsilon)
        
        # False Positive Rate
        fpr = fp / (fp + tn + self.epsilon)
        
        # True Negative Rate (Specificity)
        tnr = tn / (tn + fp + self.epsilon)
        
        # False Negative Rate
        fnr = fn / (fn + tp + self.epsilon)
        
        # Positive Predictive Value (Precision)
        ppv = tp / (tp + fp + self.epsilon)
        
        # Negative Predictive Value
        npv = tn / (tn + fn + self.epsilon)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.epsilon)
        
        return {
            'selection_rate': float(selection_rate),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'tnr': float(tnr),
            'fnr': float(fnr),
            'ppv': float(ppv),
            'npv': float(npv),
            'accuracy': float(accuracy),
            'count': len(y_true),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def _compute_parity_difference(self, group_values: Dict) -> float:
        """Compute max difference between groups (absolute)."""
        values = list(group_values.values())
        if len(values) < 2:
            return 0.0
        return float(max(values) - min(values))
    
    def _compute_parity_ratio(self, group_values: Dict) -> float:
        """Compute min/max ratio between groups."""
        values = [v for v in group_values.values() if v > 0]
        if len(values) < 2:
            return 1.0
        return float(min(values) / (max(values) + self.epsilon))
    
    def compute_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attr: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute calibration metrics per group.
        
        A well-calibrated model should have predicted probabilities
        that match the actual outcomes.
        """
        unique_groups = np.unique(sensitive_attr)
        calibration_metrics = {}
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            if len(group_y_true) == 0:
                continue
            
            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(group_y_prob, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Compute calibration per bin
            bin_counts = []
            bin_true_probs = []
            bin_pred_probs = []
            
            for i in range(n_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    bin_counts.append(np.sum(bin_mask))
                    bin_true_probs.append(np.mean(group_y_true[bin_mask]))
                    bin_pred_probs.append(np.mean(group_y_prob[bin_mask]))
            
            # Expected Calibration Error (ECE)
            ece = 0.0
            total_count = len(group_y_true)
            for count, true_prob, pred_prob in zip(bin_counts, bin_true_probs, bin_pred_probs):
                ece += (count / total_count) * abs(true_prob - pred_prob)
            
            calibration_metrics[f'group_{group}_ece'] = float(ece)
            calibration_metrics[f'group_{group}_bins'] = {
                'counts': bin_counts,
                'true_probs': bin_true_probs,
                'pred_probs': bin_pred_probs
            }
        
        # Calibration disparity (difference in ECE between groups)
        ece_values = [
            calibration_metrics[f'group_{group}_ece'] 
            for group in unique_groups 
            if f'group_{group}_ece' in calibration_metrics
        ]
        if len(ece_values) >= 2:
            calibration_metrics['calibration_disparity'] = max(ece_values) - min(ece_values)
        
        return calibration_metrics


class FairnessLoss:
    """
    Fairness-aware loss functions for training.
    """
    
    def __init__(self, lambda_fairness: float = 0.3):
        """
        Args:
            lambda_fairness: Weight for fairness regularization term
        """
        self.lambda_fairness = lambda_fairness
    
    def demographic_parity_loss(
        self,
        predictions: torch.Tensor,
        sensitive_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage similar selection rates across groups.
        
        Args:
            predictions: Model predictions (logits or probabilities) [batch_size, num_classes]
            sensitive_attr: Sensitive attribute labels [batch_size]
            
        Returns:
            Fairness loss scalar
        """
        # Get positive class probabilities
        if predictions.shape[1] == 2:
            pos_probs = torch.softmax(predictions, dim=1)[:, 1]
        else:
            pos_probs = torch.sigmoid(predictions.squeeze())
        
        unique_groups = torch.unique(sensitive_attr)
        selection_rates = []
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            if torch.sum(group_mask) > 0:
                group_selection_rate = torch.mean(pos_probs[group_mask])
                selection_rates.append(group_selection_rate)
        
        if len(selection_rates) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute variance of selection rates (lower is more fair)
        selection_rates = torch.stack(selection_rates)
        fairness_loss = torch.var(selection_rates)
        
        return self.lambda_fairness * fairness_loss
    
    def equalized_odds_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        sensitive_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage similar TPR and FPR across groups.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            labels: True labels [batch_size]
            sensitive_attr: Sensitive attribute labels [batch_size]
            
        Returns:
            Fairness loss scalar
        """
        # Get predicted labels
        if predictions.shape[1] == 2:
            pred_labels = torch.argmax(predictions, dim=1)
        else:
            pred_labels = (torch.sigmoid(predictions.squeeze()) > 0.5).long()
        
        unique_groups = torch.unique(sensitive_attr)
        tpr_list = []
        fpr_list = []
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            
            # True Positive Rate
            positive_mask = labels == 1
            group_positive_mask = group_mask & positive_mask
            if torch.sum(group_positive_mask) > 0:
                tpr = torch.sum((pred_labels == 1) & group_positive_mask).float() / torch.sum(group_positive_mask).float()
                tpr_list.append(tpr)
            
            # False Positive Rate
            negative_mask = labels == 0
            group_negative_mask = group_mask & negative_mask
            if torch.sum(group_negative_mask) > 0:
                fpr = torch.sum((pred_labels == 1) & group_negative_mask).float() / torch.sum(group_negative_mask).float()
                fpr_list.append(fpr)
        
        # Compute variance of TPR and FPR
        loss = torch.tensor(0.0, device=predictions.device)
        if len(tpr_list) >= 2:
            tpr_tensor = torch.stack(tpr_list)
            loss += torch.var(tpr_tensor)
        if len(fpr_list) >= 2:
            fpr_tensor = torch.stack(fpr_list)
            loss += torch.var(fpr_tensor)
        
        return self.lambda_fairness * loss


def compute_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attrs: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive fairness report for multiple sensitive attributes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        sensitive_attrs: Dictionary of sensitive attribute names and values
        save_path: Optional path to save report as JSON
        
    Returns:
        Complete fairness report dictionary
    """
    metrics_calculator = FairnessMetrics()
    report = {
        'overall_metrics': {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        },
        'fairness_by_attribute': {}
    }
    
    # Compute fairness metrics for each sensitive attribute
    for attr_name, attr_values in sensitive_attrs.items():
        attr_metrics = metrics_calculator.compute_all_metrics(
            y_true, y_pred, y_prob, attr_values, attr_name
        )
        calibration_metrics = metrics_calculator.compute_calibration(
            y_true, y_prob, attr_values
        )
        
        report['fairness_by_attribute'][attr_name] = {
            'fairness_metrics': attr_metrics,
            'calibration_metrics': calibration_metrics
        }
    
    # Save report if requested
    if save_path:
        import json
        with open(save_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            json.dump(convert_types(report), f, indent=2)
    
    return report
