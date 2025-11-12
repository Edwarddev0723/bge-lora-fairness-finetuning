"""
Comprehensive evaluation for fair resume matching models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.fair_lora_model import FairLoRAModel
from src.fairness_metrics import FairnessMetrics, compute_fairness_report


class FairEvaluator:
    """
    Comprehensive evaluator for fair models.
    
    Features:
    - Performance metrics (accuracy, precision, recall, F1)
    - Fairness metrics (demographic parity, equalized odds, etc.)
    - Group-wise performance analysis
    - Calibration analysis
    - Visualization of results
    """
    
    def __init__(
        self,
        model: FairLoRAModel,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained FairLoRAModel
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.fairness_metrics = FairnessMetrics()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            return_predictions: Whether to return all predictions
            
        Returns:
            Dictionary containing metrics and optionally predictions
        """
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        all_school_categories = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                school_categories = batch['school_categories'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                embeddings = outputs['embeddings']
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs[:, 1].cpu().numpy())
                all_embeddings.extend(embeddings.cpu().numpy())
                all_school_categories.extend(school_categories.cpu().numpy())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_embeddings = np.array(all_embeddings)
        all_school_categories = np.array(all_school_categories)
        
        # Compute comprehensive fairness report
        report = compute_fairness_report(
            all_labels,
            all_predictions,
            all_probabilities,
            {'school_category': all_school_categories}
        )
        
        # Add embeddings analysis
        report['embedding_analysis'] = self._analyze_embeddings(
            all_embeddings,
            all_school_categories
        )
        
        # Optionally return predictions
        if return_predictions:
            report['predictions'] = {
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'embeddings': all_embeddings,
                'school_categories': all_school_categories
            }
        
        return report
    
    def _analyze_embeddings(
        self,
        embeddings: np.ndarray,
        school_categories: np.ndarray
    ) -> Dict[str, any]:
        """
        Analyze embedding space for bias.
        
        Measures:
        - Mean embedding distance between groups
        - Standard deviation within groups
        - Silhouette score (how separable groups are)
        """
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import cdist
        
        unique_groups = np.unique(school_categories)
        
        # Group centroids
        centroids = {}
        for group in unique_groups:
            group_mask = school_categories == group
            centroids[int(group)] = np.mean(embeddings[group_mask], axis=0)
        
        # Inter-group distances
        inter_group_distances = {}
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                dist = np.linalg.norm(centroids[int(group1)] - centroids[int(group2)])
                inter_group_distances[f'{int(group1)}_to_{int(group2)}'] = float(dist)
        
        # Intra-group variance
        intra_group_variance = {}
        for group in unique_groups:
            group_mask = school_categories == group
            group_embeddings = embeddings[group_mask]
            centroid = centroids[int(group)]
            variance = np.mean(np.linalg.norm(group_embeddings - centroid, axis=1))
            intra_group_variance[int(group)] = float(variance)
        
        # Silhouette score (measures separability, -1 to 1, lower is better for fairness)
        if len(unique_groups) > 1:
            try:
                silhouette = silhouette_score(embeddings, school_categories)
            except:
                silhouette = None
        else:
            silhouette = None
        
        return {
            'inter_group_distances': inter_group_distances,
            'intra_group_variance': intra_group_variance,
            'silhouette_score': silhouette,
            'embedding_dim': embeddings.shape[1]
        }
    
    def compare_models(
        self,
        baseline_results: Dict,
        fair_results: Dict,
        save_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Compare baseline and fair models.
        
        Args:
            baseline_results: Results from baseline model evaluation
            fair_results: Results from fair model evaluation
            save_path: Optional path to save comparison report
            
        Returns:
            Comparison report
        """
        comparison = {
            'performance_comparison': {},
            'fairness_comparison': {},
            'improvement': {}
        }
        
        # Performance comparison
        baseline_perf = baseline_results['overall_metrics']
        fair_perf = fair_results['overall_metrics']
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            comparison['performance_comparison'][metric] = {
                'baseline': baseline_perf[metric],
                'fair': fair_perf[metric],
                'difference': fair_perf[metric] - baseline_perf[metric],
                'relative_change_pct': 100 * (fair_perf[metric] - baseline_perf[metric]) / baseline_perf[metric]
            }
        
        # Fairness comparison
        baseline_fairness = baseline_results['fairness_by_attribute']['school_category']['fairness_metrics']
        fair_fairness = fair_results['fairness_by_attribute']['school_category']['fairness_metrics']
        
        fairness_metrics = [
            'demographic_parity_difference',
            'equalized_odds_avg_difference',
            'equal_opportunity_difference',
            'predictive_parity_difference',
            'overall_fairness_score'
        ]
        
        for metric in fairness_metrics:
            baseline_val = baseline_fairness[metric]
            fair_val = fair_fairness[metric]
            improvement = baseline_val - fair_val  # Lower is better
            
            comparison['fairness_comparison'][metric] = {
                'baseline': baseline_val,
                'fair': fair_val,
                'improvement': improvement,
                'improvement_pct': 100 * improvement / baseline_val if baseline_val > 0 else 0
            }
        
        # Overall improvement summary
        comparison['improvement']['performance_change'] = comparison['performance_comparison']['accuracy']['difference']
        comparison['improvement']['fairness_improvement'] = comparison['fairness_comparison']['overall_fairness_score']['improvement']
        comparison['improvement']['is_pareto_improvement'] = (
            comparison['improvement']['performance_change'] >= -0.02 and  # Allow 2% accuracy drop
            comparison['improvement']['fairness_improvement'] > 0
        )
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(comparison, f, indent=2)
        
        return comparison
    
    def visualize_results(
        self,
        results: Dict,
        save_dir: Optional[Path] = None
    ):
        """
        Create visualizations of evaluation results.
        
        Args:
            results: Results from evaluate()
            save_dir: Directory to save plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        fairness_metrics = results['fairness_by_attribute']['school_category']['fairness_metrics']
        
        # 1. Fairness metrics bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = {
            'Demographic\nParity': fairness_metrics['demographic_parity_difference'],
            'Equalized\nOdds': fairness_metrics['equalized_odds_avg_difference'],
            'Equal\nOpportunity': fairness_metrics['equal_opportunity_difference'],
            'Predictive\nParity': fairness_metrics['predictive_parity_difference']
        }
        
        bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        ax.set_ylabel('Fairness Gap (lower is better)')
        ax.set_title('Fairness Metrics')
        ax.axhline(y=0.1, color='r', linestyle='--', label='Acceptable threshold')
        ax.legend()
        
        # Color bars based on threshold
        for i, bar in enumerate(bars):
            if list(metrics_to_plot.values())[i] > 0.1:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'fairness_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Group-wise performance
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract group metrics
        group_metrics_data = {}
        for key, value in fairness_metrics.items():
            if key.startswith('school_category_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    group = '_'.join(parts[2:-1])
                    metric = parts[-1]
                    if group not in group_metrics_data:
                        group_metrics_data[group] = {}
                    group_metrics_data[group][metric] = value
        
        # Plot TPR
        groups = list(group_metrics_data.keys())
        tpr_values = [group_metrics_data[g].get('tpr', 0) for g in groups]
        axes[0, 0].bar(groups, tpr_values)
        axes[0, 0].set_title('True Positive Rate by Group')
        axes[0, 0].set_ylabel('TPR')
        
        # Plot FPR
        fpr_values = [group_metrics_data[g].get('fpr', 0) for g in groups]
        axes[0, 1].bar(groups, fpr_values)
        axes[0, 1].set_title('False Positive Rate by Group')
        axes[0, 1].set_ylabel('FPR')
        
        # Plot Selection Rate
        selection_rates = [group_metrics_data[g].get('selection', 0) for g in groups]
        axes[1, 0].bar(groups, selection_rates)
        axes[1, 0].set_title('Selection Rate by Group')
        axes[1, 0].set_ylabel('Selection Rate')
        
        # Plot Accuracy
        accuracy_values = [group_metrics_data[g].get('accuracy', 0) for g in groups]
        axes[1, 1].bar(groups, accuracy_values)
        axes[1, 1].set_title('Accuracy by Group')
        axes[1, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'group_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrix by group
        if 'predictions' in results:
            predictions = results['predictions']
            y_true = predictions['labels']
            y_pred = predictions['predictions']
            school_cats = predictions['school_categories']
            
            unique_groups = np.unique(school_cats)
            n_groups = len(unique_groups)
            
            fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
            if n_groups == 1:
                axes = [axes]
            
            for idx, group in enumerate(unique_groups):
                group_mask = school_cats == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(group_y_true, group_y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
                axes[idx].set_title(f'Group {group}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Visualizations saved to {save_dir}")
    
    def print_summary(self, results: Dict):
        """Print a human-readable summary of results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Overall performance
        print("\n📊 Overall Performance:")
        overall = results['overall_metrics']
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1 Score:  {overall['f1']:.4f}")
        
        # Fairness metrics
        print("\n⚖️  Fairness Metrics:")
        fairness = results['fairness_by_attribute']['school_category']['fairness_metrics']
        print(f"  Overall Fairness Score: {fairness['overall_fairness_score']:.4f}")
        print(f"  Demographic Parity:     {fairness['demographic_parity_difference']:.4f}")
        print(f"  Equalized Odds:         {fairness['equalized_odds_avg_difference']:.4f}")
        print(f"  Equal Opportunity:      {fairness['equal_opportunity_difference']:.4f}")
        print(f"  Predictive Parity:      {fairness['predictive_parity_difference']:.4f}")
        
        # Embedding analysis
        if 'embedding_analysis' in results:
            print("\n🔍 Embedding Analysis:")
            emb = results['embedding_analysis']
            if emb['silhouette_score'] is not None:
                print(f"  Silhouette Score: {emb['silhouette_score']:.4f} (lower = less separable = more fair)")
            print(f"  Embedding Dimension: {emb['embedding_dim']}")
        
        # Group-wise breakdown
        print("\n👥 Group-wise Breakdown:")
        for key, value in fairness.items():
            if key.startswith('school_category_') and key.endswith('_count'):
                group = key.replace('school_category_', '').replace('_count', '')
                count = value
                tpr_key = f'school_category_{group}_tpr'
                fpr_key = f'school_category_{group}_fpr'
                acc_key = f'school_category_{group}_accuracy'
                
                if all(k in fairness for k in [tpr_key, fpr_key, acc_key]):
                    print(f"\n  Group {group} (n={count}):")
                    print(f"    TPR: {fairness[tpr_key]:.4f}")
                    print(f"    FPR: {fairness[fpr_key]:.4f}")
                    print(f"    Accuracy: {fairness[acc_key]:.4f}")
        
        print("\n" + "="*60 + "\n")
