import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
import time
from typing import Dict, List, Tuple, Optional


class MetricsTracker:
    """Track and compute comprehensive metrics for model evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.train_accuracies = []
        self.test_accuracies = []
        self.val_accuracies = []  # For cross-validation
        self.losses_per_trial = []
        self.epochs_to_converge = []
        self.pattern_presentations = []
        self.computation_times = []
        self.convergence_threshold = 0.95  # Default threshold
        
        # Additional classification metrics
        self.train_f1_scores = []
        self.test_f1_scores = []
        self.val_f1_scores = []
        self.train_precision_scores = []
        self.test_precision_scores = []
        self.val_precision_scores = []
        self.train_recall_scores = []
        self.test_recall_scores = []
        self.val_recall_scores = []
        self.train_mcc_scores = []
        self.test_mcc_scores = []
        self.val_mcc_scores = []
        self.train_confusion_matrices = []
        self.test_confusion_matrices = []
        self.val_confusion_matrices = []
        
    def add_trial_results(self, train_acc: float, test_acc: float, losses: List[float], 
                         epochs_converged: int, num_presentations: int, 
                         computation_time: float, val_acc: Optional[float] = None,
                         train_metrics: Optional[Dict] = None, test_metrics: Optional[Dict] = None,
                         val_metrics: Optional[Dict] = None):
        """Add results from a single trial."""
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        self.losses_per_trial.append(losses)
        self.epochs_to_converge.append(epochs_converged)
        self.pattern_presentations.append(num_presentations)
        self.computation_times.append(computation_time)
        
        # Add additional metrics if provided
        if train_metrics:
            self.train_f1_scores.append(train_metrics.get('f1', 0.0))
            self.train_precision_scores.append(train_metrics.get('precision', 0.0))
            self.train_recall_scores.append(train_metrics.get('recall', 0.0))
            self.train_mcc_scores.append(train_metrics.get('mcc', 0.0))
            self.train_confusion_matrices.append(train_metrics.get('confusion_matrix', np.array([])))
        
        if test_metrics:
            self.test_f1_scores.append(test_metrics.get('f1', 0.0))
            self.test_precision_scores.append(test_metrics.get('precision', 0.0))
            self.test_recall_scores.append(test_metrics.get('recall', 0.0))
            self.test_mcc_scores.append(test_metrics.get('mcc', 0.0))
            self.test_confusion_matrices.append(test_metrics.get('confusion_matrix', np.array([])))
        
        if val_metrics:
            self.val_f1_scores.append(val_metrics.get('f1', 0.0))
            self.val_precision_scores.append(val_metrics.get('precision', 0.0))
            self.val_recall_scores.append(val_metrics.get('recall', 0.0))
            self.val_mcc_scores.append(val_metrics.get('mcc', 0.0))
            self.val_confusion_matrices.append(val_metrics.get('confusion_matrix', np.array([])))
    
    def compute_statistics(self) -> Dict:
        """Compute comprehensive statistics across all trials."""
        train_acc_array = np.array(self.train_accuracies)
        test_acc_array = np.array(self.test_accuracies)
        
        # Basic statistics
        stats_dict = {
            'train_acc_mean': np.mean(train_acc_array),
            'train_acc_std': np.std(train_acc_array),
            'test_acc_mean': np.mean(test_acc_array),
            'test_acc_std': np.std(test_acc_array),
            'num_trials': len(self.train_accuracies)
        }
        
        # Cross-validation statistics if available
        if self.val_accuracies:
            val_acc_array = np.array(self.val_accuracies)
            stats_dict.update({
                'val_acc_mean': np.mean(val_acc_array),
                'val_acc_std': np.std(val_acc_array)
            })
        
        # Additional classification metrics
        if self.train_f1_scores:
            stats_dict.update({
                'train_f1_mean': np.mean(self.train_f1_scores),
                'train_f1_std': np.std(self.train_f1_scores),
                'train_precision_mean': np.mean(self.train_precision_scores),
                'train_precision_std': np.std(self.train_precision_scores),
                'train_recall_mean': np.mean(self.train_recall_scores),
                'train_recall_std': np.std(self.train_recall_scores),
                'train_mcc_mean': np.mean(self.train_mcc_scores),
                'train_mcc_std': np.std(self.train_mcc_scores)
            })
        
        if self.test_f1_scores:
            stats_dict.update({
                'test_f1_mean': np.mean(self.test_f1_scores),
                'test_f1_std': np.std(self.test_f1_scores),
                'test_precision_mean': np.mean(self.test_precision_scores),
                'test_precision_std': np.std(self.test_precision_scores),
                'test_recall_mean': np.mean(self.test_recall_scores),
                'test_recall_std': np.std(self.test_recall_scores),
                'test_mcc_mean': np.mean(self.test_mcc_scores),
                'test_mcc_std': np.std(self.test_mcc_scores)
            })
        
        if self.val_f1_scores:
            stats_dict.update({
                'val_f1_mean': np.mean(self.val_f1_scores),
                'val_f1_std': np.std(self.val_f1_scores),
                'val_precision_mean': np.mean(self.val_precision_scores),
                'val_precision_std': np.std(self.val_precision_scores),
                'val_recall_mean': np.mean(self.val_recall_scores),
                'val_recall_std': np.std(self.val_recall_scores),
                'val_mcc_mean': np.mean(self.val_mcc_scores),
                'val_mcc_std': np.std(self.val_mcc_scores)
            })
        
        # Average confusion matrices
        if self.train_confusion_matrices and len(self.train_confusion_matrices[0]) > 0:
            stats_dict['train_confusion_matrix_mean'] = np.mean(self.train_confusion_matrices, axis=0)
            stats_dict['train_confusion_matrix_std'] = np.std(self.train_confusion_matrices, axis=0)
        
        if self.test_confusion_matrices and len(self.test_confusion_matrices[0]) > 0:
            stats_dict['test_confusion_matrix_mean'] = np.mean(self.test_confusion_matrices, axis=0)
            stats_dict['test_confusion_matrix_std'] = np.std(self.test_confusion_matrices, axis=0)
        
        if self.val_confusion_matrices and len(self.val_confusion_matrices[0]) > 0:
            stats_dict['val_confusion_matrix_mean'] = np.mean(self.val_confusion_matrices, axis=0)
            stats_dict['val_confusion_matrix_std'] = np.std(self.val_confusion_matrices, axis=0)
        
        # 95% Confidence intervals
        train_ci = stats.t.interval(0.95, len(train_acc_array)-1, 
                                   loc=np.mean(train_acc_array), 
                                   scale=stats.sem(train_acc_array))
        test_ci = stats.t.interval(0.95, len(test_acc_array)-1, 
                                  loc=np.mean(test_acc_array), 
                                  scale=stats.sem(test_acc_array))
        
        stats_dict.update({
            'train_acc_95ci': train_ci,
            'test_acc_95ci': test_ci
        })
        
        # Best generalization and pattern presentations
        best_test_idx = np.argmax(test_acc_array)
        stats_dict.update({
            'best_test_acc': test_acc_array[best_test_idx],
            'best_test_patterns': self.pattern_presentations[best_test_idx],
            'best_test_epochs': self.epochs_to_converge[best_test_idx]
        })
        
        # Generalization factor (Test/Train for best model)
        best_train_acc = train_acc_array[best_test_idx]
        stats_dict['generalization_factor'] = test_acc_array[best_test_idx] / best_train_acc if best_train_acc > 0 else 0
        
        # Efficiency metrics
        stats_dict.update({
            'avg_pattern_presentations': np.mean(self.pattern_presentations),
            'std_pattern_presentations': np.std(self.pattern_presentations),
            'avg_computation_time': np.mean(self.computation_times),
            'std_computation_time': np.std(self.computation_times)
        })
        
        # Convergence rate
        converged_trials = sum(1 for acc in test_acc_array if acc >= self.convergence_threshold)
        stats_dict['convergence_rate'] = converged_trials / len(test_acc_array)
        
        # Average epochs to convergence (only for converged trials)
        converged_epochs = [epochs for i, epochs in enumerate(self.epochs_to_converge) 
                           if test_acc_array[i] >= self.convergence_threshold]
        if converged_epochs:
            stats_dict.update({
                'avg_epochs_to_converge': np.mean(converged_epochs),
                'std_epochs_to_converge': np.std(converged_epochs)
            })
        else:
            stats_dict.update({
                'avg_epochs_to_converge': None,
                'std_epochs_to_converge': None
            })
        
        return stats_dict
    
    def print_comprehensive_report(self):
        """Print a comprehensive report of all metrics."""
        stats = self.compute_statistics()
        
        print("=" * 80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)
        
        print(f"\nACCURACY METRICS (n={stats['num_trials']} trials)")
        print("-" * 50)
        print(f"Train Accuracy: {stats['train_acc_mean']:.4f} ± {stats['train_acc_std']:.4f}")
        print(f"Test Accuracy:  {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")
        
        if 'val_acc_mean' in stats:
            print(f"Val Accuracy:   {stats['val_acc_mean']:.4f} ± {stats['val_acc_std']:.4f}")
        
        # Additional classification metrics
        if 'test_f1_mean' in stats:
            print(f"\nCLASSIFICATION METRICS")
            print("-" * 50)
            print(f"F1 Score:")
            print(f"  Train: {stats.get('train_f1_mean', 0):.4f} ± {stats.get('train_f1_std', 0):.4f}")
            print(f"  Test:  {stats['test_f1_mean']:.4f} ± {stats['test_f1_std']:.4f}")
            if 'val_f1_mean' in stats:
                print(f"  Val:   {stats['val_f1_mean']:.4f} ± {stats['val_f1_std']:.4f}")
            
            print(f"Precision:")
            print(f"  Train: {stats.get('train_precision_mean', 0):.4f} ± {stats.get('train_precision_std', 0):.4f}")
            print(f"  Test:  {stats['test_precision_mean']:.4f} ± {stats['test_precision_std']:.4f}")
            if 'val_precision_mean' in stats:
                print(f"  Val:   {stats['val_precision_mean']:.4f} ± {stats['val_precision_std']:.4f}")
            
            print(f"Recall:")
            print(f"  Train: {stats.get('train_recall_mean', 0):.4f} ± {stats.get('train_recall_std', 0):.4f}")
            print(f"  Test:  {stats['test_recall_mean']:.4f} ± {stats['test_recall_std']:.4f}")
            if 'val_recall_mean' in stats:
                print(f"  Val:   {stats['val_recall_mean']:.4f} ± {stats['val_recall_std']:.4f}")
            
            print(f"Matthews Correlation Coefficient (MCC):")
            print(f"  Train: {stats.get('train_mcc_mean', 0):.4f} ± {stats.get('train_mcc_std', 0):.4f}")
            print(f"  Test:  {stats['test_mcc_mean']:.4f} ± {stats['test_mcc_std']:.4f}")
            if 'val_mcc_mean' in stats:
                print(f"  Val:   {stats['val_mcc_mean']:.4f} ± {stats['val_mcc_std']:.4f}")
        
        # Confusion matrices
        if 'test_confusion_matrix_mean' in stats:
            print(f"\nCONFUSION MATRICES (Mean ± Std)")
            print("-" * 50)
            test_cm_mean = stats['test_confusion_matrix_mean']
            test_cm_std = stats['test_confusion_matrix_std']
            print(f"Test Confusion Matrix:")
            for i in range(test_cm_mean.shape[0]):
                row_str = "  ["
                for j in range(test_cm_mean.shape[1]):
                    row_str += f"{test_cm_mean[i,j]:.1f}±{test_cm_std[i,j]:.1f}"
                    if j < test_cm_mean.shape[1] - 1:
                        row_str += ", "
                row_str += "]"
                print(row_str)
        
        print(f"\n95% CONFIDENCE INTERVALS")
        print("-" * 50)
        print(f"Train Accuracy: [{stats['train_acc_95ci'][0]:.4f}, {stats['train_acc_95ci'][1]:.4f}]")
        print(f"Test Accuracy:  [{stats['test_acc_95ci'][0]:.4f}, {stats['test_acc_95ci'][1]:.4f}]")
        
        print(f"\nBEST GENERALIZATION")
        print("-" * 50)
        print(f"Best Test Accuracy: {stats['best_test_acc']:.4f}")
        print(f"Pattern Presentations: {stats['best_test_patterns']}")
        print(f"Epochs to Best: {stats['best_test_epochs']}")
        print(f"Generalization Factor: {stats['generalization_factor']:.4f}")
        
        print(f"\nEFFICIENCY METRICS")
        print("-" * 50)
        print(f"Avg Pattern Presentations: {stats['avg_pattern_presentations']:.1f} ± {stats['std_pattern_presentations']:.1f}")
        print(f"Avg Computation Time: {stats['avg_computation_time']:.3f}s ± {stats['std_computation_time']:.3f}s")
        
        print(f"\nCONVERGENCE ANALYSIS")
        print("-" * 50)
        print(f"Convergence Rate (≥{self.convergence_threshold:.1%}): {stats['convergence_rate']:.1%}")
        if stats['avg_epochs_to_converge'] is not None:
            print(f"Avg Epochs to Converge: {stats['avg_epochs_to_converge']:.1f} ± {stats['std_epochs_to_converge']:.1f}")
        else:
            print("No trials reached convergence threshold")


def compute_computational_savings(baseline_time: float, current_time: float) -> Dict:
    """Compute computational cost savings vs baseline."""
    savings_ratio = (baseline_time - current_time) / baseline_time if baseline_time > 0 else 0
    speedup = baseline_time / current_time if current_time > 0 else float('inf')
    
    return {
        'baseline_time': baseline_time,
        'current_time': current_time,
        'time_savings': baseline_time - current_time,
        'savings_ratio': savings_ratio,
        'speedup_factor': speedup
    }


def compute_classification_metrics_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict:
    """Compute comprehensive classification metrics efficiently on GPU."""
    device = y_true.device
    n_classes = 3  # For Iris dataset
    
    try:
        # Compute confusion matrix on GPU
        cm = compute_confusion_matrix_gpu(y_true, y_pred, n_classes)
        
        # Compute per-class metrics
        tp = torch.diag(cm).float()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # Precision, Recall, F1 per class
        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)
        
        # Macro averages
        precision = precision_per_class.mean().item()
        recall = recall_per_class.mean().item()
        f1 = f1_per_class.mean().item()
        
        # Matthews Correlation Coefficient
        mcc = compute_mcc_gpu(cm, tp, fp, fn, tn).item()
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'confusion_matrix': cm.cpu().numpy()  # Only convert final result
        }
    except Exception as e:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'mcc': 0.0,
            'confusion_matrix': np.zeros((n_classes, n_classes))
        }


def compute_confusion_matrix_gpu(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Compute confusion matrix on GPU."""
    # Create confusion matrix using bincount
    indices = n_classes * y_true + y_pred
    cm = torch.bincount(indices, minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    return cm


def compute_mcc_gpu(cm: torch.Tensor, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> torch.Tensor:
    """Compute Matthews Correlation Coefficient on GPU for multiclass."""
    # For multiclass MCC, we use the formula based on confusion matrix
    s = tp.sum()
    if s == 0:
        return torch.tensor(0.0, device=cm.device)
    
    # Simplified MCC for multiclass
    numerator = (tp * tn - fp * fn).sum()
    denominator = torch.sqrt((tp + fp).sum() * (tp + fn).sum() * (tn + fp).sum() * (tn + fn).sum())
    
    if denominator == 0:
        return torch.tensor(0.0, device=cm.device)
    
    return numerator / denominator


def compute_accuracy_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute accuracy on GPU without CPU conversion."""
    return (y_true == y_pred).float().mean().item()


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute comprehensive classification metrics (CPU fallback)."""
    try:
        # Compute metrics with macro averaging for multi-class
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'confusion_matrix': cm
        }
    except Exception as e:
        # Return zeros if computation fails
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'mcc': 0.0,
            'confusion_matrix': np.zeros((n_classes, n_classes))
        }


def find_convergence_epoch(losses: List[float], threshold: float = 0.01, 
                          patience: int = 10) -> int:
    """Find the epoch where the model converged based on loss stabilization."""
    if len(losses) < patience:
        return len(losses)
    
    for i in range(patience, len(losses)):
        # Check if loss has been stable for 'patience' epochs
        recent_losses = losses[i-patience:i]
        loss_std = np.std(recent_losses)
        if loss_std < threshold:
            return i - patience
    
    return len(losses)  # Never converged