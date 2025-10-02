import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
from typing import Dict, List, Tuple, Optional, Callable
from .models import RegressionNet
from .metrics import MetricsTracker, find_convergence_epoch


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary with MSE, RMSE, MAE, and R² score
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


class RegressionModelEvaluator:
    """Comprehensive regression model evaluation with support for multiple learning strategies."""
    
    def __init__(self, convergence_threshold: float = 0.95):
        self.convergence_threshold = convergence_threshold
        self.metrics_tracker = MetricsTracker()
        self.metrics_tracker.convergence_threshold = convergence_threshold
    
    def evaluate_passive_learning(self, X: torch.Tensor, y: torch.Tensor, 
                                 best_params: Dict, n_trials: int = 50,
                                 use_cv: bool = True, cv_folds: int = 5,
                                 epochs: int = 1000, random_state: int = 42) -> Dict:
        """
        Evaluate passive learning with comprehensive metrics tracking for regression.
        
        Args:
            X: Input features tensor
            y: Target values tensor (continuous)
            best_params: Dictionary with best hyperparameters
            n_trials: Number of trials to run
            use_cv: Whether to use cross-validation
            cv_folds: Number of CV folds
            epochs: Maximum epochs per trial
            random_state: Base random state
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print(f"Starting {n_trials} trial evaluation for regression...")
        print(f"Parameters: {best_params}")
        print(f"Cross-validation: {'Yes' if use_cv else 'No'} ({cv_folds} folds)")
        
        self.metrics_tracker.reset()
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}", end="... ")
            
            start_time = time.time()
            
            if use_cv:
                # Cross-validation approach
                trial_results = self._run_cv_trial(X, y, best_params, 
                                                 cv_folds, epochs, random_state + trial)
            else:
                # Simple train-test split approach
                trial_results = self._run_simple_trial(X, y, best_params, 
                                                     epochs, random_state + trial)
            
            computation_time = time.time() - start_time
            
            # Add results to tracker (use negative MSE as "accuracy" for regression - higher is better)
            # We use negative MSE instead of R² because R² can be negative and unbounded
            self.metrics_tracker.add_trial_results(
                train_acc=trial_results['train_neg_mse'],  # Using -MSE as accuracy metric
                test_acc=trial_results['test_neg_mse'],
                val_acc=trial_results.get('val_neg_mse'),
                losses=trial_results['losses'],
                val_losses=trial_results.get('val_losses'),
                epochs_converged=trial_results['epochs_converged'],
                num_presentations=trial_results['num_presentations'],
                computation_time=computation_time,
                train_metrics=trial_results.get('train_metrics'),
                test_metrics=trial_results.get('test_metrics'),
                val_metrics=trial_results.get('val_metrics'),
                train_acc_curve=trial_results.get('train_r2_curve'),
                test_acc_curve=trial_results.get('test_r2_curve')
            )
            
            print(f"Test R²: {trial_results['test_r2']:.4f}, Time: {computation_time:.2f}s")
        
        # Compute and return comprehensive statistics
        results = self.metrics_tracker.compute_statistics()
        results['learning_type'] = 'passive'
        results['best_params'] = best_params
        
        return results
    
    def _run_cv_trial(self, X: torch.Tensor, y: torch.Tensor,
                     params: Dict, cv_folds: int, epochs: int, random_state: int) -> Dict:
        """Run a single trial with cross-validation for regression."""
        
        # First, create a holdout test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Cross-validation on the remaining 80%
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_val_r2s = []
        cv_train_r2s = []
        all_losses = []
        
        cv_train_metrics = []
        cv_val_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_temp)):
            X_train = X_temp[train_idx]
            X_val = X_temp[val_idx]
            y_train = y_temp[train_idx]
            y_val = y_temp[val_idx]
            
            # Train model
            model, losses, train_r2, val_r2, _, train_metrics, val_metrics, val_losses, _, _ = self._train_model(
                X_train, y_train, X_val, y_val, params, epochs
            )
            
            cv_train_r2s.append(train_r2)
            cv_val_r2s.append(val_r2)
            cv_train_metrics.append(train_metrics)
            cv_val_metrics.append(val_metrics)
            all_losses.extend(losses)
        
        # Average CV results
        avg_train_r2 = np.mean(cv_train_r2s)
        avg_val_r2 = np.mean(cv_val_r2s)
        
        # Average CV metrics
        avg_train_metrics = self._average_metrics(cv_train_metrics)
        avg_val_metrics = self._average_metrics(cv_val_metrics)
        
        # Final test on holdout set using best parameters
        final_model, final_losses, _, _, epochs_converged, _, test_metrics, final_val_losses, train_r2_curve, test_r2_curve = self._train_model(
            X_temp, y_temp, X_test, y_test, params, epochs
        )
        
        # Evaluate on test set
        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(X_test)
            test_r2 = r2_score(y_test.numpy(), test_outputs.numpy())
            test_mse = mean_squared_error(y_test.numpy(), test_outputs.numpy())
        
        # Calculate pattern presentations
        num_presentations = len(X_temp) * epochs_converged
        
        return {
            'train_r2': avg_train_r2,
            'test_r2': test_r2,
            'val_r2': avg_val_r2,
            'train_neg_mse': -avg_train_metrics.get('mse', 0),
            'test_neg_mse': -test_mse,
            'val_neg_mse': -avg_val_metrics.get('mse', 0),
            'losses': final_losses,
            'val_losses': final_val_losses,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'train_metrics': avg_train_metrics,
            'test_metrics': test_metrics,
            'val_metrics': avg_val_metrics,
            'train_r2_curve': train_r2_curve,
            'test_r2_curve': test_r2_curve
        }
    
    def _run_simple_trial(self, X: torch.Tensor, y: torch.Tensor,
                         params: Dict, epochs: int, random_state: int) -> Dict:
        """Run a single trial with simple train-test split for regression."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Train model
        model, losses, train_r2, test_r2, epochs_converged, train_metrics, test_metrics, val_losses, train_r2_curve, test_r2_curve = self._train_model(
            X_train, y_train, X_test, y_test, params, epochs
        )
        
        # Calculate pattern presentations
        num_presentations = len(X_train) * epochs_converged
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_neg_mse': -train_metrics.get('mse', 0),
            'test_neg_mse': -test_metrics.get('mse', 0),
            'losses': losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'train_r2_curve': train_r2_curve,
            'test_r2_curve': test_r2_curve
        }
    
    def _train_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_test: torch.Tensor, y_test: torch.Tensor,
                    params: Dict, epochs: int) -> Tuple:
        """Train a single regression model and return results."""
        
        # Create regression model
        model = RegressionNet(
            input_size=params.get('input_size', X_train.shape[1]),
            hidden_size=params['hidden_size'],
            output_size=1  # Single output for regression
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)
        )
        
        # Training loop
        losses = []
        val_losses = []
        train_r2_curve = []
        test_r2_curve = []
        
        for epoch in range(epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            
            
            # IMPORTANT: Add gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            
            optimizer.step()
            losses.append(loss.item())
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())
                
                # Track R² at each epoch
                train_outputs = model(X_train)
                train_r2 = r2_score(y_train.numpy(), train_outputs.numpy())
                test_r2 = r2_score(y_test.numpy(), val_outputs.numpy())
                
                train_r2_curve.append(train_r2)
                test_r2_curve.append(test_r2)
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            test_outputs = model(X_test)
            
            train_r2 = r2_score(y_train.numpy(), train_outputs.numpy())
            test_r2 = r2_score(y_test.numpy(), test_outputs.numpy())
            
            # Compute comprehensive regression metrics
            train_metrics = compute_regression_metrics(y_train.numpy(), train_outputs.numpy())
            test_metrics = compute_regression_metrics(y_test.numpy(), test_outputs.numpy())
        
        return model, losses, train_r2, test_r2, epochs_converged, train_metrics, test_metrics, val_losses, train_r2_curve, test_r2_curve
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average regression metrics across folds."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def print_report(self):
        """Print comprehensive evaluation report."""
        self.metrics_tracker.print_comprehensive_report()
    
    def get_results(self) -> Dict:
        """Get evaluation results."""
        return self.metrics_tracker.compute_statistics()


class ActiveLearningRegressionEvaluator(RegressionModelEvaluator):
    """
    Extended evaluator for active learning strategies in regression tasks.
    """
    
    def __init__(self, convergence_threshold: float = 0.95):
        super().__init__(convergence_threshold)
        self.training_set_reductions = {}
        self.training_set_sizes_by_epoch = {}
        
    def evaluate_active_learning(self, X: torch.Tensor, y: torch.Tensor,
                                best_params: Dict, strategy: str,
                                n_trials: int = 50, use_cv: bool = True,
                                cv_folds: int = 5, epochs: int = 1000,
                                random_state: int = 12, 
                                **strategy_kwargs) -> Dict:
        """
        Evaluate active learning strategies for regression.
        
        Args:
            strategy: 'output_sensitivity' for SASLA
            strategy_kwargs: Additional arguments for the active learning strategy
        """
        print(f"Starting {n_trials} trial evaluation for {strategy} active learning (regression)...")
        print(f"Parameters: {best_params}")
        print(f"Cross-validation: {'Yes' if use_cv else 'No'} ({cv_folds} folds)")
        
        self.metrics_tracker.reset()
        self.training_set_reductions = {25: [], 100: [], 200: [], 300: [], 500: [], 1000: []}
        self.final_training_set_sizes = []
        self.original_training_set_sizes = []
        self.training_set_sizes_by_epoch = {epoch: [] for epoch in [25, 100, 200, 300, 500, 1000]}
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            # Set random seed for reproducibility
            trial_seed = random_state + trial
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            
            start_time = time.time()
            
            if use_cv:
                trial_results = self._run_active_cv_trial(
                    X, y, best_params, strategy, cv_folds, 
                    epochs, trial_seed, **strategy_kwargs
                )
            else:
                trial_results = self._run_active_simple_trial(
                    X, y, best_params, strategy, 
                    epochs, trial_seed, **strategy_kwargs
                )
            
            computation_time = time.time() - start_time
            
            # Add results to tracker (use negative MSE instead of R² to avoid unbounded values)
            self.metrics_tracker.add_trial_results(
                train_acc=trial_results['train_neg_mse'],  # Using -MSE as accuracy
                test_acc=trial_results['test_neg_mse'],
                losses=trial_results['losses'],
                val_losses=trial_results.get('val_losses'),
                epochs_converged=trial_results['epochs_converged'],
                num_presentations=trial_results['num_presentations'],
                computation_time=computation_time,
                val_acc=trial_results.get('val_neg_mse', None),
                train_metrics=trial_results.get('train_metrics'),
                test_metrics=trial_results.get('test_metrics'),
                val_metrics=trial_results.get('val_metrics'),
                train_acc_curve=trial_results.get('train_r2_curve'),
                test_acc_curve=trial_results.get('test_r2_curve')
            )
            
            # Track training set size information
            self.final_training_set_sizes.append(trial_results.get('final_training_set_size', 0))
            self.original_training_set_sizes.append(trial_results.get('original_training_set_size', 0))
            
            # Track training set sizes at specific epochs
            training_set_sizes_at_epochs = trial_results.get('training_set_sizes_at_epochs', {})
            for epoch in self.training_set_sizes_by_epoch.keys():
                if epoch in training_set_sizes_at_epochs:
                    self.training_set_sizes_by_epoch[epoch].append(training_set_sizes_at_epochs[epoch])
            
            # Track training set reductions
            reduction_data = trial_results.get('training_set_reductions', {})
            if 'final_reduction_percentage' in reduction_data:
                final_reduction = reduction_data['final_reduction_percentage']
                for epoch_check in self.training_set_reductions.keys():
                    self.training_set_reductions[epoch_check].append(final_reduction)
        
        # Compute comprehensive statistics
        results = self.metrics_tracker.compute_statistics()
        results['learning_type'] = f'active_{strategy}'
        results['best_params'] = best_params
        
        # Add training set size statistics
        if self.final_training_set_sizes:
            results['final_training_set_size_mean'] = np.mean(self.final_training_set_sizes)
            results['final_training_set_size_std'] = np.std(self.final_training_set_sizes)
            results['original_training_set_size_mean'] = np.mean(self.original_training_set_sizes)
            
            # Calculate reduction percentages
            reductions = [(orig - final) / orig * 100 for orig, final in 
                         zip(self.original_training_set_sizes, self.final_training_set_sizes)]
            results['training_set_reduction_mean'] = np.mean(reductions)
            results['training_set_reduction_std'] = np.std(reductions)
        
        # Add training set sizes at specific epochs
        training_set_size_stats = {}
        for epoch, sizes in self.training_set_sizes_by_epoch.items():
            if sizes:
                training_set_size_stats[f'size_at_{epoch}_epochs'] = {
                    'mean': np.mean(sizes),
                    'std': np.std(sizes),
                    'min': np.min(sizes),
                    'max': np.max(sizes)
                }
        results['training_set_sizes_by_epoch'] = training_set_size_stats
        
        # Add training set reduction statistics
        reduction_stats = {}
        for epoch, reductions in self.training_set_reductions.items():
            if reductions:
                reduction_stats[f'reduction_at_{epoch}_epochs'] = {
                    'mean': np.mean(reductions),
                    'std': np.std(reductions),
                    'min': np.min(reductions),
                    'max': np.max(reductions)
                }
        results['training_set_reductions'] = reduction_stats
        
        return results
    
    def _run_active_cv_trial(self, X: torch.Tensor, y: torch.Tensor,
                            params: Dict, strategy: str, cv_folds: int, epochs: int, 
                            random_state: int, **strategy_kwargs) -> Dict:
        """Run a single active learning trial with cross-validation for regression."""
        
        # First, create a holdout test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Cross-validation on the remaining 80%
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_val_r2s = []
        cv_train_r2s = []
        all_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_temp)):
            X_train_fold = X_temp[train_idx]
            X_val_fold = X_temp[val_idx]
            y_train_fold = y_temp[train_idx]
            y_val_fold = y_temp[val_idx]
            
            model, losses, train_r2, val_r2, epochs_converged, reduction_data, val_losses, _, _ = self._train_active_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                params, strategy, epochs, **strategy_kwargs
            )
            
            cv_train_r2s.append(train_r2)
            cv_val_r2s.append(val_r2)
            all_losses.extend(losses)
        
        # Average CV results
        avg_train_r2 = np.mean(cv_train_r2s)
        avg_val_r2 = np.mean(cv_val_r2s)
        
        # Final test on holdout set
        final_model, final_losses, final_train_r2, _, epochs_converged, final_reduction_data, final_val_losses, _, _ = self._train_active_model(
            X_temp, y_temp, X_test, y_test, params, strategy, epochs, **strategy_kwargs
        )
        
        # Evaluate on test set  
        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(X_test)
            train_outputs = final_model(X_temp)
            final_test_r2 = r2_score(y_test.numpy(), test_outputs.numpy())
            final_test_mse = mean_squared_error(y_test.numpy(), test_outputs.numpy())
            final_train_mse = mean_squared_error(y_temp.numpy(), train_outputs.numpy())
            final_train_r2 = r2_score(y_temp.numpy(), train_outputs.numpy())
        
        num_presentations = len(X_temp) * epochs_converged
        
        # Create metrics dictionaries with R² information
        train_metrics = {
            'mse': final_train_mse,
            'r2': final_train_r2,
            'accuracy': -final_train_mse  # For compatibility
        }
        test_metrics = {
            'mse': final_test_mse,
            'r2': final_test_r2,
            'accuracy': -final_test_mse  # For compatibility
        }
        val_metrics = {
            'mse': final_test_mse,
            'r2': avg_val_r2,
            'accuracy': -final_test_mse
        }
        
        return {
            'train_r2': avg_train_r2,
            'test_r2': final_test_r2,
            'val_r2': avg_val_r2,
            'train_neg_mse': -final_train_mse,
            'test_neg_mse': -final_test_mse,
            'val_neg_mse': -final_test_mse,  # Use test MSE as proxy for val
            'losses': final_losses,
            'val_losses': final_val_losses,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'training_set_reductions': final_reduction_data,
            'final_training_set_size': final_reduction_data.get('final_labeled_size', len(X_temp)),
            'original_training_set_size': final_reduction_data.get('original_training_size', len(X_temp)),
            'training_set_sizes_at_epochs': final_reduction_data.get('training_set_sizes_at_epochs', {}),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics
        }
    
    def _run_active_simple_trial(self, X: torch.Tensor, y: torch.Tensor,
                                params: Dict, strategy: str, epochs: int, 
                                random_state: int, **strategy_kwargs) -> Dict:
        """Run a single active learning trial with simple train-test split for regression."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Train model with active learning
        model, losses, train_r2, test_r2, epochs_converged, reduction_data, val_losses, train_r2_curve, test_r2_curve = self._train_active_model(
            X_train, y_train, X_test, y_test, params, strategy, epochs, **strategy_kwargs
        )
        
        num_presentations = len(X_train) * epochs_converged
        
        # Calculate MSE for use as "accuracy" metric
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            test_outputs = model(X_test)
            train_mse = mean_squared_error(y_train.numpy(), train_outputs.numpy())
            test_mse = mean_squared_error(y_test.numpy(), test_outputs.numpy())
        
        # Create metrics dictionaries with R² information
        train_metrics = {
            'mse': train_mse,
            'r2': train_r2,
            'r2_curve': train_r2_curve,
            'accuracy': -train_mse  # For compatibility
        }
        test_metrics = {
            'mse': test_mse,
            'r2': test_r2,
            'r2_curve': test_r2_curve,
            'accuracy': -test_mse  # For compatibility
        }
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_neg_mse': -train_mse,
            'test_neg_mse': -test_mse,
            'losses': losses,
            'val_losses': val_losses,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'training_set_reductions': reduction_data,
            'final_training_set_size': reduction_data.get('final_labeled_size', len(X_train)),
            'original_training_set_size': reduction_data.get('original_training_size', len(X_train)),
            'training_set_sizes_at_epochs': reduction_data.get('training_set_sizes_at_epochs', {}),
            'train_r2_curve': train_r2_curve,
            'test_r2_curve': test_r2_curve,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def _train_active_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                           X_test: torch.Tensor, y_test: torch.Tensor,
                           params: Dict, strategy: str, epochs: int, **strategy_kwargs) -> Tuple:
        """Train a single model using active learning strategy for regression."""
        
        if strategy == 'output_sensitivity':
            return self._train_output_sensitivity_model(
                X_train, y_train, X_test, y_test, params, epochs, **strategy_kwargs
            )
        elif strategy == 'uncertainty_sampling':
            return self._train_uncertainty_sampling_model(
                X_train, y_train, X_test, y_test, params, epochs, **strategy_kwargs
            )
        elif strategy == 'ensemble_uncertainty':
            return self._train_ensemble_uncertainty_model(
                X_train, y_train, X_test, y_test, params, epochs, **strategy_kwargs
            )
        else:
            raise ValueError(f"Unknown active learning strategy: {strategy}")
    
    def _train_output_sensitivity_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                       X_test: torch.Tensor, y_test: torch.Tensor,
                                       params: Dict, epochs: int,
                                       alpha: float = 0.9,
                                       selection_interval: int = 1) -> Tuple:
        """
        Train regression model using output sensitivity active learning (SASLA approach).
        
        This implements the SASLA algorithm adapted for regression:
        - Computes exact output sensitivity using derivatives
        - Calculates pattern informativeness
        - Selects patterns with informativeness > (1-α) * average_informativeness
        """
        
        # Create regression model
        model = RegressionNet(
            input_size=params.get('input_size', X_train.shape[1]),
            hidden_size=params['hidden_size'],
            output_size=1  # Single output for regression
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)
        )
        
        # Initialize training data
        current_X_train = X_train.clone()
        current_y_train = y_train.clone()
        original_train_size = len(X_train)
        
        # Track training set reductions at specific epochs
        reduction_data_actual_size = {}
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        
        losses = []
        val_losses = []
        train_r2_curve = []
        test_r2_curve = []
        
        for epoch in range(epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(current_X_train)
            loss = criterion(outputs, current_y_train)
            
            # Backward pass
            loss.backward()
            
            # IMPORTANT: Add gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            
            # Validation step and R² tracking
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())
                
                # Track R² at each epoch
                train_outputs = model(current_X_train)
                train_r2 = r2_score(current_y_train.numpy(), train_outputs.numpy())
                test_r2 = r2_score(y_test.numpy(), val_outputs.numpy())
                
                train_r2_curve.append(train_r2)
                test_r2_curve.append(test_r2)
            
            # Apply SASLA pattern selection at specified intervals
            if epoch > 0 and epoch % selection_interval == 0:
                current_X_train, current_y_train = self._apply_sasla_selection_regression(
                    model, current_X_train, current_y_train, alpha
                )
            
            # Record training set size at checkpoints
            if epoch + 1 in epoch_checkpoints:
                current_size = len(current_X_train)
                reduction_data_actual_size[epoch + 1] = current_size

        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(current_X_train)
            test_outputs = model(X_test)
            
            # Check if we have enough training samples for stable R² calculation
            if len(current_X_train) < 3:
                print(f"Warning: Too few training samples ({len(current_X_train)}) - R² may be unstable")
            
            train_r2 = r2_score(current_y_train.numpy(), train_outputs.numpy())
            test_r2 = r2_score(y_test.numpy(), test_outputs.numpy())
            
            # Check for problematic R² values
            if test_r2 < -0.5:
                print(f"Warning: Very poor test R² ({test_r2:.4f}) - model may have collapsed. Training set size: {len(current_X_train)}/{original_train_size}")
        
        reduction_data_result = {
            'training_set_sizes_at_epochs': reduction_data_actual_size,
            'final_labeled_size': len(current_X_train),
            'original_training_size': original_train_size,
            'final_reduction_percentage': ((original_train_size - len(current_X_train)) / original_train_size) * 100
        }
        
        return model, losses, train_r2, test_r2, epochs_converged, reduction_data_result, val_losses, train_r2_curve, test_r2_curve
    
    def _apply_sasla_selection_regression(self, model: nn.Module, X_train: torch.Tensor, 
                                         y_train: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SASLA pattern selection for regression tasks.
        
        This implements the SASLA algorithm adapted for regression:
        1. Compute exact output sensitivity using derivatives
        2. Calculate pattern informativeness
        3. Select patterns with informativeness > (1-α) * average_informativeness
        
        Args:
            model: Regression neural network model
            X_train: Training input patterns
            y_train: Training target values (continuous)
            alpha: Selection constant (typically 0.9)
        """
        
        model.eval()
        
        # Get model parameters for sensitivity calculation
        params = list(model.parameters())
        if len(params) < 4:
            # Not enough layers for SASLA, return unchanged
            return X_train, y_train
            
        # Weights: input->hidden (v), hidden->output (w)
        v_weights = params[0]  # Shape: [hidden_size, input_size]
        v_bias = params[1]     # Shape: [hidden_size]
        w_weights = params[2]  # Shape: [1, hidden_size] for regression
        w_bias = params[3]     # Shape: [1]
        
        # VECTORIZED COMPUTATION FOR ALL PATTERNS AT ONCE
        with torch.no_grad():
            # Forward pass for all patterns simultaneously
            hidden_pre = torch.matmul(X_train, v_weights.T) + v_bias
            hidden_post = torch.sigmoid(hidden_pre)  # y_j, shape: [n_samples, hidden_size]
            
            output_pre = torch.matmul(hidden_post, w_weights.T) + w_bias
            # For regression, NO output activation - use raw output
            output_vals = output_pre  # o (single output), shape: [n_samples, 1]
            
            n_samples, n_hidden = hidden_post.shape
            
            # Compute weighted inputs v_ji for all patterns and hidden units
            v_ji = torch.matmul(X_train, v_weights.T)  # [n_samples, n_hidden]
            
            # For regression with linear output, sensitivity is:
            # S_o(i) = Σ_j [w_j * (1 - y_j) * y_j * v_ji]
            # (No output activation derivative term needed)
            
            # w_weights shape: [1, n_hidden], expand to [n_samples, n_hidden]
            w_expanded = w_weights.expand(n_samples, -1)
            
            # Compute sensitivity terms for each hidden unit
            hidden_expanded = hidden_post  # [n_samples, n_hidden]
            
            # sensitivity_terms: w_j * (1 - y_j) * y_j * v_ji for each sample and hidden unit
            sensitivity_terms = w_expanded * (1 - hidden_expanded) * hidden_expanded * v_ji
            
            # Sum over hidden units for each sample
            all_sensitivities = torch.sum(sensitivity_terms, dim=1, keepdim=True)  # [n_samples, 1]
            
            # For regression, informativeness is the absolute sensitivity
            informativeness_scores = torch.abs(all_sensitivities).squeeze()  # [n_samples]
        
        # Calculate selection threshold: (1-α) * average_informativeness
        avg_informativeness = torch.mean(informativeness_scores)
        selection_threshold = (1 - alpha) * avg_informativeness
        
        # Select patterns with informativeness above threshold
        selected_indices = informativeness_scores >= selection_threshold
        
        # Ensure we keep at least a minimum percentage of samples to prevent instability
        # For regression, we need enough samples for stable model training
        min_samples_pct = 0.3  # Keep at least 30% of original training set
        min_samples = max(10, int(min_samples_pct * len(X_train)))  # At least 10 samples or 30%
        num_selected = torch.sum(selected_indices).item()
        
        if num_selected < min_samples:
            # If too few samples selected, keep the top min_samples most informative ones
            top_k_values, top_k_indices = torch.topk(informativeness_scores, min_samples)
            selected_indices = torch.zeros(len(X_train), dtype=torch.bool)
            selected_indices[top_k_indices] = True
        
        # Ensure we keep at least one sample
        if not torch.any(selected_indices):
            # If no samples selected, keep the most informative one
            best_idx = torch.argmax(informativeness_scores)
            selected_indices[best_idx] = True
        
        # Return selected patterns
        return X_train[selected_indices], y_train[selected_indices]
    
    def _train_uncertainty_sampling_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                         X_test: torch.Tensor, y_test: torch.Tensor,
                                         params: Dict, epochs: int,
                                         uncertainty_method: str = 'variance') -> Tuple:
        """
        Train regression model using Pool-Based Active Learning with Uncertainty Sampling.
        
        For regression, uncertainty is measured by:
        - Prediction variance (epistemic uncertainty)
        - Prediction confidence intervals
        """
        
        # Pool-Based Active Learning Setup
        initial_labeled_size = max(10, len(X_train) // 10)  # Start with 10% or minimum 10 samples
        total_budget = len(X_train) - initial_labeled_size
        
        # Initialize labeled set L (random sampling for regression)
        indices = torch.arange(len(X_train))
        perm = torch.randperm(len(X_train))
        labeled_indices = perm[:initial_labeled_size]
        unlabeled_indices = perm[initial_labeled_size:]
        
        # Initialize L (labeled data) and U (unlabeled pool)
        L_X = X_train[labeled_indices].clone()
        L_y = y_train[labeled_indices].clone()
        
        U_X = X_train[unlabeled_indices].clone()
        U_y = y_train[unlabeled_indices].clone()
        
        # Initialize model
        model = RegressionNet(
            input_size=params.get('input_size', X_train.shape[1]),
            hidden_size=params['hidden_size'],
            output_size=1
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)
        )
        
        # Training tracking
        losses = []
        val_losses = []
        train_r2_curve = []
        test_r2_curve = []
        pool_sizes = []
        labeled_sizes = []
        
        # Track training set sizes at specific epochs
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        training_set_sizes_at_epochs = {}
        
        # Pool-Based Active Learning Algorithm
        budget_remaining = total_budget
        query_interval = max(1, total_budget // 50)  # Make ~50 queries total
        training_epochs_per_query = max(10, epochs // 50)
        
        query_iteration = 0
        total_epochs_used = 0
        
        while budget_remaining > 0 and len(U_X) > 0 and total_epochs_used < epochs:
            
            # Train model on current labeled set L
            for epoch in range(training_epochs_per_query):
                if total_epochs_used >= epochs:
                    break
                    
                # Training step
                model.train()
                optimizer.zero_grad()
                outputs = model(L_X)
                loss = criterion(outputs, L_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses.append(loss.item())
                
                # Validation step
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test)
                    val_losses.append(val_loss.item())
                    
                    # Track R² at each epoch
                    train_outputs = model(L_X)
                    train_r2 = r2_score(L_y.numpy(), train_outputs.numpy())
                    test_r2 = r2_score(y_test.numpy(), val_outputs.numpy())
                    
                    train_r2_curve.append(train_r2)
                    test_r2_curve.append(test_r2)
                
                total_epochs_used += 1
                
                # Track pool statistics
                pool_sizes.append(len(U_X))
                labeled_sizes.append(len(L_X))
                
                # Track training set size at checkpoints
                if total_epochs_used in epoch_checkpoints:
                    training_set_sizes_at_epochs[total_epochs_used] = len(L_X)
            
            # If no more unlabeled data or budget exhausted, break
            if len(U_X) == 0 or budget_remaining <= 0:
                break
            
            # Compute uncertainty for all samples in U
            # For regression, we use prediction variance as uncertainty
            model.eval()
            uncertainties = []
            
            with torch.no_grad():
                # Get predictions for all unlabeled samples
                U_predictions = model(U_X)
                
                # Calculate uncertainty based on prediction variance
                # We'll use the absolute residual as a proxy for uncertainty
                # (samples far from the current prediction trend are more uncertain)
                for i in range(len(U_X)):
                    pred = U_predictions[i].item()
                    
                    # Calculate local variance by looking at nearby predictions
                    # Higher variance = higher uncertainty
                    if len(U_predictions) > 1:
                        # Use distance from mean prediction as uncertainty measure
                        mean_pred = torch.mean(U_predictions)
                        uncertainty = abs(pred - mean_pred.item())
                    else:
                        uncertainty = abs(pred)
                    
                    uncertainties.append((uncertainty, i))
            
            # Select highest uncertainty samples
            uncertainties.sort(reverse=True, key=lambda x: x[0])
            samples_to_query = min(query_interval, len(uncertainties), budget_remaining)
            selected_indices = [idx for _, idx in uncertainties[:samples_to_query]]
            
            # Move samples from U to L
            for idx in sorted(selected_indices, reverse=True):
                L_X = torch.cat([L_X, U_X[idx:idx+1]], dim=0)
                L_y = torch.cat([L_y, U_y[idx:idx+1]], dim=0)
                
                # Remove from U
                mask = torch.ones(len(U_X), dtype=torch.bool)
                mask[idx] = False
                U_X = U_X[mask]
                U_y = U_y[mask]
            
            # Update budget
            budget_remaining -= samples_to_query
            query_iteration += 1
            
            # Reinitialize optimizer for updated labeled set
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                momentum=params.get('momentum', 0.0)
            )
        
        # Final training on complete labeled set
        remaining_epochs = epochs - total_epochs_used
        for epoch in range(remaining_epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(L_X)
            loss = criterion(outputs, L_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            losses.append(loss.item())
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())
                
                # Track R² at each epoch
                train_outputs = model(L_X)
                train_r2 = r2_score(L_y.numpy(), train_outputs.numpy())
                test_r2 = r2_score(y_test.numpy(), val_outputs.numpy())
                
                train_r2_curve.append(train_r2)
                test_r2_curve.append(test_r2)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(L_X)
            test_outputs = model(X_test)
            
            train_r2 = r2_score(L_y.numpy(), train_outputs.numpy())
            test_r2 = r2_score(y_test.numpy(), test_outputs.numpy())
        
        # Track pool-based learning statistics
        reduction_data = {
            'pool_based_queries': query_iteration,
            'initial_labeled_size': initial_labeled_size,
            'final_labeled_size': len(L_X),
            'original_training_size': len(X_train),
            'total_budget_used': total_budget - budget_remaining,
            'training_set_sizes_at_epochs': training_set_sizes_at_epochs,
            'final_reduction_percentage': ((len(X_train) - len(L_X)) / len(X_train)) * 100,
        }
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses, threshold=0.01)
        
        return model, losses, train_r2, test_r2, epochs_converged, reduction_data, val_losses, train_r2_curve, test_r2_curve
    
    def _train_ensemble_uncertainty_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                         X_test: torch.Tensor, y_test: torch.Tensor,
                                         params: Dict, epochs: int,
                                         n_ensemble: int = 3) -> Tuple:
        """
        Train ensemble of regression models using Pool-Based Active Learning.
        
        Ensemble approach for regression:
        1. Train multiple regression NNs with different random initializations
        2. At inference, average predictions across ensemble members
        3. Uncertainty = variance across ensemble predictions
        4. Select samples with highest prediction variance for labeling
        """
        
        # Pool-Based Active Learning Setup
        initial_labeled_size = max(10, len(X_train) // 10)
        total_budget = len(X_train) - initial_labeled_size
        
        # Initialize labeled set L (random sampling)
        indices = torch.arange(len(X_train))
        perm = torch.randperm(len(X_train))
        labeled_indices = perm[:initial_labeled_size]
        unlabeled_indices = perm[initial_labeled_size:]
        
        # Initialize L (labeled data) and U (unlabeled pool)
        L_X = X_train[labeled_indices].clone()
        L_y = y_train[labeled_indices].clone()
        
        U_X = X_train[unlabeled_indices].clone()
        U_y = y_train[unlabeled_indices].clone()
        
        # Initialize ensemble of models with different random seeds
        ensemble_models = []
        ensemble_optimizers = []
        
        for i in range(n_ensemble):
            torch.manual_seed(42 + i * 100)
            model = RegressionNet(
                input_size=params.get('input_size', X_train.shape[1]),
                hidden_size=params['hidden_size'],
                output_size=1
            )
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                momentum=params.get('momentum', 0.0)
            )
            ensemble_models.append(model)
            ensemble_optimizers.append(optimizer)
        
        criterion = nn.MSELoss()
        
        # Training tracking
        losses = []  # Average loss across ensemble
        val_losses = []
        train_r2_curve = []
        test_r2_curve = []
        pool_sizes = []
        labeled_sizes = []
        
        # Track training set sizes at specific epochs
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        training_set_sizes_at_epochs = {}
        
        # Pool-Based Active Learning Algorithm
        budget_remaining = total_budget
        query_interval = max(1, total_budget // 50)
        training_epochs_per_query = max(10, epochs // 50)
        
        query_iteration = 0
        total_epochs_used = 0
        
        while budget_remaining > 0 and len(U_X) > 0 and total_epochs_used < epochs:
            
            # Train all models in ensemble
            for epoch in range(training_epochs_per_query):
                if total_epochs_used >= epochs:
                    break
                
                batch_losses = []
                # Train each model in ensemble
                for model, optimizer in zip(ensemble_models, ensemble_optimizers):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(L_X)
                    loss = criterion(outputs, L_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                # Average loss across ensemble
                avg_loss = np.mean(batch_losses)
                losses.append(avg_loss)
                
                # Validation step
                val_batch_losses = []
                ensemble_train_preds = []
                ensemble_test_preds = []
                
                for model in ensemble_models:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test)
                        val_loss = criterion(val_outputs, y_test)
                        val_batch_losses.append(val_loss.item())
                        
                        train_outputs = model(L_X)
                        ensemble_train_preds.append(train_outputs)
                        ensemble_test_preds.append(val_outputs)
                
                val_losses.append(np.mean(val_batch_losses))
                
                # Calculate ensemble R² by averaging predictions
                if ensemble_train_preds and ensemble_test_preds:
                    avg_train_outputs = torch.mean(torch.stack(ensemble_train_preds), dim=0)
                    avg_test_outputs = torch.mean(torch.stack(ensemble_test_preds), dim=0)
                    
                    train_r2 = r2_score(L_y.numpy(), avg_train_outputs.numpy())
                    test_r2 = r2_score(y_test.numpy(), avg_test_outputs.numpy())
                    
                    train_r2_curve.append(train_r2)
                    test_r2_curve.append(test_r2)
                
                total_epochs_used += 1
                
                # Track pool statistics
                pool_sizes.append(len(U_X))
                labeled_sizes.append(len(L_X))
                
                # Track training set size at checkpoints
                if total_epochs_used in epoch_checkpoints:
                    training_set_sizes_at_epochs[total_epochs_used] = len(L_X)
            
            # If no more unlabeled data or budget exhausted, break
            if len(U_X) == 0 or budget_remaining <= 0:
                break
            
            # Compute ensemble uncertainty for all samples in U
            uncertainties = []
            
            for model in ensemble_models:
                model.eval()
            
            with torch.no_grad():
                for i in range(len(U_X)):
                    x_sample = U_X[i:i+1]
                    
                    # Get predictions from all ensemble members
                    ensemble_preds = []
                    for model in ensemble_models:
                        pred = model(x_sample)
                        ensemble_preds.append(pred.item())
                    
                    # Calculate uncertainty as variance across ensemble
                    pred_variance = np.var(ensemble_preds)
                    uncertainties.append((pred_variance, i))
            
            # Select highest uncertainty samples
            uncertainties.sort(reverse=True, key=lambda x: x[0])
            samples_to_query = min(query_interval, len(uncertainties), budget_remaining)
            selected_indices = [idx for _, idx in uncertainties[:samples_to_query]]
            
            # Move samples from U to L
            for idx in sorted(selected_indices, reverse=True):
                L_X = torch.cat([L_X, U_X[idx:idx+1]], dim=0)
                L_y = torch.cat([L_y, U_y[idx:idx+1]], dim=0)
                
                # Remove from U
                mask = torch.ones(len(U_X), dtype=torch.bool)
                mask[idx] = False
                U_X = U_X[mask]
                U_y = U_y[mask]
            
            # Update budget
            budget_remaining -= samples_to_query
            query_iteration += 1
            
            # Reinitialize optimizers
            ensemble_optimizers = []
            for model in ensemble_models:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    momentum=params.get('momentum', 0.0)
                )
                ensemble_optimizers.append(optimizer)
        
        # Final training on complete labeled set
        remaining_epochs = epochs - total_epochs_used
        for epoch in range(remaining_epochs):
            batch_losses = []
            for model, optimizer in zip(ensemble_models, ensemble_optimizers):
                model.train()
                optimizer.zero_grad()
                outputs = model(L_X)
                loss = criterion(outputs, L_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                batch_losses.append(loss.item())
            
            losses.append(np.mean(batch_losses))
            
            # Validation step
            val_batch_losses = []
            ensemble_train_preds = []
            ensemble_test_preds = []
            
            for model in ensemble_models:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test)
                    val_batch_losses.append(val_loss.item())
                    
                    train_outputs = model(L_X)
                    ensemble_train_preds.append(train_outputs)
                    ensemble_test_preds.append(val_outputs)
            
            val_losses.append(np.mean(val_batch_losses))
            
            # Calculate ensemble R²
            if ensemble_train_preds and ensemble_test_preds:
                avg_train_outputs = torch.mean(torch.stack(ensemble_train_preds), dim=0)
                avg_test_outputs = torch.mean(torch.stack(ensemble_test_preds), dim=0)
                
                train_r2 = r2_score(L_y.numpy(), avg_train_outputs.numpy())
                test_r2 = r2_score(y_test.numpy(), avg_test_outputs.numpy())
                
                train_r2_curve.append(train_r2)
                test_r2_curve.append(test_r2)
        
        # Final evaluation using ensemble averaging
        for model in ensemble_models:
            model.eval()
        
        with torch.no_grad():
            ensemble_train_outputs = []
            ensemble_test_outputs = []
            
            for model in ensemble_models:
                train_outputs = model(L_X)
                test_outputs = model(X_test)
                ensemble_train_outputs.append(train_outputs)
                ensemble_test_outputs.append(test_outputs)
            
            # Average predictions across ensemble
            avg_train_outputs = torch.mean(torch.stack(ensemble_train_outputs), dim=0)
            avg_test_outputs = torch.mean(torch.stack(ensemble_test_outputs), dim=0)
            
            train_r2 = r2_score(L_y.numpy(), avg_train_outputs.numpy())
            test_r2 = r2_score(y_test.numpy(), avg_test_outputs.numpy())
        
        # Track statistics
        reduction_data = {
            'pool_based_queries': query_iteration,
            'initial_labeled_size': initial_labeled_size,
            'final_labeled_size': len(L_X),
            'original_training_size': len(X_train),
            'total_budget_used': total_budget - budget_remaining,
            'training_set_sizes_at_epochs': training_set_sizes_at_epochs,
            'final_reduction_percentage': ((len(X_train) - len(L_X)) / len(X_train)) * 100,
            'ensemble_size': n_ensemble,
        }
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses, threshold=0.01)
        
        # Return first model as representative
        return ensemble_models[0], losses, train_r2, test_r2, epochs_converged, reduction_data, val_losses, train_r2_curve, test_r2_curve
