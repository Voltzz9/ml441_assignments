import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import time
from typing import Dict, List, Tuple, Optional, Callable
from .models import NeuralNet, RegressionNet
from .metrics import MetricsTracker, find_convergence_epoch, compute_classification_metrics


class UncertaintySampling:
    """Active Learning methods to sample for uncertainty using entropy-based sampling."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def entropy_based(self, prob_dist):
        """ 
        Returns the uncertainty score of a probability distribution using entropy.
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        Args:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        prob_dist = torch.clamp(prob_dist, min=epsilon)
        
        log_probs = prob_dist * torch.log2(prob_dist)  # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs)
    
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
        
        return normalized_entropy.item()
    
    def softmax(self, scores, base=math.e):
        """Returns softmax array for array of scores
        
        Converts a set of raw scores from a model (logits) into a 
        probability distribution via softmax.
            
        The probability distribution will be a set of real numbers
        such that each is in the range 0-1.0 and the sum is 1.0.
    
        Args:
            scores -- a pytorch tensor of any positive/negative real numbers.
            base -- the base for the exponential (default e)
        """
        exps = (base**scores.to(dtype=torch.float))  # exponential for each value in array
        sum_exps = torch.sum(exps)  # sum of all exponentials
        prob_dist = exps / sum_exps  # normalize exponentials 
        return prob_dist
    
    def get_uncertain_samples(self, model, X_data, y_data, method='entropy', 
                             selection_ratio=0.5, use_softmax=True):
        """Get samples via uncertainty sampling from data
    
        Args:
            model -- current Machine Learning model for this task
            X_data -- input features
            y_data -- target labels (for keeping track)
            method -- method for uncertainty sampling (default: 'entropy')
            selection_ratio -- ratio of samples to keep (0.5 = keep 50% most uncertain)
            use_softmax -- whether to apply softmax to model outputs
    
        Returns:
            Selected X and y data based on uncertainty
        """
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i in range(len(X_data)):
                x_sample = X_data[i:i+1]  # Get single sample with batch dimension
                
                # Get model predictions
                logits = model(x_sample)
                
                if use_softmax:
                    # Convert logits to probabilities
                    prob_dist = self.softmax(logits.squeeze())
                else:
                    # Assume model already outputs probabilities (sigmoid case)
                    prob_dist = logits.squeeze()
                    # Normalize to ensure it sums to 1
                    prob_dist = prob_dist / torch.sum(prob_dist)
                
                # Calculate uncertainty score
                if method == 'entropy':
                    uncertainty = self.entropy_based(prob_dist)
                # TODO add other types??
                else:
                    uncertainty = self.entropy_based(prob_dist)  # default to entropy
                
                uncertainties.append((uncertainty, i))
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(reverse=True, key=lambda x: x[0])
        
        # Select top uncertain samples
        n_select = max(1, int(len(uncertainties) * selection_ratio))
        selected_indices = [idx for _, idx in uncertainties[:n_select]]
        
        return X_data[selected_indices], y_data[selected_indices], selected_indices


class ModelEvaluator:
    """Comprehensive model evaluation with support for multiple learning strategies."""
    
    def __init__(self, convergence_threshold: float = 0.95):
        self.convergence_threshold = convergence_threshold
        self.metrics_tracker = MetricsTracker()
        self.metrics_tracker.convergence_threshold = convergence_threshold
    
    def evaluate_passive_learning(self, X: torch.Tensor, y: torch.Tensor, 
                                 best_params: Dict, n_trials: int = 50,
                                 use_cv: bool = True, cv_folds: int = 5,
                                 epochs: int = 1000, random_state: int = 42,
                                 model_class=None) -> Dict:
        """
        Evaluate passive learning with comprehensive metrics tracking.
        
        Args:
            X: Input features tensor
            y: Target labels tensor (one-hot encoded)
            best_params: Dictionary with best hyperparameters
            n_trials: Number of trials to run
            use_cv: Whether to use cross-validation
            cv_folds: Number of CV folds
            epochs: Maximum epochs per trial
            random_state: Base random state
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print(f"Starting {n_trials} trial evaluation...")
        print(f"Parameters: {best_params}")
        print(f"Cross-validation: {'Yes' if use_cv else 'No'} ({cv_folds} folds)")
        
        self.metrics_tracker.reset()
        
        # Convert one-hot to class indices for stratification
        y_indices = torch.argmax(y, dim=1)
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}", end="... ")
            
            start_time = time.time()
            
            if use_cv:
                # Cross-validation approach
                trial_results = self._run_cv_trial(X, y, y_indices, best_params, 
                                                 cv_folds, epochs, random_state + trial, model_class)
            else:
                # Simple train-test split approach
                trial_results = self._run_simple_trial(X, y, y_indices, best_params, 
                                                     epochs, random_state + trial, model_class)
            
            computation_time = time.time() - start_time
            
            # Add results to tracker
            self.metrics_tracker.add_trial_results(
                train_acc=trial_results['train_acc'],
                test_acc=trial_results['test_acc'],
                val_acc=trial_results.get('val_acc'),
                losses=trial_results['losses'],
                val_losses=trial_results.get('val_losses'),
                epochs_converged=trial_results['epochs_converged'],
                num_presentations=trial_results['num_presentations'],
                computation_time=computation_time,
                train_metrics=trial_results.get('train_metrics'),
                test_metrics=trial_results.get('test_metrics'),
                val_metrics=trial_results.get('val_metrics'),
                train_acc_curve=trial_results.get('train_acc_curve'),
                test_acc_curve=trial_results.get('test_acc_curve')
            )
            
            print(f"Test Acc: {trial_results['test_acc']:.4f}, Time: {computation_time:.2f}s")
        
        # Compute and return comprehensive statistics
        results = self.metrics_tracker.compute_statistics()
        results['learning_type'] = 'passive'
        results['best_params'] = best_params
        
        return results
    
    def _run_cv_trial(self, X: torch.Tensor, y: torch.Tensor, y_indices: torch.Tensor,
                     params: Dict, cv_folds: int, epochs: int, random_state: int,
                     model_class=None) -> Dict:
        """Run a single trial with cross-validation."""
        
        # First, create a holdout test set
        X_temp, X_test, y_temp, y_test, y_temp_idx, y_test_idx = train_test_split(
            X, y, y_indices, test_size=0.2, random_state=random_state, 
            stratify=y_indices
        )
        
        # Cross-validation on the remaining 80%
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_val_accs = []
        cv_train_accs = []
        
        cv_train_metrics = []
        cv_val_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_temp, y_temp_idx)):
            X_train = X_temp[train_idx]
            X_val = X_temp[val_idx]
            y_train = y_temp[train_idx]
            y_val = y_temp[val_idx]
            y_train_idx = y_temp_idx[train_idx]
            y_val_idx = y_temp_idx[val_idx]
            
            # Train model
            model, losses, train_acc, val_acc, _, train_metrics, val_metrics, val_losses, _, _ = self._train_model(
                X_train, y_train_idx, X_val, y_val_idx, params, epochs, model_class
            )
            
            cv_train_accs.append(train_acc)
            cv_val_accs.append(val_acc)
            cv_train_metrics.append(train_metrics)
            cv_val_metrics.append(val_metrics)
            # Note: We don't collect losses from CV folds, only from final training
        
        # Average CV results
        avg_train_acc = np.mean(cv_train_accs)
        avg_val_acc = np.mean(cv_val_accs)
        
        # Average CV metrics
        avg_train_metrics = self._average_metrics(cv_train_metrics)
        avg_val_metrics = self._average_metrics(cv_val_metrics)
        
        # Final test on holdout set using best parameters
        # Train on all temp data for final test
        final_model, final_losses, _, _, epochs_converged, _, test_metrics, final_val_losses, train_acc_curve, test_acc_curve = self._train_model(
            X_temp, y_temp_idx, X_test, y_test_idx, params, epochs, model_class
        )
        
        # Evaluate on test set
        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(X_test)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
        
        # Calculate pattern presentations
        num_presentations = len(X_temp) * epochs_converged
        
        return {
            'train_acc': avg_train_acc,
            'test_acc': test_acc,
            'val_acc': avg_val_acc,
            'losses': final_losses,
            'val_losses': final_val_losses,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'train_metrics': avg_train_metrics,
            'test_metrics': test_metrics,
            'val_metrics': avg_val_metrics,
            'train_acc_curve': train_acc_curve,
            'test_acc_curve': test_acc_curve
        }
    
    def _run_simple_trial(self, X: torch.Tensor, y: torch.Tensor, y_indices: torch.Tensor,
                         params: Dict, epochs: int, random_state: int, model_class=None) -> Dict:
        """Run a single trial with simple train-test split."""
        
        X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
            X, y, y_indices, test_size=0.2, random_state=random_state, 
            stratify=y_indices
        )
        
        # Train model
        model, losses, train_acc, test_acc, epochs_converged, train_metrics, test_metrics, val_losses, train_acc_curve, test_acc_curve = self._train_model(
            X_train, y_train_idx, X_test, y_test_idx, params, epochs, model_class
        )
        
        # Calculate pattern presentations
        num_presentations = len(X_train) * epochs_converged
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'losses': losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'train_acc_curve': train_acc_curve,
            'test_acc_curve': test_acc_curve
        }
    
    def _train_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                    X_test: torch.Tensor, y_test_idx: torch.Tensor,
                    params: Dict, epochs: int, model_class=None) -> Tuple:
        """Train a single model and return results."""
        
        # unpack input and output sizes from params (if not present, use default)
        input_size = params.get('input_size', 4)
        output_size = params.get('output_size', 3)
        
        # Default to NeuralNet if no model class specified
        if model_class is None:
            model_class = NeuralNet
        
        # Handle different model types
        if model_class == RegressionNet:
            # For regression, use raw target values (no one-hot encoding)
            y_train_target = y_train_idx.float().unsqueeze(1) if y_train_idx.dim() == 1 else y_train_idx.float()
            y_test_target = y_test_idx.float().unsqueeze(1) if y_test_idx.dim() == 1 else y_test_idx.float()
            
            # Create regression model
            model = model_class(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                output_size=1  # Single output for regression
            )
        else:
            # For classification, convert to one-hot encoding
            y_train_onehot = torch.zeros(len(y_train_idx), output_size)
            y_train_onehot[range(len(y_train_idx)), y_train_idx] = 1
            # Scale to [0.1, 0.9] like in active learning
            y_train_target = y_train_onehot * 0.8 + 0.1
            
            # Convert y_test_idx to one-hot for validation loss calculation
            y_test_onehot = torch.zeros(len(y_test_idx), output_size)
            y_test_onehot[range(len(y_test_idx)), y_test_idx] = 1
            # Scale to [0.1, 0.9] like training data
            y_test_target = y_test_onehot * 0.8 + 0.1
            
            # Create classification model
            model = model_class(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                output_size=output_size,
                use_mse=True
            )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)  # Default to 0.0 if not specified
        )
        
        # Training loop
        losses = []
        val_losses = []
        train_acc_curve = []
        test_acc_curve = []
        
        for epoch in range(epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train_target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test_target)
                val_losses.append(val_loss.item())
                
                # Track accuracy at each epoch - handle both regression and classification
                if model_class == RegressionNet:
                    # For regression, calculate R² score or MSE as "accuracy"
                    train_outputs = model(X_train)
                    train_mse = nn.MSELoss()(train_outputs, y_train_target)
                    test_mse = nn.MSELoss()(val_outputs, y_test_target)
                    
                    # Use negative MSE as "accuracy" (higher is better)
                    train_epoch_acc = -train_mse.item()
                    test_epoch_acc = -test_mse.item()
                else:
                    # For classification, calculate accuracy as usual
                    train_outputs = model(X_train)
                    train_pred = torch.argmax(train_outputs, dim=1)
                    test_pred = torch.argmax(val_outputs, dim=1)
                    
                    train_epoch_acc = accuracy_score(y_train_idx.numpy(), train_pred.numpy())
                    test_epoch_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
                
                train_acc_curve.append(train_epoch_acc)
                test_acc_curve.append(test_epoch_acc)
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            test_outputs = model(X_test)
            
            if model_class == RegressionNet:
                # For regression, use MSE-based metrics
                train_mse = nn.MSELoss()(train_outputs, y_train_target)
                test_mse = nn.MSELoss()(test_outputs, y_test_target)
                
                train_acc = -train_mse.item()  # Use negative MSE as "accuracy"
                test_acc = -test_mse.item()
                
                # Create dummy metrics for regression (since classification metrics don't apply)
                train_metrics = {'accuracy': train_acc, 'f1_score': train_acc, 'precision': train_acc, 'recall': train_acc}
                test_metrics = {'accuracy': test_acc, 'f1_score': test_acc, 'precision': test_acc, 'recall': test_acc}
            else:
                # For classification, use regular accuracy
                train_pred = torch.argmax(train_outputs, dim=1)
                test_pred = torch.argmax(test_outputs, dim=1)
                
                train_acc = accuracy_score(y_train_idx.numpy(), train_pred.numpy())
                test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
                
                # Compute comprehensive classification metrics
                train_metrics = compute_classification_metrics(y_train_idx.numpy(), train_pred.numpy())
                test_metrics = compute_classification_metrics(y_test_idx.numpy(), test_pred.numpy())
        
        return model, losses, train_acc, test_acc, epochs_converged, train_metrics, test_metrics, val_losses, train_acc_curve, test_acc_curve
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average classification metrics across folds."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key == 'confusion_matrix':
                # Average confusion matrices
                cms = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(cms, axis=0)
            else:
                # Average scalar metrics
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def print_report(self):
        """Print comprehensive evaluation report."""
        self.metrics_tracker.print_comprehensive_report()
    
    def get_results(self) -> Dict:
        """Get evaluation results."""
        return self.metrics_tracker.compute_statistics()


class ActiveLearningEvaluator(ModelEvaluator):
    """
    Extended evaluator for active learning strategies.
    
    Includes specialized plotting methods for analyzing active learning performance,
    particularly F1 score progression as the number of labeled instances increases
    during uncertainty sampling.
    
    Example usage for F1 vs instances plotting:
    
    ```python
    # Run uncertainty sampling evaluation
    evaluator = ActiveLearningEvaluator()
    results = evaluator.evaluate_active_learning(
        X, y, best_params, 'uncertainty_sampling', n_trials=10
    )
    
    # Plot F1 score vs number of instances (uses all trials)
    evaluator.plot_f1_vs_instances(results, 
        title="F1 Score vs Labeled Instances - Uncertainty Sampling",
        save_path="f1_vs_instances.png"
    )
    ```
    """
    
    def __init__(self, convergence_threshold: float = 0.95):
        super().__init__(convergence_threshold)
        # Add additional tracking for active learning metrics
        self.training_set_reductions = {}  # Store reduction percentages at different epochs
        self.f1_curves_data = []  # Store F1 vs instances data from all trials
        
    def evaluate_active_learning(self, X: torch.Tensor, y: torch.Tensor,
                                best_params: Dict, strategy: str,
                                n_trials: int = 50, use_cv: bool = True,
                                cv_folds: int = 5, epochs: int = 1000,
                                random_state: int = 12, 
                                track_generalization_per_presentation: bool = False,
                                model_class=None,
                                **strategy_kwargs) -> Dict:
        """
        Evaluate active learning strategies.
        
        Args:
            strategy: 'uncertainty_sampling', 'output_sensitivity', or 'ensemble_uncertainty'
            strategy_kwargs: Additional arguments for the active learning strategy
                - For 'ensemble_uncertainty': n_ensemble (int, default=3) 
        """
        print(f"Starting {n_trials} trial evaluation for {strategy} active learning...")
        print(f"Parameters: {best_params}")
        print(f"Cross-validation: {'Yes' if use_cv else 'No'} ({cv_folds} folds)")
        
        self.metrics_tracker.reset()
        self.training_set_reductions = {25: [], 100: [], 200: [], 300: [], 500: [], 1000: []}
        self.final_training_set_sizes = []
        self.original_training_set_sizes = []
        self.training_set_sizes_by_epoch = {epoch: [] for epoch in [25, 100, 200, 300, 500, 1000]}
        self.f1_curves_data = []  # Reset F1 curve data
        
        # Convert one-hot to class indices for stratification
        y_indices = torch.argmax(y, dim=1)
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            # Set random seed for reproducibility
            trial_seed = random_state + trial
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            
            start_time = time.time()
            
            if use_cv:
                trial_results = self._run_active_cv_trial(
                    X, y, y_indices, best_params, strategy, cv_folds, 
                    epochs, trial_seed, **strategy_kwargs
                )
            else:
                trial_results = self._run_active_simple_trial(
                    X, y, y_indices, best_params, strategy, 
                    epochs, trial_seed, **strategy_kwargs
                )
            
            computation_time = time.time() - start_time
            
            # Add results to tracker
            self.metrics_tracker.add_trial_results(
                train_acc=trial_results['train_acc'],
                test_acc=trial_results['test_acc'],
                losses=trial_results['losses'],
                val_losses=trial_results.get('val_losses'),
                epochs_converged=trial_results['epochs_converged'],
                num_presentations=trial_results['num_presentations'],
                computation_time=computation_time,
                val_acc=trial_results.get('val_acc', None),
                train_metrics=trial_results.get('train_metrics'),
                test_metrics=trial_results.get('test_metrics'),
                val_metrics=trial_results.get('val_metrics'),
                train_acc_curve=trial_results.get('train_acc_curve'),
                test_acc_curve=trial_results.get('test_acc_curve')
            )
            
            # Track training set size information
            self.final_training_set_sizes.append(trial_results.get('final_training_set_size', 0))
            self.original_training_set_sizes.append(trial_results.get('original_training_set_size', 0))
            
            # Store F1 curve data for uncertainty sampling
            if strategy == 'uncertainty_sampling':
                f1_curve_data = trial_results.get('training_set_reductions', {}).get('f1_scores_vs_instances', [])
                if f1_curve_data:
                    self.f1_curves_data.append(f1_curve_data)
            
            # Track training set sizes at specific epochs
            training_set_sizes_at_epochs = trial_results.get('training_set_sizes_at_epochs', {})
            for epoch in self.training_set_sizes_by_epoch.keys():
                if epoch in training_set_sizes_at_epochs:
                    self.training_set_sizes_by_epoch[epoch].append(training_set_sizes_at_epochs[epoch])
            
            # Track training set reductions (for backward compatibility)
            reduction_data = trial_results.get('training_set_reductions', {})
            if 'final_reduction_percentage' in reduction_data:
                # For now, store the final reduction at all epochs (will be updated with proper epoch tracking)
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
        
        # Add training set reduction statistics (backward compatibility)
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
    
    def plot_f1_vs_instances(self, results: Dict, title: str = "F1 Score vs Number of Labeled Instances", 
                            figsize: Tuple[int, int] = (10, 6), 
                            save_path: Optional[str] = None):
        """
        Plot F1 score vs number of labeled instances for uncertainty sampling.
        
        Args:
            results: Results dictionary from evaluate_active_learning() with uncertainty_sampling strategy
            title: Title for the plot
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Cannot plot F1 vs instances.")
            return
        
        # Check if this is uncertainty sampling results
        if 'active_uncertainty_sampling' not in results.get('learning_type', ''):
            print("This plotting method is specifically for uncertainty sampling results.")
            return
        
        # Check if we have stored F1 curves data
        if not self.f1_curves_data:
            print("No F1 vs instances data available. Make sure you ran uncertainty_sampling evaluation.")
            return
        
        # Use the multi-trial plotting method with stored data
        trial_results_list = []
        for f1_curve in self.f1_curves_data:
            trial_result = {'training_set_reductions': {'f1_scores_vs_instances': f1_curve}}
            trial_results_list.append(trial_result)
        
        self.plot_f1_vs_instances_multi_trial(
            trial_results_list, title=title, figsize=figsize, 
            save_path=save_path, show_individual=True, confidence_bands=True
        )
        
    def plot_f1_vs_instances_from_trial(self, trial_results: Dict, 
                                       title: str = "F1 Score vs Number of Labeled Instances",
                                       figsize: Tuple[int, int] = (10, 6),
                                       save_path: Optional[str] = None):
        """
        Plot F1 score vs number of labeled instances from a single trial.
        
        Args:
            trial_results: Results from a single uncertainty sampling trial
            title: Title for the plot
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Cannot plot F1 vs instances.")
            return
        
        # Extract F1 data from trial results
        f1_data = trial_results.get('training_set_reductions', {}).get('f1_scores_vs_instances', [])
        
        if not f1_data:
            print("No F1 vs instances data found in trial results.")
            print("Make sure you're using uncertainty_sampling strategy and the data was collected.")
            return
        
        # Extract instances and F1 scores
        instances = [point[0] for point in f1_data]
        f1_scores = [point[1] for point in f1_data]
        
        if len(instances) == 0:
            print("No F1 data points found.")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(instances, f1_scores, 'b-o', linewidth=2, markersize=6, 
                label='F1 Score', markerfacecolor='white', markeredgecolor='blue')
        
        ax.set_xlabel('Number of Labeled Instances')
        ax.set_ylabel('F1 Score (Macro Average)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis to show full range from 0 to 1
        ax.set_ylim(0, 1)
        
        # Add some statistics as text
        max_f1 = max(f1_scores)
        final_f1 = f1_scores[-1]
        initial_instances = instances[0]
        final_instances = instances[-1]
        
        textstr = f'Initial: {initial_instances} instances, F1={f1_scores[0]:.3f}\n'
        textstr += f'Final: {final_instances} instances, F1={final_f1:.3f}\n'
        textstr += f'Max F1: {max_f1:.3f}\n'
        textstr += f'Improvement: {final_f1 - f1_scores[0]:.3f}'
        
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"F1 vs instances plot saved to {save_path}")
        
        plt.show()
    
    def plot_f1_vs_instances_multi_trial(self, all_trial_results: List[Dict],
                                        title: str = "F1 Score vs Number of Labeled Instances (Multi-Trial)",
                                        figsize: Tuple[int, int] = (12, 8),
                                        save_path: Optional[str] = None,
                                        show_individual: bool = False,
                                        confidence_bands: bool = True):
        """
        Plot F1 score vs number of labeled instances across multiple trials with confidence bands.
        
        Args:
            all_trial_results: List of results from multiple uncertainty sampling trials
            title: Title for the plot
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
            show_individual: Whether to show individual trial curves
            confidence_bands: Whether to show confidence bands around mean
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Cannot plot F1 vs instances.")
            return
        
        # Extract F1 data from all trials
        all_curves = []
        for trial in all_trial_results:
            f1_data = trial.get('training_set_reductions', {}).get('f1_scores_vs_instances', [])
            if f1_data:
                instances = [point[0] for point in f1_data]
                f1_scores = [point[1] for point in f1_data]
                all_curves.append((instances, f1_scores))
        
        if not all_curves:
            print("No F1 vs instances data found in any trial results.")
            return
        
        # Find common instance points (interpolate if necessary)
        all_instances = sorted(set().union(*[curve[0] for curve in all_curves]))
        
        # Interpolate F1 scores for common instance points
        from scipy.interpolate import interp1d
        interpolated_curves = []
        
        for instances, f1_scores in all_curves:
            if len(instances) > 1:  # Need at least 2 points for interpolation
                # Create interpolation function
                f = interp1d(instances, f1_scores, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                # Interpolate at common instance points
                interp_f1 = f(all_instances)
                interpolated_curves.append(interp_f1)
        
        if not interpolated_curves:
            print("Not enough data points for interpolation.")
            return
        
        # Convert to numpy array for easier statistics
        curves_array = np.array(interpolated_curves)
        mean_f1 = np.mean(curves_array, axis=0)
        std_f1 = np.std(curves_array, axis=0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Show individual trials if requested
        if show_individual:
            for i, f1_curve in enumerate(interpolated_curves):
                ax.plot(all_instances, f1_curve, 'lightgray', alpha=0.5, linewidth=1)
        
        # Plot mean curve
        ax.plot(all_instances, mean_f1, 'b-o', linewidth=3, markersize=6,
                label=f'Mean F1 Score (n={len(interpolated_curves)} trials)',
                markerfacecolor='white', markeredgecolor='blue')
        
        # Add confidence bands
        if confidence_bands:
            ax.fill_between(all_instances, mean_f1 - std_f1, mean_f1 + std_f1,
                           alpha=0.2, color='blue', label='±1 Standard Deviation')
        
        ax.set_xlabel('Number of Labeled Instances')
        ax.set_ylabel('F1 Score (Macro Average)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis to show full range
        ax.set_ylim(0, 1)
        
        # Add statistics
        initial_f1 = mean_f1[0]
        final_f1 = mean_f1[-1]
        max_f1 = max(mean_f1)
        initial_instances = all_instances[0]
        final_instances = all_instances[-1]
        
        textstr = f'Trials: {len(interpolated_curves)}\n'
        textstr += f'Initial: {initial_instances} instances, F1={initial_f1:.3f}±{std_f1[0]:.3f}\n'
        textstr += f'Final: {final_instances} instances, F1={final_f1:.3f}±{std_f1[-1]:.3f}\n'
        textstr += f'Max Mean F1: {max_f1:.3f}\n'
        textstr += f'Mean Improvement: {final_f1 - initial_f1:.3f}'
        
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-trial F1 vs instances plot saved to {save_path}")
        
        plt.show()
    
    def _run_active_cv_trial(self, X: torch.Tensor, y: torch.Tensor, y_indices: torch.Tensor,
                            params: Dict, strategy: str, cv_folds: int, epochs: int, 
                            random_state: int, **strategy_kwargs) -> Dict:
        """Run a single active learning trial with cross-validation."""
        
        # First, create a holdout test set
        X_temp, X_test, y_temp, y_test, y_temp_idx, y_test_idx = train_test_split(
            X, y, y_indices, test_size=0.2, random_state=random_state,
            stratify=y_indices
        )
        
        # Cross-validation on the remaining 80%
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_val_accs = []
        cv_train_accs = []
        all_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_temp, y_temp_idx)):
            X_train_fold = X_temp[train_idx]
            X_val_fold = X_temp[val_idx]
            y_train_fold_idx = y_temp_idx[train_idx]
            y_val_fold_idx = y_temp_idx[val_idx]
            
            model, losses, train_acc, val_acc, epochs_converged, reduction_data, val_losses, _, _ = self._train_active_model(
                X_train_fold, y_train_fold_idx, X_val_fold, y_val_fold_idx,
                params, strategy, epochs, **strategy_kwargs
            )
            
            cv_train_accs.append(train_acc)
            cv_val_accs.append(val_acc)
            all_losses.extend(losses)
        
        # Average CV results
        avg_train_acc = np.mean(cv_train_accs)
        avg_val_acc = np.mean(cv_val_accs)
        
        # Final test on holdout set
        final_model, final_losses, _, _, epochs_converged, final_reduction_data, final_val_losses, _, _ = self._train_active_model(
            X_temp, y_temp_idx, X_test, y_test_idx, params, strategy, epochs, **strategy_kwargs
        )
        
        # Evaluate on test set
        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(X_test)
            test_pred = torch.argmax(test_outputs, dim=1)
            final_test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
        
        num_presentations = len(X_temp) * epochs_converged
        
        return {
            'train_acc': avg_train_acc,
            'test_acc': final_test_acc,
            'val_acc': avg_val_acc,
            'losses': final_losses,
            'val_losses': final_val_losses,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'training_set_reductions': final_reduction_data,
            'final_training_set_size': final_reduction_data.get('final_labeled_size', len(X_temp)),
            'original_training_set_size': final_reduction_data.get('original_training_size', len(X_temp)),
            'training_set_sizes_at_epochs': final_reduction_data.get('training_set_sizes_at_epochs', {})
        }
    
    def _run_active_simple_trial(self, X: torch.Tensor, y: torch.Tensor, y_indices: torch.Tensor,
                                params: Dict, strategy: str, epochs: int, 
                                random_state: int, **strategy_kwargs) -> Dict:
        """Run a single active learning trial with simple train-test split."""
        
        X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
            X, y, y_indices, test_size=0.2, random_state=random_state,
            stratify=y_indices
        )
        
        # Train model with active learning
        model, losses, train_acc, test_acc, epochs_converged, reduction_data, val_losses, train_acc_curve, test_acc_curve = self._train_active_model(
            X_train, y_train_idx, X_test, y_test_idx, params, strategy, epochs, **strategy_kwargs
        )
        
        num_presentations = len(X_train) * epochs_converged
        
        # if Strategy is output_sensitivity
        if strategy == 'output_sensitivity':
            return {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'losses': losses,
                'val_losses': val_losses,
                'epochs_converged': epochs_converged,
                'num_presentations': num_presentations,
                'training_set_reductions': reduction_data,
                'final_training_set_size': reduction_data.get('final_labeled_size', len(X_train)),
                'original_training_set_size': reduction_data.get('original_training_size', len(X_train)),
                'training_set_sizes_at_epochs': reduction_data.get('training_set_sizes_at_epochs', {}),
                'train_acc_curve': train_acc_curve,
                'test_acc_curve': test_acc_curve
            }
        else:
            return {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'losses': losses,
                'val_losses': val_losses,
                'epochs_converged': epochs_converged,
                'num_presentations': num_presentations,
                'training_set_reductions': reduction_data,
                'final_training_set_size': reduction_data.get('final_labeled_size', len(X_train)),
                'original_training_set_size': reduction_data.get('original_training_size', len(X_train)),
                'training_set_sizes_at_epochs': reduction_data.get('training_set_sizes_at_epochs', {}),
                'train_acc_curve': train_acc_curve,
                'test_acc_curve': test_acc_curve
            }
    
    def _train_active_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                           X_test: torch.Tensor, y_test_idx: torch.Tensor,
                           params: Dict, strategy: str, epochs: int, model_class=None, **strategy_kwargs) -> Tuple:
        """Train a single model using active learning strategy."""
        
        if strategy == 'output_sensitivity':
            return self._train_output_sensitivity_model(
                X_train, y_train_idx, X_test, y_test_idx, params, epochs, model_class, **strategy_kwargs
            )
        elif strategy == 'uncertainty_sampling':
            return self._train_uncertainty_sampling_model(
                X_train, y_train_idx, X_test, y_test_idx, params, epochs, model_class, **strategy_kwargs
            )
        elif strategy == 'ensemble_uncertainty':
            return self._train_ensemble_uncertainty_model(
                X_train, y_train_idx, X_test, y_test_idx, params, epochs, model_class, **strategy_kwargs
            )
        else:
            raise ValueError(f"Unknown active learning strategy: {strategy}")
    
    def _train_output_sensitivity_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                                       X_test: torch.Tensor, y_test_idx: torch.Tensor,
                                       params: Dict, epochs: int, model_class=None,
                                       alpha: float = 0.9,
                                       selection_interval: int = 1) -> Tuple:
        """
        Train model using output sensitivity active learning (SASLA approach).
        
        This implements the true SASLA algorithm from the paper:
        - Computes exact output sensitivity using derivatives (Equation 4)
        - Calculates pattern informativeness (Equations 1-2)
        - Selects patterns with informativeness > (1-α) * average_informativeness
        - Uses α = 0.9 as selection constant (conservative approach)
        """
        
        # Create model - use model_class if provided, otherwise default to NeuralNet
        if model_class is None:
            model_class = NeuralNet
            
        if model_class == RegressionNet:
            # For regression
            y_train_target = y_train_idx.float().unsqueeze(1) if y_train_idx.dim() == 1 else y_train_idx.float()
            model = model_class(
                input_size=params.get('input_size', 4),
                hidden_size=params['hidden_size'],
                output_size=1
            )
        else:
            # For classification - convert to one-hot and scale
            y_train_onehot = torch.zeros(len(y_train_idx), params.get('output_size', 3))
            y_train_onehot[range(len(y_train_idx)), y_train_idx] = 1
            # Scale to [0.1, 0.9] like in other active learning methods
            y_train_target = y_train_onehot * 0.8 + 0.1
            
            model = model_class(
                input_size=params.get('input_size', 4),
                hidden_size=params['hidden_size'],
                output_size=params.get('output_size', 3),
                use_mse=True
            )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)
        )
        
        # Initialize training data (make copies to avoid modifying originals)
        current_X_train = X_train.clone()
        current_y_train_idx = y_train_idx.clone()
        current_y_train_onehot = y_train_onehot.clone()
        original_train_size = len(X_train)
        
        # Convert y_test_idx to one-hot for validation loss calculation
        y_test_onehot = torch.zeros(len(y_test_idx), params.get('output_size', 3))
        y_test_onehot[range(len(y_test_idx)), y_test_idx] = 1
        y_test_onehot = y_test_onehot * 0.8 + 0.1
        
        # Track training set reductions at specific epochs
        reduction_data = {}
        reduction_data_actual_size = {}
        
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        
        losses = []
        val_losses = []
        train_acc_curve = []
        test_acc_curve = []
        
        for epoch in range(epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(current_X_train)
            loss = criterion(outputs, current_y_train_onehot)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Validation step and accuracy tracking
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test_onehot)
                val_losses.append(val_loss.item())
                
                # Track accuracy at each epoch
                train_outputs = model(current_X_train)
                train_pred = torch.argmax(train_outputs, dim=1)
                test_pred = torch.argmax(val_outputs, dim=1)
                
                train_epoch_acc = accuracy_score(current_y_train_idx.numpy(), train_pred.numpy())
                test_epoch_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
                
                train_acc_curve.append(train_epoch_acc)
                test_acc_curve.append(test_epoch_acc)
            
            # Apply SASLA pattern selection at specified intervals
            if epoch > 0 and epoch % selection_interval == 0:
                current_X_train, current_y_train_idx = self._apply_sasla_selection(
                    model, current_X_train, current_y_train_idx, alpha
                )
                # Update one-hot encoded labels to match selected patterns
                current_y_train_onehot = torch.zeros(len(current_y_train_idx), params.get('output_size', 3))
                current_y_train_onehot[range(len(current_y_train_idx)), current_y_train_idx] = 1
                current_y_train_onehot = current_y_train_onehot * 0.8 + 0.1
            
            # Record training set size at checkpoints
            if epoch + 1 in epoch_checkpoints:
                current_size = len(current_X_train)
                reduction_percentage = ((original_train_size - current_size) / original_train_size) * 100
                reduction_data[epoch + 1] = reduction_percentage
                reduction_data_actual_size[epoch + 1] = current_size


        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(current_X_train)
            test_outputs = model(X_test)
            
            train_pred = torch.argmax(train_outputs, dim=1)
            test_pred = torch.argmax(test_outputs, dim=1)
            
            train_acc = accuracy_score(current_y_train_idx.numpy(), train_pred.numpy())
            test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
            
        reduction_data_result = {
            'training_set_sizes_at_epochs': reduction_data_actual_size,
        }
        
        return model, losses, train_acc, test_acc, epochs_converged, reduction_data_result, val_losses, train_acc_curve, test_acc_curve
    
    def _apply_sasla_selection(self, model: nn.Module, X_train: torch.Tensor, 
                              y_train_idx: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SASLA pattern selection based on output sensitivity analysis.
        
        This implements the true SASLA algorithm from the paper using vectorized operations:
        1. Compute exact output sensitivity using derivatives (Equation 4)
        2. Calculate pattern informativeness (Equations 1-2) 
        3. Select patterns with informativeness > (1-α) * average_informativeness
        
        Args:
            model: Neural network model
            X_train: Training input patterns
            y_train_idx: Training target indices  
            alpha: Selection constant (typically 0.9)
        """
        
        model.eval()
        
        # Get model parameters for sensitivity calculation
        # Assuming a simple 2-layer network: input -> hidden -> output
        # Extract weights and biases
        params = list(model.parameters())
        if len(params) < 4:
            # Not enough layers for SASLA, return unchanged
            return X_train, y_train_idx
            
        # Weights: input->hidden (v), hidden->output (w)
        v_weights = params[0]  # Shape: [hidden_size, input_size]
        v_bias = params[1]     # Shape: [hidden_size]
        w_weights = params[2]  # Shape: [output_size, hidden_size] 
        w_bias = params[3]     # Shape: [output_size]
        
        # VECTORIZED COMPUTATION FOR ALL PATTERNS AT ONCE
        with torch.no_grad():
            # Forward pass for all patterns simultaneously
            # X_train shape: [n_samples, input_size]
            hidden_pre = torch.matmul(X_train, v_weights.T) + v_bias  # [n_samples, hidden_size]
            hidden_post = torch.sigmoid(hidden_pre)  # y_j in paper, shape: [n_samples, hidden_size]
            
            output_pre = torch.matmul(hidden_post, w_weights.T) + w_bias  # [n_samples, output_size]
            output_probs = torch.sigmoid(output_pre)  # o_k in paper, shape: [n_samples, output_size]
            
            n_samples, n_hidden = hidden_post.shape
            n_outputs = output_probs.shape[1]
            
            # Compute weighted inputs v_ji for all patterns and hidden units
            # v_ji = Σ(v_weights[j, :] * X_train[i, :]) for each sample i and hidden unit j
            v_ji = torch.matmul(X_train, v_weights.T)  # [n_samples, n_hidden]
            
            # Compute sensitivity terms for all combinations
            # For each sample i, output k, hidden unit j:
            # term_ijk = w_kj * (1 - y_j) * y_j * v_ji
            
            # Expand tensors for broadcasting
            # hidden_post: [n_samples, n_hidden] -> [n_samples, 1, n_hidden]
            # w_weights: [n_outputs, n_hidden] -> [1, n_outputs, n_hidden]
            # v_ji: [n_samples, n_hidden] -> [n_samples, 1, n_hidden]
            
            hidden_expanded = hidden_post.unsqueeze(1)  # [n_samples, 1, n_hidden]
            w_expanded = w_weights.unsqueeze(0)         # [1, n_outputs, n_hidden]
            v_ji_expanded = v_ji.unsqueeze(1)           # [n_samples, 1, n_hidden]
            
            # Compute sensitivity terms for all (sample, output, hidden) combinations
            # Shape: [n_samples, n_outputs, n_hidden]
            sensitivity_terms = w_expanded * (1 - hidden_expanded) * hidden_expanded * v_ji_expanded
            
            # Sum over hidden units for each (sample, output) pair
            # Shape: [n_samples, n_outputs]
            sensitivity_sums = torch.sum(sensitivity_terms, dim=2)
            
            # Compute complete sensitivity: (1 - o_k) * o_k * sensitivity_sum
            # output_probs shape: [n_samples, n_outputs]
            all_sensitivities = (1 - output_probs) * output_probs * sensitivity_sums
            
            # Compute informativeness for each pattern
            # Sum-norm: sum of absolute sensitivities across all outputs
            sum_norms = torch.sum(torch.abs(all_sensitivities), dim=1)  # [n_samples]
            
            # Max-norm: maximum absolute sensitivity across outputs
            max_norms = torch.max(torch.abs(all_sensitivities), dim=1)[0]  # [n_samples]
            
            # Pattern informativeness (Equation 2): max of sum-norm and max-norm
            informativeness_scores = torch.maximum(sum_norms, max_norms)  # [n_samples]
        
        # Calculate selection threshold: (1-α) * average_informativeness
        avg_informativeness = torch.mean(informativeness_scores)
        selection_threshold = (1 - alpha) * avg_informativeness
        
        # Select patterns with informativeness above threshold
        selected_indices = informativeness_scores >= selection_threshold
        
        # Ensure we keep at least one sample per class
        for class_label in torch.unique(y_train_idx):
            class_mask = (y_train_idx == class_label)
            class_selected = selected_indices & class_mask
            
            if not torch.any(class_selected):
                # If no samples selected for this class, keep the most informative one
                class_informativeness = informativeness_scores[class_mask]
                best_idx = torch.argmax(class_informativeness)
                class_indices = torch.where(class_mask)[0]
                selected_indices[class_indices[best_idx]] = True
        
        # Return selected patterns
        return X_train[selected_indices], y_train_idx[selected_indices]
    
    def _train_uncertainty_sampling_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                                         X_test: torch.Tensor, y_test_idx: torch.Tensor,
                                         params: Dict, epochs: int,
                                         uncertainty_method: str = 'entropy') -> Tuple:
        """
        Train model using Pool-Based Active Learning with Uncertainty Sampling.
        
        Implementation of Algorithm 1 Pool-Based Active Learning:
        1. Start with small labeled set L and large unlabeled pool U
        2. For each query iteration:
           - Compute utility (uncertainty) for all samples in U
           - Select highest utility sample x* and query its label
           - Move x* from U to L
           - Retrain model on updated L
        3. Repeat until budget is exhausted
        """
        # Initialize uncertainty sampling
        uncertainty_sampler = UncertaintySampling(verbose=False)
        
        # Convert y_train_idx to one-hot for MSE loss
        y_train_onehot = torch.zeros(len(y_train_idx), params.get('output_size', 3))
        y_train_onehot[range(len(y_train_idx)), y_train_idx] = 1
        # Scale to [0.1, 0.9] like in your passive learning
        y_train_onehot = y_train_onehot * 0.8 + 0.1
        
        # Pool-Based Active Learning Setup
        # L: labeled data (starts small)
        # U: unlabeled data pool (starts with most data)
        initial_labeled_size = max(10, len(X_train) // 10)  # Start with 10% or minimum 10 samples
        total_budget = len(X_train) - initial_labeled_size  # Budget = remaining samples
        
        # Initialize labeled set L (stratified sampling to ensure class balance)
        from sklearn.model_selection import train_test_split
        
        # Create initial labeled set with stratified sampling
        indices = torch.arange(len(X_train))
        labeled_indices, unlabeled_indices = train_test_split(
            indices.numpy(), 
            test_size=total_budget, 
            random_state=42,
            stratify=y_train_idx.numpy()
        )
        
        # Convert back to tensors
        labeled_indices = torch.tensor(labeled_indices)
        unlabeled_indices = torch.tensor(unlabeled_indices)
        
        # Initialize L (labeled data) and U (unlabeled pool)
        L_X = X_train[labeled_indices].clone()
        L_y = y_train_onehot[labeled_indices].clone()
        L_y_idx = y_train_idx[labeled_indices].clone()
        
        U_X = X_train[unlabeled_indices].clone()
        U_y = y_train_onehot[unlabeled_indices].clone()
        U_y_idx = y_train_idx[unlabeled_indices].clone()
        
        # Initialize model
        model = NeuralNet(
            input_size=params.get('input_size', 4),
            hidden_size=params['hidden_size'],
            output_size=params.get('output_size', 3),
            use_mse=True
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params['momentum']
        )
        
        # Convert y_test_idx to one-hot for validation loss calculation
        y_test_onehot = torch.zeros(len(y_test_idx), params.get('output_size', 3))
        y_test_onehot[range(len(y_test_idx)), y_test_idx] = 1
        y_test_onehot = y_test_onehot * 0.8 + 0.1
        
        # Training tracking
        losses = []
        val_losses = []
        train_acc_curve = []
        test_acc_curve = []
        pool_sizes = []
        labeled_sizes = []
        
        # Track F1 scores vs number of labeled instances for plotting
        f1_scores_vs_instances = []  # List of (num_instances, f1_score) tuples
        
        # Track training set sizes at specific epochs for comparison
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        training_set_sizes_at_epochs = {}
        
        # Pool-Based Active Learning Algorithm
        budget_remaining = total_budget
        query_interval = max(1, total_budget // 50)  # Make ~50 queries total
        training_epochs_per_query = max(10, epochs // 50)  # Training epochs between queries
        
        query_iteration = 0
        total_epochs_used = 0
        
        # Evaluate initial F1 score with initial labeled set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_pred = torch.argmax(test_outputs, dim=1)
            from sklearn.metrics import f1_score
            initial_f1 = f1_score(y_test_idx.numpy(), test_pred.numpy(), average='macro', zero_division=0)
            f1_scores_vs_instances.append((len(L_X), initial_f1))
        
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
                optimizer.step()
                
                losses.append(loss.item())
                
                # Validation step and accuracy tracking
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test_onehot)
                    val_losses.append(val_loss.item())
                    
                    # Track accuracy at each epoch
                    train_outputs = model(L_X)
                    train_pred = torch.argmax(train_outputs, dim=1)
                    test_pred = torch.argmax(val_outputs, dim=1)
                    
                    train_epoch_acc = accuracy_score(L_y_idx.numpy(), train_pred.numpy())
                    test_epoch_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
                    
                    train_acc_curve.append(train_epoch_acc)
                    test_acc_curve.append(test_epoch_acc)
                
                total_epochs_used += 1
                
                # Track pool statistics
                pool_sizes.append(len(U_X))
                labeled_sizes.append(len(L_X))
                
                # Track training set size at specific epoch checkpoints
                if total_epochs_used in epoch_checkpoints:
                    training_set_sizes_at_epochs[total_epochs_used] = len(L_X)
            
            # If no more unlabeled data or budget exhausted, break
            if len(U_X) == 0 or budget_remaining <= 0:
                break
                
            # Compute utility (uncertainty) for all samples in U
            model.eval()
            uncertainties = []
            
            with torch.no_grad():
                for i in range(len(U_X)):
                    x_sample = U_X[i:i+1]  # Single sample
                    output = model(x_sample)
                    
                    # Convert output to probability distribution
                    if uncertainty_method == 'entropy':
                        # Apply softmax to get probabilities
                        prob_dist = uncertainty_sampler.softmax(output.squeeze())
                        utility = uncertainty_sampler.entropy_based(prob_dist)
                    else:
                        # Default to entropy
                        prob_dist = uncertainty_sampler.softmax(output.squeeze())
                        utility = uncertainty_sampler.entropy_based(prob_dist)
                    
                    uncertainties.append((utility, i))
            
            # Pick highest utility samples (query multiple samples at once for efficiency)
            uncertainties.sort(reverse=True, key=lambda x: x[0])  # Sort by utility (highest first)
            samples_to_query = min(query_interval, len(uncertainties), budget_remaining)
            
            # Query labels for highest utility samples
            selected_indices = [idx for _, idx in uncertainties[:samples_to_query]]
            
            # Move samples from U to L (simulate querying labels)
            for idx in sorted(selected_indices, reverse=True):  # Reverse order to maintain indices
                # Move sample from U to L
                L_X = torch.cat([L_X, U_X[idx:idx+1]], dim=0)
                L_y = torch.cat([L_y, U_y[idx:idx+1]], dim=0)
                L_y_idx = torch.cat([L_y_idx, U_y_idx[idx:idx+1]], dim=0)
                
                # Remove from U
                mask = torch.ones(len(U_X), dtype=torch.bool)
                mask[idx] = False
                U_X = U_X[mask]
                U_y = U_y[mask]
                U_y_idx = U_y_idx[mask]
            
            # Update budget
            budget_remaining -= samples_to_query
            query_iteration += 1
            
            # Evaluate F1 score on test set after adding new samples
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_pred = torch.argmax(test_outputs, dim=1)
                # Compute F1 score
                from sklearn.metrics import f1_score
                current_f1 = f1_score(y_test_idx.numpy(), test_pred.numpy(), average='macro', zero_division=0)
                f1_scores_vs_instances.append((len(L_X), current_f1))
            
            # Reinitialize optimizer for updated labeled set
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                momentum=params['momentum']
            )
            
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Query {query_iteration}: Added {samples_to_query} samples, "
                      f"L size: {len(L_X)}, U size: {len(U_X)}, Budget remaining: {budget_remaining}")
        
        # Final training on complete labeled set
        remaining_epochs = epochs - total_epochs_used
        for epoch in range(remaining_epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(L_X)
            loss = criterion(outputs, L_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Validation step and accuracy tracking
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test_onehot)
                val_losses.append(val_loss.item())
                
                # Track accuracy at each epoch  
                train_outputs = model(L_X)
                train_pred = torch.argmax(train_outputs, dim=1)
                test_pred = torch.argmax(val_outputs, dim=1)
                
                train_epoch_acc = accuracy_score(L_y_idx.numpy(), train_pred.numpy())
                test_epoch_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
                
                train_acc_curve.append(train_epoch_acc)
                test_acc_curve.append(test_epoch_acc)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy (on final labeled set L)
            train_outputs = model(L_X)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_acc = accuracy_score(L_y_idx.numpy(), train_pred.numpy())
            
            # Test accuracy (on full test set)
            y_test_onehot = torch.zeros(len(y_test_idx), params.get('output_size', 3))
            y_test_onehot[range(len(y_test_idx)), y_test_idx] = 1
            y_test_onehot = y_test_onehot * 0.8 + 0.1
            
            test_outputs = model(X_test)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
        
        # Track pool-based learning statistics
        reduction_data = {
            'pool_based_queries': query_iteration,
            'initial_labeled_size': initial_labeled_size,
            'final_labeled_size': len(L_X),
            'original_training_size': len(X_train),
            'total_budget_used': total_budget - budget_remaining,
            'avg_pool_size': np.mean(pool_sizes) if pool_sizes else 0,
            'avg_labeled_size': np.mean(labeled_sizes) if labeled_sizes else initial_labeled_size,
            'training_set_sizes_at_epochs': training_set_sizes_at_epochs,
            'final_reduction_percentage': ((len(X_train) - len(L_X)) / len(X_train)) * 100,
            'f1_scores_vs_instances': f1_scores_vs_instances,  # Add F1 score tracking
            'query_history': {
                'pool_sizes': pool_sizes,
                'labeled_sizes': labeled_sizes,
                'total_epochs': total_epochs_used
            }
        }
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses, threshold=0.01)
        
        return model, losses, train_acc, test_acc, epochs_converged, reduction_data, val_losses, train_acc_curve, test_acc_curve

    def _train_ensemble_uncertainty_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                                         X_test: torch.Tensor, y_test_idx: torch.Tensor,
                                         params: Dict, epochs: int,
                                         n_ensemble: int = 3) -> Tuple:
        """
        Train ensemble of models using Pool-Based Active Learning with Uncertainty Sampling.
        
        Ensemble approach:
        1. Train multiple NNs with different random initializations
        2. At inference, average predictions across ensemble members
        3. Uncertainty = variance across ensemble predictions
        4. Select samples with highest prediction variance for labeling
        
        Args:
            n_ensemble: Number of models in ensemble (kept small for efficiency)
        """
        # Convert y_train_idx to one-hot for MSE loss
        y_train_onehot = torch.zeros(len(y_train_idx), params.get('output_size', 3))
        y_train_onehot[range(len(y_train_idx)), y_train_idx] = 1
        y_train_onehot = y_train_onehot * 0.8 + 0.1
        
        # Pool-Based Active Learning Setup
        initial_labeled_size = max(10, len(X_train) // 10)
        total_budget = len(X_train) - initial_labeled_size
        
        # Initialize labeled set L (stratified sampling)
        from sklearn.model_selection import train_test_split
        
        indices = torch.arange(len(X_train))
        labeled_indices, unlabeled_indices = train_test_split(
            indices.numpy(), 
            test_size=total_budget, 
            random_state=42,
            stratify=y_train_idx.numpy()
        )
        
        labeled_indices = torch.tensor(labeled_indices)
        unlabeled_indices = torch.tensor(unlabeled_indices)
        
        # Initialize L (labeled data) and U (unlabeled pool)
        L_X = X_train[labeled_indices].clone()
        L_y = y_train_onehot[labeled_indices].clone()
        L_y_idx = y_train_idx[labeled_indices].clone()
        
        U_X = X_train[unlabeled_indices].clone()
        U_y = y_train_onehot[unlabeled_indices].clone()
        U_y_idx = y_train_idx[unlabeled_indices].clone()
        
        # Initialize ensemble of models with different random seeds
        ensemble_models = []
        ensemble_optimizers = []
        
        for i in range(n_ensemble):
            # Set different random seed for each model initialization
            torch.manual_seed(42 + i * 100)
            model = NeuralNet(
                input_size=params.get('input_size', 4),
                hidden_size=params['hidden_size'],
                output_size=params.get('output_size', 3),
                use_mse=True
            )
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                momentum=params['momentum']
            )
            ensemble_models.append(model)
            ensemble_optimizers.append(optimizer)
        
        criterion = nn.MSELoss()
        
        # Convert y_test_idx to one-hot for validation loss calculation
        y_test_onehot = torch.zeros(len(y_test_idx), params.get('output_size', 3))
        y_test_onehot[range(len(y_test_idx)), y_test_idx] = 1
        y_test_onehot = y_test_onehot * 0.8 + 0.1
        
        # Training tracking
        losses = []  # Average loss across ensemble
        val_losses = []  # Average validation loss across ensemble
        train_acc_curve = []
        test_acc_curve = []
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
            
            # Train all models in ensemble on current labeled set L
            epoch_losses = []
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
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                # Average loss across ensemble
                avg_loss = np.mean(batch_losses)
                losses.append(avg_loss)
                epoch_losses.append(avg_loss)
                
                # Validation step - compute average validation loss across ensemble
                val_batch_losses = []
                ensemble_train_preds = []
                ensemble_test_preds = []
                
                for model in ensemble_models:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test)
                        val_loss = criterion(val_outputs, y_test_onehot)
                        val_batch_losses.append(val_loss.item())
                        
                        # Get predictions for accuracy calculation
                        train_outputs = model(L_X)
                        ensemble_train_preds.append(torch.argmax(train_outputs, dim=1))
                        ensemble_test_preds.append(torch.argmax(val_outputs, dim=1))
                
                avg_val_loss = np.mean(val_batch_losses)
                val_losses.append(avg_val_loss)
                
                # Calculate ensemble accuracy by averaging predictions
                # Stack predictions from all models and take majority vote
                if ensemble_train_preds and ensemble_test_preds:
                    train_pred_stack = torch.stack(ensemble_train_preds)  # [n_ensemble, n_samples]
                    test_pred_stack = torch.stack(ensemble_test_preds)
                    
                    # Take mode (most frequent prediction) across ensemble
                    train_final_pred = torch.mode(train_pred_stack, dim=0)[0]
                    test_final_pred = torch.mode(test_pred_stack, dim=0)[0]
                    
                    train_epoch_acc = accuracy_score(L_y_idx.numpy(), train_final_pred.numpy())
                    test_epoch_acc = accuracy_score(y_test_idx.numpy(), test_final_pred.numpy())
                    
                    train_acc_curve.append(train_epoch_acc)
                    test_acc_curve.append(test_epoch_acc)
                
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
            
            # Set all models to eval mode
            for model in ensemble_models:
                model.eval()
                
            with torch.no_grad():
                for i in range(len(U_X)):
                    x_sample = U_X[i:i+1]
                    
                    # Get predictions from all ensemble members
                    ensemble_outputs = []
                    for model in ensemble_models:
                        output = model(x_sample)
                        # Convert to probabilities using softmax
                        prob_dist = torch.softmax(output.squeeze(), dim=0)
                        ensemble_outputs.append(prob_dist)
                    
                    # Stack predictions: shape [n_ensemble, n_classes]
                    ensemble_preds = torch.stack(ensemble_outputs)
                    
                    # Calculate uncertainty as variance across ensemble predictions
                    # Higher variance = higher uncertainty = more informative
                    pred_variance = torch.var(ensemble_preds, dim=0)  # Variance per class
                    uncertainty = torch.sum(pred_variance).item()     # Total variance as uncertainty measure
                    
                    uncertainties.append((uncertainty, i))
            
            # Select highest uncertainty samples
            uncertainties.sort(reverse=True, key=lambda x: x[0])
            samples_to_query = min(query_interval, len(uncertainties), budget_remaining)
            selected_indices = [idx for _, idx in uncertainties[:samples_to_query]]
            
            # Move samples from U to L
            for idx in sorted(selected_indices, reverse=True):
                L_X = torch.cat([L_X, U_X[idx:idx+1]], dim=0)
                L_y = torch.cat([L_y, U_y[idx:idx+1]], dim=0)
                L_y_idx = torch.cat([L_y_idx, U_y_idx[idx:idx+1]], dim=0)
                
                # Remove from U
                mask = torch.ones(len(U_X), dtype=torch.bool)
                mask[idx] = False
                U_X = U_X[mask]
                U_y = U_y[mask]
                U_y_idx = U_y_idx[mask]
            
            # Update budget
            budget_remaining -= samples_to_query
            query_iteration += 1
            
            # Reinitialize optimizers for updated labeled set
            ensemble_optimizers = []
            for model in ensemble_models:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    momentum=params['momentum']
                )
                ensemble_optimizers.append(optimizer)
        
        # Final training on complete labeled set
        remaining_epochs = epochs - total_epochs_used
        for epoch in range(remaining_epochs):
            # Training step
            batch_losses = []
            for model, optimizer in zip(ensemble_models, ensemble_optimizers):
                model.train()
                optimizer.zero_grad()
                outputs = model(L_X)
                loss = criterion(outputs, L_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            losses.append(np.mean(batch_losses))
            
            # Validation step with accuracy tracking
            val_batch_losses = []
            ensemble_train_preds = []
            ensemble_test_preds = []
            
            for model in ensemble_models:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test_onehot)
                    val_batch_losses.append(val_loss.item())
                    
                    # Get predictions for accuracy calculation
                    train_outputs = model(L_X)
                    ensemble_train_preds.append(torch.argmax(train_outputs, dim=1))
                    ensemble_test_preds.append(torch.argmax(val_outputs, dim=1))
            
            val_losses.append(np.mean(val_batch_losses))
            
            # Calculate ensemble accuracy by majority vote
            if ensemble_train_preds and ensemble_test_preds:
                train_pred_stack = torch.stack(ensemble_train_preds)
                test_pred_stack = torch.stack(ensemble_test_preds)
                
                train_final_pred = torch.mode(train_pred_stack, dim=0)[0]
                test_final_pred = torch.mode(test_pred_stack, dim=0)[0]
                
                train_epoch_acc = accuracy_score(L_y_idx.numpy(), train_final_pred.numpy())
                test_epoch_acc = accuracy_score(y_test_idx.numpy(), test_final_pred.numpy())
                
                train_acc_curve.append(train_epoch_acc)
                test_acc_curve.append(test_epoch_acc)
        
        # Final evaluation using ensemble averaging
        for model in ensemble_models:
            model.eval()
            
        with torch.no_grad():
            # Training accuracy (ensemble average on labeled set L)
            train_ensemble_outputs = []
            for model in ensemble_models:
                train_outputs = model(L_X)
                train_ensemble_outputs.append(train_outputs)
            
            # Average predictions across ensemble
            avg_train_outputs = torch.mean(torch.stack(train_ensemble_outputs), dim=0)
            train_pred = torch.argmax(avg_train_outputs, dim=1)
            train_acc = accuracy_score(L_y_idx.numpy(), train_pred.numpy())
            
            # Test accuracy (ensemble average on test set)
            test_ensemble_outputs = []
            for model in ensemble_models:
                test_outputs = model(X_test)
                test_ensemble_outputs.append(test_outputs)
            
            avg_test_outputs = torch.mean(torch.stack(test_ensemble_outputs), dim=0)
            test_pred = torch.argmax(avg_test_outputs, dim=1)
            test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
        
        # Track ensemble-specific statistics
        reduction_data = {
            'pool_based_queries': query_iteration,
            'initial_labeled_size': initial_labeled_size,
            'final_labeled_size': len(L_X),
            'original_training_size': len(X_train),
            'total_budget_used': total_budget - budget_remaining,
            'avg_pool_size': np.mean(pool_sizes) if pool_sizes else 0,
            'avg_labeled_size': np.mean(labeled_sizes) if labeled_sizes else initial_labeled_size,
            'training_set_sizes_at_epochs': training_set_sizes_at_epochs,
            'final_reduction_percentage': ((len(X_train) - len(L_X)) / len(X_train)) * 100,
            'ensemble_size': n_ensemble,
            'query_history': {
                'pool_sizes': pool_sizes,
                'labeled_sizes': labeled_sizes,
                'total_epochs': total_epochs_used
            }
        }
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses, threshold=0.01)
        
        # Return first model as representative (all models trained similarly)
        return ensemble_models[0], losses, train_acc, test_acc, epochs_converged, reduction_data, val_losses, train_acc_curve, test_acc_curve


def compare_learning_strategies(results_list: List[Dict], strategy_names: List[str]):
    """Compare multiple learning strategies."""
    print("\n" + "=" * 100)
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)
    
    comparison_data = []
    for results, name in zip(results_list, strategy_names):
        comparison_data.append({
            'Strategy': name,
            'Test Acc': f"{results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}",
            'Train Acc': f"{results['train_acc_mean']:.4f} ± {results['train_acc_std']:.4f}",
            'Gen Factor': f"{results['generalization_factor']:.4f}",
            'Conv Rate': f"{results['convergence_rate']:.1%}",
            'Avg Time': f"{results['avg_computation_time']:.3f}s"
        })
    
    # Print comparison table
    headers = ['Strategy', 'Test Acc', 'Train Acc', 'Gen Factor', 'Conv Rate', 'Avg Time']
    col_widths = [max(len(str(row[col])) for row in [dict(zip(headers, headers))] + comparison_data) + 2 
                  for col in headers]
    
    # Print header
    header_row = "|".join(f"{header:^{width}}" for header, width in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in comparison_data:
        data_row = "|".join(f"{row[col]:^{width}}" for col, width in zip(headers, col_widths))
        print(data_row)