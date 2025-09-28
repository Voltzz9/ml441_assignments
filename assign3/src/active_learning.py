import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import time
from typing import Dict, List, Tuple, Optional, Callable
from .models import IrisNet
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
                                 epochs: int = 1000, random_state: int = 42) -> Dict:
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
                                                 cv_folds, epochs, random_state + trial)
            else:
                # Simple train-test split approach
                trial_results = self._run_simple_trial(X, y, y_indices, best_params, 
                                                     epochs, random_state + trial)
            
            computation_time = time.time() - start_time
            
            # Add results to tracker
            self.metrics_tracker.add_trial_results(
                train_acc=trial_results['train_acc'],
                test_acc=trial_results['test_acc'],
                val_acc=trial_results.get('val_acc'),
                losses=trial_results['losses'],
                epochs_converged=trial_results['epochs_converged'],
                num_presentations=trial_results['num_presentations'],
                computation_time=computation_time,
                train_metrics=trial_results.get('train_metrics'),
                test_metrics=trial_results.get('test_metrics'),
                val_metrics=trial_results.get('val_metrics')
            )
            
            print(f"Test Acc: {trial_results['test_acc']:.4f}, Time: {computation_time:.2f}s")
        
        # Compute and return comprehensive statistics
        results = self.metrics_tracker.compute_statistics()
        results['learning_type'] = 'passive'
        results['best_params'] = best_params
        
        return results
    
    def _run_cv_trial(self, X: torch.Tensor, y: torch.Tensor, y_indices: torch.Tensor,
                     params: Dict, cv_folds: int, epochs: int, random_state: int) -> Dict:
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
        all_losses = []
        
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
            model, losses, train_acc, val_acc, _, train_metrics, val_metrics = self._train_model(
                X_train, y_train_idx, X_val, y_val_idx, params, epochs
            )
            
            cv_train_accs.append(train_acc)
            cv_val_accs.append(val_acc)
            cv_train_metrics.append(train_metrics)
            cv_val_metrics.append(val_metrics)
            all_losses.extend(losses)
        
        # Average CV results
        avg_train_acc = np.mean(cv_train_accs)
        avg_val_acc = np.mean(cv_val_accs)
        
        # Average CV metrics
        avg_train_metrics = self._average_metrics(cv_train_metrics)
        avg_val_metrics = self._average_metrics(cv_val_metrics)
        
        # Final test on holdout set using best parameters
        # Train on all temp data for final test
        final_model, final_losses, _, _, epochs_converged, _, test_metrics = self._train_model(
            X_temp, y_temp_idx, X_test, y_test_idx, params, epochs
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
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations,
            'train_metrics': avg_train_metrics,
            'test_metrics': test_metrics,
            'val_metrics': avg_val_metrics
        }
    
    def _run_simple_trial(self, X: torch.Tensor, y: torch.Tensor, y_indices: torch.Tensor,
                         params: Dict, epochs: int, random_state: int) -> Dict:
        """Run a single trial with simple train-test split."""
        
        X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
            X, y, y_indices, test_size=0.2, random_state=random_state, 
            stratify=y_indices
        )
        
        # Train model
        model, losses, train_acc, test_acc, epochs_converged, train_metrics, test_metrics = self._train_model(
            X_train, y_train_idx, X_test, y_test_idx, params, epochs
        )
        
        # Calculate pattern presentations
        num_presentations = len(X_train) * epochs_converged
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'losses': losses,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'epochs_converged': epochs_converged,
            'num_presentations': num_presentations
        }
    
    def _train_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                    X_test: torch.Tensor, y_test_idx: torch.Tensor,
                    params: Dict, epochs: int) -> Tuple:
        """Train a single model and return results."""
        
        # Create model
        model = IrisNet(
            input_size=4,
            hidden_size=params['hidden_size'],
            output_size=3
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)  # Default to 0.0 if not specified
        )
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train_idx)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            test_outputs = model(X_test)
            
            train_pred = torch.argmax(train_outputs, dim=1)
            test_pred = torch.argmax(test_outputs, dim=1)
            
            train_acc = accuracy_score(y_train_idx.numpy(), train_pred.numpy())
            test_acc = accuracy_score(y_test_idx.numpy(), test_pred.numpy())
        
        # Compute comprehensive classification metrics
        train_metrics = compute_classification_metrics(y_train_idx.numpy(), train_pred.numpy())
        test_metrics = compute_classification_metrics(y_test_idx.numpy(), test_pred.numpy())
        
        return model, losses, train_acc, test_acc, epochs_converged, train_metrics, test_metrics
    
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
    """Extended evaluator for active learning strategies."""
    
    def __init__(self, convergence_threshold: float = 0.95):
        super().__init__(convergence_threshold)
        # Add additional tracking for active learning metrics
        self.training_set_reductions = {}  # Store reduction percentages at different epochs
        
    def evaluate_active_learning(self, X: torch.Tensor, y: torch.Tensor,
                                best_params: Dict, strategy: str,
                                n_trials: int = 50, use_cv: bool = True,
                                cv_folds: int = 5, epochs: int = 1000,
                                random_state: int = 12, **strategy_kwargs) -> Dict:
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
                epochs_converged=trial_results['epochs_converged'],
                num_presentations=trial_results['num_presentations'],
                computation_time=computation_time,
                val_acc=trial_results.get('val_acc', None),
                train_metrics=trial_results.get('train_metrics'),
                test_metrics=trial_results.get('test_metrics'),
                val_metrics=trial_results.get('val_metrics')
            )
            
            # Track training set size information
            self.final_training_set_sizes.append(trial_results.get('final_training_set_size', 0))
            self.original_training_set_sizes.append(trial_results.get('original_training_set_size', 0))
            
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
            
            model, losses, train_acc, val_acc, epochs_converged, reduction_data = self._train_active_model(
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
        final_model, final_losses, _, _, epochs_converged, final_reduction_data = self._train_active_model(
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
        model, losses, train_acc, test_acc, epochs_converged, reduction_data = self._train_active_model(
            X_train, y_train_idx, X_test, y_test_idx, params, strategy, epochs, **strategy_kwargs
        )
        
        num_presentations = len(X_train) * epochs_converged
        
        # if Strategy is output_sensitivity
        if strategy == 'output_sensitivity':
            return {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'losses': losses,
                'epochs_converged': epochs_converged,
                'num_presentations': num_presentations,
                'training_set_reductions': reduction_data,
                'final_training_set_size': reduction_data.get('final_labeled_size', len(X_train)),
                'original_training_set_size': reduction_data.get('original_training_size', len(X_train)),
                'training_set_sizes_at_epochs': reduction_data.get('training_set_sizes_at_epochs', {})
            }
        else:
            return {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'losses': losses,
                'epochs_converged': epochs_converged,
                'num_presentations': num_presentations,
                'training_set_reductions': reduction_data,
                'final_training_set_size': reduction_data.get('final_labeled_size', len(X_train)),
                'original_training_set_size': reduction_data.get('original_training_size', len(X_train)),
                'training_set_sizes_at_epochs': reduction_data.get('training_set_sizes_at_epochs', {})
            }
    
    def _train_active_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                           X_test: torch.Tensor, y_test_idx: torch.Tensor,
                           params: Dict, strategy: str, epochs: int, **strategy_kwargs) -> Tuple:
        """Train a single model using active learning strategy."""
        
        if strategy == 'output_sensitivity':
            return self._train_output_sensitivity_model(
                X_train, y_train_idx, X_test, y_test_idx, params, epochs, **strategy_kwargs
            )
        elif strategy == 'uncertainty_sampling':
            return self._train_uncertainty_sampling_model(
                X_train, y_train_idx, X_test, y_test_idx, params, epochs, **strategy_kwargs
            )
        elif strategy == 'ensemble_uncertainty':
            return self._train_ensemble_uncertainty_model(
                X_train, y_train_idx, X_test, y_test_idx, params, epochs, **strategy_kwargs
            )
        else:
            raise ValueError(f"Unknown active learning strategy: {strategy}")
    
    def _train_output_sensitivity_model(self, X_train: torch.Tensor, y_train_idx: torch.Tensor,
                                       X_test: torch.Tensor, y_test_idx: torch.Tensor,
                                       params: Dict, epochs: int, 
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
        
        # Create model
        model = IrisNet(
            input_size=4,
            hidden_size=params['hidden_size'],
            output_size=3
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.0)
        )
        
        # Initialize training data (make copies to avoid modifying originals)
        current_X_train = X_train.clone()
        current_y_train_idx = y_train_idx.clone()
        original_train_size = len(X_train)
        
        # Track training set reductions at specific epochs
        reduction_data = {}
        reduction_data_actual_size = {}
        
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        
        losses = []
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(current_X_train)
            loss = criterion(outputs, current_y_train_idx)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Apply SASLA pattern selection at specified intervals
            if epoch > 0 and epoch % selection_interval == 0:
                current_X_train, current_y_train_idx = self._apply_sasla_selection(
                    model, current_X_train, current_y_train_idx, alpha
                )
            
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
        
        return model, losses, train_acc, test_acc, epochs_converged, reduction_data_result
    
    def _apply_sasla_selection(self, model: nn.Module, X_train: torch.Tensor, 
                              y_train_idx: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SASLA pattern selection based on output sensitivity analysis.
        
        This implements the true SASLA algorithm from the paper:
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
        
        # Compute pattern informativeness for each training sample
        informativeness_scores = []
        
        for i in range(len(X_train)):
            pattern = X_train[i:i+1]  # Shape: [1, input_size]
            target_class = y_train_idx[i].item()
            # Forward pass to get activations
            with torch.no_grad():
                # Hidden layer activations (before activation function)
                hidden_pre = torch.matmul(pattern, v_weights.T) + v_bias  # [1, hidden_size]
                # Apply activation function (assuming sigmoid for SASLA)
                hidden_post = torch.sigmoid(hidden_pre)  # y_j in paper
                
                # Output layer activations (before softmax)
                output_pre = torch.matmul(hidden_post, w_weights.T) + w_bias  # [1, output_size]
                # Apply softmax to get output probabilities
                output_probs = torch.sigmoid(output_pre)  # o_k in paper
            
            # Compute output sensitivity for target class (Equation 4)
            # S_oz(k,i) = (1 - o_k) * o_k * Σ_j [w_kj * (1 - y_j) * y_j * v_ji]
            
            target_output_prob = output_probs[0, target_class]  # o_k for target class
            
            # Compute the sum over hidden units
            sensitivity_sum = 0.0
            for j in range(hidden_post.shape[1]):  # For each hidden unit
                y_j = hidden_post[0, j].item()  # Hidden activation
                w_kj = w_weights[target_class, j].item()  # Weight from hidden j to output k
                v_ji = v_weights[j, :].dot(pattern[0]).item()  # Weighted input to hidden j
                
                sensitivity_sum += w_kj * (1 - y_j) * y_j * v_ji
            
            # Complete sensitivity calculation
            output_sensitivity = (1 - target_output_prob) * target_output_prob * sensitivity_sum
            
            # Compute pattern informativeness (Equations 1-2)
            # First compute sum-norm: sum of absolute sensitivities for all outputs
            sum_norm = 0.0
            for k in range(output_probs.shape[1]):  # For each output class
                o_k = output_probs[0, k].item()
                class_sensitivity_sum = 0.0
                
                for j in range(hidden_post.shape[1]):
                    y_j = hidden_post[0, j].item()
                    w_kj = w_weights[k, j].item()
                    v_ji = v_weights[j, :].dot(pattern[0]).item()
                    class_sensitivity_sum += w_kj * (1 - y_j) * y_j * v_ji
                
                class_sensitivity = (1 - o_k) * o_k * class_sensitivity_sum
                sum_norm += abs(class_sensitivity)
            
            # Max-norm: maximum absolute sensitivity across outputs
            max_sensitivity = abs(output_sensitivity.item())
            for k in range(output_probs.shape[1]):
                if k != target_class:
                    o_k = output_probs[0, k].item()
                    class_sensitivity_sum = 0.0
                    
                    for j in range(hidden_post.shape[1]):
                        y_j = hidden_post[0, j].item()
                        w_kj = w_weights[k, j].item()
                        v_ji = v_weights[j, :].dot(pattern[0]).item()
                        class_sensitivity_sum += w_kj * (1 - y_j) * y_j * v_ji
                    
                    class_sensitivity = (1 - o_k) * o_k * class_sensitivity_sum
                    max_sensitivity = max(max_sensitivity, abs(class_sensitivity))
            
            # Pattern informativeness (Equation 2): max-norm of sum-norm
            informativeness = max(sum_norm, max_sensitivity)
            informativeness_scores.append(informativeness)
        
        # Convert to numpy for easier manipulation
        informativeness_scores = np.array(informativeness_scores)
        
        # Calculate selection threshold: (1-α) * average_informativeness
        avg_informativeness = np.mean(informativeness_scores)
        selection_threshold = (1 - alpha) * avg_informativeness
        
        # Select patterns with informativeness above threshold
        selected_indices = informativeness_scores >= selection_threshold

        
        # Ensure we keep at least one sample per class
        for class_label in torch.unique(y_train_idx):
            class_mask = (y_train_idx == class_label)
            class_selected = selected_indices & class_mask.numpy()
            
            if not np.any(class_selected):
                # If no samples selected for this class, keep the most informative one
                class_informativeness = informativeness_scores[class_mask.numpy()]
                best_idx = np.argmax(class_informativeness)
                class_indices = torch.where(class_mask)[0]
                selected_indices[class_indices[best_idx]] = True
        
        # Return selected patterns
        selected_tensor = torch.tensor(selected_indices, dtype=torch.bool)
        return X_train[selected_tensor], y_train_idx[selected_tensor]
    
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
        
        # Convert y_train_idx to one-hot for MSE loss (matching your approach)
        y_train_onehot = torch.zeros(len(y_train_idx), 3)
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
        model = IrisNet(
            input_size=4,
            hidden_size=params['hidden_size'],
            output_size=3,
            use_mse=True
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params['momentum']
        )
        
        # Training tracking
        losses = []
        pool_sizes = []
        labeled_sizes = []
        
        # Track training set sizes at specific epochs for comparison
        epoch_checkpoints = [25, 100, 200, 300, 500, 1000]
        training_set_sizes_at_epochs = {}
        
        # Pool-Based Active Learning Algorithm
        budget_remaining = total_budget
        query_interval = max(1, total_budget // 50)  # Make ~50 queries total
        training_epochs_per_query = max(10, epochs // 50)  # Training epochs between queries
        
        query_iteration = 0
        total_epochs_used = 0
        
        while budget_remaining > 0 and len(U_X) > 0 and total_epochs_used < epochs:
            
            # Train model on current labeled set L
            for epoch in range(training_epochs_per_query):
                if total_epochs_used >= epochs:
                    break
                    
                model.train()
                optimizer.zero_grad()
                outputs = model(L_X)
                loss = criterion(outputs, L_y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
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
            model.train()
            optimizer.zero_grad()
            outputs = model(L_X)
            loss = criterion(outputs, L_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy (on final labeled set L)
            train_outputs = model(L_X)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_acc = accuracy_score(L_y_idx.numpy(), train_pred.numpy())
            
            # Test accuracy (on full test set)
            y_test_onehot = torch.zeros(len(y_test_idx), 3)
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
            'query_history': {
                'pool_sizes': pool_sizes,
                'labeled_sizes': labeled_sizes,
                'total_epochs': total_epochs_used
            }
        }
        
        # Find convergence epoch
        epochs_converged = find_convergence_epoch(losses, threshold=0.01)
        
        return model, losses, train_acc, test_acc, epochs_converged, reduction_data

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
        y_train_onehot = torch.zeros(len(y_train_idx), 3)
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
            model = IrisNet(
                input_size=4,
                hidden_size=params['hidden_size'],
                output_size=3,
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
        
        # Training tracking
        losses = []  # Average loss across ensemble
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
        return ensemble_models[0], losses, train_acc, test_acc, epochs_converged, reduction_data


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