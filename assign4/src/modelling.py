# imports
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, jaccard_score)
from time import time
import tracemalloc


def evaluate_isolation_forest(X, y_true, params, param_name=None, param_value=None):
    """
    Evaluate Isolation Forest with given parameters
    
    Args:
        X: Feature matrix
        y_true: True binary labels (1 for anomaly, 0 for normal)
        params: Dictionary of parameters for IsolationForest
        param_name: Name of parameter being varied (for tracking)
        param_value: Value of parameter being varied (for tracking)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    start_time = time()
    
    # Ensure random_state is set but avoid duplicate keyword errors
    model_params = params.copy()
    model_params.setdefault('random_state', 42)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(**model_params)
    y_pred = iso_forest.fit_predict(X)
    
    # Convert predictions (-1 for anomaly, 1 for normal) to binary (1 for anomaly, 0 for normal)
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Get anomaly scores (lower score = more anomalous)
    anomaly_scores = iso_forest.score_samples(X)
    # Convert to anomaly scores (higher = more anomalous)
    anomaly_scores = -anomaly_scores
    
    training_time = time() - start_time
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
    except Exception:
        roc_auc = np.nan
    
    try:
        pr_auc = average_precision_score(y_true, anomaly_scores)
    except Exception:
        pr_auc = np.nan
    
    results = {
        'param_name': param_name,
        'param_value': param_value,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'training_time': training_time,
        'n_anomalies_detected': y_pred_binary.sum(),
        'params': model_params
    }
    
    return results



def evaluate_isolation_forest_with_std(X, y_true, params, param_name=None, param_value=None, n_runs=10):
    """
    Evaluate Isolation Forest with given parameters, averaging over multiple runs to account for randomness.
    
    Args:
        X: Feature matrix
        y_true: True binary labels (1 for anomaly, 0 for normal)
        params: Dictionary of parameters for IsolationForest
        param_name: Name of parameter being varied (for tracking)
        param_value: Value of parameter being varied (for tracking)
        n_runs: Number of runs to average results over (default: 10)
    Returns:
        Dictionary containing mean, std, min, max of evaluation metrics
    """
    # Store results from all runs
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_roc_aucs = []
    all_pr_aucs = []
    all_training_times = []
    all_n_anomalies = []
    all_avg_path_lengths = []
    all_predictions = []
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time()
    
    for run in range(n_runs):
        # Update random state for each run
        model_params = params.copy()
        model_params['random_state'] = 42 + run
        
        run_start_time = time()
        
        # Train Isolation Forest
        iso_forest = IsolationForest(**model_params)
        y_pred = iso_forest.fit_predict(X)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to binary (1 for anomaly, 0 for normal)
        y_pred_binary = (y_pred == -1).astype(int)
        all_predictions.append(y_pred_binary)
        
        # Get anomaly scores (lower score = more anomalous)
        anomaly_scores = iso_forest.score_samples(X)
        # Convert to anomaly scores (higher = more anomalous)
        anomaly_scores = -anomaly_scores
        
        # Get average path length
        avg_path_length = np.mean(iso_forest.decision_function(X))
        all_avg_path_lengths.append(avg_path_length)
        
        run_time = time() - run_start_time
        all_training_times.append(run_time)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_n_anomalies.append(y_pred_binary.sum())
        
        try:
            roc_auc = roc_auc_score(y_true, anomaly_scores)
            all_roc_aucs.append(roc_auc)
        except Exception:
            all_roc_aucs.append(np.nan)
        
        try:
            pr_auc = average_precision_score(y_true, anomaly_scores)
            all_pr_aucs.append(pr_auc)
        except Exception:
            all_pr_aucs.append(np.nan)
    
    total_training_time = time() - start_time
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage_mb = peak / 1024 / 1024  # Convert to MB
    
    # Calculate Jaccard similarity between consecutive runs
    jaccard_similarities = []
    for i in range(len(all_predictions) - 1):
        jaccard_sim = jaccard_score(all_predictions[i], all_predictions[i+1])
        jaccard_similarities.append(jaccard_sim)
    
    # Compile results with statistics
    results = {
        'param_name': param_name,
        'param_value': param_value,
        
        # Precision statistics
        'precision_mean': np.mean(all_precisions),
        'precision_std': np.std(all_precisions),
        'precision_min': np.min(all_precisions),
        'precision_max': np.max(all_precisions),
        
        # Recall statistics
        'recall_mean': np.mean(all_recalls),
        'recall_std': np.std(all_recalls),
        'recall_min': np.min(all_recalls),
        'recall_max': np.max(all_recalls),
        
        # F1 statistics
        'f1_score_mean': np.mean(all_f1s),
        'f1_score_std': np.std(all_f1s),
        'f1_score_min': np.min(all_f1s),
        'f1_score_max': np.max(all_f1s),
        
        # ROC-AUC statistics
        'roc_auc_mean': np.nanmean(all_roc_aucs),
        'roc_auc_std': np.nanstd(all_roc_aucs),
        'roc_auc_min': np.nanmin(all_roc_aucs),
        'roc_auc_max': np.nanmax(all_roc_aucs),
        
        # PR-AUC statistics
        'pr_auc_mean': np.nanmean(all_pr_aucs),
        'pr_auc_std': np.nanstd(all_pr_aucs),
        'pr_auc_min': np.nanmin(all_pr_aucs),
        'pr_auc_max': np.nanmax(all_pr_aucs),
        
        # Average path length statistics
        'avg_path_length_mean': np.mean(all_avg_path_lengths),
        'avg_path_length_std': np.std(all_avg_path_lengths),
        'avg_path_length_min': np.min(all_avg_path_lengths),
        'avg_path_length_max': np.max(all_avg_path_lengths),
        
        # Training time statistics
        'training_time_mean': np.mean(all_training_times),
        'training_time_std': np.std(all_training_times),
        'training_time_total': total_training_time,
        
        # Number of anomalies statistics
        'n_anomalies_detected_mean': np.mean(all_n_anomalies),
        'n_anomalies_detected_std': np.std(all_n_anomalies),
        'n_anomalies_detected_min': np.min(all_n_anomalies),
        'n_anomalies_detected_max': np.max(all_n_anomalies),
        
        # Jaccard similarity statistics
        'jaccard_similarity_mean': np.mean(jaccard_similarities) if jaccard_similarities else np.nan,
        'jaccard_similarity_std': np.std(jaccard_similarities) if jaccard_similarities else np.nan,
        'jaccard_similarity_min': np.min(jaccard_similarities) if jaccard_similarities else np.nan,
        'jaccard_similarity_max': np.max(jaccard_similarities) if jaccard_similarities else np.nan,
        
        # Memory usage
        'memory_usage_mb': memory_usage_mb,
        
        # Number of runs
        'n_runs': n_runs,
        'params': params
    }
    
    return results