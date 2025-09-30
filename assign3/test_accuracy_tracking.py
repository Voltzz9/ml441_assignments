#!/usr/bin/env python3
"""
Test script to verify accuracy tracking functionality.
This script tests all learning strategies to ensure accuracy curves are properly tracked.
"""

import torch
import numpy as np
import sys
import os

# Add src to path and import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.active_learning import ModelEvaluator, ActiveLearningEvaluator
from src.data_utils import load_iris_data

def test_accuracy_tracking():
    """Test accuracy tracking across all learning strategies."""
    print("Testing Accuracy Tracking Functionality")
    print("=" * 50)
    
    # Load test data
    X, y = load_iris_data()
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test parameters (small for quick testing)
    best_params = {
        'hidden_size': 10,
        'learning_rate': 0.01,
        'weight_decay': 0.0001,
        'momentum': 0.9
    }
    
    n_trials = 2  # Small number for testing
    epochs = 50   # Reduced epochs for quick testing
    
    print(f"\nTesting with {n_trials} trials, {epochs} epochs each")
    print(f"Parameters: {best_params}")
    
    # Test Passive Learning
    print("\n1. Testing Passive Learning...")
    try:
        evaluator = ModelEvaluator()
        passive_results = evaluator.evaluate_passive_learning(
            X, y, best_params, n_trials=n_trials, epochs=epochs, use_cv=False
        )
        
        # Check if accuracy curves are available
        accuracy_curves = evaluator.metrics_tracker.get_accuracy_curves()
        if accuracy_curves:
            train_curves, test_curves = accuracy_curves
            print(f"✓ Passive Learning: Got {len(train_curves)} train curves, {len(test_curves)} test curves")
            print(f"  - Average curve length: {np.mean([len(curve) for curve in train_curves]):.1f} epochs")
            print(f"  - First trial final accuracy: {train_curves[0][-1]:.3f} (train), {test_curves[0][-1]:.3f} (test)")
        else:
            print("✗ Passive Learning: No accuracy curves found!")
            
    except Exception as e:
        print(f"✗ Passive Learning failed: {e}")
    
    # Test Active Learning strategies
    active_evaluator = ActiveLearningEvaluator()
    
    strategies = [
        ('output_sensitivity', {}),
        ('uncertainty_sampling', {}),
        ('ensemble_uncertainty', {'n_ensemble': 2})  # Small ensemble for testing
    ]
    
    for i, (strategy, kwargs) in enumerate(strategies, 2):
        print(f"\n{i}. Testing Active Learning - {strategy}...")
        try:
            results = active_evaluator.evaluate_active_learning(
                X, y, best_params, strategy, 
                n_trials=n_trials, epochs=epochs, use_cv=False, **kwargs
            )
            
            # Check if accuracy curves are available
            accuracy_curves = active_evaluator.metrics_tracker.get_accuracy_curves()
            if accuracy_curves:
                train_curves, test_curves = accuracy_curves
                print(f"✓ {strategy}: Got {len(train_curves)} train curves, {len(test_curves)} test curves")
                print(f"  - Average curve length: {np.mean([len(curve) for curve in train_curves]):.1f} epochs")
                print(f"  - First trial final accuracy: {train_curves[0][-1]:.3f} (train), {test_curves[0][-1]:.3f} (test)")
                
                # For uncertainty sampling, check F1 curve data
                if strategy == 'uncertainty_sampling' and active_evaluator.f1_curves_data:
                    print(f"  - F1 vs instances data: {len(active_evaluator.f1_curves_data)} trials")
                    print(f"  - First trial F1 points: {len(active_evaluator.f1_curves_data[0])}")
            else:
                print(f"✗ {strategy}: No accuracy curves found!")
                
        except Exception as e:
            print(f"✗ {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Accuracy Tracking Test Complete!")

if __name__ == "__main__":
    test_accuracy_tracking()