"""
Model Prediction Module
=======================

This module handles predictions using trained SYN scan detection models.

Usage:
    1. Load trained model
    2. Preprocess new network traffic data
    3. Make predictions
    4. Interpret results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib


def load_model(filepath: str) -> Tuple[Any, list]:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model
    
    Returns:
        Tuple of (model, feature_names)
    """
    model_data = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model_data['model'], model_data.get('feature_names', [])


def predict(model, X: pd.DataFrame, feature_names: list = None) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        X: Feature matrix (must have same features as training)
        feature_names: Expected feature names (for validation)
    
    Returns:
        Array of predictions (0=normal, 1=attack)
    """
    # Ensure correct feature order if names provided
    if feature_names:
        X = X[feature_names]
    
    # Handle missing values
    X = X.fillna(0)
    
    predictions = model.predict(X)
    return predictions


def predict_proba(model, X: pd.DataFrame, feature_names: list = None) -> np.ndarray:
    """
    Get prediction probabilities.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Expected feature names
    
    Returns:
        Array of probabilities [P(normal), P(attack)]
    """
    if feature_names:
        X = X[feature_names]
    
    X = X.fillna(0)
    
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        # For models without predict_proba, use predictions
        preds = model.predict(X)
        return np.column_stack([1 - preds, preds])


def predict_with_details(model, X: pd.DataFrame, 
                        feature_names: list,
                        threshold: float = 0.5) -> pd.DataFrame:
    """
    Make predictions with detailed output.
    
    Returns DataFrame with:
    - prediction: 0 or 1
    - probability: probability of attack
    - confidence: how confident the model is
    - risk_level: Low/Medium/High based on probability
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Expected feature names
        threshold: Classification threshold
    
    Returns:
        DataFrame with predictions and details
    """
    # Get probabilities
    proba = predict_proba(model, X, feature_names)
    attack_prob = proba[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': (attack_prob >= threshold).astype(int),
        'attack_probability': attack_prob,
        'confidence': np.abs(attack_prob - 0.5) * 2,  # 0 to 1 scale
    })
    
    # Add risk level
    conditions = [
        attack_prob < 0.3,
        attack_prob < 0.7,
        attack_prob >= 0.7
    ]
    risk_levels = ['Low', 'Medium', 'High']
    results['risk_level'] = np.select(conditions, risk_levels, default='Medium')
    
    return results


def analyze_predictions(predictions: np.ndarray, 
                       original_data: pd.DataFrame = None) -> Dict:
    """
    Analyze prediction results.
    
    Args:
        predictions: Array of predictions
        original_data: Optional original data for context
    
    Returns:
        Dictionary with analysis results
    """
    total = len(predictions)
    attacks = predictions.sum()
    normal = total - attacks
    
    analysis = {
        'total_samples': total,
        'detected_attacks': int(attacks),
        'normal_traffic': int(normal),
        'attack_percentage': 100 * attacks / total if total > 0 else 0
    }
    
    print("\n" + "=" * 50)
    print("PREDICTION ANALYSIS")
    print("=" * 50)
    print(f"Total time windows analyzed: {total}")
    print(f"Detected SYN scan attacks:   {attacks} ({analysis['attack_percentage']:.1f}%)")
    print(f"Normal traffic:              {normal} ({100-analysis['attack_percentage']:.1f}%)")
    print("=" * 50)
    
    return analysis


def predict_on_new_traffic(model_path: str,
                          raw_data_path: str,
                          window_seconds: int = 1) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete prediction pipeline for new traffic data.
    
    This is the main function to use for detecting SYN scans in new data.
    
    Pipeline:
    1. Load trained model
    2. Preprocess new traffic (aggregate into time windows)
    3. Make predictions
    4. Return results with analysis
    
    Args:
        model_path: Path to trained model
        raw_data_path: Path to new traffic CSV
        window_seconds: Time window size
    
    Returns:
        Tuple of (results_df, analysis_dict)
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocess import preprocess_pipeline
    from features.build_features import select_features
    
    print("=" * 50)
    print("SYN SCAN DETECTION - Prediction")
    print("=" * 50)
    
    # Load model
    print("\n[1/3] Loading model...")
    model, feature_names = load_model(model_path)
    
    # Preprocess new data
    print("\n[2/3] Preprocessing new traffic data...")
    aggregated = preprocess_pipeline(raw_data_path, window_seconds)
    
    # Select features
    X, _ = select_features(aggregated)
    
    # Make predictions
    print("\n[3/3] Making predictions...")
    results = predict_with_details(model, X, feature_names)
    
    # Combine with original data
    results_full = pd.concat([
        aggregated[['ip.src', 'ip.dst', 'time_window', 'syn_count', 
                   'ack_count', 'unique_dst_ports', 'packet_count']].reset_index(drop=True),
        results
    ], axis=1)
    
    # Analyze
    analysis = analyze_predictions(results['prediction'].values, aggregated)
    
    return results_full, analysis


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "syn_scan_detector.joblib"
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    
    if model_path.exists() and raw_path.exists():
        results, analysis = predict_on_new_traffic(
            str(model_path),
            str(raw_path)
        )
        
        print("\nSample of detected attacks:")
        attacks = results[results['prediction'] == 1]
        if len(attacks) > 0:
            print(attacks.head(10))
        else:
            print("No attacks detected")
    else:
        print("Model or data file not found")
        print(f"Model: {model_path} - {'exists' if model_path.exists() else 'MISSING'}")
        print(f"Data: {raw_path} - {'exists' if raw_path.exists() else 'MISSING'}")
