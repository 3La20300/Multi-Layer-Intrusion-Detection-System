"""
Model Training Module
=====================

This module handles training ML models for SYN scan detection.

Recommended Models:
- Random Forest: Handles ratios & thresholds well, good interpretability
- XGBoost: Best accuracy for tabular data
- Isolation Forest: For anomaly detection without labels

Key Insight:
    The model learns patterns like:
    - syn_count >> ack_count → attack
    - High unique_dst_ports + high syn_count → port scan
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler


def create_random_forest(n_estimators: int = 100,
                        max_depth: int = 10,
                        min_samples_split: int = 5,
                        min_samples_leaf: int = 2,
                        random_state: int = 42,
                        class_weight: str = "balanced") -> RandomForestClassifier:
    """
    Create Random Forest classifier for SYN scan detection.
    
    Random Forest is ideal because:
    - Handles non-linear relationships (syn_count vs ack_count ratio)
    - Robust to outliers
    - Provides feature importance for interpretability
    - Works well with imbalanced data (using class_weight)
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed
        class_weight: Handle imbalanced classes
    
    Returns:
        Configured RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )


def create_xgboost(n_estimators: int = 100,
                   max_depth: int = 6,
                   learning_rate: float = 0.1,
                   random_state: int = 42,
                   scale_pos_weight: float = 1.0):
    """
    Create XGBoost classifier for SYN scan detection.
    
    XGBoost often provides the best accuracy for:
    - Tabular data with mixed features
    - Complex decision boundaries
    
    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        random_state: Random seed
        scale_pos_weight: Balance positive/negative classes
    
    Returns:
        Configured XGBClassifier
    """
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    except ImportError:
        print("XGBoost not installed. Using Random Forest instead.")
        return create_random_forest()


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                model_type: str = "random_forest",
                model_params: Optional[Dict] = None) -> Any:
    """
    Train a model for SYN scan detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: "random_forest" or "xgboost"
        model_params: Optional model parameters
    
    Returns:
        Trained model
    """
    print(f"\nTraining {model_type} model...")
    
    # Default parameters
    params = model_params or {}
    
    # Create model
    if model_type == "random_forest":
        model = create_random_forest(**params)
    elif model_type == "xgboost":
        model = create_xgboost(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    print(f"Model trained successfully!")
    return model


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                        cv: int = 5) -> Dict[str, float]:
    """
    Perform cross-validation on the model.
    
    Args:
        model: Trained or untrained model
        X: Feature matrix
        y: Labels
        cv: Number of folds
    
    Returns:
        Dict with mean and std of scores
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    result = {
        'cv_mean_f1': scores.mean(),
        'cv_std_f1': scores.std(),
        'cv_scores': scores
    }
    
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return result


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    This helps explain WHY the model detects SYN scans:
    - Which features are most important?
    - Does it match our domain knowledge?
    
    Expected important features:
    - syn_ack_ratio (high = incomplete connections)
    - syn_count (high = many SYN packets)
    - unique_dst_ports (high = port scanning)
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importances:")
        print("-" * 40)
        for _, row in importance.head(10).iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"{row['feature']:25s} {row['importance']:.4f} {bar}")
        
        return importance
    else:
        print("Model does not have feature_importances_ attribute")
        return pd.DataFrame()


def save_model(model, filepath: str, feature_names: list = None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
        feature_names: Optional list of feature names
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    
    joblib.dump(model_data, filepath)
    print(f"\nModel saved to: {filepath}")


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


def train_pipeline(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  feature_names: list,
                  model_type: str = "random_forest",
                  model_save_path: Optional[str] = None) -> Tuple[Any, Dict]:
    """
    Complete training pipeline.
    
    Pipeline steps:
    1. Train model
    2. Cross-validate
    3. Get feature importance
    4. Save model (optional)
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: Feature names
        model_type: Type of model to train
        model_save_path: Path to save model (optional)
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print("=" * 50)
    print("SYN SCAN DETECTION - Model Training")
    print("=" * 50)
    
    # Train model
    model = train_model(X_train, y_train, model_type)
    
    # Cross-validation
    cv_results = cross_validate_model(model, X_train, y_train)
    
    # Feature importance
    importance = get_feature_importance(model, feature_names)
    
    # Save model
    if model_save_path:
        save_model(model, model_save_path, feature_names)
    
    metrics = {
        **cv_results,
        'feature_importance': importance
    }
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    return model, metrics


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocess import preprocess_pipeline
    from features.build_features import build_features_pipeline
    
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    model_path = project_root / "models" / "syn_scan_detector.joblib"
    
    if raw_path.exists():
        # Preprocess
        aggregated = preprocess_pipeline(str(raw_path))
        
        # Build features
        X_train, X_test, y_train, y_test, feature_names, _ = \
            build_features_pipeline(aggregated)
        
        # Train
        model, metrics = train_pipeline(
            X_train, y_train, X_test, y_test, feature_names,
            model_type="random_forest",
            model_save_path=str(model_path)
        )
    else:
        print(f"Raw data file not found: {raw_path}")
