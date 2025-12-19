"""
SQL Injection Model Training Module
===================================

This module handles training ML models for SQL injection detection.

Key Features Used:
- has_sql_keywords: Contains SQL keywords
- has_comment: Contains SQL comments
- has_or_true: Contains OR 1=1 pattern
- special_char_count: Count of special characters
- payload_length: Length of payload
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def create_sqli_model(n_estimators: int = 100,
                     max_depth: int = 10,
                     random_state: int = 42,
                     class_weight: str = "balanced") -> RandomForestClassifier:
    """
    Create Random Forest classifier for SQL injection detection.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        class_weight: Handle imbalanced classes
    
    Returns:
        Configured RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )


def train_sqli_model(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    model_params: Optional[Dict] = None) -> RandomForestClassifier:
    """
    Train SQL injection detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_params: Optional model parameters
    
    Returns:
        Trained model
    """
    print("\nTraining SQL injection detection model...")
    
    params = model_params or {}
    model = create_sqli_model(**params)
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")
    return model


def cross_validate_sqli(model, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5) -> Dict[str, float]:
    """
    Perform cross-validation.
    
    Args:
        model: Model to validate
        X: Feature matrix
        y: Labels
        cv: Number of folds
    
    Returns:
        Dict with CV scores
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
    
    For SQL injection, important features should be:
    - has_sql_keywords
    - special_char_count
    - has_or_true
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importances (SQL Injection):")
        print("-" * 40)
        for _, row in importance.head(10).iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"{row['feature']:25s} {row['importance']:.4f} {bar}")
        
        return importance
    return pd.DataFrame()


def save_sqli_model(model, filepath: str, feature_names: list = None):
    """
    Save trained SQL injection model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save
        feature_names: Feature names
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'model_type': 'sqli_detector'
    }
    
    joblib.dump(model_data, filepath)
    print(f"\nSQL Injection model saved to: {filepath}")


def load_sqli_model(filepath: str) -> Tuple[Any, list]:
    """
    Load SQL injection model from disk.
    
    Args:
        filepath: Path to saved model
    
    Returns:
        Tuple of (model, feature_names)
    """
    model_data = joblib.load(filepath)
    print(f"SQL Injection model loaded from: {filepath}")
    return model_data['model'], model_data.get('feature_names', [])


def train_sqli_pipeline(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       feature_names: list,
                       model_save_path: Optional[str] = None) -> Tuple[Any, Dict]:
    """
    Complete SQL injection model training pipeline.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: Feature names
        model_save_path: Path to save model
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print("=" * 50)
    print("SQL INJECTION - Model Training")
    print("=" * 50)
    
    # Train model
    model = train_sqli_model(X_train, y_train)
    
    # Cross-validation
    cv_results = cross_validate_sqli(model, X_train, y_train)
    
    # Feature importance
    importance = get_feature_importance(model, feature_names)
    
    # Save model
    if model_save_path:
        save_sqli_model(model, model_save_path, feature_names)
    
    metrics = {
        **cv_results,
        'feature_importance': importance
    }
    
    print("\n" + "=" * 50)
    print("SQL Injection model training complete!")
    print("=" * 50)
    
    return model, metrics


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocess_sqli import preprocess_http_pipeline
    from features.build_features_sqli import build_sqli_features_pipeline
    
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    model_path = project_root / "models" / "sqli_detector.joblib"
    
    if raw_path.exists():
        # Preprocess
        http_df = preprocess_http_pipeline(str(raw_path))
        
        # Build features
        X_train, X_test, y_train, y_test, feature_names, _ = \
            build_sqli_features_pipeline(http_df)
        
        # Train
        model, metrics = train_sqli_pipeline(
            X_train, y_train, X_test, y_test, feature_names,
            model_save_path=str(model_path)
        )
    else:
        print(f"Raw data file not found: {raw_path}")
