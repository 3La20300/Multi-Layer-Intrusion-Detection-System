"""
Model Evaluation Module
=======================

This module handles comprehensive evaluation of trained SYN scan detection models.

Metrics used:
- Accuracy: Overall correctness
- Precision: Of predicted attacks, how many are real attacks?
- Recall: Of actual attacks, how many did we detect?
- F1 Score: Balance of precision and recall
- ROC-AUC: Overall model discrimination ability

For SYN scan detection:
- High RECALL is important (don't miss attacks!)
- Precision should also be reasonable (avoid too many false alarms)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
from sklearn.feature_selection import f_classif, SelectKBest
from scipy.stats import spearmanr, pearsonr


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    
    # if y_proba is not None and len(np.unique(y_true)) > 1:
    #     metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall (Detection Rate)',
        'f1': 'F1 Score',
        'roc_auc': 'ROC-AUC'
    }
    
    for key, value in metrics.items():
        name = metric_names.get(key, key)
        bar = "█" * int(value * 30)
        print(f"{name:25s}: {value:.4f} {bar}")
    
    print("=" * 50)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    
    target_names = ['Normal Traffic', 'SYN Scan Attack']
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot confusion matrix.
    
    The confusion matrix shows:
    - True Positives (TP): Correctly detected attacks
    - True Negatives (TN): Correctly identified normal traffic
    - False Positives (FP): Normal traffic flagged as attack
    - False Negatives (FN): Missed attacks (most dangerous!)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix - SYN Scan Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Add interpretation
    tn, fp, fn, tp = cm.ravel()
    plt.figtext(0.5, -0.1, 
                f'TP={tp} (Detected attacks) | TN={tn} (Correct normal)\n'
                f'FP={fp} (False alarms) | FN={fn} (Missed attacks)',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def compute_anova_scores(X: pd.DataFrame, 
                         y: np.ndarray,
                         feature_names: list) -> pd.DataFrame:
    """
    Compute ANOVA F-scores for feature selection.
    
    ANOVA (Analysis of Variance) tests if there's a significant 
    relationship between each feature and the target class.
    
    Higher F-score = more discriminative feature for classification.
    Lower p-value = more statistically significant.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    
    Returns:
        DataFrame with ANOVA results sorted by F-score
    """
    X_filled = X.fillna(0)
    
    # Compute ANOVA F-scores and p-values
    f_scores, p_values = f_classif(X_filled, y)
    
    # Create results DataFrame
    anova_df = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)
    
    # Add significance indicator
    anova_df['significant'] = anova_df['p_value'] < 0.05
    
    return anova_df


def compute_correlation_matrix(X: pd.DataFrame,
                               y: np.ndarray,
                               feature_names: list) -> pd.DataFrame:
    """
    Compute correlation between features and target.
    
    Uses Spearman correlation (works for non-linear relationships).
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    
    Returns:
        DataFrame with correlation results
    """
    X_filled = X.fillna(0)
    correlations = []
    
    for i, feature in enumerate(feature_names):
        corr, p_val = spearmanr(X_filled.iloc[:, i], y)
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_val
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    return corr_df


def plot_anova_scores(anova_df: pd.DataFrame,
                     top_n: int = 10,
                     save_path: Optional[str] = None,
                     show: bool = True):
    """
    Plot ANOVA F-scores for feature selection.
    
    Args:
        anova_df: DataFrame with ANOVA results
        top_n: Number of top features to show
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    if anova_df.empty:
        print("No ANOVA data to plot")
        return
    
    top_features = anova_df.head(top_n)
    
    plt.figure(figsize=(12, 6))
    
    # Color based on significance
    colors = ['#2E7D32' if sig else '#C62828' for sig in top_features['significant']]
    
    bars = plt.barh(range(len(top_features)), top_features['f_score'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('ANOVA F-Score (Higher = More Discriminative)')
    plt.title('ANOVA Feature Selection - SYN Scan Detection\n(Green = p<0.05, Red = Not Significant)')
    plt.gca().invert_yaxis()
    
    # Add value labels with p-values
    for bar, (_, row) in zip(bars, top_features.iterrows()):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'F={row["f_score"]:.2f}, p={row["p_value"]:.2e}',
                va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ANOVA scores plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_correlation_heatmap(X: pd.DataFrame,
                            feature_names: list,
                            save_path: Optional[str] = None,
                            show: bool = True):
    """
    Plot correlation heatmap between features.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    X_filled = X.fillna(0)
    X_filled.columns = feature_names
    
    corr_matrix = X_filled.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0,
                square=True, linewidths=0.5,
                annot_kws={'size': 8})
    
    plt.title('Feature Correlation Heatmap - SYN Scan Detection')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_precision_recall_metrics(metrics: Dict[str, float],
                                  title: str = "SYN Scan Detection",
                                  save_path: Optional[str] = None,
                                  show: bool = True):
    """
    Plot precision, recall, F1 and accuracy as a bar chart.
    
    Args:
        metrics: Dictionary with precision, recall, f1, accuracy
        title: Title for the plot
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    # Metrics to plot
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    values = [metrics.get(k, 0) for k in metric_keys]
    
    # Colors: green gradient for good metrics
    colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2%}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Precision & Recall Metrics - {title}', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Score')
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend explaining metrics
    legend_text = "Precision: Of predicted attacks, how many are real?\n" \
                  "Recall: Of actual attacks, how many did we detect?"
    ax.text(0.5, -0.15, legend_text, transform=ax.transAxes,
            ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Precision/Recall metrics saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 10,
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    Plot feature importance from the model.
    
    This shows which features are most important for detection:
    - syn_ack_ratio should be high (key indicator!)
    - syn_count and unique_dst_ports should also be important
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    if importance_df.empty:
        print("No feature importance data to plot")
        return
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features)))
    
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - SYN Scan Detection')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels
    for bar, val in zip(bars, top_features['importance']):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def evaluate_model(model, 
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  feature_names: list,
                  output_dir: Optional[str] = None,
                  show_plots: bool = True) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    
    Returns:
        Dictionary with all evaluation results
    """
    print("=" * 50)
    print("SYN SCAN DETECTION - Model Evaluation")
    print("=" * 50)
    
    # Make predictions
    X_test = X_test.fillna(0)
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics)
    
    # Print classification report
    print_classification_report(y_test, y_pred)
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = str(output_path / "confusion_matrix.png") if output_dir else None
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path, show=show_plots)
    
    # Plot Precision & Recall metrics
    pr_path = str(output_path / "precision_recall_metrics.png") if output_dir else None
    plot_precision_recall_metrics(metrics, title="SYN Scan Detection", save_path=pr_path, show=show_plots)
    
    # ANOVA Analysis for Feature Selection
    print("\n" + "=" * 50)
    print("ANOVA FEATURE SELECTION ANALYSIS")
    print("=" * 50)
    anova_df = compute_anova_scores(X_test, y_test, feature_names)
    print("\nTop 10 Features by ANOVA F-Score:")
    print(anova_df.head(10).to_string(index=False))
    
    anova_path = str(output_path / "anova_feature_scores.png") if output_dir else None
    plot_anova_scores(anova_df, save_path=anova_path, show=show_plots)
    
    # Correlation Analysis
    print("\n" + "=" * 50)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 50)
    corr_df = compute_correlation_matrix(X_test, y_test, feature_names)
    print("\nFeature-Target Correlations:")
    print(corr_df.head(10).to_string(index=False))
    
    corr_path = str(output_path / "correlation_heatmap.png") if output_dir else None
    plot_correlation_heatmap(X_test, feature_names, save_path=corr_path, show=show_plots)
    
    # Get and plot feature importance
    importance_df = pd.DataFrame()
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fi_path = str(output_path / "feature_importance.png") if output_dir else None
        plot_feature_importance(importance_df, save_path=fi_path, show=show_plots)
    
    results = {
        'metrics': metrics,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': importance_df,
        'anova_scores': anova_df,
        'correlations': corr_df,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocess import preprocess_pipeline
    from features.build_features import build_features_pipeline
    from models.train import train_model
    
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    output_dir = project_root / "reports" / "figures"
    
    if raw_path.exists():
        # Preprocess
        aggregated = preprocess_pipeline(str(raw_path))
        
        # Build features
        X_train, X_test, y_train, y_test, feature_names, _ = \
            build_features_pipeline(aggregated)
        
        # Train
        model = train_model(X_train, y_train, "random_forest")
        
        # Evaluate
        results = evaluate_model(
            model, X_test, y_test, feature_names,
            output_dir=str(output_dir),
            show_plots=False
        )
    else:
        print(f"Raw data file not found: {raw_path}")
