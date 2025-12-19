"""
SQL Injection Model Evaluation Module
=====================================

This module handles evaluation of SQL injection detection models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import f_classif
from scipy.stats import spearmanr


def evaluate_sqli_model(model,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       feature_names: list,
                       output_dir: Optional[str] = None,
                       show_plots: bool = False) -> Dict[str, Any]:
    """
    Evaluate SQL injection detection model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    
    Returns:
        Dictionary with evaluation results
    """
    print("=" * 50)
    print("SQL INJECTION - Model Evaluation")
    print("=" * 50)
    
    # Predictions
    X_test = X_test.fillna(0)
    y_pred = model.predict(X_test)
    
    # Probabilities
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_proba is not None and len(np.unique(y_test)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    # Print metrics
    print("\n" + "=" * 50)
    print("SQL Injection Detection Performance")
    print("=" * 50)
    for key, value in metrics.items():
        bar = "█" * int(value * 30)
        print(f"{key:25s}: {value:.4f} {bar}")
    print("=" * 50)
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 50)
    target_names = ['Normal Request', 'SQL Injection']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ANOVA Analysis
    print("\n" + "=" * 50)
    print("ANOVA FEATURE SELECTION ANALYSIS")
    print("=" * 50)
    X_filled = X_test.fillna(0)
    f_scores, p_values = f_classif(X_filled, y_test)
    
    anova_df = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'p_value': p_values,
        'significant': p_values < 0.05
    }).sort_values('f_score', ascending=False)
    
    print("\nTop 10 Features by ANOVA F-Score:")
    print(anova_df.head(10).to_string(index=False))
    
    # Correlation Analysis
    print("\n" + "=" * 50)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 50)
    correlations = []
    for i, feature in enumerate(feature_names):
        corr, p_val = spearmanr(X_filled.iloc[:, i], y_test)
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_val
        })
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    print("\nFeature-Target Correlations:")
    print(corr_df.head(10).to_string(index=False))
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Normal', 'SQLi'],
                    yticklabels=['Normal', 'SQLi'])
        plt.title('Confusion Matrix - SQL Injection Detection')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(output_path / "sqli_confusion_matrix.png", dpi=150)
        print(f"\nConfusion matrix saved to: {output_path / 'sqli_confusion_matrix.png'}")
        
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot Precision & Recall metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        values = [metrics.get(k, 0) for k in metric_keys]
        colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.2)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision & Recall Metrics - SQL Injection Detection', fontsize=14, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        legend_text = "Precision: Of predicted attacks, how many are real?\\n" \
                      "Recall: Of actual attacks, how many did we detect?"
        ax.text(0.5, -0.15, legend_text, transform=ax.transAxes,
                ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(output_path / "sqli_precision_recall_metrics.png", dpi=150, bbox_inches='tight')
        print(f"Precision/Recall metrics saved to: {output_path / 'sqli_precision_recall_metrics.png'}")
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot ANOVA scores
        top_anova = anova_df.head(10)
        plt.figure(figsize=(12, 6))
        colors = ['#2E7D32' if sig else '#C62828' for sig in top_anova['significant']]
        bars = plt.barh(range(len(top_anova)), top_anova['f_score'], color=colors)
        plt.yticks(range(len(top_anova)), top_anova['feature'])
        plt.xlabel('ANOVA F-Score (Higher = More Discriminative)')
        plt.title('ANOVA Feature Selection - SQL Injection Detection\n(Green = p<0.05, Red = Not Significant)')
        plt.gca().invert_yaxis()
        for bar, (_, row) in zip(bars, top_anova.iterrows()):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'F={row["f_score"]:.2f}, p={row["p_value"]:.2e}',
                    va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path / "sqli_anova_scores.png", dpi=150)
        print(f"ANOVA scores saved to: {output_path / 'sqli_anova_scores.png'}")
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot correlation heatmap
        X_filled.columns = feature_names
        corr_matrix = X_filled.corr()
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True, linewidths=0.5,
                    annot_kws={'size': 8})
        plt.title('Feature Correlation Heatmap - SQL Injection Detection')
        plt.tight_layout()
        plt.savefig(output_path / "sqli_correlation_heatmap.png", dpi=150)
        print(f"Correlation heatmap saved to: {output_path / 'sqli_correlation_heatmap.png'}")
        if show_plots:
            plt.show()
        plt.close()
        
        # Feature importance plot
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(importance)))
            plt.barh(range(len(importance)), importance['importance'], color=colors)
            plt.yticks(range(len(importance)), importance['feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance - SQL Injection Detection')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path / "sqli_feature_importance.png", dpi=150)
            print(f"Feature importance saved to: {output_path / 'sqli_feature_importance.png'}")
            
            if show_plots:
                plt.show()
            plt.close()
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'anova_scores': anova_df,
        'correlations': corr_df,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    print("\n" + "=" * 50)
    print("SQL Injection evaluation complete!")
    print("=" * 50)
    
    return results
