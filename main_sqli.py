"""
SQL Injection Detector - Main Entry Point
==========================================

This script runs the complete SQL injection detection pipeline.

Key Concept:
    SQL injection detection works at the APPLICATION layer.
    We analyze HTTP request URIs for malicious patterns like:
    - SQL keywords (SELECT, UNION, DROP, etc.)
    - Comments (--,  #, /*)
    - Boolean tricks (OR 1=1)
    - Special characters (', ", ;)

Usage:
    python main_sqli.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.preprocess_sqli import preprocess_http_pipeline
from features.build_features_sqli import build_sqli_features_pipeline
from models.train_sqli import train_sqli_pipeline
from evaluation.evaluate_sqli import evaluate_sqli_model
from utils.io import save_csv, ensure_dir


def main():
    """
    Main function to run SQL injection detection pipeline.
    """
    print("\n" + "=" * 60)
    print("   SQL INJECTION DETECTOR - Application Layer IDS")
    print("=" * 60)
    print("\nKey Concept:")
    print("  ✓ SQL injection detected from HTTP request patterns")
    print("  ✓ Lexical features (keywords, comments, special chars)")
    print("  ✓ Random Forest learns attack signatures")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent
    raw_data_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    processed_path = project_root / "data" / "processed" / "http_features_labeled.csv"
    model_path = project_root / "models" / "sqli_detector.joblib"
    reports_path = project_root / "reports" / "figures"
    
    # Ensure directories exist
    ensure_dir(str(project_root / "data" / "processed"))
    ensure_dir(str(project_root / "models"))
    ensure_dir(str(reports_path))
    
    # Check if raw data exists
    if not raw_data_path.exists():
        print(f"\n ERROR: Raw data file not found!")
        print(f"   Expected: {raw_data_path}")
        return None, None
    
    # ===========================================
    # STEP 1: Preprocess HTTP Data
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 1: HTTP DATA PREPROCESSING")
    print("=" * 60)
    print("\nExtracting and decoding HTTP request URIs...")
    
    http_df = preprocess_http_pipeline(str(raw_data_path))
    
    if len(http_df) == 0:
        print("\n No HTTP requests found in the data!")
        return None, None
    
    # ===========================================
    # STEP 2: Feature Engineering & Labeling
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 2: SQL INJECTION FEATURE EXTRACTION")
    print("=" * 60)
    print("\nExtracting features from HTTP payloads:")
    print("  • SQL keywords (SELECT, UNION, DROP...)")
    print("  • Comment patterns (--, #, /*)")
    print("  • Boolean tricks (OR 1=1)")
    print("  • Special character counts")
    
    X_train, X_test, y_train, y_test, feature_names, labeled_df = \
        build_sqli_features_pipeline(http_df, test_size=0.3)
    
    # Save labeled data with URI columns included
    clean_columns = ['frame.time_epoch', 'ip.src', 'ip.dst', 'http.request.method', 
                     'http.request.uri', 'decoded_uri', '_ws.col.protocol'] + feature_names + ['label']
    clean_columns = [c for c in clean_columns if c in labeled_df.columns]
    clean_df = labeled_df[clean_columns].copy()
    save_csv(clean_df, str(processed_path))
    
    # ===========================================
    # STEP 3: Model Training
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)
    print("\nTraining Random Forest for SQL injection detection...")
    
    model, train_metrics = train_sqli_pipeline(
        X_train, y_train, X_test, y_test, feature_names,
        model_save_path=str(model_path)
    )
    
    # ===========================================
    # STEP 4: Model Evaluation
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)
    
    eval_results = evaluate_sqli_model(
        model, X_test, y_test, feature_names,
        output_dir=str(reports_path),
        show_plots=False
    )
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("   SQL INJECTION DETECTION - SUMMARY")
    print("=" * 60)
    print(f"\n Data:")
    print(f"   • HTTP requests analyzed: {len(labeled_df)}")
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Testing samples: {len(X_test)}")
    
    print(f"\n Model Performance:")
    metrics = eval_results['metrics']
    print(f"   • Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   • Precision: {metrics['precision']:.2%}")
    print(f"   • Recall:    {metrics['recall']:.2%} (Detection Rate)")
    print(f"   • F1 Score:  {metrics['f1']:.2%}")
    if 'roc_auc' in metrics:
        print(f"   • ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n Saved files:")
    print(f"   • Labeled HTTP data: {processed_path}")
    print(f"   • Trained model: {model_path}")
    print(f"   • Evaluation plots: {reports_path}/")
    
    print("\n" + "=" * 60)
    print("   SQL INJECTION DETECTOR READY!")
    print("=" * 60 + "\n")
    
    return model, eval_results


if __name__ == "__main__":
    model, results = main()
