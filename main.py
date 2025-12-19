"""
SYN Scan Detector - Main Entry Point
====================================

This is the main script to run the complete SYN scan detection pipeline.

Usage:
    python main.py

Pipeline Steps:
1. Load raw packet data from CSV
2. Preprocess and aggregate into time windows
3. Create labels using heuristic rules
4. Train Random Forest classifier
5. Evaluate model performance
6. Save model and results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.preprocess import preprocess_pipeline
from features.build_features import build_features_pipeline
from models.train import train_pipeline
from evaluation.evaluate import evaluate_model
from utils.io import load_config, save_csv, ensure_dir


def main():
    """
    Main function to run the complete SYN scan detection pipeline.
    """
    print("\n" + "=" * 60)
    print("   SYN SCAN DETECTOR - ML-Based Detection System")
    print("=" * 60)
    print("\nKey Concept:")
    print("   SYN scanning is detected from PATTERNS, not single packets")
    print("   Time-window aggregation captures temporal behavior")
    print("   Random Forest learns the attack signatures")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent
    config_path = project_root / "configs" / "config.yaml"
    raw_data_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    processed_path = project_root / "data" / "processed" / "aggregated_features.csv"
    model_path = project_root / "models" / "syn_scan_detector.joblib"
    reports_path = project_root / "reports" / "figures"
    
    # Ensure directories exist
    ensure_dir(str(project_root / "data" / "processed"))
    ensure_dir(str(project_root / "models"))
    ensure_dir(str(reports_path))
    
    # Check if raw data exists
    if not raw_data_path.exists():
        print(f"\n  ERROR: Raw data file not found!")
        print(f"   Expected: {raw_data_path}")
        print("\nPlease place your raw packet CSV file in the data/raw/ directory.")
        return
    
    # Load configuration
    if config_path.exists():
        config = load_config(str(config_path))
        window_seconds = config.get('aggregation', {}).get('time_window_seconds', 1)
        syn_threshold = config.get('labeling', {}).get('syn_count_threshold', 5)
        ports_threshold = config.get('labeling', {}).get('unique_ports_threshold', 3)
        model_type = config.get('model', {}).get('type', 'random_forest')
    else:
        # Default values
        window_seconds = 1
        syn_threshold = 5
        ports_threshold = 3
        model_type = 'random_forest'
    
    # ===========================================
    # STEP 1: Preprocess Data
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    print(f"\nAggregating packets into {window_seconds}-second time windows...")
    print("This transforms raw packets → aggregated features")
    
    aggregated_df = preprocess_pipeline(
        str(raw_data_path),
        window_seconds=window_seconds,
        min_packets=2
    )
    
    # Save processed data
    save_csv(aggregated_df, str(processed_path))
    
    # ===========================================
    # STEP 2: Feature Engineering & Labeling
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING & LABELING")
    print("=" * 60)
    print(f"\nLabeling rules:")
    print(f"  • SYN count > {syn_threshold}")
    print(f"  • Unique ports > {ports_threshold}")
    print(f"  • High SYN/ACK ratio")
    
    X_train, X_test, y_train, y_test, feature_names, labeled_df = \
        build_features_pipeline(
            aggregated_df,
            syn_threshold=syn_threshold,
            ports_threshold=ports_threshold,
            test_size=0.2
        )
    
    # ===========================================
    # STEP 3: Model Training
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)
    print(f"\nTraining {model_type} classifier...")
    print("The model learns patterns like:")
    print("  • syn_count >> ack_count → attack")
    print("  • Many unique ports + SYN packets → port scan")
    
    model, train_metrics = train_pipeline(
        X_train, y_train, X_test, y_test, feature_names,
        model_type=model_type,
        model_save_path=str(model_path)
    )
    
    # ===========================================
    # STEP 4: Model Evaluation
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)
    
    eval_results = evaluate_model(
        model, X_test, y_test, feature_names,
        output_dir=str(reports_path),
        show_plots=False  # Set to True to display plots
    )
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"\n  Data:")
    print(f"   • Raw packets processed: {len(aggregated_df)} time windows created")
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
    
    print(f"\n  Saved files:")
    print(f"   • Processed data: {processed_path}")
    print(f"   • Trained model:  {model_path}")
    print(f"   • Evaluation plots: {reports_path}/")
    
    print("\n" + "=" * 60)
    print("   SYN SCAN DETECTOR READY!")
    print("=" * 60)
    print("\nTo make predictions on new data, use:")
    print("  from src.models.predict import predict_on_new_traffic")
    print("  results, analysis = predict_on_new_traffic(model_path, new_data_path)")
    print("\n")
    
    return model, eval_results


if __name__ == "__main__":
    model, results = main()
