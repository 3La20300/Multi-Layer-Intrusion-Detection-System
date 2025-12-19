"""
Multi-Layer IDS Prediction Script
=================================

This script takes new raw packet data and predicts:
1. SYN Scan attacks (Network Layer)
2. SQL Injection attacks (Application Layer)

Usage:
    python predict.py <path_to_raw_csv>
    
Example:
    python predict.py data/raw/new_capture.csv
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.preprocess import preprocess_pipeline
from data.preprocess_sqli import preprocess_http_pipeline
from features.build_features import select_features as select_syn_features
from features.build_features_sqli import extract_features_batch as extract_sqli_features


# ============================================================
# CORE PREDICTION FUNCTIONS (Reusable - accept DataFrames)
# ============================================================

def load_model(model_path: str):
    """Load model and return (model, feature_names)."""
    model_data = joblib.load(model_path)
    model = model_data['model'] if isinstance(model_data, dict) else model_data
    feature_names = model_data.get('feature_names', []) if isinstance(model_data, dict) else []
    return model, feature_names


def predict_syn_scan_from_df(aggregated_df: pd.DataFrame, model_path: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Predict SYN scan attacks from already preprocessed/aggregated DataFrame.
    
    Args:
        aggregated_df: DataFrame with aggregated time-window features
        model_path: Path to trained model (default: models/syn_scan_detector.joblib)
        verbose: Whether to print progress messages
    
    Returns:
        DataFrame with predictions added
    """
    # Load model
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "syn_scan_detector.joblib"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model, feature_names = load_model(model_path)
    
    if len(aggregated_df) == 0:
        return pd.DataFrame()
    
    # Select features
    available_features = [col for col in feature_names if col in aggregated_df.columns]
    X = aggregated_df[available_features].fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add predictions to DataFrame
    result_df = aggregated_df.copy()
    result_df['prediction'] = predictions
    result_df['attack_probability'] = probabilities
    result_df['prediction_label'] = result_df['prediction'].map({0: 'Normal', 1: 'SYN Scan'})
    
    return result_df


def predict_sqli_from_df(features_df: pd.DataFrame, model_path: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Predict SQL injection attacks from already extracted features DataFrame.
    
    Args:
        features_df: DataFrame with extracted SQLi features
        model_path: Path to trained model (default: models/sqli_detector.joblib)
        verbose: Whether to print progress messages
    
    Returns:
        DataFrame with predictions added
    """
    # Load model
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "sqli_detector.joblib"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model, feature_names = load_model(model_path)
    
    if len(features_df) == 0:
        return pd.DataFrame()
    
    # Select features
    available_features = [col for col in feature_names if col in features_df.columns]
    X = features_df[available_features].fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add predictions to DataFrame
    result_df = features_df.copy()
    result_df['prediction'] = predictions
    result_df['attack_probability'] = probabilities
    result_df['prediction_label'] = result_df['prediction'].map({0: 'Normal', 1: 'SQL Injection'})
    
    return result_df


# ============================================================
# SYN SCAN DETECTION (File-based wrapper)
# ============================================================


def predict_syn_scan(raw_csv_path: str, model_path: str = None) -> pd.DataFrame:
    """
    Predict SYN scan attacks from raw packet data (file-based wrapper).
    
    Args:
        raw_csv_path: Path to raw CSV file with packet data
        model_path: Path to trained model (default: models/syn_scan_detector.joblib)
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "=" * 50)
    print("SYN SCAN DETECTION")
    print("=" * 50)
    
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "syn_scan_detector.joblib"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("  Run 'python main.py' first to train the model.")
        return None
    
    print(f"Model loaded from {model_path}")
    
    # Preprocess data
    print("\n[1/3] Preprocessing packet data...")
    try:
        aggregated_df = preprocess_pipeline(raw_csv_path)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None
    
    if len(aggregated_df) == 0:
        print("No data after preprocessing")
        return None
    
    print(f"   → {len(aggregated_df)} time windows created")
    
    # Use core prediction function
    print("\n[2/3] Selecting features...")
    print("\n[3/3] Making predictions...")
    result_df = predict_syn_scan_from_df(aggregated_df, model_path)
    
    # Summary
    attack_count = result_df['prediction'].sum()
    total_count = len(result_df)
    
    print("\n" + "-" * 50)
    print("RESULTS:")
    print(f"   Total time windows: {total_count}")
    print(f"   SYN Scan detected:  {attack_count} ({100*attack_count/total_count:.1f}%)")
    print(f"   Normal traffic:     {total_count - attack_count} ({100*(total_count-attack_count)/total_count:.1f}%)")
    
    if attack_count > 0:
        print("\n SYN SCAN ATTACKS DETECTED!")
        print("\nAttack sources:")
        attacks = result_df[result_df['prediction'] == 1]
        for _, row in attacks.head(10).iterrows():
            src = row.get('ip.src', row.get('src_ip', 'Unknown'))
            dst = row.get('ip.dst', row.get('dst_ip', 'Unknown'))
            ports = row.get('unique_dst_ports', 0)
            print(f"   • {src} → {dst} (scanned {int(ports)} ports)")
    
    return result_df


# ============================================================
# SQL INJECTION DETECTION
# ============================================================


def predict_sqli(raw_csv_path: str, model_path: str = None) -> pd.DataFrame:
    """
    Predict SQL injection attacks from raw packet data (file-based wrapper).
    
    Args:
        raw_csv_path: Path to raw CSV file with packet data
        model_path: Path to trained model (default: models/sqli_detector.joblib)
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "=" * 50)
    print("SQL INJECTION DETECTION")
    print("=" * 50)
    
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "sqli_detector.joblib"
    
    if not Path(model_path).exists():
        print(f" Model not found: {model_path}")
        print("   Run 'python main_sqli.py' first to train the model.")
        return None
    
    print(f" Model loaded from {model_path}")
    
    # Preprocess data
    print("\n[1/3] Extracting HTTP requests...")
    try:
        http_df = preprocess_http_pipeline(raw_csv_path)
    except Exception as e:
        print(f" Preprocessing failed: {e}")
        return None
    
    if len(http_df) == 0:
        print(" No HTTP requests found in data")
        return None
    
    print(f"   → {len(http_df)} HTTP requests extracted")
    
    # Extract features
    print("\n[2/3] Extracting SQL injection features...")
    features_df = extract_sqli_features(http_df)
    
    print(f"   → Features extracted")
    
    # Use core prediction function
    print("\n[3/3] Making predictions...")
    result_df = predict_sqli_from_df(features_df, model_path)
    
    # Summary
    attack_count = result_df['prediction'].sum()
    total_count = len(result_df)
    
    print("\n" + "-" * 50)
    print("RESULTS:")
    print(f"   Total HTTP requests: {total_count}")
    print(f"   SQLi detected:       {attack_count} ({100*attack_count/total_count:.1f}%)")
    print(f"   Normal requests:     {total_count - attack_count} ({100*(total_count-attack_count)/total_count:.1f}%)")
    
    if attack_count > 0:
        print("\nSQL INJECTION ATTACKS DETECTED!")
        print("\nSample malicious requests:")
        attacks = result_df[result_df['prediction'] == 1]
        for _, row in attacks.head(5).iterrows():
            src = row.get('ip.src', 'Unknown')
            uri = row.get('decoded_uri', row.get('http.request.uri', 'Unknown'))
            # Truncate long URIs
            if len(str(uri)) > 80:
                uri = str(uri)[:80] + "..."
            print(f"   • [{src}] {uri}")
    
    return result_df


# ============================================================
# UNIFIED PREDICTION
# ============================================================

def predict_all(raw_csv_path: str, output_dir: str = None) -> dict:
    """
    Run both SYN scan and SQL injection detection on raw packet data.
    
    Args:
        raw_csv_path: Path to raw CSV file
        output_dir: Directory to save results (optional)
    
    Returns:
        Dictionary with results for both detectors
    """
    print("\n" + "=" * 60)
    print("   MULTI-LAYER INTRUSION DETECTION SYSTEM")
    print("=" * 60)
    print(f"\nAnalyzing: {raw_csv_path}")
    
    results = {}
    
    # SYN Scan Detection
    syn_results = predict_syn_scan(raw_csv_path)
    if syn_results is not None:
        results['syn_scan'] = syn_results
    
    # SQL Injection Detection
    sqli_results = predict_sqli(raw_csv_path)
    if sqli_results is not None:
        results['sqli'] = sqli_results
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if 'syn_scan' in results:
            syn_out = output_path / "syn_scan_predictions.csv"
            results['syn_scan'].to_csv(syn_out, index=False)
            print(f"\n SYN scan results saved to: {syn_out}")
        
        if 'sqli' in results:
            sqli_out = output_path / "sqli_predictions.csv"
            results['sqli'].to_csv(sqli_out, index=False)
            print(f" SQLi results saved to: {sqli_out}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("   DETECTION SUMMARY")
    print("=" * 60)
    
    if 'syn_scan' in results:
        syn_attacks = results['syn_scan']['prediction'].sum()
        print(f"\n SYN Scan Detection:")
        print(f"   • {syn_attacks} attack(s) detected")
    
    if 'sqli' in results:
        sqli_attacks = results['sqli']['prediction'].sum()
        print(f"\n SQL Injection Detection:")
        print(f"   • {sqli_attacks} attack(s) detected")
    
    print("\n" + "=" * 60)
    
    return results


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """Main entry point for prediction script."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_raw_csv> [output_dir]")
        print("\nExample:")
        print("  python predict.py data/raw/new_capture.csv")
        print("  python predict.py data/raw/new_capture.csv results/")
        print("\nThe input CSV should have columns like:")
        print("  - frame.time_epoch")
        print("  - ip.src, ip.dst")
        print("  - tcp.flags.syn, tcp.flags.ack")
        print("  - tcp.dstport")
        print("  - http.request.uri (for SQL injection detection)")
        return
    
    raw_csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if file exists
    if not Path(raw_csv_path).exists():
        print(f" File not found: {raw_csv_path}")
        return
    
    # Run predictions
    results = predict_all(raw_csv_path, output_dir)
    
    return results


if __name__ == "__main__":
    main()
