"""
Feature Building Module
=======================

This module handles feature engineering and labeling for SYN scan detection.

Key Concept:
    SYN scan detection relies on temporal patterns, not single packets.
    Features are extracted from aggregated time windows to capture:
    - High SYN count with low ACK count
    - Many unique destination ports (port scanning)
    - Very short time deltas (burst traffic)
    - Low connection completion rate
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def create_labels(df: pd.DataFrame,
                 syn_threshold: int = 5,
                 ports_threshold: int = 3,
                 syn_ack_ratio_threshold: float = 3.0,
                 use_combination: bool = True) -> pd.DataFrame:
    """
    Create labels for SYN scan detection using heuristic rules.
    
    SYN Scan Detection Logic:
    A time window is labeled as SYN scan (attack=1) if:
    - syn_count > threshold AND unique_ports > threshold
    OR
    - syn_ack_ratio > ratio_threshold (many SYNs, few ACKs)
    
    This mimics what a SYN scan looks like:
    - Many SYN packets sent rapidly
    - To many different ports
    - Without completing TCP handshakes (no ACKs back)
    
    Args:
        df: Aggregated DataFrame from preprocessing
        syn_threshold: Minimum SYN packets to consider attack
        ports_threshold: Minimum unique ports to consider port scan
        syn_ack_ratio_threshold: SYN/ACK ratio threshold
        use_combination: Use AND combination of rules
    
    Returns:
        DataFrame with 'label' column (1=attack, 0=normal)
    """
    df = df.copy()
    
    if use_combination:
        # Rule: High SYN count AND many unique ports
        # This is the classic port scanning pattern
        condition1 = (df['syn_count'] > syn_threshold) & \
                     (df['unique_dst_ports'] > ports_threshold)
        
        # Rule: Very high SYN/ACK ratio (incomplete connections)
        condition2 = df['syn_ack_ratio'] > syn_ack_ratio_threshold
        
        # Combine with OR (either pattern indicates attack)
        df['label'] = ((condition1) | (condition2)).astype(int)
    else:
        # Simpler rule: just high SYN count
        df['label'] = (df['syn_count'] > syn_threshold).astype(int)
    
    # Print label distribution
    attack_count = df['label'].sum()
    normal_count = len(df) - attack_count
    print(f"\nLabel distribution:")
    print(f"  Normal (0): {normal_count} ({100*normal_count/len(df):.1f}%)")
    print(f"  Attack (1): {attack_count} ({100*attack_count/len(df):.1f}%)")
    
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features for ML model.
    
    These features capture the temporal patterns of SYN scanning:
    
    Primary Detection Features:
    - syn_count: High in SYN scan
    - ack_count: Low in SYN scan
    - syn_ack_ratio: High in SYN scan (key indicator!)
    - unique_dst_ports: High in port scanning
    
    Secondary Features:
    - rst_count: High when probing closed ports
    - packet_count: Traffic volume
    - packets_per_second: Burst traffic indicator
    - completion_ratio: Low in SYN scan
    
    Args:
        df: Labeled DataFrame
    
    Returns:
        Tuple of (feature DataFrame, feature names list)
    """
    feature_columns = [
        # Primary SYN scan indicators
        'syn_count',
        'ack_count', 
        'syn_ack_ratio',
        'unique_dst_ports',
        
        # Secondary indicators
        'rst_count',
        'fin_count',
        'unique_src_ports',
        'packet_count',
        'total_bytes', # if high traffic volume means scanning
        
        # Derived features
        'syn_percentage',
        'completion_ratio', # low = incomplete connections (possible scan)
        'ports_per_packet',
        'rst_ratio', # high = many closed ports probed , rst/total packets
        
        # Timing features
        'mean_time_delta',
        'packets_per_second',
        'mean_packet_size'
    ]
    
    # Only include columns that exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"\nSelected {len(available_features)} features for training:")
    for i, feat in enumerate(available_features, 1): 
        print(f"  {i}. {feat}")
    
    return df[available_features], available_features


def prepare_train_test_data(df: pd.DataFrame,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple:
    """
    Prepare data for training and testing.
    
    Args:
        df: Labeled DataFrame with features
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    from sklearn.model_selection import train_test_split
    
    # Get features and labels
    X, feature_names = select_features(df)
    y = df['label']
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Train attack ratio: {y_train.mean():.2%}")
    print(f"  Test attack ratio: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_names


def build_features_pipeline(aggregated_df: pd.DataFrame,
                           syn_threshold: int = 5,
                           ports_threshold: int = 3,
                           test_size: float = 0.2) -> Tuple:
    """
    Complete feature building pipeline.
    
    Pipeline steps:
    1. Create labels using heuristic rules
    2. Select relevant features
    3. Split into train/test sets
    
    Args:
        aggregated_df: Preprocessed aggregated DataFrame
        syn_threshold: SYN count threshold for labeling
        ports_threshold: Unique ports threshold for labeling
        test_size: Test set fraction
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, labeled_df)
    """
    print("=" * 50)
    print("SYN SCAN DETECTION - Feature Engineering")
    print("=" * 50)
    
    # Step 1: Create labels
    print("\n[1/2] Creating labels using heuristic rules...")
    labeled_df = create_labels(
        aggregated_df,
        syn_threshold=syn_threshold,
        ports_threshold=ports_threshold
    )
    
    # Step 2: Prepare train/test data
    print("\n[2/2] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_names = prepare_train_test_data(
        labeled_df, test_size=test_size
    )
    
    print("\n" + "=" * 50)
    print("Feature engineering complete!")
    print("=" * 50)
    
    return X_train, X_test, y_train, y_test, feature_names, labeled_df


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocess import preprocess_pipeline
    
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    
    if raw_path.exists():
        # Preprocess
        aggregated = preprocess_pipeline(str(raw_path))
        
        # Build features
        X_train, X_test, y_train, y_test, feature_names, labeled_df = \
            build_features_pipeline(aggregated)
        
        print("\nFeature matrix shape:", X_train.shape)
        print("Sample features:")
        print(X_train.head())
    else:
        print(f"Raw data file not found: {raw_path}")
