"""
SQL Injection Feature Building Module
=====================================

This module handles feature extraction for SQL injection detection.

Key Concept:
    SQL injection attacks contain specific patterns:
    - SQL keywords (SELECT, UNION, DROP, etc.)
    - Comments (--,  #, /*)
    - Boolean tricks (OR 1=1, OR 'a'='a')
    - Special characters (', ", ;, =, %)
    
    We extract lexical and statistical features from HTTP URIs.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List


# SQL Injection patterns
SQL_KEYWORDS = r"(select|union|drop|insert|update|delete|where|from|into|values|table|database|schema|concat|group_concat|load_file|outfile|benchmark|sleep|waitfor|case|when|having|order|by|limit|and|or|xor|not|null|like|in|between|exists|exec|execute|xp_cmdshell|information_schema)"
COMMENTS = r"(--|#|/\*|\*/)" # SQL comment indicators: --, #, /*, */
# Boolean injection patterns : OR 1=1, OR 'a'='a', AND 1=1, etc.
OR_TRUE = r"('|\")?\s*(or|and)\s+('?1'?\s*=\s*'?1'?|'?true'?|'a'\s*=\s*'a')"
QUOTES = r"('|\")"
ENCODED_CHARS = r"(%27|%22|%3d|%3b|%2d%2d|%23)"  # URL-encoded special chars


def extract_sql_features(payload: str) -> pd.Series:
    """
    Extract features from a single HTTP payload/URI.
    
    Features extracted:
    - has_sql_keywords: Contains SQL keywords (SELECT, UNION, etc.)
    - has_comment: Contains SQL comment indicators
    - has_or_true: Contains boolean injection patterns
    - has_quotes: Contains quote characters
    - special_char_count: Count of special characters
    - payload_length: Length of the payload
    - digit_ratio: Ratio of digits in payload
    - keyword_count: Number of SQL keywords found
    - has_encoded_chars: Contains URL-encoded attack chars
    
    Args:
        payload: HTTP request URI string
    
    Returns:
        Series with extracted features
    """
    payload = str(payload).lower()
    
    # Find all SQL keywords
    keywords_found = re.findall(SQL_KEYWORDS, payload, re.IGNORECASE)
    
    return pd.Series({
        # Binary indicators
        'has_sql_keywords': int(bool(re.search(SQL_KEYWORDS, payload, re.IGNORECASE))),
        'has_comment': int(bool(re.search(COMMENTS, payload))),
        'has_or_true': int(bool(re.search(OR_TRUE, payload, re.IGNORECASE))),
        'has_quotes': int(bool(re.search(QUOTES, payload))),
        'has_encoded_chars': int(bool(re.search(ENCODED_CHARS, payload, re.IGNORECASE))),
        
        # Count features
        'special_char_count': sum(payload.count(c) for c in "'\";=%()"),
        'keyword_count': len(keywords_found),
        
        # Length features
        'payload_length': len(payload),
        'param_count': payload.count('&') + payload.count('?'),
        
        # Ratio features
        'digit_ratio': sum(c.isdigit() for c in payload) / (len(payload) + 1),
        'special_ratio': sum(payload.count(c) for c in "'\";=%()") / (len(payload) + 1),
    })


def extract_features_batch(df: pd.DataFrame, uri_column: str = 'decoded_uri') -> pd.DataFrame:
    """
    Extract SQL injection features from all HTTP requests.
    
    Args:
        df: DataFrame with HTTP requests
        uri_column: Column containing the URI to analyze
    
    Returns:
        DataFrame with original data + extracted features
    """
    print(f"Extracting features from {len(df)} HTTP requests...")
    
    # Reset index first to ensure alignment
    df = df.reset_index(drop=True)
    
    # Extract features for each URI
    features = df[uri_column].apply(extract_sql_features)
    
    # Combine with original data (indices now match)
    result = pd.concat([df, features], axis=1)
    
    print("Feature extraction complete!")
    return result


def create_labels(df: pd.DataFrame,
                 keyword_required: bool = True,
                 special_char_threshold: int = 2) -> pd.DataFrame:
    """
    Create labels for SQL injection detection.
    
    Labeling Logic:
    A request is labeled as SQL injection (label=1) if:
    - Contains SQL keywords AND has special characters > threshold
    OR
    - Contains OR 1=1 pattern (definite injection)
    OR
    - Contains SQL comments with keywords
    
    Args:
        df: DataFrame with extracted features
        keyword_required: Require SQL keywords for positive label
        special_char_threshold: Min special chars required
    
    Returns:
        DataFrame with 'label' column
    """
    df = df.copy()
    
    # Rule 1: SQL keywords + special characters
    condition1 = (df['has_sql_keywords'] == 1) & \
                 (df['special_char_count'] > special_char_threshold)
    
    # Rule 2: OR 1=1 pattern (definite injection attempt)
    condition2 = df['has_or_true'] == 1
    
    # Rule 3: Comments with SQL keywords (often used in injection)
    condition3 = (df['has_comment'] == 1) & (df['has_sql_keywords'] == 1)
    
    # Combine rules
    df['label'] = ((condition1) | (condition2) | (condition3)).astype(int)
    
    # Print distribution
    attack_count = df['label'].sum()
    normal_count = len(df) - attack_count
    print(f"\nLabel distribution:")
    print(f"  Normal (0): {normal_count} ({100*normal_count/len(df):.1f}%)")
    print(f"  SQLi Attack (1): {attack_count} ({100*attack_count/len(df):.1f}%)")
    
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features for ML model.
    
    Args:
        df: Labeled DataFrame
    
    Returns:
        Tuple of (feature DataFrame, feature names list)
    """
    feature_columns = [
        'has_sql_keywords',
        'has_comment',
        'has_or_true',
        'has_quotes',
        'has_encoded_chars',
        'special_char_count',
        'keyword_count',
        'payload_length',
        'param_count',
        'digit_ratio',
        'special_ratio'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"\nSelected {len(available_features)} features:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i}. {feat}")
    
    return df[available_features], available_features


def prepare_train_test_data(df: pd.DataFrame,
                           test_size: float = 0.3,
                           random_state: int = 42) -> Tuple:
    """
    Prepare data for training and testing.
    
    Args:
        df: Labeled DataFrame with features
        test_size: Fraction of data for testing
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    from sklearn.model_selection import train_test_split
    
    X, feature_names = select_features(df)
    y = df['label']
    
    # Handle NaN values
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


def build_sqli_features_pipeline(http_df: pd.DataFrame,
                                 special_char_threshold: int = 2,
                                 test_size: float = 0.3) -> Tuple:
    """
    Complete feature building pipeline for SQL injection detection.
    
    Args:
        http_df: Preprocessed HTTP DataFrame
        special_char_threshold: Threshold for labeling
        test_size: Test set fraction
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, labeled_df)
    """
    print("=" * 50)
    print("SQL INJECTION - Feature Engineering")
    print("=" * 50)
    
    # Step 1: Extract features
    print("\n[1/3] Extracting SQL injection features...")
    df_features = extract_features_batch(http_df)
    
    # Step 2: Create labels
    print("\n[2/3] Creating labels...")
    labeled_df = create_labels(df_features, 
                               special_char_threshold=special_char_threshold)
    
    # Step 3: Prepare train/test data
    print("\n[3/3] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_names = prepare_train_test_data(
        labeled_df, test_size=test_size
    )
    
    print("\n" + "=" * 50)
    print("Feature engineering complete!")
    print("=" * 50)
    
    return X_train, X_test, y_train, y_test, feature_names, labeled_df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocess_sqli import preprocess_http_pipeline
    
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    
    if raw_path.exists():
        # Preprocess HTTP data
        http_df = preprocess_http_pipeline(str(raw_path))
        
        # Build features
        X_train, X_test, y_train, y_test, feature_names, labeled_df = \
            build_sqli_features_pipeline(http_df)
        
        print("\nSample SQL injection attempts:")
        attacks = labeled_df[labeled_df['label'] == 1]
        if len(attacks) > 0:
            print(attacks[['decoded_uri', 'has_sql_keywords', 'special_char_count', 'label']].head())
    else:
        print(f"Raw data file not found: {raw_path}")
