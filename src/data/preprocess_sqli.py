"""
SQL Injection Data Preprocessing Module
=======================================

This module handles loading and preprocessing HTTP request data
for SQL injection detection.

Key Concept:
    SQL injection detection works at the APPLICATION layer.
    We analyze HTTP request URIs for malicious patterns.
"""

import pandas as pd
import numpy as np
from urllib.parse import unquote
from pathlib import Path
from typing import Optional


def load_raw_data(filepath: str, on_bad_lines: str = 'skip') -> pd.DataFrame:
    """
    Load raw packet data from CSV file.
    
    Args:
        filepath: Path to the raw CSV file
        on_bad_lines: How to handle malformed lines
    
    Returns:
        DataFrame with raw packet data
    """
    df = pd.read_csv(filepath, on_bad_lines=on_bad_lines)
    print(f"Loaded {len(df)} packets from {filepath}")
    return df


def extract_http_requests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract HTTP requests from packet data.
    
    Filters only rows that contain HTTP request URIs.
    
    Args:
        df: Raw DataFrame with all packets
    
    Returns:
        DataFrame with only HTTP requests
    """
    # Filter rows with HTTP request URIs
    http_df = df[df['http.request.uri'].notna()].copy()
    
    print(f"Extracted {len(http_df)} HTTP requests")
    return http_df


def decode_uri(uri: str) -> str:
    """
    URL-decode the URI to reveal actual payload.
    
    Example: %27 → ' (single quote)
            %20 → space
            %3D → =
    
    Args:
        uri: URL-encoded URI string
    
    Returns:
        Decoded URI string
    """
    try:
        # Decode multiple times to handle double encoding
        decoded = unquote(unquote(str(uri)))
        return decoded.lower()
    except:
        return str(uri).lower()


def clean_http_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HTTP request data.
    
    Steps:
    1. URL-decode the request URIs
    2. Handle missing values
    3. Keep relevant columns
    
    Args:
        df: HTTP requests DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # URL decode the request URIs
    df['decoded_uri'] = df['http.request.uri'].apply(decode_uri)
    
    # Keep relevant columns
    relevant_cols = ['frame.time_epoch', 'ip.src', 'ip.dst', 
                     'http.request.method', 'http.request.uri', 
                     'decoded_uri', '_ws.col.protocol']
    
    available_cols = [col for col in relevant_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['decoded_uri', 'ip.src'], keep='first')
    
    print(f"After cleaning: {len(df)} unique HTTP requests")
    return df


def preprocess_http_pipeline(raw_filepath: str) -> pd.DataFrame:
    """
    Complete HTTP preprocessing pipeline.
    
    Pipeline steps:
    1. Load raw packet data
    2. Extract HTTP requests
    3. Clean and decode URIs
    
    Args:
        raw_filepath: Path to raw CSV file
    
    Returns:
        Cleaned HTTP DataFrame ready for feature extraction
    """
    print("=" * 50)
    print("SQL INJECTION DETECTION - HTTP Preprocessing")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n[1/3] Loading raw packet data...")
    df = load_raw_data(raw_filepath)
    
    # Step 2: Extract HTTP requests
    print("\n[2/3] Extracting HTTP requests...")
    http_df = extract_http_requests(df)
    
    # Step 3: Clean data
    print("\n[3/3] Cleaning and decoding URIs...")
    http_df = clean_http_data(http_df)
    
    print("\n" + "=" * 50)
    print("HTTP Preprocessing complete!")
    print("=" * 50)
    
    return http_df


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    
    if raw_path.exists():
        http_df = preprocess_http_pipeline(str(raw_path))
        print("\nSample decoded URIs:")
        print(http_df['decoded_uri'].head(10))
    else:
        print(f"Raw data file not found: {raw_path}")
