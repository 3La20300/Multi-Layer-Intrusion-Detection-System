"""
Data Preprocessing Module
=========================

This module handles loading and cleaning raw packet data from CSV files.
It prepares the data for time-window aggregation.

Key Concept:
    SYN scanning cannot be detected from a single packet.
    Raw packets must be aggregated into time windows to detect patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_raw_data(filepath: str, on_bad_lines: str = 'skip') -> pd.DataFrame:
    """
    Load raw packet data from CSV file.
    
    Args:
        filepath: Path to the raw CSV file
        on_bad_lines: How to handle malformed lines ('skip', 'error', 'warn')
    
    Returns:
        DataFrame with raw packet data
    """
    df = pd.read_csv(filepath, on_bad_lines=on_bad_lines)
    print(f"Loaded {len(df)} packets from {filepath}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw packet data.
    
    Steps:
    1. Convert boolean flag columns (True/False strings to actual booleans)
    2. Handle missing values
    3. Ensure correct data types
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Convert string True/False to boolean for flag columns
    flag_columns = ['tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.fin', 'tcp.flags.reset']
    for col in flag_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower() == 'true'
    
    # Convert numeric columns
    numeric_columns = ['frame.time_epoch', 'frame.time_delta', 'tcp.srcport', 
                       'tcp.dstport', 'tcp.len', 'frame.len']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical values
    critical_columns = ['frame.time_epoch', 'ip.src', 'ip.dst']
    df = df.dropna(subset=[col for col in critical_columns if col in df.columns])
    
    print(f"After cleaning: {len(df)} packets")
    return df


def create_time_windows(df: pd.DataFrame, window_seconds: int = 1) -> pd.DataFrame:
    """
    Create time window identifier for each packet.
    
    This groups packets into fixed time windows (e.g., 1-second intervals)
    which is essential for detecting SYN scanning patterns.
    
    Args:
        df: Cleaned DataFrame with packet data
        window_seconds: Size of time window in seconds
    
    Returns:
        DataFrame with time_window column added
    """
    df = df.copy()
    
    # Create time window by truncating epoch time
    df['time_window'] = (df['frame.time_epoch'] // window_seconds).astype(int)
    
    return df


def aggregate_by_time_window(df: pd.DataFrame, 
                             min_packets: int = 2) -> pd.DataFrame:
    """
    Aggregate packets by (src_ip, dst_ip, time_window).
    
    This is the CORE of SYN scan detection:
    - Group packets into time windows
    - Calculate aggregated features that reveal scanning patterns
    
    Features extracted:
    - syn_count: Number of SYN packets (high in SYN scan)
    - ack_count: Number of ACK packets (low in SYN scan)
    - rst_count: Number of RST packets (closed port indicator)
    - fin_count: Number of FIN packets
    - unique_dst_ports: Number of unique destination ports (high in port scan)
    - unique_src_ports: Number of unique source ports
    - packet_count: Total packets in window
    - total_bytes: Total data transferred
    - mean_time_delta: Average time between packets
    - syn_ack_ratio: SYN/ACK ratio (high indicates incomplete connections)
    
    Args:
        df: DataFrame with time_window column
        min_packets: Minimum packets required in a window
    
    Returns:
        Aggregated DataFrame with one row per (src, dst, time_window)
    """
    # Group by source IP, destination IP, and time window
    grouped = df.groupby(['ip.src', 'ip.dst', 'time_window'])
    
    # Aggregate features
    aggregated = grouped.agg(
        # Flag counts
        syn_count=('tcp.flags.syn', 'sum'),
        ack_count=('tcp.flags.ack', 'sum'),
        rst_count=('tcp.flags.reset', 'sum'),
        fin_count=('tcp.flags.fin', 'sum'),
        
        # Port analysis
        unique_dst_ports=('tcp.dstport', 'nunique'),
        unique_src_ports=('tcp.srcport', 'nunique'),
        
        # Traffic volume
        packet_count=('frame.len', 'count'),
        total_bytes=('frame.len', 'sum'),
        mean_packet_size=('frame.len', 'mean'),
        
        # Timing analysis
        mean_time_delta=('frame.time_delta', 'mean'),
        min_time_delta=('frame.time_delta', 'min'),
        max_time_delta=('frame.time_delta', 'max'),
        
        # Window timing
        window_start=('frame.time_epoch', 'min'),
        window_end=('frame.time_epoch', 'max')
    ).reset_index() # Reset index to turn groupby keys into columns
    
    # Calculate derived features
    # SYN/ACK ratio - high ratio indicates incomplete connections (SYN scan pattern)
    aggregated['syn_ack_ratio'] = aggregated['syn_count'] / (aggregated['ack_count'] + 1)
    
    # SYN percentage of total packets
    aggregated['syn_percentage'] = aggregated['syn_count'] / aggregated['packet_count']
    
    # Connection completion indicator (low = SYN scan) , 1 syn: 1 ack/fin to complete
    aggregated['completion_ratio'] = (aggregated['ack_count'] + aggregated['fin_count']) / (aggregated['syn_count'] + 1)
    
    # Ports per packet (high = port scanning)
    aggregated['ports_per_packet'] = aggregated['unique_dst_ports'] / (aggregated['packet_count'] + 1)
    
    # RST ratio (high = many closed ports probed)
    aggregated['rst_ratio'] = aggregated['rst_count'] / (aggregated['packet_count'] ) 
    
    # Window duration
    aggregated['window_duration'] = aggregated['window_end'] - aggregated['window_start']
    
    # Packets per second (burst indicator)
    aggregated['packets_per_second'] = aggregated['packet_count'] / (aggregated['window_duration'] + 0.001)
    
    # Filter windows with minimum packets
    aggregated = aggregated[aggregated['packet_count'] >= min_packets]
    
    print(f"Created {len(aggregated)} aggregated time windows")
    return aggregated


def preprocess_pipeline(raw_filepath: str, 
                       window_seconds: int = 1,
                       min_packets: int = 2) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Pipeline steps:
    1. Load raw packet data
    2. Clean and validate data
    3. Create time windows
    4. Aggregate features by time window
    
    Args:
        raw_filepath: Path to raw CSV file
        window_seconds: Time window size in seconds
        min_packets: Minimum packets per window
    
    Returns:
        Aggregated DataFrame ready for feature engineering
    """
    print("=" * 50)
    print("SYN SCAN DETECTION - Data Preprocessing")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n[1/4] Loading raw packet data...")
    df = load_raw_data(raw_filepath)
    
    # Step 2: Clean data
    print("\n[2/4] Cleaning data...")
    df = clean_data(df)
    
    # Step 3: Create time windows
    print(f"\n[3/4] Creating {window_seconds}-second time windows...")
    df = create_time_windows(df, window_seconds)
    
    # Step 4: Aggregate
    print("\n[4/4] Aggregating features by time window...")
    aggregated = aggregate_by_time_window(df, min_packets)
    
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"Raw packets: {len(df)} → Aggregated windows: {len(aggregated)}")
    print("=" * 50)
    
    return aggregated


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / "data" / "raw" / "project_features_raw9.0.csv"
    
    if raw_path.exists():
        aggregated = preprocess_pipeline(str(raw_path))
        print("\nAggregated features sample:")
        print(aggregated.head())
        print("\nFeature columns:")
        print(aggregated.columns.tolist())
    else:
        print(f"Raw data file not found: {raw_path}")
