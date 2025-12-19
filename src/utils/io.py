"""
I/O Utility Functions
=====================

Helper functions for file I/O operations.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False):
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        index: Whether to save index
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    print(f"Saved to: {filepath}")


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def ensure_dir(path: str):
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
