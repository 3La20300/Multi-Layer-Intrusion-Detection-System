"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocess import (
    clean_data, 
    create_time_windows, 
    aggregate_by_time_window
)


class TestCleanData:
    """Tests for clean_data function."""
    
    def test_converts_boolean_flags(self):
        """Test that string boolean values are converted to actual booleans."""
        df = pd.DataFrame({
            'frame.time_epoch': [1.0, 2.0],
            'ip.src': ['192.168.1.1', '192.168.1.2'],
            'ip.dst': ['10.0.0.1', '10.0.0.2'],
            'tcp.flags.syn': ['True', 'False'],
            'tcp.flags.ack': ['False', 'True']
        })
        
        result = clean_data(df)
        
        assert result['tcp.flags.syn'].dtype == bool
        assert result['tcp.flags.syn'].iloc[0] == True
        assert result['tcp.flags.syn'].iloc[1] == False
    
    def test_drops_missing_critical_values(self):
        """Test that rows with missing critical values are dropped."""
        df = pd.DataFrame({
            'frame.time_epoch': [1.0, None, 3.0],
            'ip.src': ['192.168.1.1', '192.168.1.2', None],
            'ip.dst': ['10.0.0.1', '10.0.0.2', '10.0.0.3']
        })
        
        result = clean_data(df)
        
        # Only first row should remain (has all critical values)
        assert len(result) == 1


class TestCreateTimeWindows:
    """Tests for create_time_windows function."""
    
    def test_creates_time_window_column(self):
        """Test that time_window column is created."""
        df = pd.DataFrame({
            'frame.time_epoch': [1.5, 1.8, 2.1, 2.9, 3.5]
        })
        
        result = create_time_windows(df, window_seconds=1)
        
        assert 'time_window' in result.columns
        assert list(result['time_window']) == [1, 1, 2, 2, 3]


class TestAggregateByTimeWindow:
    """Tests for aggregate_by_time_window function."""
    
    def test_aggregates_syn_count(self):
        """Test that SYN packets are correctly counted."""
        df = pd.DataFrame({
            'ip.src': ['A', 'A', 'A'],
            'ip.dst': ['B', 'B', 'B'],
            'time_window': [1, 1, 1],
            'tcp.flags.syn': [True, True, False],
            'tcp.flags.ack': [False, False, True],
            'tcp.flags.reset': [False, False, False],
            'tcp.flags.fin': [False, False, False],
            'tcp.srcport': [1000, 1001, 1002],
            'tcp.dstport': [80, 22, 443],
            'frame.len': [100, 100, 100],
            'frame.time_delta': [0.1, 0.1, 0.1],
            'frame.time_epoch': [1.0, 1.1, 1.2]
        })
        
        result = aggregate_by_time_window(df, min_packets=1)
        
        assert len(result) == 1
        assert result['syn_count'].iloc[0] == 2
        assert result['ack_count'].iloc[0] == 1
        assert result['unique_dst_ports'].iloc[0] == 3
    
    def test_filters_by_min_packets(self):
        """Test that windows with few packets are filtered out."""
        df = pd.DataFrame({
            'ip.src': ['A', 'B', 'B', 'B'],
            'ip.dst': ['X', 'Y', 'Y', 'Y'],
            'time_window': [1, 2, 2, 2],
            'tcp.flags.syn': [True, True, True, True],
            'tcp.flags.ack': [False, False, False, False],
            'tcp.flags.reset': [False, False, False, False],
            'tcp.flags.fin': [False, False, False, False],
            'tcp.srcport': [1000, 1001, 1002, 1003],
            'tcp.dstport': [80, 22, 23, 24],
            'frame.len': [100, 100, 100, 100],
            'frame.time_delta': [0.1, 0.1, 0.1, 0.1],
            'frame.time_epoch': [1.0, 2.0, 2.1, 2.2]
        })
        
        result = aggregate_by_time_window(df, min_packets=2)
        
        # Only second window has 3 packets >= min_packets(2)
        assert len(result) == 1
        assert result['ip.src'].iloc[0] == 'B'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
