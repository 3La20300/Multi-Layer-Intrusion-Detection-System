"""
SYN Scan Detector
=================

A machine learning project to detect SYN scanning attacks using 
time-window-based feature aggregation.

Key Concept:
- SYN scanning cannot be detected from a single packet
- Detection is based on temporal patterns across multiple packets
- Features are aggregated over 1-second time windows
"""

__version__ = "1.0.0"
__author__ = "Network Security Project Team"
