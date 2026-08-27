"""
src/peer_comparison.py — backward-compat shim
Old: from src.peer_comparison import ...
New: from src.analytics.peer_analysis import analyze_peers
"""
from src.analytics.peer_analysis import analyze_peers

__all__ = ["analyze_peers"]
