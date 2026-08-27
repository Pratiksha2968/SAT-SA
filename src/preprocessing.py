"""
src/preprocessing.py — backward-compat shim
Old imports: from src.preprocessing import preprocess_data
New location: src.preprocessing.features
"""
from src.preprocessing.features import preprocess_data, engineer_features

__all__ = ["preprocess_data", "engineer_features"]
