"""
src/data_loader.py — backward-compat shim
Old imports: from src.data_loader import load_soc_data
New location: src.ingestion.loader
This shim re-exports to keep existing app/tests working.
"""
from src.ingestion.loader import load_soc_data, load_csv, load_dataset, load_json

__all__ = ["load_soc_data", "load_csv", "load_dataset", "load_json"]
