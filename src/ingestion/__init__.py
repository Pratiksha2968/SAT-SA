"""src/ingestion — CSV/JSON intake + validation."""
from .loader import load_dataset, load_csv, load_json
from .validator import validate_dataset, ValidationResult

__all__ = ["load_dataset", "load_csv", "load_json", "validate_dataset", "ValidationResult"]
