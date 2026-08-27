"""
src/ingestion/loader.py — safe CSV/JSON ingestion
- Enforces size/row limits from config
- Validates after load (via validator)
- Supports file paths and Streamlit UploadedFile (file-like)
- Returns DataFrame + ValidationResult + audit dict
"""
from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Tuple, BinaryIO
import pandas as pd
import yaml

from src.ingestion.validator import validate_dataset, ValidationResult
from src.utils.constants import MAX_FILE_MB, MAX_ROWS
from src.utils.logging_config import get_logger

logger = get_logger("satsa.ingestion")

CONFIG_PATH = Path("config/thresholds.yaml")


def _load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open() as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Config load failed, using defaults: {e}")
    return {}


def _check_limits(file_bytes: bytes | None, df: pd.DataFrame | None = None) -> list:
    issues = []
    cfg = _load_config()
    max_mb = cfg.get("ingestion", {}).get("max_file_mb", MAX_FILE_MB)
    max_rows = cfg.get("ingestion", {}).get("max_rows", MAX_ROWS)
    if file_bytes is not None and len(file_bytes) > max_mb * 1024 * 1024:
        issues.append(f"File size {len(file_bytes)/1024/1024:.1f} MB exceeds limit {max_mb} MB")
    if df is not None and len(df) > max_rows:
        issues.append(f"Row count {len(df)} exceeds limit {max_rows}")
    return issues


def load_csv(source: str | Path | BinaryIO, **read_kwargs) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Load CSV from path or file-like object.
    Safe: limits, dtype handling, no code execution.
    """
    # Read raw bytes first for size check if file-like
    file_bytes = None
    df: pd.DataFrame
    try:
        if hasattr(source, "read"):
            # Streamlit UploadedFile — read bytes then parse
            pos = source.tell() if hasattr(source, "tell") else 0
            file_bytes = source.read() if isinstance(source.read(0), (bytes, str)) else None
            # Handle str vs bytes
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode()
            if file_bytes is not None:
                limit_issues = _check_limits(file_bytes)
                if limit_issues:
                    raise ValueError("; ".join(limit_issues))
                df = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)
            else:
                # fallback: reset and read directly
                if hasattr(source, "seek"):
                    source.seek(pos)
                df = pd.read_csv(source, **read_kwargs)
        else:
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            # Size check via filesystem
            size = p.stat().st_size
            limit_issues = _check_limits(b" " * size if size else None)
            if limit_issues:
                raise ValueError("; ".join(limit_issues))
            df = pd.read_csv(p, **read_kwargs)

        # Row limit
        row_issues = _check_limits(None, df)
        if row_issues:
            raise ValueError("; ".join(row_issues))

        logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} cols")

    except Exception as e:
        logger.error(f"CSV load failed: {e}")
        raise

    v = validate_dataset(df)
    return df, v


def load_json(source: str | Path | BinaryIO, **read_kwargs) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load JSON (list of records) into DataFrame."""
    try:
        if hasattr(source, "read"):
            raw = source.read()
            if isinstance(raw, bytes):
                raw = raw.decode()
            data = json.loads(raw)
        else:
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            with p.open() as f:
                data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"Loaded JSON: {len(df)} rows")
    except Exception as e:
        logger.error(f"JSON load failed: {e}")
        raise

    _check_limits(None, df)
    v = validate_dataset(df)
    return df, v


def load_dataset(source: str | Path | BinaryIO, file_type: str | None = None, **kwargs) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Unified loader — infers type from extension if not given.
    Supports .csv and .json. For Streamlit uploads, pass file_type="csv".
    """
    # Streamlit UploadedFile has .name
    name = getattr(source, "name", str(source) if isinstance(source, (str, Path)) else "")
    ext = file_type or (Path(name).suffix.lower().lstrip(".") if name else "csv")
    if ext == "json":
        return load_json(source, **kwargs)
    else:
        return load_csv(source, **kwargs)


# Backward-compat shim: old app.py imports load_soc_data
def load_soc_data(file_path: str | Path) -> pd.DataFrame:
    """Legacy helper — returns DataFrame only (logs validation warnings)."""
    df, v = load_csv(file_path)
    if not v.is_valid:
        logger.warning(f"Validation errors on load: {v.summary_text()}")
    elif v.warnings:
        logger.info(f"Validation warnings: {v.summary_text()}")
    return df
