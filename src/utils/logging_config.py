"""
src/utils/logging_config.py — structured logging + audit trail
Offline, no external services. Writes JSONL audit records to logs/.
"""
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def get_logger(name: str = "satsa") -> logging.Logger:
    """Return a configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def dataset_fingerprint(df) -> str:
    """Short hash of dataset for audit (first 12 hex chars)."""
    try:
        # Use shape + column hash + first/last alert_id for stable fingerprint
        raw = f"{df.shape}:{list(df.columns)}:{df['alert_id'].iloc[0] if len(df) else ''}:{df['alert_id'].iloc[-1] if len(df) else ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def audit_record(
    dataset_path: str | None,
    df,
    validation_result: dict | None = None,
    config_version: str = "thresholds.yaml",
    extra: dict[str, Any] | None = None,
) -> dict:
    """Build a serializable audit record."""
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path or "in-memory/upload",
        "dataset_fingerprint": dataset_fingerprint(df) if df is not None else "none",
        "rows": int(len(df)) if df is not None else 0,
        "cses": int(df["cse_id"].nunique()) if df is not None and "cse_id" in df.columns else 0,
        "config_version": config_version,
        "validation": validation_result or {},
    }
    if extra:
        rec.update(extra)
    return rec


def write_audit(record: dict, log_dir: str = "logs") -> Path | None:
    """Append audit record as JSONL. Fails silently (audit must not crash app)."""
    try:
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / "audit.jsonl"
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return out
    except Exception:
        return None
