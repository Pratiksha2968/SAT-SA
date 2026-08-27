"""
src/preprocessing/features.py — SAT-SA normalization + supervisory features
Vectorized, no loops. Single entry point: preprocess_data(df).
Produces a normalized internal schema + documented derived columns.

Derived fields (all minutes unless noted):
- acknowledgement_delay   = acknowledged - created
- investigation_delay     = investigation_start - acknowledged
- investigation_duration  = investigation_end - investigation_start
- resolution_time         = closure - created  (aka closure_minutes)
- escalation_delay        = closure - created where escalated else NaN
- rapid_closure           = CRITICAL and resolution_time < threshold
- missing_investigation   = not investigation_evidence
- workflow_completeness   = count of completed steps / 5
- activity_coverage       = observed/expected*100 (clipped 0-100)
- activity_gap            = expected - observed
- after_hours             = created hour not in 09-18
- is_false_positive       = disposition == "False Positive"
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.utils.constants import (
    TIME_COLUMNS,
    BOOLEAN_COLUMNS,
    BOOLEAN_TRUE_SET,
    BOOLEAN_FALSE_SET,
    SEVERITY_ALLOWLIST,
)

CONFIG_PATH = Path("config/thresholds.yaml")


def _load_thresholds() -> dict:
    defaults = {"rapid_closure_minutes": 5, "activity_coverage_threshold": 50, "activity_coverage_clip_max": 100}
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open() as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("preprocessing", defaults)
    except Exception:
        pass
    return defaults


def _normalize_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string booleans safely (avoids bool('False')==True trap)."""
    for col in BOOLEAN_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].dtype == bool:
            continue
        # Map strings case-insensitively
        s = df[col].astype(str).str.strip().str.lower()
        # True where in true set, False where in false set, NaN otherwise → False
        mapped = s.map(lambda x: True if x in BOOLEAN_TRUE_SET else (False if x in BOOLEAN_FALSE_SET else np.nan))
        # If unmapped (<2% expected), coerce to False with warning (validator already flags)
        df[col] = mapped.fillna(False).astype(bool)
    return df


def _normalize_severity(df: pd.DataFrame) -> pd.DataFrame:
    if "severity" in df.columns:
        df["severity"] = df["severity"].astype(str).str.upper().str.strip()
        # Invalid values kept as-is for validator warning, but downstream treats as non-CRITICAL
    return df


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    for col in TIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize + engineer features. Vectorized.
    Never mutates input. Returns new DataFrame with derived columns.
    Invalid rows are kept but flagged (see _invalid_* cols); pipeline won't crash.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    df = df.copy()
    cfg = _load_thresholds()
    rapid_thresh = float(cfg.get("rapid_closure_minutes", 5))
    clip_max = float(cfg.get("activity_coverage_clip_max", 100))

    # 1. Normalize
    df = _parse_times(df)
    df = _normalize_severity(df)
    df = _normalize_booleans(df)

    # 2. Deduplicate — keep first, flag extras (validator already errors, but don't drop silently)
    # We keep all rows; downstream aggregations use alert_id uniqueness via nunique where needed.

    # 3. Derived: time deltas (vectorized)
    # Use .dt.total_seconds()/60 for minutes, preserves NaT as NaN
    if {"acknowledged_time", "created_time"}.issubset(df.columns):
        df["acknowledgement_delay"] = (df["acknowledged_time"] - df["created_time"]).dt.total_seconds() / 60
        # Also keep legacy name for backward compat
        df["acknowledgement_minutes"] = df["acknowledgement_delay"]

    if {"investigation_start", "acknowledged_time"}.issubset(df.columns):
        df["investigation_delay"] = (df["investigation_start"] - df["acknowledged_time"]).dt.total_seconds() / 60

    if {"investigation_end", "investigation_start"}.issubset(df.columns):
        df["investigation_duration"] = (df["investigation_end"] - df["investigation_start"]).dt.total_seconds() / 60
        df["investigation_minutes"] = df["investigation_duration"]  # legacy

    if {"closure_time", "created_time"}.issubset(df.columns):
        df["resolution_time"] = (df["closure_time"] - df["created_time"]).dt.total_seconds() / 60
        df["closure_minutes"] = df["resolution_time"]  # legacy

    if {"closure_time", "created_time"}.issubset(df.columns):
        # Escalation delay only where escalated==True
        esc_delay = (df["closure_time"] - df["created_time"]).dt.total_seconds() / 60
        df["escalation_delay"] = np.where(df.get("escalated", False), esc_delay, np.nan)

    # 4. Flags
    if "severity" in df.columns and "resolution_time" in df.columns:
        df["rapid_closure"] = (df["severity"] == "CRITICAL") & (df["resolution_time"] < rapid_thresh)
    else:
        df["rapid_closure"] = False

    if "investigation_evidence" in df.columns:
        df["missing_investigation"] = ~df["investigation_evidence"].astype(bool)
        df["investigation_present"] = df["investigation_evidence"].astype(bool)
    else:
        df["missing_investigation"] = True
        df["investigation_present"] = False

    # 5. Activity coverage (negative-space core)
    if {"observed_activity", "expected_activity"}.issubset(df.columns):
        exp = pd.to_numeric(df["expected_activity"], errors="coerce")
        obs = pd.to_numeric(df["observed_activity"], errors="coerce")
        # Avoid div/0
        df["activity_coverage"] = np.where(exp > 0, obs / exp * 100, np.nan)
        df["activity_coverage"] = pd.Series(df["activity_coverage"]).clip(lower=0, upper=clip_max)
        df["activity_gap"] = exp - obs
        # Also clip helper for risk scoring
        df["_activity_gap_pct"] = (100 - df["activity_coverage"]).clip(lower=0, upper=100)
    else:
        df["activity_coverage"] = np.nan
        df["activity_gap"] = np.nan

    # 6. Workflow completeness (5 steps: ack, investigate, escalate-if-critical, close, evidence)
    # Each step 0/1 → completeness 0.0-1.0
    steps = pd.DataFrame(index=df.index)
    steps["has_ack"] = df["acknowledged_time"].notna().astype(int) if "acknowledged_time" in df.columns else 0
    steps["has_investigation"] = df["investigation_present"].astype(int) if "investigation_present" in df.columns else 0
    steps["has_escalation_or_not_needed"] = np.where(
        df["severity"] == "CRITICAL",
        df.get("escalated", False).astype(int),
        1,  # non-critical doesn't require escalation
    )
    steps["has_closure"] = df["closure_time"].notna().astype(int) if "closure_time" in df.columns else 0
    steps["has_evidence"] = df["investigation_present"].astype(int) if "investigation_present" in df.columns else 0
    df["workflow_completeness"] = steps.sum(axis=1) / 5.0
    df["workflow_steps_completed"] = steps.sum(axis=1).astype(int)

    # 7. After-hours & false-positive helpers
    if "created_time" in df.columns:
        hrs = df["created_time"].dt.hour
        df["after_hours"] = ~hrs.between(9, 18)
    else:
        df["after_hours"] = False

    if "disposition" in df.columns:
        df["is_false_positive"] = df["disposition"].astype(str).str.strip() == "False Positive"
    else:
        df["is_false_positive"] = False

    # 8. Invalid-record flags (for reporting, not dropping)
    df["_invalid_time_sequence"] = False
    try:
        ct = df["created_time"]
        at = df["acknowledged_time"]
        cl = df["closure_time"]
        bad = (cl < ct) | (at < ct)  # ack before create is also suspicious
        df["_invalid_time_sequence"] = bad.fillna(False)
    except Exception:
        pass

    # 9. Ensure boolean dtypes for downstream
    for col in ["rapid_closure", "missing_investigation", "after_hours", "is_false_positive"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for preprocess_data — for explicit feature-engineering imports."""
    return preprocess_data(df)
