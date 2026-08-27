"""
src/analytics/behavioural.py — CSE Digital Profile
Persistent behavioural fingerprint per CSE: workload, handling, investigation,
escalation, evidence, closure, execution gaps, negative-space, peer context.

Uses neutral supervisory terminology:
  Normal | Review Recommended | Supervisory Attention Required | High Priority Review
Never labels as malicious/dishonest.

Vectorized, deterministic, offline.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# Thresholds for profile labels (can be moved to YAML later)
WORKLOAD_BANDS = {"LOW": 300, "MEDIUM": 700}  # alerts per period
INVEST_QUALITY = {"HIGH": 0.85, "MEDIUM": 0.60}
ESCALATION_QUALITY = {"HIGH": 0.18, "MEDIUM": 0.10}
EVIDENCE_QUALITY = {"HIGH": 0.85, "MEDIUM": 0.60}
CLOSURE_FAST_RATE_HIGH = 0.10  # >10% rapid = anomalous


def _label_rate(rate: float, high_thr: float, med_thr: float) -> str:
    if rate >= high_thr:
        return "HIGH"
    if rate >= med_thr:
        return "MEDIUM"
    return "LOW"


def _overall_label(row: pd.Series) -> str:
    """Map aggregate signals to supervisory terminology."""
    signals = 0
    if row.get("investigation_rate", 1) < 0.60:
        signals += 1
    if row.get("escalation_rate", 1) < 0.08:
        signals += 1
    if row.get("evidence_completeness", 1) < 0.60:
        signals += 1
    if row.get("execution_gap_rate", 0) > 0.08:
        signals += 1
    if row.get("negative_space_rate", 0) > 0.40:
        signals += 1
    if row.get("rapid_closure_rate", 0) > 0.10:
        signals += 1
    if signals >= 4:
        return "High Priority Review"
    if signals >= 2:
        return "Supervisory Attention Required"
    if signals >= 1:
        return "Review Recommended"
    return "Normal"


def build_behavioural_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-CSE behavioural profile.

    Expects preprocessed df with: severity, investigation_present, escalated,
    resolution_time, rapid_closure, negative_space, activity_coverage, disposition.
    Returns DataFrame with one row per CSE, measurable features + labels.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure cols
    if "investigation_present" not in df.columns:
        df = df.copy()
        df["investigation_present"] = df.get("investigation_evidence", False).fillna(False).astype(bool) if "investigation_evidence" in df.columns else False
    if "rapid_closure" not in df.columns and "resolution_time" in df.columns:
        df = df.copy()
        df["rapid_closure"] = (df["severity"].astype(str).str.upper() == "CRITICAL") & (df["resolution_time"] < 5)
    if "is_false_positive" not in df.columns:
        df = df.copy()
        df["is_false_positive"] = df.get("disposition", "").astype(str).str.strip() == "False Positive"

    # Per-CSE aggregates (vectorized groupby)
    grp = df.groupby("cse_id")
    profile = pd.DataFrame({
        "total_alerts": grp.size(),
        "critical_alerts": grp.apply(lambda g: (g["severity"].astype(str).str.upper() == "CRITICAL").sum(), include_groups=False),
        "critical_rate": grp.apply(lambda g: (g["severity"].astype(str).str.upper() == "CRITICAL").mean(), include_groups=False),
        "avg_resolution_time": grp["resolution_time"].mean() if "resolution_time" in df.columns else np.nan,
        "median_resolution_time": grp["resolution_time"].median() if "resolution_time" in df.columns else np.nan,
        "investigation_rate": grp["investigation_present"].mean(),
        "escalation_rate": grp["escalated"].mean() if "escalated" in df.columns else 0,
        "evidence_completeness": grp["investigation_present"].mean(),  # same as investigation in current schema
        "false_positive_rate": grp["is_false_positive"].mean(),
        "rapid_closure_rate": grp["rapid_closure"].mean() if "rapid_closure" in df.columns else 0,
        "execution_gap_rate": grp["rapid_closure"].mean() if "rapid_closure" in df.columns else 0,  # proxy until rules integrated
        "negative_space_rate": grp["negative_space"].mean() if "negative_space" in df.columns else 0,
        "avg_activity_coverage": grp["activity_coverage"].mean() if "activity_coverage" in df.columns else np.nan,
        "avg_workflow_completeness": grp["workflow_completeness"].mean() if "workflow_completeness" in df.columns else np.nan,
        "after_hours_rate": grp["after_hours"].mean() if "after_hours" in df.columns else 0,
    }).reset_index()

    # Derived labels
    profile["workload_label"] = profile["total_alerts"].apply(
        lambda n: "HIGH" if n > WORKLOAD_BANDS["MEDIUM"] else ("MEDIUM" if n > WORKLOAD_BANDS["LOW"] else "LOW")
    )
    profile["investigation_quality"] = profile["investigation_rate"].apply(lambda r: _label_rate(r, INVEST_QUALITY["HIGH"], INVEST_QUALITY["MEDIUM"]))
    profile["escalation_behaviour"] = profile["escalation_rate"].apply(lambda r: _label_rate(r, ESCALATION_QUALITY["HIGH"], ESCALATION_QUALITY["MEDIUM"]))
    profile["evidence_label"] = profile["evidence_completeness"].apply(lambda r: _label_rate(r, EVIDENCE_QUALITY["HIGH"], EVIDENCE_QUALITY["MEDIUM"]))
    profile["closure_behaviour"] = np.where(profile["rapid_closure_rate"] > CLOSURE_FAST_RATE_HIGH, "ANOMALOUS", "Normal")
    profile["profile_status"] = profile.apply(_overall_label, axis=1)

    # Round for display
    for col in ["critical_rate", "investigation_rate", "escalation_rate", "evidence_completeness", "false_positive_rate", "rapid_closure_rate", "execution_gap_rate", "negative_space_rate", "avg_activity_coverage"]:
        if col in profile.columns:
            profile[col] = profile[col].round(3)

    if "avg_resolution_time" in profile.columns:
        profile["avg_resolution_time"] = profile["avg_resolution_time"].round(1)
        profile["median_resolution_time"] = profile["median_resolution_time"].round(1)

    profile = profile.sort_values("profile_status", key=lambda s: s.map({"High Priority Review": 0, "Supervisory Attention Required": 1, "Review Recommended": 2, "Normal": 3})).reset_index(drop=True)

    return profile
