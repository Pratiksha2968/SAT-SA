"""
src/analytics/negative_space.py — Advanced Negative-Space Intelligence
Answers: What should have happened but didn't? (not just null check)

Two complementary lenses:
  1) Workflow gaps  — expected SOC steps missing per severity/asset policy
  2) Activity gaps  — observed_activity << expected_activity

Expected workflow policy (severity-driven, configurable):
  CRITICAL: [Ack, Investigation, Escalation, Resolution, Evidence]
  HIGH:     [Ack, Investigation, Resolution, Evidence]  (Escalation optional)
  MEDIUM:   [Ack, Investigation, Resolution]
  LOW:      [Ack, Resolution]

We distinguish:
  Observed Missing Data (column is null)
vs
  Expected Activity Not Observed (step required by policy but absent) — the supervisory gap

Vectorized, offline, no ML.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/thresholds.yaml")

# Default policy — mirrored in thresholds.yaml under negative_space.workflow
DEFAULT_WORKFLOW = {
    "CRITICAL": ["Ack", "Investigation", "Escalation", "Resolution", "Evidence"],
    "HIGH": ["Ack", "Investigation", "Resolution", "Evidence"],
    "MEDIUM": ["Ack", "Investigation", "Resolution"],
    "LOW": ["Ack", "Resolution"],
}

STEP_TO_COLUMN = {
    "Ack": "acknowledged_time",
    "Investigation": "investigation_present",  # derived bool
    "Escalation": "escalated",                  # required only for CRITICAL
    "Resolution": "closure_time",
    "Evidence": "investigation_present",
}

# Human-readable missing reason per step
STEP_REASON = {
    "Ack": "Alert not acknowledged",
    "Investigation": "Investigation not performed",
    "Escalation": "CRITICAL alert not escalated (expected)",
    "Resolution": "Resolution/closure not recorded",
    "Evidence": "Investigation evidence not recorded",
}


def _load_policy() -> tuple[dict, float]:
    """Load workflow policy and activity threshold from config."""
    workflow = DEFAULT_WORKFLOW
    act_thr = 50.0
    try:
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            ns = cfg.get("negative_space", {})
            if "workflow" in ns:
                # YAML workflow keys are upper severity
                workflow = {k.upper(): v for k, v in ns["workflow"].items()}
            if "activity_threshold" in ns:
                act_thr = float(ns["activity_threshold"])
            else:
                # fallback to preprocessing threshold
                act_thr = float(cfg.get("preprocessing", {}).get("activity_coverage_threshold", 50))
    except Exception:
        pass
    return workflow, act_thr


def analyze_negative_space(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_flagged: input + columns: expected_steps, missing_steps, negative_space, negative_space_reason, gap_count, gap_severity
      gaps_detail: one row per alert with missing steps (for evidence drill-down)
      summary: per-CSE negative-space stats
    """
    if df is None or df.empty:
        return df.copy() if df is not None else df, pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    workflow, act_thr = _load_policy()

    # Ensure derived cols present (safe if caller forgot preprocessing)
    if "investigation_present" not in df.columns:
        if "investigation_evidence" in df.columns:
            df["investigation_present"] = df["investigation_evidence"].fillna(False).astype(bool)
        else:
            df["investigation_present"] = False
    if "activity_coverage" not in df.columns:
        if {"expected_activity", "observed_activity"}.issubset(df.columns):
            exp = pd.to_numeric(df["expected_activity"], errors="coerce")
            obs = pd.to_numeric(df["observed_activity"], errors="coerce")
            df["activity_coverage"] = np.where(exp > 0, obs / exp * 100, np.nan)
            df["activity_coverage"] = pd.Series(df["activity_coverage"]).clip(0, 100)
        else:
            df["activity_coverage"] = np.nan

    # Vectorized missing-step detection
    def _is_present(step: str, row_idx) -> pd.Series:
        col = STEP_TO_COLUMN[step]
        if step in ("Investigation", "Evidence"):
            return df.loc[row_idx, col].fillna(False).astype(bool) if col in df.columns else pd.Series(False, index=row_idx)
        if step == "Escalation":
            return df.loc[row_idx, col].fillna(False).astype(bool) if col in df.columns else pd.Series(False, index=row_idx)
        # Ack / Resolution are timestamp presence
        return df.loc[row_idx, col].notna() if col in df.columns else pd.Series(False, index=row_idx)

    # Apply per-severity expected steps
    # Build arrays of missing steps (loop over severity groups vectorized per group)
    df["expected_steps"] = ""
    df["missing_steps"] = ""
    df["missing_count"] = 0
    df["negative_space"] = False  # preserve legacy boolean for compatibility
    df["negative_space_reason"] = ""
    df["gap_severity"] = "None"  # None / LOW / MEDIUM / HIGH / CRITICAL

    for sev, expected in workflow.items():
        mask = df["severity"].astype(str).str.upper() == sev
        if mask.sum() == 0:
            continue
        idx = df.index[mask]
        df.loc[idx, "expected_steps"] = " → ".join(expected)

        missing_lists = []
        for step in expected:
            present = _is_present(step, idx)
            # For Escalation, only required for CRITICAL per policy already; others skip
            missing = ~present
            missing_lists.append(missing)

        # Combine per-row missing steps
        # Create DataFrame of bools per step for this group
        miss_df = pd.DataFrame({step: missing for step, missing in zip(expected, missing_lists)}, index=idx)
        # Build comma-separated missing steps per row
        missing_str = miss_df.apply(lambda r: ", ".join([s for s, m in r.items() if m]), axis=1)
        df.loc[idx, "missing_steps"] = missing_str
        df.loc[idx, "missing_count"] = miss_df.sum(axis=1)

    # Activity gap additionally counts as negative-space if coverage < threshold
    # We treat it as separate signal but also flags negative_space boolean
    activity_gap_flag = (df["activity_coverage"] < act_thr).fillna(False)
    # If workflow gaps exist OR activity gap, flag
    workflow_gap_flag = df["missing_count"] > 0
    df["negative_space"] = workflow_gap_flag | activity_gap_flag

    # Gap severity scoring: count missing + activity gap + criticality weight
    def _gap_sev(row) -> str:
        cnt = row["missing_count"]
        sev = str(row["severity"]).upper()
        cov = row.get("activity_coverage", 100)
        if cnt == 0 and cov >= act_thr:
            return "None"
        # Critical missing escalation/evidence is higher
        missing = str(row["missing_steps"])
        if ("Escalation" in missing and sev == "CRITICAL") or ("Evidence" in missing and sev == "CRITICAL"):
            return "CRITICAL"
        if cnt >= 3:
            return "HIGH"
        if cnt >= 2 or cov < 30:
            return "MEDIUM"
        if cnt == 1 or cov < act_thr:
            return "LOW"
        return "LOW"

    df["gap_severity"] = df.apply(_gap_sev, axis=1)

    # Reason: combine workflow + activity
    def _reason(row) -> str:
        parts = []
        if row["missing_count"] > 0:
            parts.append(f"Missing: {row['missing_steps']} (expected {row['expected_steps']})")
        if row.get("activity_coverage", 100) < act_thr:
            parts.append(f"Low coverage {row['activity_coverage']:.1f}% < {act_thr:.0f}% (expected {row.get('expected_activity','?')}, observed {row.get('observed_activity','?')})")
        return " | ".join(parts) if parts else ""

    df["negative_space_reason"] = df.apply(_reason, axis=1)

    # Keep legacy column for old risk_scoring compatibility: already set negative_space bool
    # Also preserve activity_coverage already

    # Detail view: one row per gap (only flagged alerts)
    gaps_detail = df[df["negative_space"]].copy()
    # Select evidence columns for drill-down
    keep = ["cse_id", "alert_id", "asset_id", "severity", "asset_criticality", "alert_type",
            "created_time", "acknowledged_time", "investigation_start", "investigation_end",
            "closure_time", "escalated", "investigation_present", "disposition",
            "expected_steps", "missing_steps", "missing_count", "gap_severity",
            "expected_activity", "observed_activity", "activity_coverage", "negative_space_reason"]
    keep = [c for c in keep if c in gaps_detail.columns]
    gaps_detail = gaps_detail[keep].sort_values(["gap_severity", "missing_count"], ascending=[False, False]).reset_index(drop=True)

    # Per-CSE summary
    summary_rows = []
    for cse, grp in df.groupby("cse_id"):
        total = len(grp)
        flagged = int(grp["negative_space"].sum())
        avg_cov = grp["activity_coverage"].mean()
        avg_missing = grp["missing_count"].mean()
        workflow_gaps = int((grp["missing_count"] > 0).sum())
        activity_gaps = int(activity_gap_flag[grp.index].sum())
        sev_counts = grp["gap_severity"].value_counts().to_dict()
        summary_rows.append({
            "cse_id": cse,
            "total_alerts": total,
            "negative_space": flagged,
            "negative_space_rate": round(flagged / total * 100, 2) if total else 0,
            "workflow_gaps": workflow_gaps,
            "activity_gaps": activity_gaps,
            "avg_activity_coverage": round(avg_cov, 2) if pd.notna(avg_cov) else np.nan,
            "avg_missing_steps": round(avg_missing, 2),
            "gap_severity_breakdown": str(sev_counts),
            "critical_gaps": int((grp["gap_severity"] == "CRITICAL").sum()),
            "high_gaps": int((grp["gap_severity"] == "HIGH").sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("negative_space_rate", ascending=False).reset_index(drop=True) if summary_rows else pd.DataFrame()

    return df, gaps_detail, summary
