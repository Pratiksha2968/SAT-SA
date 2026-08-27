"""
src/analytics/process_mining.py — SAT-SA Process Intelligence
Reconstructs observed SOC workflows per alert, aggregates paths,
identifies deviations from expected workflow.

Example:
  Expected CRITICAL: Alert → Ack → Investigation → Escalation → Resolution → Evidence
  Observed path C:  Alert → Ack → Resolution  (10% of cases) → Deviation

Uses only timestamp/flag presence — no heavy mining lib, fully offline, vectorized.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# Canonical step order (Alert is implicit start)
STEP_ORDER = ["Ack", "Investigation", "Escalation", "Resolution", "Evidence"]
STEP_LABEL = {
    "Ack": "Ack",
    "Investigation": "Investigation",
    "Escalation": "Escalation",
    "Resolution": "Resolution",
    "Evidence": "Evidence",
}

EXPECTED_CRITICAL = "Alert → Ack → Investigation → Escalation → Resolution → Evidence"


def _observed_path(row: pd.Series) -> str:
    """Build path string from row's present steps."""
    parts = ["Alert"]
    # Ack
    if pd.notna(row.get("acknowledged_time")):
        parts.append("Ack")
    # Investigation
    if bool(row.get("investigation_present", False)):
        parts.append("Investigation")
    # Escalation
    if bool(row.get("escalated", False)):
        parts.append("Escalation")
    # Resolution
    if pd.notna(row.get("closure_time")):
        parts.append("Resolution")
    # Evidence (same as investigation_present in current schema, but keep distinct label)
    if bool(row.get("investigation_present", False)):
        parts.append("Evidence")
    # Deduplicate consecutive? Investigation+Evidence both present gives ...Investigation→Escalation→Resolution→Evidence
    # If no investigation, path is Alert → Ack → Resolution  (the classic deviation)
    return " → ".join(parts)


def _conforms(path: str, severity: str) -> bool:
    """Simple conformance check: critical must contain Investigation+Escalation+Evidence, high must have Investigation+Evidence."""
    sev = str(severity).upper()
    has = lambda s: s in path
    if sev == "CRITICAL":
        return has("Investigation") and has("Escalation") and has("Evidence")
    if sev == "HIGH":
        return has("Investigation") and has("Evidence")
    if sev == "MEDIUM":
        return has("Investigation")
    # LOW: just needs Ack+Resolution
    return has("Ack") and has("Resolution")


def mine_processes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_with_path: input + observed_path, conforms, missing_steps (recomputed)
      path_stats: global path frequency
      per_cse_path_stats: per-CSE path frequency with conformance flag
      deviations: paths that are deviation (non-conforming & >2% share or critical missing)
    """
    if df is None or df.empty:
        return df.copy() if df is not None else df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.copy()

    # Ensure preprocessing cols
    if "investigation_present" not in df.columns:
        df["investigation_present"] = df.get("investigation_evidence", False).fillna(False).astype(bool) if "investigation_evidence" in df.columns else False

    df["observed_path"] = df.apply(_observed_path, axis=1)
    df["conforms"] = df.apply(lambda r: _conforms(r["observed_path"], r.get("severity", "")), axis=1)

    # Missing steps string for convenience (reuse negative-space logic quickly)
    def _missing(r) -> str:
        expected_order = {
            "CRITICAL": ["Ack", "Investigation", "Escalation", "Resolution", "Evidence"],
            "HIGH": ["Ack", "Investigation", "Resolution", "Evidence"],
            "MEDIUM": ["Ack", "Investigation", "Resolution"],
            "LOW": ["Ack", "Resolution"],
        }.get(str(r.get("severity", "")).upper(), ["Ack", "Resolution"])
        present = set(r["observed_path"].split(" → "))
        missing = [s for s in expected_order if s not in present]
        return ", ".join(missing) if missing else ""

    df["process_missing"] = df.apply(_missing, axis=1)

    # Global path stats
    path_stats = (
        df.groupby("observed_path")
        .agg(count=("alert_id", "count"), cses=("cse_id", "nunique"), avg_resolution=("resolution_time", "mean"))
        .reset_index()
    )
    total = len(df)
    path_stats["share_pct"] = (path_stats["count"] / total * 100).round(2)
    path_stats["conforms_example"] = path_stats["observed_path"].apply(lambda p: _conforms(p, "CRITICAL"))
    # Example conformance is illustrative; per-row conforms is authoritative
    path_stats = path_stats.sort_values("count", ascending=False).reset_index(drop=True)

    # Per-CSE path stats
    per_cse = (
        df.groupby(["cse_id", "observed_path"])
        .agg(count=("alert_id", "count"), conforms_rate=("conforms", "mean"))
        .reset_index()
    )
    # Add share within CSE
    cse_totals = df.groupby("cse_id").size().to_dict()
    per_cse["cse_total"] = per_cse["cse_id"].map(cse_totals)
    per_cse["share_within_cse_pct"] = (per_cse["count"] / per_cse["cse_total"] * 100).round(2)
    per_cse["is_deviation_path"] = ~per_cse["observed_path"].apply(lambda p: _conforms(p, "CRITICAL"))  # flag if not fully conforming for critical
    # But flag only if path is actually present in a critical context — keep simple: deviation if not conforms
    per_cse = per_cse.sort_values(["cse_id", "count"], ascending=[True, False]).reset_index(drop=True)

    # Deviations: paths that are non-conforming and significant
    # Criteria: global share >2% OR per-CSE share >10% and missing critical steps
    deviations = path_stats[(~path_stats["observed_path"].apply(lambda p: _conforms(p, "CRITICAL"))) & (path_stats["share_pct"] > 2)].copy()
    deviations["deviation_reason"] = deviations["observed_path"].apply(
        lambda p: "Missing: " + ", ".join([s for s in ["Investigation", "Escalation", "Evidence"] if s not in p])
    )

    return df, path_stats, per_cse, deviations
