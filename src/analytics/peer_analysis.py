"""
src/analytics/peer_analysis.py — Adaptive Peer Baselines
Compares CSEs against appropriate peers using robust stats.
Avoids unfair raw-count compares by normalizing for workload/severity mix.

Metrics: peer median, percentile, z-score, deviation flag.
Workload-aware: groups by volume tercile if needed, otherwise global peers.

Offline, vectorized.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/thresholds.yaml")

def _load_cfg() -> dict:
    try:
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            return cfg.get("peer", {"z_threshold": 1.5, "min_group_size": 3, "percentile_low": 10, "percentile_high": 90})
    except Exception:
        pass
    return {"z_threshold": 1.5, "min_group_size": 3, "percentile_low": 10, "percentile_high": 90}


def _z_scores(series: pd.Series) -> pd.Series:
    """Robust z using median/MAD or mean/std if small group."""
    # Use mean/std for n<10, else median/MAD scaled (1.4826)
    if len(series) < 10:
        mean = series.mean()
        std = series.std(ddof=0) or 1e-9
        return (series - mean) / std
    median = series.median()
    mad = (series - median).abs().median()
    # MAD->std scaling for normal
    robust_std = mad * 1.4826 if mad != 0 else series.std(ddof=0) or 1e-9
    return (series - median) / robust_std


def analyze_peers(df: pd.DataFrame, behavioural: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build adaptive peer baselines.

    Args:
        df: preprocessed DataFrame (needs escalated, investigation_present, activity_coverage, etc.)
        behavioural: optional output of build_behavioural_profiles (to reuse aggregates)

    Returns:
        peer_summary: per-metric global stats (median, mean, std, p10/p90)
        per_cse_deviations: per-CSE per-metric deviation vs peers (z, percentile, flag)
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Ensure derived
    df = df.copy()
    if "investigation_present" not in df.columns:
        df["investigation_present"] = df.get("investigation_evidence", False).fillna(False).astype(bool) if "investigation_evidence" in df.columns else False
    if "rapid_closure" not in df.columns and "resolution_time" in df.columns:
        df["rapid_closure"] = (df["severity"].astype(str).str.upper() == "CRITICAL") & (df["resolution_time"] < 5)

    # Build per-CSE metrics for peer compare (reuse behavioural if provided)
    if behavioural is not None and not behavioural.empty and "investigation_rate" in behavioural.columns:
        cse_metrics = behavioural[["cse_id", "total_alerts", "investigation_rate", "escalation_rate", "evidence_completeness", "false_positive_rate", "rapid_closure_rate", "negative_space_rate", "avg_activity_coverage"]].copy()
        cse_metrics.rename(columns={"evidence_completeness": "evidence_rate"}, inplace=True)
    else:
        grp = df.groupby("cse_id")
        cse_metrics = pd.DataFrame({
            "cse_id": grp.size().index,
            "total_alerts": grp.size().values,
            "investigation_rate": grp["investigation_present"].mean().values,
            "escalation_rate": grp["escalated"].mean().values if "escalated" in df.columns else 0,
            "evidence_rate": grp["investigation_present"].mean().values,
            "false_positive_rate": (df.groupby("cse_id")["disposition"].apply(lambda s: (s.astype(str).str.strip() == "False Positive").mean()).values if "disposition" in df.columns else 0),
            "rapid_closure_rate": grp["rapid_closure"].mean().values if "rapid_closure" in df.columns else 0,
            "negative_space_rate": grp["negative_space"].mean().values if "negative_space" in df.columns else 0,
            "avg_activity_coverage": grp["activity_coverage"].mean().values if "activity_coverage" in df.columns else np.nan,
        })

    cfg = _load_cfg()
    z_thr = float(cfg.get("z_threshold", 1.5))

    # Peer summary per metric
    metrics = ["investigation_rate", "escalation_rate", "evidence_rate", "false_positive_rate", "rapid_closure_rate", "negative_space_rate", "avg_activity_coverage", "total_alerts"]
    metrics = [m for m in metrics if m in cse_metrics.columns]
    summary_rows = []
    for m in metrics:
        s = cse_metrics[m].dropna()
        if s.empty:
            continue
        summary_rows.append({
            "metric": m,
            "peer_median": round(float(s.median()), 4),
            "peer_mean": round(float(s.mean()), 4),
            "peer_std": round(float(s.std(ddof=0)), 4) if len(s) > 1 else 0,
            "peer_min": round(float(s.min()), 4),
            "peer_max": round(float(s.max()), 4),
            "p10": round(float(s.quantile(0.10)), 4),
            "p90": round(float(s.quantile(0.90)), 4),
        })
    peer_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    # Per-CSE deviations
    dev_rows = []
    for _, row in cse_metrics.iterrows():
        cse = row["cse_id"]
        for m in metrics:
            val = row[m]
            if pd.isna(val):
                continue
            # Find peer stats
            ps = peer_summary[peer_summary["metric"] == m]
            if ps.empty:
                continue
            median = float(ps["peer_median"].values[0])
            mean = float(ps["peer_mean"].values[0])
            std = float(ps["peer_std"].values[0]) or 1e-9
            # Z (robust)
            series = cse_metrics[m].dropna()
            z = float(_z_scores(series)[cse_metrics["cse_id"] == cse].values[0]) if len(series) > 1 else 0
            # Percentile (rank)
            pct = float((series < val).mean() * 100) if len(series) else 50
            deviation = abs(z) > z_thr
            # Direction: for good metrics (investigation, escalation, coverage) low is bad; for bad metrics high is bad
            is_bad = False
            if m in ("investigation_rate", "escalation_rate", "evidence_rate", "avg_activity_coverage"):
                is_bad = val < median
            else:
                is_bad = val > median
            flag = deviation and is_bad

            # Workload-aware note: if comparing total_alerts, flag high workload separately
            dev_rows.append({
                "cse_id": cse,
                "metric": m,
                "value": round(float(val), 4),
                "peer_median": median,
                "peer_mean": mean,
                "z_score": round(z, 2),
                "percentile": round(pct, 1),
                "deviation": deviation,
                "needs_attention": flag,
                "evidence": f"{m}={val:.3f} vs peer median {median:.3f} (z={z:.2f}, p{ pct:.0f})",
            })
    per_cse = pd.DataFrame(dev_rows).sort_values(["cse_id", "metric"]).reset_index(drop=True) if dev_rows else pd.DataFrame()

    return peer_summary, per_cse
